"""Actual SQLite persistence, namespace fencing, failure visibility and pools."""

from datetime import datetime

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError

from truthound.stores.backends.database import DatabaseStore, PoolingConfig
from truthound.stores.base import StoreQuery, StoreReadError, StoreWriteError
from truthound.stores.results import ResultStatus, ValidationResult

pytestmark = pytest.mark.fault


def record(identity="same", *, asset="synthetic", tags=None):
    return ValidationResult(
        run_id=identity,
        data_asset=asset,
        run_time=datetime(2026, 1, 1),
        status=ResultStatus.FAILURE,
        tags=tags or {},
    )


@pytest.mark.parametrize("pooled", [False, True])
def test_same_run_id_in_distinct_namespaces_cannot_replace_or_delete_other_owner(tmp_path, pooled):
    url = "sqlite:///" + str(tmp_path / "namespaces.db")
    first = DatabaseStore(url, namespace="first", use_pool_manager=pooled)
    second = DatabaseStore(url, namespace="second", use_pool_manager=pooled)
    try:
        first.save(record(asset="first-private-asset"))
        second.save(record(asset="second-private-asset"))
        assert first.get("same").data_asset == "first-private-asset"
        assert second.get("same").data_asset == "second-private-asset"
        assert first.count() == second.count() == 1
        second.delete("same")
        assert first.exists("same") and not second.exists("same")
    finally:
        first.close()
        second.close()


def test_legacy_global_unique_key_rejects_collision_without_rewriting_schema_or_owner(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE validation_results (
            id INTEGER PRIMARY KEY, run_id VARCHAR(255) NOT NULL UNIQUE,
            data_asset VARCHAR(500) NOT NULL, run_time DATETIME NOT NULL,
            status VARCHAR(50) NOT NULL, namespace VARCHAR(255) NOT NULL,
            tags_json TEXT, data_json TEXT, created_at DATETIME, updated_at DATETIME
        )""")
        original_schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='validation_results'"
        ).fetchone()[0]
    url = "sqlite:///" + str(path)
    first, second = DatabaseStore(url, namespace="first"), DatabaseStore(url, namespace="second")
    try:
        first.save(record(asset="first-private-asset"))
        with pytest.raises(StoreWriteError):
            second.save(record(asset="second-private-asset"))
        assert first.get("same").data_asset == "first-private-asset"
        assert not second.exists("same")
        with sqlite3.connect(path) as connection:
            assert (
                connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name='validation_results'"
                ).fetchone()[0]
                == original_schema
            )
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("pooled", [False, True])
def test_file_database_reopens_after_owned_pool_is_closed(tmp_path, pooled):
    store = DatabaseStore("sqlite:///" + str(tmp_path / "reopen.db"), use_pool_manager=pooled)
    try:
        store.save(record())
        previous = store._engine
        store.close()
        assert store.get("same").run_id == "same"
        assert store._engine is not previous
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["exists", "list_ids", "query", "count"])
def test_sql_failure_is_not_an_empty_success(tmp_path, operation):
    store = DatabaseStore(
        "sqlite:///" + str(tmp_path / "fault.db"), pooling=PoolingConfig(enable_health_checks=False)
    )
    store.save(record())

    def failed(*args):
        raise OperationalError("synthetic", {}, RuntimeError("synthetic-private-db-detail"))

    event.listen(store._engine, "before_cursor_execute", failed)
    try:
        with pytest.raises(StoreReadError) as error:
            if operation == "exists":
                store.exists("same")
            elif operation == "query":
                store.query(StoreQuery())
            else:
                getattr(store, operation)()
        assert "synthetic-private-db-detail" not in str(error.value)
    finally:
        event.remove(store._engine, "before_cursor_execute", failed)
        store.close()


def test_query_rejects_corrupt_record_instead_of_skipping_it(tmp_path):
    store = DatabaseStore("sqlite:///" + str(tmp_path / "corrupt.db"))
    try:
        store.save(record())
        with store._engine.begin() as connection:
            connection.execute(
                text("UPDATE validation_results SET data_json=:value"),
                {"value": "invalid synthetic JSON"},
            )
        with pytest.raises(StoreReadError):
            store.query(StoreQuery())
    finally:
        store.close()


def test_tags_sorting_and_pagination_are_applied_to_matching_records(tmp_path):
    store = DatabaseStore("sqlite:///" + str(tmp_path / "filters.db"))
    try:
        store.save(record("a", asset="z", tags={"selected": "yes"}))
        store.save(record("b", asset="a", tags={"selected": "no"}))
        store.save(record("c", asset="b", tags={"selected": "yes"}))
        query = StoreQuery(
            tags={"selected": "yes"}, order_by="data_asset", ascending=True, offset=1, limit=1
        )
        assert store.list_ids(query) == ["a"]
        assert [item.run_id for item in store.query(query)] == ["a"]
        assert store.count(StoreQuery(tags={"selected": "no"})) == 1
        assert store.query(StoreQuery(limit=0)) == []
    finally:
        store.close()


def test_initial_pool_connection_and_recycling_are_counted(tmp_path):
    store = DatabaseStore(
        "sqlite:///" + str(tmp_path / "metrics.db"),
        pooling=PoolingConfig(enable_health_checks=False),
    )
    try:
        store.save(record())
        assert store.pool_metrics.total_created == 1
        assert store.pool_metrics.current_size == 1
        assert store.recycle_connections() == 1
        assert store.get("same").run_id == "same"
        assert store.pool_metrics.total_created == 2
    finally:
        store.close()


def test_failed_session_cleanup_is_visible_and_retains_pool_for_retry(tmp_path):
    store = DatabaseStore(
        "sqlite:///" + str(tmp_path / "close-failure.db"),
        pooling=PoolingConfig(enable_health_checks=False),
    )
    store.initialize()
    manager = store._pool_manager
    first, second = manager.get_session(), manager.get_session()
    first.execute(text("SELECT 1"))
    native_close = first.close
    closed = []
    native_second_close = second.close

    def fail_close():
        raise RuntimeError("synthetic-private-session-close")

    def second_close():
        native_second_close()
        closed.append("second")

    first.close, second.close = fail_close, second_close
    try:
        with pytest.raises(ExceptionGroup):
            store.close()
        assert closed == ["second"]
        assert store._pool_manager is manager and not manager._disposed
        with pytest.raises(RuntimeError, match="cleanup"):
            manager.get_session()
        first.close = native_close
        store.close()
        assert store._pool_manager is None and manager._disposed
    finally:
        first.close = native_close
        store.close()


@pytest.mark.parametrize("pooled", [False, True])
def test_table_prefix_isolates_schema_and_records_without_touching_default_table(tmp_path, pooled):
    from sqlalchemy import inspect

    url = "sqlite:///" + str(tmp_path / "prefixes.db")
    stores = [
        DatabaseStore(url, namespace="shared", table_prefix=prefix, use_pool_manager=pooled)
        for prefix in ("", "first_", "second_")
    ]
    try:
        for index, store in enumerate(stores):
            store.save(record(asset=f"synthetic-{index}"))
        assert set(inspect(stores[0]._engine).get_table_names()) == {
            "validation_results",
            "first_validation_results",
            "second_validation_results",
        }
        for index, store in enumerate(stores):
            assert store.get("same").data_asset == f"synthetic-{index}"
        stores[1].delete("same")
        assert stores[0].exists("same") and stores[2].exists("same")
        stores[2].close()
        assert stores[2].get("same").data_asset == "synthetic-2"
    finally:
        for store in stores:
            store.close()


def test_initial_sql_failure_does_not_leak_provider_detail_or_owned_engine(tmp_path, monkeypatch):
    from sqlalchemy import create_engine

    from truthound.stores.backends import database
    from truthound.stores.base import StoreConnectionError

    engine = create_engine("sqlite:///" + str(tmp_path / "initial-failure.db"))
    disposed = []

    def fail(*args):
        raise OperationalError(
            "synthetic-private-initial-sql", {}, RuntimeError("synthetic-private-credential")
        )

    event.listen(engine, "before_cursor_execute", fail)
    event.listen(engine, "engine_disposed", lambda value: disposed.append(True))
    monkeypatch.setattr(database, "create_engine", lambda *args, **kwargs: engine)
    store = DatabaseStore(
        "sqlite:///" + str(tmp_path / "initial-failure.db"), use_pool_manager=False
    )
    try:
        with pytest.raises(StoreConnectionError) as error:
            store.initialize()
        assert "synthetic-private-" not in str(error.value)
        assert disposed == [True] and store._engine is None
    finally:
        store.close()
