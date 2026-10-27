"""Native pagination and child lifetimes at the same contracts used by MCP."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from truthound.execution.base import BaseExecutionEngine
from truthound.execution.polars_engine import PolarsExecutionEngine
from truthound.stores.backends.memory import MemoryStore
from truthound.stores.base import StoreQuery
from truthound.stores.results import ResultStatus, ValidationResult
from truthound.stores.versioning.store import VersionedStore

pytestmark = pytest.mark.contract


def records(store):
    for index in range(5):
        store.save(
            ValidationResult(
                f"run-{index}",
                datetime(2026, 1, 1) + timedelta(days=index),
                "local",
                ResultStatus.SUCCESS,
            )
        )


def test_iter_query_never_yields_past_limit_when_batch_is_larger():
    store = MemoryStore()
    records(store)
    result = list(store.iter_query(StoreQuery(offset=1, limit=2), batch_size=4))
    assert [row.run_id for row in result] == ["run-3", "run-2"]
    assert list(store.iter_query(StoreQuery(limit=0), batch_size=2)) == []
    with pytest.raises(ValueError):
        list(store.iter_query(StoreQuery(limit=2), batch_size=0))


def test_versioned_store_exposes_wrapped_backend_configuration_after_save():
    base = MemoryStore(namespace="project")
    wrapped = VersionedStore(base)
    records(wrapped)
    assert wrapped.config is base.config
    assert wrapped.get("run-0").data_asset == "local"


class TrackingEngine(PolarsExecutionEngine):
    count_matching = BaseExecutionEngine.count_matching

    def __init__(self, frame, events, fail=False):
        super().__init__(frame)
        self.events = events
        self.fail = fail

    def sample(self, n=1000, seed=None):
        return TrackingEngine(self.lazyframe.limit(n), self.events, self.fail)

    def filter_by_condition(self, condition):
        return self.sample(1)

    def count_rows(self):
        if self.fail:
            raise RuntimeError("synthetic computation failed")
        return super().count_rows()

    def to_numpy(self, columns=None):
        if self.fail:
            raise RuntimeError("synthetic conversion failed")
        return super().to_numpy(columns)

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, *args):
        self.events.append("close")
        return super().__exit__(*args)


@pytest.mark.parametrize("fail", [False, True])
def test_default_filter_count_closes_native_temporary_even_on_error(fail):
    events = []
    engine = TrackingEngine(pl.DataFrame({"x": [1, 2]}).lazy(), events, fail)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic"):
            engine.count_matching("selected")
    else:
        assert engine.count_matching("selected") == 1
    assert events == ["enter", "close"]


def test_default_safe_numpy_closes_sample_without_closing_parent():
    events = []
    engine = TrackingEngine(pl.DataFrame({"x": [1, 2]}).lazy(), events)
    assert engine.to_numpy_safe(["x"], max_rows=1).tolist() == [[1]]
    assert events == ["enter", "close"]
    assert engine.count_rows() == 2


def test_sql_filtered_view_preserves_lazy_pushdown_sampling_and_parent_connection(tmp_path):
    import sqlite3

    from truthound.datasources.sql.sqlite import SQLiteDataSource
    from truthound.execution.sql_engine import SQLExecutionEngine

    database = tmp_path / "filter.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE items (x INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(-1,), (1,), (2,), (None,)])
    with SQLiteDataSource(database=str(database), table="items") as source:
        engine = SQLExecutionEngine(source)
        with engine.filter_by_condition("x > 0") as filtered:
            assert filtered.supports_sql_pushdown
            assert filtered.count_rows() == 2
            assert filtered.get_columns() == ["x"]
            assert filtered.get_column_values("x") == [1, 2]
            assert filtered.datasource.row_count == 2
            assert filtered.sample(1).count_rows() == 1
            assert filtered.filter_by_condition("x > 1").count_rows() == 1
        assert engine.count_rows() == 4


def test_retention_wrapper_keeps_backend_configuration_after_cleanup():
    from truthound.stores.retention.store import RetentionStore

    base = MemoryStore(namespace="retained-project")
    store = RetentionStore(base)
    records(store)
    assert store.run_cleanup(dry_run=True).items_scanned == 5
    assert store.config is base.config
