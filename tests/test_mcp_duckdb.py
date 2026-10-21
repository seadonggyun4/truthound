"""Real native DuckDB engine, hardened connections and owned-copy failures."""

import hashlib
from dataclasses import fields

import duckdb
import pytest

from tests.test_mcp_sources import call, invoke, read
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


@pytest.mark.fault
def test_missing_driver_discovery_neither_imports_native_provider_nor_opens_session(
    tmp_path, monkeypatch
):
    import builtins
    from importlib.metadata import PackageNotFoundError

    from truthound.mcp import duckdb_source

    def missing(_name):
        raise PackageNotFoundError("duckdb")

    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"duckdb", "truthound.datasources.sql.duckdb"}, (
            "Discovery must not load a provider"
        )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(duckdb_source, "version", missing)
    monkeypatch.setattr(builtins, "__import__", guarded)
    service = MCPService(MCPPolicy((tmp_path,), profile="core"))
    try:
        page = call(service, "capabilities_list", limit=100)
        source = next(
            row for row in page["operations"] if row["operation_id"] == "sources_open_duckdb"
        )
        assert source["availability"] == "missing_dependency"
        description = call(service, "capability_describe", operation_id="sources_open_duckdb")
        assert description["required_extras"] == ["mcp", "duckdb"]
        assert service._session is None
    finally:
        service.close()


def database(path):
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE facts AS SELECT * FROM (VALUES (1),(NULL),(3)) t(x)")


def test_native_duckdb_sql_canonical_result_snapshot_and_parent_release(tmp_path):
    import truthound as th
    from truthound.datasources.sql.duckdb import DuckDBDataSource, DuckDBDataSourceConfig
    from truthound.mcp.operations import _context
    from truthound.mcp.sources import DuckDBConfig
    from truthound.validators import get_validator

    assert set(DuckDBConfig.model_fields) | {"database", "read_only"} == {
        field.name for field in fields(DuckDBDataSourceConfig)
    }
    path = tmp_path / "data.duckdb"
    database(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {"name": "native-duckdb", "threads": 1, "memory_limit": "64MB", "fetch_size": 2}
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        source = call(service, "sources_open_duckdb", path=str(path), table="facts", config=config)[
            "resource_id"
        ]
        worker = service._session._sources.workers[source]
        entry = service._session.resources.get(source, "source")
        snapshot = service._session.resources.get(entry.metadata["asset_id"], "artifact")
        assert snapshot.sha256 == digest and snapshot.media_type == "application/vnd.duckdb"
        actual = read(
            service,
            call(
                service,
                "sources_validate",
                source_id=source,
                checks=[{"validator": "null", "settings": {"columns": ["x"]}}],
                result_format="complete",
            )["artifact"]["resource_id"],
        )
        with DuckDBDataSource(
            database=str(path),
            query='SELECT * FROM "main"."facts"',
            read_only=True,
            config=DuckDBDataSourceConfig(**config),
        ) as native:
            expected = th.check(
                source=native,
                validators=[get_validator("null")(columns=["x"])],
                result_format="complete",
                context=_context(str(tmp_path / "direct.ipc")),
            ).to_dict()
        for key in (
            "checks",
            "issues",
            "execution_issues",
            "row_count",
            "column_count",
            "source",
            "execution_mode",
            "planned_execution_mode",
            "result_format",
        ):
            assert actual[key] == expected[key], key
        assert not actual["execution_issues"] and actual["issues"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with duckdb.connect(str(path)) as connection:
            connection.execute("INSERT INTO facts VALUES (4)")
        inspection = read(
            service, call(service, "sources_inspect", source_id=source)["resource_id"]
        )
        assert inspection["row_count"] == 3
        sample = call(service, "sources_sample", source_id=source, n=2, seed=42)["resource_id"]
        assert call(service, "assets_page", asset_id=sample)["total_rows"] == 2
        call(service, "resources_release", resource_id=source)
        assert worker.closed and worker.process.poll() is not None
        assert not snapshot.path.exists()
        assert invoke(service, "assets_page", asset_id=sample)["execution_status"] == "failed"
    finally:
        service.close()


@pytest.mark.fault
def test_native_duckdb_guards_write_external_view_and_configuration_changes(tmp_path):
    from truthound.mcp.duckdb_source import open_source
    from truthound.mcp.sources import DuckDBConfig

    path = tmp_path / "data.duckdb"
    database(path)
    outside = tmp_path / "not-granted.csv"
    outside.write_text("x\n987654\n", encoding="utf-8")
    literal = "'" + str(outside).replace("'", "''") + "'"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE VIEW external_view AS SELECT * FROM read_csv(" + literal + ")")
    source = open_source(str(path), "facts", DuckDBConfig(memory_limit="64MB").model_dump())
    try:
        with source._get_connection() as connection:
            for sql in (
                "INSERT INTO facts VALUES (4)",
                "SELECT * FROM external_view",
                "SET enable_external_access=true",
                "SET autoload_known_extensions=true",
            ):
                with pytest.raises(duckdb.Error):
                    connection.execute(sql).fetchall()
        assert source.row_count == 3
    finally:
        source.__exit__(None, None, None)


@pytest.mark.fault
def test_duckdb_rejects_views_literal_injection_bad_config_and_corruption_without_leaks(tmp_path):
    path = tmp_path / "data.duckdb"
    database(path)
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE VIEW view_not_table AS SELECT * FROM facts")
    service = MCPService(MCPPolicy((tmp_path,), profile="core"))
    try:
        for table in ("view_not_table", "absent", "facts' OR TRUE --"):
            failed = invoke(service, "sources_open_duckdb", path=str(path), table=table)
            assert failed["issues"] == [{"code": "unsupported_table"}]
            assert not service._session._sources.workers
            assert not service._session.resources._entries
            assert not list(service._session.resources.root.iterdir())
        assert invoke(
            service,
            "sources_open_duckdb",
            path=str(path),
            table="facts",
            config={"read_only": False},
        )["issues"] == [{"code": "invalid_arguments"}]
        broken = tmp_path / "broken.duckdb"
        broken.write_bytes(b"synthetic malformed database")
        assert invoke(service, "sources_open_duckdb", path=str(broken), table="facts")[
            "issues"
        ] == [{"code": "native_source_error"}]
        assert not service._session.resources._entries
        assert not list(service._session.resources.root.iterdir())
    finally:
        service.close()


@pytest.mark.fault
def test_duckdb_wal_and_quota_rejection_do_not_connect_or_copy_unbounded_input(tmp_path):
    path = tmp_path / "data.duckdb"
    database(path)
    wal = tmp_path / "data.duckdb.wal"
    wal.write_bytes(b"synthetic WAL marker")
    service = MCPService(MCPPolicy((tmp_path,), profile="core", max_session_bytes=1024))
    try:
        assert invoke(service, "sources_open_duckdb", path=str(path), table="facts")["issues"] == [
            {"code": "unsupported_database_mode"}
        ]
        wal.unlink()
        assert invoke(service, "sources_open_duckdb", path=str(path), table="facts")["issues"] == [
            {"code": "file_too_large"}
        ]
        assert not service._session.resources._entries
        assert not service._session._sources.workers
        assert not list(service._session.resources.root.iterdir())
    finally:
        service.close()
