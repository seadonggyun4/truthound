"""Actual native datasource lifecycle, sample semantics and revoked ownership."""

import asyncio
import base64
import json
from dataclasses import asdict, fields

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.datasources.polars_source import PolarsDataSource, PolarsDataSourceConfig
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.sources import PolarsConfig
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


def invoke(service, name, **arguments):
    return asyncio.run(service.call_operation_async(name, arguments))


def call(service, name, **arguments):
    response = invoke(service, name, **arguments)
    assert response["execution_status"] == "succeeded", response
    return response["data"]


def read(service, identity):
    chunks, offset = [], 0
    while offset is not None:
        value = call(service, "artifacts_read", resource_id=identity, offset=offset)
        chunks.append(base64.b64decode(value["base64"]))
        offset = value["next_offset"]
    return decode(json.loads(b"".join(chunks)))


def test_native_source_lifecycle_full_config_metadata_sampling_and_parent_release(tmp_path):
    assert set(PolarsConfig.model_fields) == {
        field.name for field in fields(PolarsDataSourceConfig)
    }
    frame = pl.DataFrame({"x": list(range(20)), "text": ["value"] * 20})
    config = {
        "name": "native-owned",
        "max_rows": 10,
        "max_memory_mb": 8,
        "sample_size": 4,
        "sample_seed": 13,
        "cache_schema": False,
        "strict_types": True,
        "metadata": {"purpose": "synthetic"},
        "rechunk": True,
        "streaming": True,
    }
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[
                {"name": "x", "dtype": {"kind": "Int64"}, "values": frame["x"].to_list()},
                {"name": "text", "dtype": {"kind": "String"}, "values": frame["text"].to_list()},
            ],
        )["resource_id"]
        source = call(service, "sources_open_polars", asset_id=asset, config=config)["resource_id"]
        worker = service._session._sources.workers[source]
        with PolarsDataSource(frame.lazy(), config=PolarsDataSourceConfig(**config)) as native:
            import truthound as th
            from truthound.mcp.operations import _context
            from truthound.validators import get_validator

            validation = call(
                service,
                "sources_validate",
                source_id=source,
                checks=[
                    {"validator": "mean_between", "settings": {"columns": ["x"], "max_value": 1}}
                ],
                result_format="complete",
            )
            actual_run = read(service, validation["artifact"]["resource_id"])
            direct_run = th.check(
                source=native,
                validators=[get_validator("mean_between")(columns=["x"], max_value=1)],
                result_format="complete",
                context=_context(str(tmp_path / "direct/input.ipc")),
            ).to_dict()
            assert direct_run["issues"] and not direct_run["execution_issues"]
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
                assert actual_run[key] == direct_run[key], key
            actual = read(
                service, call(service, "sources_inspect", source_id=source)["resource_id"]
            )
            assert actual == {
                "name": native.name,
                "config": asdict(native.config),
                "schema": {k: v.value for k, v in native.schema.items()},
                "columns": native.columns,
                "row_count": native.row_count,
                "capabilities": sorted(v.value for v in native.capabilities),
                "needs_sampling": native.needs_sampling(),
                "numeric_columns": native.get_numeric_columns(),
                "string_columns": native.get_string_columns(),
                "datetime_columns": native.get_datetime_columns(),
            }
            missing = read(
                service,
                call(
                    service,
                    "sources_validate_columns",
                    source_id=source,
                    required_columns=["x", "missing"],
                )["resource_id"],
            )
            assert missing == {"missing_columns": native.validate_columns(["x", "missing"])}
            connected = read(
                service,
                call(service, "sources_validate_connection", source_id=source)["resource_id"],
            )
            assert connected == {"connected": native.validate_connection()}
            sampled = call(service, "sources_sample", source_id=source, n=4, seed=13)["resource_id"]
            rows = decode(call(service, "assets_page", asset_id=sampled)["rows"])
            assert rows == native.sample(4, seed=13).to_polars_lazyframe().collect().to_dicts()
        failed = invoke(service, "sources_check_size_limits", source_id=source)
        assert failed["execution_status"] == "failed"
        assert failed["issues"] == [{"code": "native_source_error"}]
        # A failed read-only size check must not destroy the resident source.
        assert not worker.closed
        call(service, "resources_release", resource_id=asset)
        assert worker.closed and worker.process.poll() is not None
        assert invoke(service, "sources_inspect", source_id=source)["execution_status"] == "failed"
        assert invoke(service, "assets_page", asset_id=sampled)["execution_status"] == "failed"
    finally:
        service.close()


@pytest.mark.fault
def test_native_source_timeout_invalidates_owned_process_and_session_close_reaps(
    tmp_path, monkeypatch
):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 2]}],
        )["resource_id"]
        source = call(service, "sources_open_polars", asset_id=asset)["resource_id"]
        worker = service._session._sources.workers[source]

        # Exercise real cancellation handling around the exchange, not a fake Core.
        def broken_exchange(_request):
            raise TimeoutError

        monkeypatch.setattr(worker, "_exchange", broken_exchange)
        failed = invoke(service, "sources_inspect", source_id=source)
        assert failed["execution_status"] == "failed"
        assert worker.closed and worker.process.poll() is not None
        other = call(service, "sources_open_polars", asset_id=asset)["resource_id"]
        owned = service._session._sources.workers[other]
        forged = invoke(service, "sources_open_polars", asset_id=asset, config={"separator": ";"})
        assert forged["issues"] == [{"code": "invalid_arguments"}]
        service.close()
        assert owned.closed and owned.process.poll() is not None
    finally:
        service.close()


def test_sqlite_native_source_engine_canonical_run_snapshot_and_cleanup(tmp_path):
    import hashlib
    import sqlite3

    import truthound as th
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig
    from truthound.mcp.operations import _context
    from truthound.mcp.sources import SQLiteConfig
    from truthound.validators import get_validator

    assert set(SQLiteConfig.model_fields) | {"database"} == {
        field.name for field in fields(SQLiteDataSourceConfig)
    }
    path = tmp_path / "input.db"
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE "odd name" (x INTEGER, text TEXT)')
        connection.executemany(
            'INSERT INTO "odd name" VALUES (?, ?)', [(1, "a"), (None, "b"), (3, "c")]
        )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {
        "name": "native-sqlite",
        "fetch_size": 2,
        "materialization_row_limit": 100,
        "max_rows": 100,
    }
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        source = call(
            service, "sources_open_sqlite", path=str(path), table="odd name", config=config
        )["resource_id"]
        entry = service._session.resources.get(source, "source")
        snapshot = service._session.resources.get(entry.metadata["asset_id"], "artifact")
        assert snapshot.media_type == "application/vnd.sqlite3"
        worker = service._session._sources.workers[source]
        with SQLiteDataSource(
            database=str(path),
            query='SELECT * FROM "odd name"',
            config=SQLiteDataSourceConfig(**config),
        ) as native:
            direct = th.check(
                source=native,
                validators=[get_validator("null")(columns=["x"])],
                result_format="complete",
                context=_context(str(tmp_path / "direct/input.ipc")),
            ).to_dict()
            result = call(
                service,
                "sources_validate",
                source_id=source,
                checks=[{"validator": "null", "settings": {"columns": ["x"]}}],
                result_format="complete",
            )
            actual = read(service, result["artifact"]["resource_id"])
            assert direct["issues"] and not direct["execution_issues"]
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
                assert actual[key] == direct[key], key
            inspection = read(
                service, call(service, "sources_inspect", source_id=source)["resource_id"]
            )
            assert inspection["row_count"] == native.row_count
            assert inspection["capabilities"] == sorted(
                value.value for value in native.capabilities
            )
            assert inspection["config"]["fetch_size"] == 2
        assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        # Later database changes must not change an existing source's identity.
        with sqlite3.connect(path) as connection:
            connection.execute('INSERT INTO "odd name" VALUES (4, "d")')
        assert (
            read(service, call(service, "sources_inspect", source_id=source)["resource_id"])[
                "row_count"
            ]
            == 3
        )
        call(service, "resources_release", resource_id=snapshot.id)
        assert worker.closed and worker.process.poll() is not None
        assert invoke(service, "sources_inspect", source_id=source)["execution_status"] == "failed"
        assert (
            before != hashlib.sha256(path.read_bytes()).hexdigest()
        )  # only our explicit INSERT changed it
    finally:
        service.close()


@pytest.mark.fault
@pytest.mark.parametrize("failure", ["wal", "table", "budget"])
def test_sqlite_admission_failure_has_no_original_write_or_leaked_owned_resources(
    tmp_path, failure
):
    import hashlib
    import sqlite3

    path = tmp_path / "input.db"
    with sqlite3.connect(path) as connection:
        if failure == "wal":
            connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("CREATE TABLE items (x INTEGER)")
        connection.execute("INSERT INTO items VALUES (1)")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    files = {item.name for item in tmp_path.iterdir()}
    policy = MCPPolicy(
        (tmp_path,), profile="core", max_file_bytes=1024 if failure == "budget" else 100 * 1024**2
    )
    service = MCPService(policy)
    try:
        result = invoke(
            service,
            "sources_open_sqlite",
            path=str(path),
            table="items; DROP TABLE items" if failure == "table" else "items",
        )
        expected = {
            "wal": "operator_configuration_required",
            "table": "invalid_arguments",
            "budget": "file_too_large",
        }[failure]
        assert result["execution_status"] == "failed"
        assert result["issues"] == [{"code": expected}]
        assert before == hashlib.sha256(path.read_bytes()).hexdigest()
        assert {item.name for item in tmp_path.iterdir()} == files
        assert not list(service._session.resources.root.iterdir())
        assert not service._session._sources.workers
    finally:
        service.close()
