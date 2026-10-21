"""Core provider/report re-entry, bounded output and no source-side effects."""

import asyncio
import base64
import hashlib
import json
import sqlite3
from datetime import datetime

import polars as pl
import pytest

pytest.importorskip("mcp")

from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService
from truthound.mcp.tables import decode

pytestmark = pytest.mark.contract


@pytest.fixture
def adapter(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    yield service
    service.close()


def call(adapter, name, **arguments):
    result = asyncio.run(adapter.call_operation_async(name, arguments))
    assert result["execution_status"] == "succeeded", result
    return result["data"]


def content(adapter, handle):
    chunks, offset = [], 0
    while offset is not None:
        page = call(adapter, "artifacts_read", resource_id=handle, offset=offset, limit=32768)
        chunks.append(base64.b64decode(page["base64"]))
        offset = page["next_offset"]
    return b"".join(chunks)


def test_sqlite_backup_matches_core_without_mutation_and_quotes_identifiers(tmp_path, adapter):
    import truthound as th
    from truthound.datasources.sql.sqlite import SQLiteDataSource

    path = tmp_path / "input.sqlite"
    table = 'data"; DROP TABLE other; --'
    quoted = '"' + table.replace('"', '""') + '"'
    connection = sqlite3.connect(path)
    connection.execute(f"CREATE TABLE {quoted} (id INTEGER, email TEXT)")
    connection.executemany(f"INSERT INTO {quoted} VALUES (?, ?)", [(1, "a@example.com"), (2, None)])
    connection.commit()
    connection.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with SQLiteDataSource(database=str(path), query=f"SELECT * FROM {quoted}") as source:
        direct = th.read(source)
    handle = call(adapter, "assets_sqlite", path=str(path), table=table)["resource_id"]
    exported = call(adapter, "artifacts_export", resource_id=handle, format="ipc")["resource_id"]
    import io

    assert pl.read_ipc(io.BytesIO(content(adapter, exported))).equals(direct)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["input.sqlite"]
    missing = asyncio.run(
        adapter.call_operation_async("assets_sqlite", {"path": str(path), "table": "missing"})
    )
    assert missing["issues"] == [{"code": "invalid_arguments"}]


def test_sqlite_wal_and_oversized_materialization_are_not_silent_success(tmp_path, adapter):
    path = tmp_path / "input.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE data (x INTEGER)")
        connection.executemany("INSERT INTO data VALUES (?)", [(1,), (2,)])
        connection.commit()
        result = asyncio.run(
            adapter.call_operation_async(
                "assets_sqlite",
                {"path": str(path), "table": "data", "materialization_row_limit": 1},
            )
        )
        assert result["execution_status"] == "failed"
        connection.execute("PRAGMA journal_mode = WAL")
        result = asyncio.run(
            adapter.call_operation_async("assets_sqlite", {"path": str(path), "table": "data"})
        )
        assert result["issues"] == [{"code": "operator_configuration_required"}]
    finally:
        connection.close()


def test_full_file_config_profile_scan_report_and_localized_docs_reentry(tmp_path, adapter):
    import truthound as th
    from truthound.core.results import ValidationRunResult
    from truthound.datadocs import generate_validation_report
    from truthound.datasources.polars_source import FileDataSource, FileDataSourceConfig
    from truthound.reporters import get_reporter

    path = tmp_path / "input.csv"
    path.write_text("email;x\na@example.com;1\nb@example.com;2\n", encoding="utf-8")
    options = {
        "separator": ";",
        "sample_seed": 13,
        "sample_size": 2,
        "max_rows": 20,
        "max_memory_mb": 8,
        "cache_schema": False,
        "strict_types": True,
        "metadata": {"source": "synthetic"},
    }
    handle = call(adapter, "assets_capture", path=str(path), options=options, sample_size=1)[
        "resource_id"
    ]
    expected = th.read(FileDataSource(path, config=FileDataSourceConfig(**options)), sample_size=1)
    for tool, api in (("assets_profile", th.profile), ("assets_scan", th.scan)):
        artifact = call(adapter, tool, asset_id=handle)["resource_id"]
        actual = decode(json.loads(content(adapter, artifact)))
        assert actual == api(expected).to_dict()
    result = call(adapter, "assets_validate", asset_id=handle, checks=[{"validator": "null"}])
    run_handle = result["artifact"]["resource_id"]
    run = ValidationRunResult.from_dict(decode(json.loads(content(adapter, run_handle))))
    render_started = datetime.now().replace(microsecond=0)
    rendered = call(
        adapter,
        "artifacts_render",
        resource_id=run_handle,
        format="json",
        settings={"indent": 4, "sort_keys": True},
    )["resource_id"]
    actual_json = content(adapter, rendered).decode()
    expected_json = get_reporter("json", indent=4, sort_keys=True).render(run)
    render_finished = datetime.now()
    # Independent processes render at different instants. Verify both measured
    # timestamps instead of depending on coincidental same-second execution.
    actual_report, expected_report = json.loads(actual_json), json.loads(expected_json)
    for report in (actual_report, expected_report):
        stamp = datetime.strptime(report["generated_at"], "%Y-%m-%d %H:%M:%S")
        assert render_started <= stamp <= render_finished
    expected_report["generated_at"] = actual_report["generated_at"]
    assert actual_report == expected_report
    # Preserve formatting parity too; normalize only the verified render clock.
    assert actual_json == expected_json.replace(
        json.loads(expected_json)["generated_at"], actual_report["generated_at"], 1
    )
    docs = call(
        adapter, "artifacts_datadocs", resource_id=run_handle, locale="ko", title="검증 결과"
    )["resource_id"]
    assert content(adapter, docs).decode() == generate_validation_report(
        run, locale="ko", title="검증 결과"
    )
    assert not (tmp_path / ".truthound").exists()
    refused = asyncio.run(
        adapter.call_operation_async(
            "artifacts_render",
            {
                "resource_id": run_handle,
                "format": "html",
                "settings": {"template_path": "/etc/passwd"},
            },
        )
    )
    assert refused["issues"] == [{"code": "operator_configuration_required"}]
