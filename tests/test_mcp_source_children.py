"""Retained native samples share their real owner and close before its provider."""

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_sources import call, invoke, read
from truthound.mcp.policy import MCPPolicy
from truthound.mcp.service import MCPService

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def test_native_polars_sample_children_nested_validation_snapshot_and_release(tmp_path):
    import polars as pl

    from truthound.datasources.polars_source import PolarsDataSource

    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": list(range(10))}],
        )["resource_id"]
        parent = call(service, "sources_open_polars", asset_id=asset)["resource_id"]
        child = call(service, "sources_sample_source", source_id=parent, n=4, seed=42)[
            "resource_id"
        ]
        sibling = call(service, "sources_sample_source", source_id=parent, n=3, seed=42)[
            "resource_id"
        ]
        nested = call(service, "sources_sample_source", source_id=child, n=2, seed=42)[
            "resource_id"
        ]
        controller = service._session._sources
        assert controller.workers[parent] is controller.workers[child] is controller.workers[nested]
        sample = PolarsDataSource(pl.DataFrame({"x": list(range(10))}).lazy()).sample(4, 42)
        assert (
            read(service, call(service, "sources_inspect", source_id=child)["resource_id"])[
                "row_count"
            ]
            == sample.row_count
        )
        validation = call(
            service,
            "sources_validate",
            source_id=nested,
            checks=[{"validator": "not_null", "settings": {"columns": ["x"]}}],
        )
        run = read(service, validation["artifact"]["resource_id"])
        assert run["row_count"] == 2 and not run["execution_issues"]
        snapshot = call(service, "sources_sample", source_id=nested, n=1, seed=42)["resource_id"]
        assert call(service, "assets_page", asset_id=snapshot)["total_rows"] == 1
        call(service, "resources_release", resource_id=child)
        assert not controller.workers[parent].closed
        assert invoke(service, "sources_inspect", source_id=nested)["issues"] == [
            {"code": "invalid_handle"}
        ]
        assert read(
            service, call(service, "sources_validate_connection", source_id=sibling)["resource_id"]
        )["connected"]
        worker = controller.workers[parent]
        call(service, "resources_release", resource_id=parent)
        assert worker.closed and not controller.workers
    finally:
        service.close()


def test_actual_async_samples_keep_shared_loop_until_nested_children_disconnect(
    runtime, monkeypatch
):
    adapter, _, marker, _, _ = runtime
    request = plugins.prepare_source(runtime, monkeypatch, "async-source-factory")
    parent = call(adapter, "sources_open_binding", **request)["resource_id"]
    child = call(adapter, "sources_sample_source", source_id=parent, n=2)["resource_id"]
    nested = call(adapter, "sources_sample_source", source_id=child, n=1)["resource_id"]
    validation = call(
        adapter,
        "sources_validate",
        source_id=nested,
        checks=[{"validator": "null", "settings": {"columns": ["x"]}}],
    )
    assert read(adapter, validation["artifact"]["resource_id"])["row_count"] == 1
    assert marker.read_text().splitlines().count("async-connect") == 3
    call(adapter, "resources_release", resource_id=child)
    assert marker.read_text().splitlines().count("async-disconnect") == 2
    assert "teardown" not in marker.read_text()
    assert read(
        adapter, call(adapter, "sources_validate_connection", source_id=parent)["resource_id"]
    )["connected"]
    call(adapter, "resources_release", resource_id=parent)
    events = marker.read_text().splitlines()
    assert events.count("async-disconnect") == 3 and events[-1] == "teardown"


@pytest.mark.fault
def test_native_sample_limit_rejects_without_closing_existing_children(tmp_path):
    service = MCPService(MCPPolicy((tmp_path,), profile="core", allow_sensitive_results=True))
    try:
        asset = call(
            service,
            "assets_inline",
            columns=[{"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 2]}],
        )["resource_id"]
        parent = call(service, "sources_open_polars", asset_id=asset)["resource_id"]
        children = [
            call(service, "sources_sample_source", source_id=parent, n=1, seed=42)["resource_id"]
            for _ in range(16)
        ]
        failed = invoke(service, "sources_sample_source", source_id=parent, n=1, seed=42)
        assert failed["issues"] == [{"code": "resource_exhausted"}]
        assert not service._session._sources.workers[parent].closed
        call(service, "resources_release", resource_id=children[0])
        extra = call(service, "sources_sample_source", source_id=parent, n=1, seed=42)[
            "resource_id"
        ]
        assert (
            read(service, call(service, "sources_inspect", source_id=extra)["resource_id"])[
                "row_count"
            ]
            == 1
        )
    finally:
        service.close()


@pytest.mark.fault
def test_stalled_native_child_cleanup_kills_shared_owner_without_hanging(runtime, monkeypatch):
    import time

    adapter, _, marker, _, _ = runtime
    marker.with_name("slow-async-child-close").write_text("local test")
    request = plugins.prepare_source(runtime, monkeypatch, "async-source-factory")
    parent = call(adapter, "sources_open_binding", **request)["resource_id"]
    child = call(adapter, "sources_sample_source", source_id=parent, n=1)["resource_id"]
    worker = adapter._session._sources.workers[parent]
    start = time.monotonic()
    assert call(adapter, "resources_release", resource_id=child)["released"]
    assert time.monotonic() - start < 6
    assert worker.closed and worker.process.poll() is not None
    assert "async-child-close-blocked" in marker.read_text()
    assert invoke(adapter, "sources_validate_connection", source_id=parent)["issues"] == [
        {"code": "invalid_handle"}
    ]
