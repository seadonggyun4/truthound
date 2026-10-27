"""Actual installed async factory lifecycle and canonical native-result parity."""

import sys

import pytest

from tests import test_mcp_plugin_bindings as plugins

runtime = plugins.runtime
invoke = plugins.invoke
pytestmark = pytest.mark.contract


def test_async_native_factory_validation_sample_and_release(runtime, monkeypatch, tmp_path):
    import truthound as th
    from truthound.datasources.adapters import AsyncToSyncAdapter
    from truthound.mcp.operations import _context
    from truthound.validators import get_validator

    adapter, _, marker, _, _ = runtime
    request = plugins.prepare_source(runtime, monkeypatch, "async-source-factory")
    opened = invoke(adapter, "sources_open_binding", **request)
    assert opened["execution_status"] == "succeeded", opened
    identity = opened["data"]["resource_id"]
    inspected = invoke(adapter, "sources_inspect", source_id=identity)
    assert inspected["execution_status"] == "succeeded", inspected
    view = plugins.source_artifact(adapter, inspected["data"]["resource_id"])
    assert view["config"] is None and view["row_count"] == 3
    result = invoke(
        adapter,
        "sources_validate",
        source_id=identity,
        checks=[{"validator": "null", "settings": {"columns": ["x"]}}],
        result_format="complete",
    )
    assert result["execution_status"] == "succeeded", result
    actual = plugins.source_artifact(adapter, result["data"]["artifact"]["resource_id"])
    # The fixture is an installed operator plugin, imported here only after the
    # approved worker result exists. Both paths use its same native async source.
    from mcp_operator_fixture import AsyncNativeSource

    try:
        with AsyncToSyncAdapter(AsyncNativeSource()) as direct_source:
            direct = th.check(
                source=direct_source,
                validators=[get_validator("null")(columns=["x"])],
                context=_context(str(tmp_path / "direct/run.ipc")),
                result_format="complete",
            ).to_dict()
    finally:
        sys.modules.pop("mcp_operator_fixture", None)
    for key in (
        "checks",
        "issues",
        "execution_issues",
        "source",
        "row_count",
        "column_count",
        "execution_mode",
        "planned_execution_mode",
        "result_format",
    ):
        assert actual[key] == direct[key], key
    sample = invoke(adapter, "sources_sample", source_id=identity, n=2)
    assert sample["execution_status"] == "succeeded", sample
    connection = invoke(adapter, "sources_validate_connection", source_id=identity)
    assert plugins.source_artifact(adapter, connection["data"]["resource_id"])["connected"]
    assert (
        invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
        == "succeeded"
    )
    events = marker.read_text().splitlines()
    assert events.count("async-connect") == events.count("async-disconnect") == 3
    assert events[-1] == "teardown"


@pytest.mark.fault
def test_async_factory_schema_failure_closes_source_and_is_not_retried(runtime, monkeypatch):
    adapter, _, marker, _, _ = runtime
    request = plugins.prepare_source(runtime, monkeypatch, "failed-async-source")
    result = invoke(adapter, "sources_open_binding", **request)
    assert result["execution_status"] == "outcome_unknown", result
    events = marker.read_text().splitlines()
    assert events == ["import", "setup", "async-connect", "async-disconnect", "teardown"]
    assert "synthetic-connection-secret" not in str(result)
    assert (
        invoke(adapter, "sources_open_binding", **request)["execution_status"] == "outcome_unknown"
    )
    assert marker.read_text().splitlines() == events
