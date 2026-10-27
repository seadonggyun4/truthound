"""Real installed native monitor: periodic work, unknown effects and revocation."""

import json
import sys
import time
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved
from truthound.mcp.approval import ApprovalStore

runtime = plugins.runtime
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch):
    from truthound.mcp import model_process

    adapter, control, _, _, _ = runtime
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    document = json.loads((control / "bindings.json").read_text())
    document["plugins"]["fixture"]["capabilities"]["ml-factory"] = "ml-factory"
    document["ml_services"] = {"local": "ml-factory"}
    (control / "bindings.json").write_text(json.dumps(document))
    return approved(
        runtime, "ml_services_open", binding_id="local", idempotency_key="ml-open-local"
    )["data"]["resource_id"]


def call(runtime, service, command, key=None, failed=False, **arguments):
    request = dict(
        binding_id="local",
        service_id=service,
        command=command,
        idempotency_key=key or "ml-command-" + command,
        **arguments,
    )
    pending = plugins.invoke(runtime[0], "ml_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(runtime[1])
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    response = plugins.invoke(runtime[0], "ml_services_execute", **request)
    assert response["execution_status"] == ("failed" if failed else "succeeded"), response
    payload = plugins.source_artifact(runtime[0], response["data"]["resource_id"])
    assert "synthetic-private-" not in json.dumps(payload), payload
    return response, payload


def asset(runtime, latency=1.0):
    columns = [
        {"name": "x", "dtype": {"kind": "Int64"}, "values": [1, 2]},
        {"name": "prediction", "dtype": {"kind": "Int64"}, "values": [1, 0]},
        {"name": "actual", "dtype": {"kind": "Int64"}, "values": [1, 1]},
        {"name": "latency", "dtype": {"kind": "Float64"}, "values": [latency, latency]},
    ]
    return plugins.invoke(runtime[0], "assets_inline", columns=columns)["data"]["resource_id"]


def test_native_monitor_periodic_collection_feedback_alert_history_and_acknowledged_shutdown(
    runtime, monkeypatch
):
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    call(runtime, service, "register", model_id="local")
    call(runtime, service, "rule_add", rule_id="latency")
    call(runtime, service, "handler_add", handler_id="local")
    call(runtime, service, "reference", model_id="local", asset_id=source)
    _, recorded = call(runtime, service, "record", model_id="local", asset_id=source)
    prediction = recorded["predictions"][0]["prediction_id"]
    assert recorded["recorded"] == 2 and recorded["predictions"][0]["pending"]
    call(runtime, service, "feedback", model_id="local", prediction_id=prediction, asset_id=source)
    call(runtime, service, "start")
    deadline = time.monotonic() + 5
    while "ml-alert" not in runtime[2].read_text() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (
        runtime[2].read_text().count("ml-store") == 1
        and runtime[2].read_text().count("ml-alert") == 1
    )
    _, dashboard = call(runtime, service, "dashboard", model_id="local")
    assert dashboard["metrics"]["latency_ms"] == 1
    assert (
        len(dashboard["alert_history"]) == 1
        and dashboard["alert_history"][0]["rule_id"] == "latency"
    )
    call(runtime, service, "evaluate", model_id="local")
    assert runtime[2].read_text().count("ml-alert") == 1
    call(runtime, service, "shutdown")
    assert "ml-close" in runtime[2].read_text()
    rejected = plugins.invoke(
        runtime[0],
        "ml_services_execute",
        binding_id="local",
        service_id=service,
        command="status",
        idempotency_key="ml-after-shutdown",
    )
    assert rejected["issues"] == [{"code": "invalid_handle"}]


@pytest.mark.parametrize(
    "fault,marker", [("ml-commit-failed", "ml-store"), ("ml-alert-failed", "ml-alert")]
)
def test_unknown_native_effect_stays_failed_and_is_not_automatically_replayed(
    runtime, monkeypatch, fault, marker
):
    runtime[2].with_name(fault).touch()
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    call(runtime, service, "register", model_id="local")
    call(runtime, service, "rule_add", rule_id="latency")
    call(runtime, service, "handler_add", handler_id="local")
    call(runtime, service, "record", model_id="local", asset_id=source)
    response, result = call(runtime, service, "collect", model_id="local", failed=True)
    assert response["issues"] == [{"code": "native_monitor_incomplete"}]
    assert result["state"]["models"]["local"]["uncertain"]
    replay = plugins.invoke(
        runtime[0],
        "ml_services_execute",
        binding_id="local",
        service_id=service,
        command="collect",
        model_id="local",
        idempotency_key="ml-command-collect",
    )
    assert replay == response
    call(runtime, service, "collect", key="ml-second-collect", model_id="local", failed=True)
    assert runtime[2].read_text().count(marker) == 1
    _, discarded = call(runtime, service, "discard_uncertain", model_id="local")
    assert discarded["discarded"] == 2 and not discarded["state"]["models"]["local"]["uncertain"]
    call(runtime, service, "evaluate", model_id="local")
    assert runtime[2].read_text().count(marker) == 1


def test_input_release_stops_owned_timers_and_revokes_derived_results(runtime, monkeypatch):
    service = prepare(runtime, monkeypatch)
    source = asset(runtime)
    call(runtime, service, "register", model_id="local")
    response, _ = call(runtime, service, "record", model_id="local", asset_id=source)
    worker = runtime[0]._session._administration["ml_services"].workers[service]
    plugins.invoke(runtime[0], "resources_release", resource_id=source)
    assert worker.closed and worker.process.poll() is not None
    assert "ml-store" not in runtime[2].read_text()
    rejected = plugins.invoke(
        runtime[0], "artifacts_read", resource_id=response["data"]["resource_id"]
    )
    assert rejected["issues"] == [{"code": "invalid_handle"}]
