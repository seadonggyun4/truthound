"""Installed operator factories exercise native scheduled and escalation effects."""

import asyncio
import json
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from truthound.mcp.approval import ApprovalStore

pytestmark = pytest.mark.contract
runtime = plugins.runtime
invoke = plugins.invoke


def approve(adapter, control, operation, **request):
    pending = invoke(adapter, operation, **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    result = invoke(adapter, operation, **request)
    assert result["execution_status"] == "succeeded", result
    return result


def prepare(runtime, monkeypatch, kind, *, approved=True):
    from truthound.mcp import model_process

    adapter, control, marker, _, _ = runtime
    monkeypatch.setattr(
        model_process,
        "sys",
        SimpleNamespace(
            executable=adapter._worker_argv()[0],
            platform=sys.platform,
            _base_executable=sys._base_executable,
        ),
    )
    path = control / "bindings.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    capability = "checkpoint-runner-factory" if kind == "runner" else "escalation-factory"
    config["plugins"]["fixture"]["capabilities"][capability] = capability
    config["checkpoints"] = {"service": capability}
    path.write_text(json.dumps(config), encoding="utf-8")
    request = {"kind": kind, "binding_id": "service", "idempotency_key": "service-open"}
    if not approved:
        pending = invoke(adapter, "checkpoint_services_open", **request)
        assert pending["issues"] == [{"code": "approval_required"}]
        assert not marker.exists()
        return request
    response = approve(adapter, control, "checkpoint_services_open", **request)
    effects = marker.read_bytes()
    assert invoke(adapter, "checkpoint_services_open", **request) == response
    assert marker.read_bytes() == effects
    return response["data"]["resource_id"]


def command(runtime, identity, action, *, key=None, **kwargs):
    adapter, control, _, _, _ = runtime
    response = approve(
        adapter,
        control,
        "checkpoint_services_execute",
        binding_id="service",
        service_id=identity,
        command=action,
        idempotency_key=key or f"command-{action}",
        **kwargs,
    )
    return plugins.source_artifact(adapter, response["data"]["resource_id"])


def test_runner_uses_native_schedule_and_retains_results(runtime, monkeypatch, tmp_path):
    from truthound.checkpoint import Checkpoint
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig

    adapter, _, marker, _, _ = runtime
    identity = prepare(runtime, monkeypatch, "runner")
    assert "checkpoint-action" not in marker.read_text(encoding="utf-8")
    assert command(runtime, identity, "start")["running"] is True
    deadline = time.monotonic() + 10
    rows = []
    index = 0
    while time.monotonic() < deadline and not rows:
        rows = command(runtime, identity, "results", key=f"results-{index:04}")["results"]
        index += 1
    assert len(rows) == 1
    assert command(runtime, identity, "stop")["running"] is False
    with SQLiteDataSource(
        database=str(tmp_path / "operator.db"),
        table="items",
        config=SQLiteDataSourceConfig(name="approved-source"),
    ) as source:
        direct = Checkpoint(
            name="approved-checkpoint", data_source=source, validators=["null"]
        ).run()
    assert rows[0]["status"] == direct.status.value
    for key in ("checks", "issues", "execution_issues", "row_count", "column_count"):
        assert rows[0]["validation_run"][key] == direct.validation_run.to_dict()[key]
    assert rows[0]["metadata"]["trigger"] == "approved-schedule"
    assert marker.read_text(encoding="utf-8").count("checkpoint-action") == 1
    assert command(runtime, identity, "results", key="drained-results")["results"] == []
    worker = adapter._session._checkpoint_services.workers[identity]
    invoke(adapter, "resources_release", resource_id=identity)
    assert worker.closed and worker.process.poll() is not None


def native_engine():
    from truthound.checkpoint.escalation.engine import EscalationEngine, EscalationEngineConfig
    from truthound.checkpoint.escalation.protocols import (
        EscalationLevel,
        EscalationPolicy,
        EscalationTarget,
    )
    from truthound.checkpoint.escalation.scheduler import SchedulerConfig, SchedulerType

    async def notify(record, level, targets):
        return True

    engine = EscalationEngine(
        EscalationEngineConfig(
            scheduler_config=SchedulerConfig(scheduler_type=SchedulerType.ASYNCIO),
            default_notification_handler=notify,
        )
    )
    engine.register_policy(
        EscalationPolicy(
            name="approved-policy",
            levels=[
                EscalationLevel(level=1, targets=[EscalationTarget.user("local")]),
                EscalationLevel(
                    level=2, delay_minutes=60, targets=[EscalationTarget.user("local")]
                ),
            ],
        )
    )
    return engine


@pytest.mark.parametrize("action", ["acknowledge", "resolve", "cancel", "escalate"])
def test_escalation_native_transition_and_notification_parity(runtime, monkeypatch, action):
    _, _, marker, _, _ = runtime
    identity = prepare(runtime, monkeypatch, "escalation")
    assert "escalation-notify" not in marker.read_text(encoding="utf-8")
    assert command(runtime, identity, "start")["running"] is True
    initial = command(
        runtime, identity, "trigger", incident_id="incident", policy_name="approved-policy"
    )
    record_id = initial["record"]["id"]
    actual = command(runtime, identity, action, record_id=record_id)

    async def direct_run():
        engine = native_engine()
        await engine.start()
        try:
            created = await engine.trigger("incident", policy_name="approved-policy")
            if action == "escalate":
                result = await engine.escalate(created.record.id)
            else:
                result = await getattr(engine, action)(created.record.id, "mcp-approved-operator")
            return result.to_dict()
        finally:
            await engine.stop()

    direct = asyncio.run(direct_run())
    for key in ("success", "action", "level", "targets_notified", "message", "error", "metadata"):
        assert actual[key] == direct[key]
    for key in ("state", "current_level", "notification_count", "incident_id", "policy_name"):
        assert actual["record"][key] == direct["record"][key]
    retained = command(runtime, identity, "record", record_id=record_id)
    assert retained["record"]["state"] == actual["record"]["state"]
    assert marker.read_text(encoding="utf-8").count("escalation-notify") == (
        2 if action == "escalate" else 1
    )
    assert command(runtime, identity, "stop")["running"] is False


@pytest.mark.parametrize("change", ["binding", "package", "policy", "release"])
def test_service_revocation_reaps_worker(runtime, monkeypatch, change):
    adapter, control, _, code, _ = runtime
    identity = prepare(runtime, monkeypatch, "runner")
    worker = adapter._session._checkpoint_services.workers[identity]
    if change == "binding":
        (control / "bindings.json").write_text("{}", encoding="utf-8")
    elif change == "package":
        code.write_text(code.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    elif change == "policy":
        adapter.policy = replace(adapter.policy, max_items=99)
    else:
        invoke(adapter, "resources_release", resource_id=identity)
    result = invoke(
        adapter,
        "checkpoint_services_execute",
        binding_id="service",
        service_id=identity,
        command="start",
        idempotency_key="revoked-start",
    )
    assert result["execution_status"] == "failed"
    worker.process.wait(timeout=5)
    assert worker.closed


def test_service_approval_and_closed_contract(runtime, monkeypatch):
    adapter, _, marker, _, _ = runtime
    request = prepare(runtime, monkeypatch, "runner", approved=False)
    assert invoke(adapter, "checkpoint_services_open", **request, factory="os.system")[
        "issues"
    ] == [{"code": "invalid_arguments"}]
    assert not marker.exists()


def test_lost_command_receipt_never_repeats_notification(runtime, monkeypatch):
    from truthound.mcp.policy import MCPError

    adapter, control, marker, _, _ = runtime
    identity = prepare(runtime, monkeypatch, "escalation")
    worker = adapter._session._checkpoint_services.workers[identity]
    request = dict(
        binding_id="service",
        service_id=identity,
        command="trigger",
        incident_id="uncertain",
        idempotency_key="uncertain-trigger",
    )
    pending = invoke(adapter, "checkpoint_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    original = worker.call

    async def lose_response(*args):
        await original(*args)
        raise MCPError("worker_error")

    monkeypatch.setattr(worker, "call", lose_response)
    result = invoke(adapter, "checkpoint_services_execute", **request)
    assert result["issues"] == [{"code": "outcome_unknown"}]
    effects = marker.read_bytes()
    assert effects.count(b"escalation-notify") == 1
    assert (
        invoke(adapter, "checkpoint_services_execute", **request)["execution_status"] != "succeeded"
    )
    assert marker.read_bytes() == effects
    assert worker.closed and worker.process.poll() is not None


def test_native_escalation_rejection_is_not_delivery_success(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "escalation")
    result = command(runtime, identity, "trigger", incident_id="local", policy_name="absent")
    direct = asyncio.run(native_engine().trigger("local", policy_name="absent")).to_dict()
    assert result == direct
    assert result["success"] is False
    adapter, _, marker, _, _ = runtime
    assert "escalation-notify" not in marker.read_text(encoding="utf-8")
    for extra in ({"context": {}}, {"force": True}, {"acknowledged_by": "someone"}):
        result = invoke(
            adapter,
            "checkpoint_services_execute",
            binding_id="service",
            service_id=identity,
            command="status",
            idempotency_key="bad-command",
            **extra,
        )
        assert result["issues"] == [{"code": "invalid_arguments"}]


def test_command_replay_does_not_redeliver_or_drain(runtime, monkeypatch):
    adapter, control, marker, _, _ = runtime
    identity = prepare(runtime, monkeypatch, "escalation")
    request = dict(
        binding_id="service",
        service_id=identity,
        command="trigger",
        incident_id="once",
        idempotency_key="trigger-once",
    )
    result = approve(adapter, control, "checkpoint_services_execute", **request)
    effects = marker.read_bytes()
    assert effects.count(b"escalation-notify") == 1
    assert invoke(adapter, "checkpoint_services_execute", **request) == result
    assert marker.read_bytes() == effects
