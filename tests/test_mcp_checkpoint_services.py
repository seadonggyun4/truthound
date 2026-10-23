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


def test_policy_edit_query_remove_matches_native_and_preserves_targets(runtime, monkeypatch):
    from dataclasses import replace as copy_policy

    identity = prepare(runtime, monkeypatch, "escalation")
    engine = native_engine()
    original = engine.get_policy("approved-policy")
    patch = dict(
        enabled=False,
        cooldown_minutes=30,
        max_escalations=2,
        severity_filter=["critical"],
        business_hours_only=True,
        business_hours_start=8,
        business_hours_end=17,
        business_days=[0, 2, 4],
        timezone="UTC",
    )
    engine.register_policy(copy_policy(original, **patch))
    actual = command(
        runtime, identity, "policy_update", policy_name="approved-policy", policy_update=patch
    )
    assert actual["policy"] == engine.get_policy("approved-policy").to_dict()
    assert actual["policy"]["levels"] == original.to_dict()["levels"]
    assert command(runtime, identity, "policy", policy_name="approved-policy") == actual
    denied = command(runtime, identity, "trigger", incident_id="disabled")
    assert denied == asyncio.run(engine.trigger("disabled")).to_dict()
    assert not denied["success"]
    assert command(runtime, identity, "policy_remove", policy_name="approved-policy") == {
        "removed": engine.unregister_policy("approved-policy")
    }
    assert command(
        runtime, identity, "policy_remove", key="remove-again", policy_name="approved-policy"
    ) == {"removed": False}
    assert command(
        runtime, identity, "policy", key="query-removed", policy_name="approved-policy"
    ) == {"policy": None}


@pytest.mark.parametrize("kind", ["runner", "escalation"])
def test_missing_native_management_target_never_reports_success(runtime, monkeypatch, kind):
    identity = prepare(runtime, monkeypatch, kind)
    adapter, control, marker, _, _ = runtime
    request = dict(binding_id="service", service_id=identity, idempotency_key="missing-target")
    if kind == "runner":
        request.update(command="run_once", checkpoint_name="missing")
    else:
        request.update(
            command="policy_update", policy_name="missing", policy_update={"enabled": False}
        )
    pending = invoke(adapter, "checkpoint_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    worker = adapter._session._checkpoint_services.workers[identity]
    response = invoke(adapter, "checkpoint_services_execute", **request)
    assert response["issues"] == [{"code": "outcome_unknown"}]
    effects = marker.read_bytes()
    assert b"checkpoint-action" not in effects and b"escalation-notify" not in effects
    assert (
        invoke(adapter, "checkpoint_services_execute", **request)["execution_status"] != "succeeded"
    )
    assert marker.read_bytes() == effects
    assert worker.closed
    assert "escalation-notify" not in runtime[2].read_text(encoding="utf-8")


def test_incident_and_active_records_follow_native_resolution(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "escalation")
    command(runtime, identity, "start")
    initial = command(
        runtime, identity, "trigger", incident_id="query-incident", policy_name="approved-policy"
    )["record"]
    active = command(runtime, identity, "active_records", policy_name="approved-policy")
    incident = command(runtime, identity, "incident_records", incident_id="query-incident")
    assert len(active["records"]) == len(incident["records"]) == 1

    def stable(record):
        return {k: v for k, v in record.items() if k != "duration_seconds"}

    assert stable(active["records"][0]) == stable(incident["records"][0]) == stable(initial)
    assert (
        0
        <= initial["duration_seconds"]
        <= active["records"][0]["duration_seconds"]
        <= incident["records"][0]["duration_seconds"]
    )
    command(runtime, identity, "resolve", record_id=initial["id"])
    assert command(runtime, identity, "active_records", key="resolved-active")["records"] == []
    retained = command(
        runtime, identity, "incident_records", key="resolved-incident", incident_id="query-incident"
    )["records"]

    async def direct():
        engine = native_engine()
        await engine.start()
        try:
            result = await engine.trigger("query-incident", policy_name="approved-policy")
            await engine.resolve(result.record.id, "mcp-approved-operator")
            assert engine.get_active_escalations() == []
            return engine.get_records_for_incident("query-incident")[0].to_dict()
        finally:
            await engine.stop()

    expected = asyncio.run(direct())
    assert len(retained) == 1
    for key in ("state", "current_level", "incident_id", "policy_name", "notification_count"):
        assert retained[0][key] == expected[key]
    command(runtime, identity, "stop")


@pytest.mark.parametrize("action", ["run_once", "run_all"])
def test_runner_immediate_execution_and_removal_native_parity(
    runtime, monkeypatch, tmp_path, action
):
    from truthound.checkpoint import Checkpoint
    from truthound.checkpoint.runner import CheckpointRunner
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig

    identity = prepare(runtime, monkeypatch, "runner")
    args = {"checkpoint_name": "approved-checkpoint"} if action == "run_once" else {}
    actual = command(runtime, identity, action, **args)
    with SQLiteDataSource(
        database=str(tmp_path / "operator.db"),
        table="items",
        config=SQLiteDataSourceConfig(name="approved-source"),
    ) as source:
        runner = CheckpointRunner()
        runner.add_checkpoint(
            Checkpoint(name="approved-checkpoint", data_source=source, validators=["null"])
        )
        direct = (
            runner.run_once("approved-checkpoint") if action == "run_once" else runner.run_all()[0]
        )
        row = actual["result"] if action == "run_once" else actual["results"][0]
        assert row["status"] == direct.status.value
        for key in ("checks", "issues", "execution_issues", "row_count", "column_count"):
            assert row["validation_run"][key] == direct.validation_run.to_dict()[key]
        assert command(
            runtime, identity, "remove_checkpoint", checkpoint_name="approved-checkpoint"
        ) == {"removed": runner.remove_checkpoint("approved-checkpoint")}
        assert command(
            runtime,
            identity,
            "remove_checkpoint",
            key="remove-again",
            checkpoint_name="approved-checkpoint",
        ) == {"removed": False}
        assert (
            command(runtime, identity, "run_all", key="empty-run")["results"]
            == runner.run_all()
            == []
        )
    assert runtime[2].read_text(encoding="utf-8").count("checkpoint-action") == 1


def test_policy_edit_rejects_new_destinations_and_wrong_arguments(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "escalation")
    adapter = runtime[0]
    for patch in (
        {},
        {"enabled": None},
        {"enabled": "yes"},
        {"timezone": 123},
        {"business_days": [7]},
        {"levels": []},
        {"targets": ["https://example.com"]},
    ):
        response = invoke(
            adapter,
            "checkpoint_services_execute",
            binding_id="service",
            service_id=identity,
            command="policy_update",
            policy_name="approved-policy",
            policy_update=patch,
            idempotency_key="invalid-policy",
        )
        assert response["issues"] == [{"code": "invalid_arguments"}]
    assert command(runtime, identity, "policy", policy_name="approved-policy")["policy"] == (
        native_engine().get_policy("approved-policy").to_dict()
    )


def test_policy_edit_requires_exact_approval_and_replay_does_not_restore(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch, "escalation")
    adapter, control, _, _, _ = runtime
    request = dict(
        binding_id="service",
        service_id=identity,
        command="policy_update",
        policy_name="approved-policy",
        policy_update={"enabled": False},
        idempotency_key="disable-policy",
    )
    pending = invoke(adapter, "checkpoint_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    assert command(runtime, identity, "policy", policy_name="approved-policy")["policy"]["enabled"]
    store = ApprovalStore(control)
    try:
        store.approve(pending["data"]["plan_id"])
    finally:
        store.close()
    altered = {**request, "policy_update": {"enabled": True}}
    assert (
        invoke(adapter, "checkpoint_services_execute", **altered)["execution_status"] != "succeeded"
    )
    result = invoke(adapter, "checkpoint_services_execute", **request)
    assert result["execution_status"] == "succeeded"
    command(runtime, identity, "policy_remove", policy_name="approved-policy")
    assert invoke(adapter, "checkpoint_services_execute", **request) == result
    assert command(
        runtime, identity, "policy", key="after-replay", policy_name="approved-policy"
    ) == {"policy": None}
