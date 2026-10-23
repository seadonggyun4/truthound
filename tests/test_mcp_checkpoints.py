"""Real installed factories, native Checkpoint parity, approval and lifecycle."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from truthound.mcp.approval import ApprovalStore

pytestmark = pytest.mark.contract
runtime = plugins.runtime
invoke = plugins.invoke


def prepare(runtime, monkeypatch, *, approve=True):
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
    config["plugins"]["fixture"]["capabilities"]["checkpoint-factory"] = "checkpoint-factory"
    config["checkpoints"] = {"daily": "checkpoint-factory"}
    path.write_text(json.dumps(config), encoding="utf-8")
    request = {"binding_id": "daily", "idempotency_key": "native-checkpoint"}
    pending = invoke(adapter, "checkpoints_run", **request)
    assert pending["issues"] == [{"code": "approval_required"}], pending
    assert "operator_configured_native_action_retries" in pending["data"]["effects"]
    assert not marker.exists()
    if approve:
        store = ApprovalStore(control)
        try:
            store.approve(pending["data"]["plan_id"])
        finally:
            store.close()
    return request


def test_native_checkpoint_result_and_action_cleanup(runtime, monkeypatch, tmp_path):
    from truthound.checkpoint import Checkpoint, CheckpointResult
    from truthound.datasources.sql.sqlite import SQLiteDataSource, SQLiteDataSourceConfig

    adapter, _, marker, _, _ = runtime
    request = prepare(runtime, monkeypatch)
    response = invoke(adapter, "checkpoints_run", **request)
    assert response["execution_status"] == "succeeded", response
    result = plugins.source_artifact(adapter, response["data"]["resource_id"])
    with SQLiteDataSource(
        database=str(tmp_path / "operator.db"),
        table="items",
        config=SQLiteDataSourceConfig(name="approved-source"),
    ) as source:
        direct = Checkpoint(
            name="approved-checkpoint", data_source=source, validators=["null"]
        ).run()
    native = CheckpointResult.from_dict(result)
    assert native.status == direct.status
    assert native.validation_run is not None and direct.validation_run is not None
    for key in ("checks", "issues", "execution_issues", "row_count", "column_count"):
        assert native.validation_run.to_dict()[key] == direct.validation_run.to_dict()[key]
    assert result["action_results"][0]["status"] == "success"
    assert result["action_results"][0]["details"] == {"status": direct.status.value}
    effects = marker.read_text(encoding="utf-8")
    assert effects.splitlines()[-3:] == ["source-close", "checkpoint-close", "teardown"]
    assert effects.count("checkpoint-action") == 1
    assert invoke(adapter, "checkpoints_run", **request) == response
    assert marker.read_text(encoding="utf-8") == effects


@pytest.mark.parametrize("change", ["binding", "package", "policy", "release"])
def test_checkpoint_artifact_authority_is_revocable(runtime, monkeypatch, change):
    adapter, control, _, code, _ = runtime
    request = prepare(runtime, monkeypatch)
    response = invoke(adapter, "checkpoints_run", **request)
    assert response["execution_status"] == "succeeded", response
    identity = response["data"]["resource_id"]
    if change == "binding":
        (control / "bindings.json").write_text("{}", encoding="utf-8")
    elif change == "package":
        code.write_text(code.read_text(encoding="utf-8") + "\n# altered\n", encoding="utf-8")
    elif change == "policy":
        adapter.policy = replace(adapter.policy, max_items=99)
    else:
        assert (
            invoke(adapter, "resources_release", resource_id=identity)["execution_status"]
            == "succeeded"
        )
    assert invoke(adapter, "artifacts_read", resource_id=identity)["execution_status"] == "failed"
    assert invoke(adapter, "checkpoints_run", **request)["execution_status"] == "failed"


def test_checkpoint_approval_and_closed_input(runtime, monkeypatch):
    adapter, _, marker, _, _ = runtime
    request = prepare(runtime, monkeypatch, approve=False)
    assert invoke(adapter, "checkpoints_run", **request)["issues"] == [
        {"code": "approval_required"}
    ]
    for extra in (
        {"factory": "os.system"},
        {"context": {}},
        {"actions": []},
        {"data_source": "private"},
    ):
        assert invoke(adapter, "checkpoints_run", **request, **extra)["issues"] == [
            {"code": "invalid_arguments"}
        ]
    assert not marker.exists()


@pytest.mark.parametrize("router", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_native_actions_router_failure_and_retry(runtime, monkeypatch, tmp_path, router, fail):
    adapter, _, marker, _, _ = runtime
    (tmp_path / "checkpoint-settings.json").write_text(
        json.dumps({"router": router, "fail": fail, "retries": 1}), encoding="utf-8"
    )
    request = prepare(runtime, monkeypatch)
    response = invoke(adapter, "checkpoints_run", **request)
    assert response["execution_status"] == "succeeded", response
    result = plugins.source_artifact(adapter, response["data"]["resource_id"])
    assert result["action_results"][0]["status"] == ("error" if fail else "success")
    if fail:
        assert result["status"] == "error"
        assert result["action_results"][0]["error"] == "synthetic-action-error"
    effects = marker.read_text(encoding="utf-8")
    assert effects.count("checkpoint-action") == (2 if fail else 1)
    assert invoke(adapter, "checkpoints_run", **request) == response
    assert marker.read_text(encoding="utf-8") == effects


@pytest.mark.parametrize("fault", ["wrong", "block"])
def test_uncertain_checkpoint_is_not_reexecuted(runtime, monkeypatch, tmp_path, fault):
    from truthound.mcp import checkpoints

    adapter, _, marker, _, _ = runtime
    (tmp_path / "checkpoint-settings.json").write_text(json.dumps({fault: True}), encoding="utf-8")
    if fault == "block":
        adapter.policy = replace(adapter.policy, timeout_seconds=2)
    request = prepare(runtime, monkeypatch)
    created = []
    original = checkpoints.CheckpointProcess

    def track(ttl):
        worker = original(ttl)
        created.append(worker)
        return worker

    monkeypatch.setattr(checkpoints, "CheckpointProcess", track)
    for _ in range(2):
        result = invoke(adapter, "checkpoints_run", **request)
        assert result["issues"] == [{"code": "outcome_unknown"}], result
    assert len(created) == 1 and created[0].closed
    assert created[0].process.poll() is not None
    if marker.exists():
        assert "checkpoint-action" not in marker.read_text(encoding="utf-8")
