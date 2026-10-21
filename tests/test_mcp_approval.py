"""Exact operator authority and uncertain persistent effects, with real Core IO."""

import asyncio
import json

import pytest

pytest.importorskip("mcp")

from truthound.mcp.approval import ApprovalStore
from truthound.mcp.policy import MCPError, MCPPolicy
from truthound.mcp.service import MCPService

pytestmark = pytest.mark.contract


def invoke(service, name, **arguments):
    return asyncio.run(service.call_operation_async(name, arguments))


@pytest.fixture
def operator(tmp_path):
    data, workspace, control = (tmp_path / name for name in ("data", "workspace", "control"))
    for path in (data, workspace, control):
        path.mkdir(mode=0o700)
    service = MCPService(
        MCPPolicy(
            (data,), profile="core", workspace_directories=(workspace,), control_directory=control
        )
    )
    yield service, workspace, control
    service.close()


def test_core_workspace_requires_out_of_band_exact_approval_and_deduplicates(operator):
    service, workspace, control = operator
    opened = invoke(service, "contexts_open", path=str(workspace))
    assert opened["execution_status"] == "succeeded", opened
    context = opened["data"]["context_id"]
    assert not (workspace / ".truthound").exists()
    requested = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="create-workspace-1"
    )
    assert requested["issues"] == [{"code": "approval_required"}], requested
    plan = requested["data"]["plan_id"]
    assert not (workspace / ".truthound").exists()
    forged = invoke(
        service,
        "contexts_initialize",
        context_id=context,
        idempotency_key="create-workspace-1",
        approved=True,
    )
    assert forged["issues"] == [{"code": "invalid_arguments"}]
    store = ApprovalStore(control)
    try:
        review = store.review(plan)
        assert review["request"]["root"] == str(workspace)
        store.approve(plan)
    finally:
        store.close()
    result = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="create-workspace-1"
    )
    assert result["execution_status"] == "succeeded" and not result["data"]["replayed"], result
    assert (workspace / ".truthound/config.yaml").is_file()
    before = (workspace / ".truthound/config.yaml").stat().st_mtime_ns
    repeated = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="create-workspace-1"
    )
    assert (
        repeated["data"]["replayed"]
        and (workspace / ".truthound/config.yaml").stat().st_mtime_ns == before
    )
    assert ".approval-key" not in json.dumps(result) and "signature" not in json.dumps(result)


def test_changed_plan_and_revoked_operator_receipt_never_execute(operator):
    service, workspace, control = operator
    context = invoke(service, "contexts_open", path=str(workspace))["data"]["context_id"]
    first = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="exact-plan-1"
    )
    plan = first["data"]["plan_id"]
    store = ApprovalStore(control)
    try:
        store.approve(plan)
        (workspace / "truthound.yaml").write_text(
            "context:\n  persist_docs: false\n", encoding="utf-8"
        )
        changed = invoke(
            service, "contexts_initialize", context_id=context, idempotency_key="exact-plan-1"
        )
        assert changed["issues"] == [{"code": "approval_required"}]
        assert changed["data"]["plan_id"] != plan
        store.approve(changed["data"]["plan_id"])
        store.revoke(changed["data"]["plan_id"])
        denied = invoke(
            service, "contexts_initialize", context_id=context, idempotency_key="exact-plan-1"
        )
        assert denied["execution_status"] != "succeeded"
        assert not (workspace / ".truthound").exists()
    finally:
        store.close()


def test_worker_timeout_preserves_indeterminate_effect_and_blocks_retry(operator, monkeypatch):
    service, workspace, control = operator
    context = invoke(service, "contexts_open", path=str(workspace))["data"]["context_id"]
    requested = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="uncertain-effect-1"
    )
    plan = requested["data"]["plan_id"]
    store = ApprovalStore(control)
    store.approve(plan)
    store.close()

    async def timed_out(*args):
        raise TimeoutError

    monkeypatch.setattr(service, "_execute_request", timed_out)
    result = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="uncertain-effect-1"
    )
    assert result["execution_status"] == "outcome_unknown"
    assert result["data"]["state"] == "outcome_unknown"
    repeated = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="uncertain-effect-1"
    )
    assert repeated["execution_status"] == "outcome_unknown"
    assert not (workspace / ".truthound").exists()


def test_context_symlink_escape_and_cross_session_plan_are_denied(operator, tmp_path):
    service, workspace, control = operator
    context = invoke(service, "contexts_open", path=str(workspace))["data"]["context_id"]
    plan = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="owned-plan-1"
    )["data"]["plan_id"]
    other = MCPService(service.policy)
    try:
        assert invoke(other, "plans_status", plan_id=plan)["issues"] == [{"code": "invalid_plan"}]
    finally:
        other.close()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".truthound").symlink_to(outside, target_is_directory=True)
    result = invoke(
        service, "contexts_initialize", context_id=context, idempotency_key="owned-plan-1"
    )
    assert result["issues"] == [{"code": "access_denied"}]
    assert not list(outside.iterdir())


def test_expired_forged_and_conflicting_intents_cannot_authorize(tmp_path):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    now = [1.0]
    store = ApprovalStore(control, clock=lambda: now[0])
    try:
        plan = store.prepare(
            "operation",
            key="key",
            request={"target": "one"},
            policy="policy",
            session="session",
            effects=["workspace_write"],
        )
        with pytest.raises(MCPError, match="idempotency_conflict"):
            store.prepare(
                "operation",
                key="key",
                request={"target": "two"},
                policy="policy",
                session="session",
                effects=["workspace_write"],
            )
        store.approve(plan["plan_id"])
        store.connection.execute(
            "UPDATE plans SET signature='model-invented-signature' WHERE identity=?",
            (plan["plan_id"],),
        )
        with pytest.raises(MCPError, match="approval_required"):
            store.begin(plan["plan_id"], session="session", policy="policy")
        now[0] += 301
        with pytest.raises(MCPError, match="plan_expired"):
            store.begin(plan["plan_id"], session="session", policy="policy")
    finally:
        store.close()
