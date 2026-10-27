"""AI lifecycle uses native compilation and distinct operator review/apply approvals."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests import test_mcp_plugin_bindings as plugins
from tests.test_mcp_brokers import approved

runtime = plugins.runtime
invoke = plugins.invoke
pytestmark = pytest.mark.contract


def prepare(runtime, monkeypatch):
    from truthound.mcp import model_process
    from truthound.mcp.ai_services import AIServices

    created = []
    original = AIServices.process_type

    def tracked(ttl):
        worker = original(ttl)
        created.append(worker)
        return worker

    monkeypatch.setattr(AIServices, "process_type", staticmethod(tracked))

    adapter, control, marker, _, _ = runtime
    root = control.parent / "ai-workspace"
    root.mkdir()
    adapter.policy = replace(adapter.policy, workspace_directories=(root,))
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
    settings = json.loads(path.read_text())
    settings["plugins"]["fixture"]["capabilities"]["ai-factory"] = "ai-factory"
    settings["ai_services"] = {"local-ai": "ai-factory"}
    path.write_text(json.dumps(settings), encoding="utf-8")
    try:
        result = approved(
            runtime, "ai_services_open", binding_id="local-ai", idempotency_key="ai-open-01"
        )
    except AssertionError as error:
        allowed = {
            "worker_error",
            "timeout",
            "invalid_handle",
            "exchange_failed",
            "access_denied",
            "invalid_binding",
        }
        error.add_note(
            json.dumps(
                {
                    "ai_owners": [
                        {
                            "failure_category": worker.failure_category
                            if getattr(worker, "failure_category", None) in allowed
                            else "other",
                            "exit_status_after_cleanup": getattr(
                                worker, "failure_exit_status_after_cleanup", None
                            ),
                        }
                        for worker in created
                    ]
                }
            )
        )
        raise
    assert "ai-provider" not in marker.read_text()
    return result["data"]["resource_id"]


def command(runtime, identity, action, *, key=None, **arguments):
    response = approved(
        runtime,
        "ai_services_execute",
        binding_id="local-ai",
        service_id=identity,
        command=action,
        idempotency_key=key or "ai-command-" + action,
        **arguments,
    )
    return plugins.source_artifact(runtime[0], response["data"]["resource_id"])


def test_native_ai_compile_review_apply_and_analysis_lifecycle(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch)
    adapter, control, marker, _, _ = runtime
    request = dict(
        binding_id="local-ai",
        service_id=identity,
        command="suggest",
        prompt="Check uniqueness of x",
        idempotency_key="ai-suggestion-request",
    )
    before = marker.read_bytes()
    pending = invoke(adapter, "ai_services_execute", **request)
    assert pending["issues"] == [{"code": "approval_required"}]
    assert marker.read_bytes() == before
    response = approved(runtime, "ai_services_execute", **request)
    proposal = plugins.source_artifact(adapter, response["data"]["resource_id"])
    assert proposal["checks"][0]["validator_name"] == "unique"
    assert "workspace_root" not in proposal
    assert marker.read_text().count("ai-provider") == 1
    assert invoke(adapter, "ai_services_execute", **request) == response
    assert marker.read_text().count("ai-provider") == 1
    proposal_id = proposal["artifact_id"]
    decision = command(
        runtime,
        identity,
        "approve",
        proposal_id=proposal_id,
        expected_sha256=proposal["artifact_sha256"],
    )
    assert decision["proposal"]["approval_status"] == "approved"
    applied = command(
        runtime,
        identity,
        "apply",
        proposal_id=proposal_id,
        expected_sha256=decision["artifact_sha256"],
    )
    assert applied["proposal"]["approval_status"] == "applied"
    assert applied["changed"] is True
    from truthound.ai import show_proposal
    from truthound.context import TruthoundContext

    context = TruthoundContext(control.parent / "ai-workspace")
    native = show_proposal(proposal_id, context=context)
    assert applied["proposal"]["checks"] == [item.model_dump(mode="json") for item in native.checks]
    assert len(command(runtime, identity, "history", proposal_id=proposal_id)["events"]) == 2
    analysis = command(runtime, identity, "explain")
    assert analysis["summary"] == "Local run analysis"
    stored = command(runtime, identity, "analysis", analysis_id=analysis["artifact_id"])
    assert stored == analysis
    assert len(command(runtime, identity, "list_analyses")["analyses"]) == 1
    assert marker.read_text().count("ai-provider") == 2


def test_ai_wire_cannot_supply_provider_model_actor_or_path(runtime, monkeypatch):
    identity = prepare(runtime, monkeypatch)
    adapter = runtime[0]
    before = runtime[2].read_bytes()
    for arguments in (
        {"provider": "paid"},
        {"model": "arbitrary"},
        {"actor_id": "admin"},
        {"run_id": "../private"},
    ):
        result = invoke(
            adapter,
            "ai_services_execute",
            binding_id="local-ai",
            service_id=identity,
            command="explain",
            idempotency_key="reject-ai-overrides",
            **arguments,
        )
        assert result["issues"] == [{"code": "invalid_arguments"}]
    assert runtime[2].read_bytes() == before


def test_ai_application_rejects_a_stale_approved_proposal_revision(runtime, monkeypatch):
    from truthound.ai import show_proposal
    from truthound.context import TruthoundContext
    from truthound.mcp.approval import ApprovalStore

    identity = prepare(runtime, monkeypatch)
    proposal = command(runtime, identity, "suggest", prompt="Check uniqueness of x")
    decision = command(
        runtime,
        identity,
        "approve",
        proposal_id=proposal["artifact_id"],
        expected_sha256=proposal["artifact_sha256"],
    )
    assert decision["artifact_sha256"] != proposal["artifact_sha256"]
    request = dict(
        binding_id="local-ai",
        service_id=identity,
        command="apply",
        proposal_id=proposal["artifact_id"],
        expected_sha256=proposal["artifact_sha256"],
        idempotency_key="stale-proposal-apply",
    )
    pending = invoke(runtime[0], "ai_services_execute", **request)
    store = ApprovalStore(runtime[1])
    store.approve(pending["data"]["plan_id"])
    store.close()
    result = invoke(runtime[0], "ai_services_execute", **request)
    assert result["issues"] == [{"code": "outcome_unknown"}]
    context = TruthoundContext(runtime[1].parent / "ai-workspace")
    assert show_proposal(proposal["artifact_id"], context=context).approval_status == "approved"
