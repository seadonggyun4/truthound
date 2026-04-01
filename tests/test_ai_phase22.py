from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import truthound as th
from truthound.cli import app
from truthound.context import TruthoundContext
from truthound.core.suite import ValidationSuite

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract


def _proposal_input_ref(source_key: str):
    from truthound.ai import InputRef

    return InputRef(
        kind="schema_summary",
        ref=f"schema-summary:{source_key}",
        hash="schemahash001",
        redacted=True,
        metadata={"column_count": 4, "observed_count": 3},
    )


def _current_suite_snapshot():
    from truthound.ai import SuiteCheckSnapshot, ValidationSuiteSnapshot

    current_check = SuiteCheckSnapshot(
        check_id="null",
        check_key='null||{}',
        validator_name="null",
        category="completeness",
        columns=[],
        params={},
        tags=["completeness"],
        rationale="Always validate completeness across discovered columns.",
        origin="current",
    )
    return ValidationSuiteSnapshot(
        suite_name="truthound-auto-suite",
        check_count=1,
        schema_check_present=False,
        evidence_mode="summary",
        min_severity=None,
        checks=[current_check],
    )


def _added_snapshot(*, check_key: str, validator_name: str, category: str, columns: list[str], params: dict[str, object], rationale: str):
    from truthound.ai import SuiteCheckSnapshot

    return SuiteCheckSnapshot(
        check_id=check_key,
        check_key=check_key,
        validator_name=validator_name,
        category=category,
        columns=columns,
        params=params,
        tags=[category],
        rationale=rationale,
        origin="proposal",
    )


def _proposal_diff_preview(*, added_snapshot, current_snapshot=None):
    from truthound.ai import ValidationSuiteDiffCounts, ValidationSuiteDiffPreview, ValidationSuiteSnapshot

    current = current_snapshot or _current_suite_snapshot()
    proposed = ValidationSuiteSnapshot(
        suite_name=current.suite_name,
        check_count=current.check_count + 1,
        schema_check_present=current.schema_check_present,
        evidence_mode=current.evidence_mode,
        min_severity=current.min_severity,
        checks=[*current.checks, added_snapshot],
    )
    return ValidationSuiteDiffPreview(
        current_suite=current,
        proposed_suite=proposed,
        added=[added_snapshot],
        counts=ValidationSuiteDiffCounts(
            added=1,
            already_present=0,
            conflicts=0,
            rejected=0,
        ),
    )


def _make_ready_proposal(
    tmp_path: Path,
    *,
    source_key: str,
    validator_name: str,
    category: str,
    columns: list[str],
    params: dict[str, object],
    rationale: str,
):
    from truthound._applied_suite import canonical_check_key
    from truthound.ai import CompiledProposalCheck, SuiteProposalArtifact

    check_key = canonical_check_key(
        validator_name=validator_name,
        columns=columns,
        params=params,
    )
    added_snapshot = _added_snapshot(
        check_key=check_key,
        validator_name=validator_name,
        category=category,
        columns=columns,
        params=params,
        rationale=rationale,
    )
    return SuiteProposalArtifact(
        source_key=source_key,
        input_refs=[_proposal_input_ref(source_key)],
        model_provider="fake-openai",
        model_name="gpt-fake",
        prompt_hash=f"prompt-hash-{validator_name}",
        created_by="phase22-test",
        workspace_root=str(tmp_path),
        summary=f"Proposal for {validator_name}.",
        rationale="Reviewable proposal for dashboard lifecycle hardening.",
        checks=[
            CompiledProposalCheck(
                check_key=check_key,
                validator_name=validator_name,
                category=category,
                columns=columns,
                params=params,
                rationale=rationale,
            )
        ],
        risks=[],
        compile_status="ready",
        diff_preview=_proposal_diff_preview(added_snapshot=added_snapshot),
        compiled_check_count=1,
        rejected_check_count=0,
        compiler_errors=[],
    )


def test_truthound_root_probe_and_ai_root_exports_are_available():
    import truthound.ai as ai_namespace

    status = th.get_ai_support_status()

    assert th.has_ai_support() is True
    assert status.ready is True
    assert status.provider_name == "openai"
    assert callable(ai_namespace.list_proposals)
    assert callable(ai_namespace.show_proposal)
    assert callable(ai_namespace.list_analyses)
    assert callable(ai_namespace.show_analysis)
    assert callable(ai_namespace.approve_proposal)
    assert callable(ai_namespace.reject_proposal)
    assert callable(ai_namespace.apply_proposal)
    assert callable(ai_namespace.list_proposal_approval_events)
    assert issubclass(ai_namespace.ProviderConfigurationError, Exception)
    assert issubclass(ai_namespace.ProviderTransportError, Exception)
    assert issubclass(ai_namespace.ProviderResponseError, Exception)


def test_proposal_lifecycle_transitions_and_history_are_recorded(tmp_path: Path):
    from truthound.ai import AIArtifactStore, ActorRef, approve_proposal, list_proposal_approval_events, reject_proposal

    data = {
        "order_id": [1, 2, 3],
        "refund_rate": [10, 20, 30],
        "status": ["pending", "approved", "pending"],
    }
    context = TruthoundContext(tmp_path)
    proposal = _make_ready_proposal(
        tmp_path,
        source_key=context.resolve_source_key(data=data),
        validator_name="between",
        category="distribution",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 100},
        rationale="Refund rate should stay between zero and one hundred.",
    )
    AIArtifactStore(context).write_proposal(proposal)
    actor = ActorRef(actor_id="user-001", actor_name="Truthound Operator")

    approved = approve_proposal(
        proposal.artifact_id,
        actor=actor,
        comment="Looks good for controlled rollout.",
        context=context,
    )
    approved_noop = approve_proposal(
        proposal.artifact_id,
        actor=actor,
        comment="Repeated approve should no-op.",
        context=context,
    )
    rejected = reject_proposal(
        proposal.artifact_id,
        actor=actor,
        comment="Hold until after QA review.",
        context=context,
    )

    assert approved.proposal.approval_status == "approved"
    assert approved.changed is True
    assert approved.event is not None
    assert approved_noop.changed is False
    assert approved_noop.event is None
    assert rejected.proposal.approval_status == "rejected"
    assert rejected.changed is True

    history = list_proposal_approval_events(proposal.artifact_id, context=context)
    assert [event.action for event in history] == ["reject", "approve"]
    assert AIArtifactStore(context).read_proposal(proposal.artifact_id).approval_status == "rejected"


def test_apply_requires_approved_proposal(tmp_path: Path):
    from truthound.ai import AIArtifactStore, ActorRef, ProposalStateError, apply_proposal, reject_proposal

    data = {
        "order_id": [1, 2, 3],
        "refund_rate": [10, 20, 30],
        "status": ["pending", "approved", "pending"],
    }
    context = TruthoundContext(tmp_path)
    proposal = _make_ready_proposal(
        tmp_path,
        source_key=context.resolve_source_key(data=data),
        validator_name="between",
        category="distribution",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 100},
        rationale="Refund rate should stay between zero and one hundred.",
    )
    AIArtifactStore(context).write_proposal(proposal)
    actor = ActorRef(actor_id="user-001", actor_name="Truthound Operator")

    with pytest.raises(ProposalStateError, match="approved"):
        apply_proposal(proposal.artifact_id, actor=actor, context=context)

    reject_proposal(
        proposal.artifact_id,
        actor=actor,
        comment="Do not apply rejected proposal.",
        context=context,
    )
    with pytest.raises(ProposalStateError, match="approved"):
        apply_proposal(proposal.artifact_id, actor=actor, context=context)


def test_apply_persists_suite_record_and_th_check_consumes_applied_checks(tmp_path: Path):
    from truthound.ai import AIArtifactStore, ActorRef, approve_proposal, apply_proposal

    data = {
        "order_id": [1, 2, 3],
        "refund_rate": [10, 20, 30],
        "status": ["pending", "approved", "pending"],
        "customer_email": ["a@example.com", "b@example.com", "c@example.com"],
    }
    context = TruthoundContext(tmp_path)
    proposal = _make_ready_proposal(
        tmp_path,
        source_key=context.resolve_source_key(data=data),
        validator_name="between",
        category="distribution",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 100},
        rationale="Refund rate should stay between zero and one hundred.",
    )
    AIArtifactStore(context).write_proposal(proposal)
    actor = ActorRef(actor_id="user-001", actor_name="Truthound Operator")

    approve_proposal(
        proposal.artifact_id,
        actor=actor,
        comment="Approved for apply.",
        context=context,
    )
    apply_result = apply_proposal(
        proposal.artifact_id,
        actor=actor,
        context=context,
    )

    assert apply_result.changed is True
    assert apply_result.proposal.approval_status == "applied"
    assert apply_result.applied_check_count == 1
    assert context.suites_index_path.exists()
    index_payload = json.loads(context.suites_index_path.read_text(encoding="utf-8"))
    assert proposal.source_key in index_payload

    run_result = th.check(data, context=context)
    assert any(
        check.name == "between" and check.metadata.get("applied_suite") is True
        for check in run_result.checks
    )

    runner = CliRunner()
    doctor_result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])
    assert doctor_result.exit_code == 0


def test_runtime_skips_exact_duplicate_and_prefers_applied_conflict(tmp_path: Path):
    from truthound.ai import AIArtifactStore, ActorRef, approve_proposal, apply_proposal

    data = {
        "order_id": [1, 2, 3],
        "refund_rate": [10, 20, 30],
        "status": ["pending", "approved", "pending"],
        "customer_email": ["a@example.com", "b@example.com", "c@example.com"],
    }
    context = TruthoundContext(tmp_path)
    actor = ActorRef(actor_id="user-001", actor_name="Truthound Operator")

    duplicate_proposal = _make_ready_proposal(
        tmp_path,
        source_key=context.resolve_source_key(data=data),
        validator_name="unique",
        category="uniqueness",
        columns=["customer_email", "order_id", "refund_rate"],
        params={},
        rationale="Key-like columns should remain unique across the auto-suite signature.",
    )
    AIArtifactStore(context).write_proposal(duplicate_proposal)
    approve_proposal(
        duplicate_proposal.artifact_id,
        actor=actor,
        comment="Approved duplicate proposal.",
        context=context,
    )
    apply_proposal(
        duplicate_proposal.artifact_id,
        actor=actor,
        context=context,
    )

    duplicate_suite = ValidationSuite.from_legacy(
        context=context,
        data=data,
        validators=None,
    )
    unique_checks = [spec for spec in duplicate_suite.checks if spec.name == "unique"]
    assert len(unique_checks) == 1
    assert not any(spec.metadata.get("applied_suite") is True for spec in unique_checks)

    conflict_context = TruthoundContext(tmp_path / "conflict")
    conflict_proposal = _make_ready_proposal(
        tmp_path / "conflict",
        source_key=conflict_context.resolve_source_key(data=data),
        validator_name="range",
        category="distribution",
        columns=["order_id", "refund_rate"],
        params={"min_value": 0, "max_value": 100},
        rationale="Use an explicit applied range for the deterministic numeric signature.",
    )
    AIArtifactStore(conflict_context).write_proposal(conflict_proposal)
    approve_proposal(
        conflict_proposal.artifact_id,
        actor=actor,
        comment="Approved conflicting proposal.",
        context=conflict_context,
    )
    apply_proposal(
        conflict_proposal.artifact_id,
        actor=actor,
        context=conflict_context,
    )

    conflict_suite = ValidationSuite.from_legacy(
        context=conflict_context,
        data=data,
        validators=None,
    )
    range_checks = [spec for spec in conflict_suite.checks if spec.name == "range"]
    assert len(range_checks) == 1
    assert range_checks[0].metadata.get("applied_suite") is True
    assert range_checks[0].metadata["config"]["max_value"] == 100


def test_explicit_validators_ignore_applied_suite(tmp_path: Path):
    from truthound.ai import AIArtifactStore, ActorRef, approve_proposal, apply_proposal

    data = {
        "order_id": [1, 2, 3],
        "refund_rate": [10, 20, 30],
        "status": ["pending", "approved", "pending"],
    }
    context = TruthoundContext(tmp_path)
    proposal = _make_ready_proposal(
        tmp_path,
        source_key=context.resolve_source_key(data=data),
        validator_name="between",
        category="distribution",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 100},
        rationale="Refund rate should stay between zero and one hundred.",
    )
    AIArtifactStore(context).write_proposal(proposal)
    actor = ActorRef(actor_id="user-001", actor_name="Truthound Operator")
    approve_proposal(
        proposal.artifact_id,
        actor=actor,
        comment="Approved explicit-validator ignore test.",
        context=context,
    )
    apply_proposal(
        proposal.artifact_id,
        actor=actor,
        context=context,
    )

    explicit_suite = ValidationSuite.from_legacy(
        context=context,
        data=data,
        validators=["null"],
    )

    assert [spec.name for spec in explicit_suite.checks] == ["null"]
    assert not any(spec.metadata.get("applied_suite") is True for spec in explicit_suite.checks)


def test_cli_lifecycle_commands_and_history_surface_results(tmp_path: Path, monkeypatch):
    from truthound.ai import AIArtifactStore

    data = {
        "order_id": [1, 2, 3],
        "refund_rate": [10, 20, 30],
        "status": ["pending", "approved", "pending"],
    }
    context = TruthoundContext(tmp_path)
    proposal = _make_ready_proposal(
        tmp_path,
        source_key=context.resolve_source_key(data=data),
        validator_name="between",
        category="distribution",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 100},
        rationale="Refund rate should stay between zero and one hundred.",
    )
    AIArtifactStore(context).write_proposal(proposal)
    monkeypatch.setattr("truthound.cli_modules.ai.get_context", lambda: context)
    runner = CliRunner()

    approve_result = runner.invoke(
        app,
        [
            "ai",
            "proposals",
            "approve",
            proposal.artifact_id,
            "--actor-id",
            "user-001",
            "--actor-name",
            "Truthound Operator",
            "--comment",
            "Approved from CLI.",
            "--json",
        ],
    )
    assert approve_result.exit_code == 0
    approve_payload = json.loads(approve_result.output)
    assert approve_payload["proposal"]["approval_status"] == "approved"

    apply_result = runner.invoke(
        app,
        [
            "ai",
            "proposals",
            "apply",
            proposal.artifact_id,
            "--actor-id",
            "user-001",
            "--actor-name",
            "Truthound Operator",
            "--yes",
            "--json",
        ],
    )
    assert apply_result.exit_code == 0
    apply_payload = json.loads(apply_result.output)
    assert apply_payload["proposal"]["approval_status"] == "applied"
    assert apply_payload["applied_check_count"] == 1

    history_result = runner.invoke(
        app,
        ["ai", "proposals", "history", proposal.artifact_id, "--json"],
    )
    assert history_result.exit_code == 0
    history_payload = json.loads(history_result.output)
    assert [item["action"] for item in history_payload] == ["apply", "approve"]


def test_doctor_reports_invalid_applied_suites_index(tmp_path: Path):
    context = TruthoundContext(tmp_path)
    context.ensure_suites_workspace()
    context.suites_index_path.write_text("{not-json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])

    assert result.exit_code == 1
    assert "applied-suites-index-invalid" in result.output
