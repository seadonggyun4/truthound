from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from truthound.cli import app
from truthound.context import TruthoundContext

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract


def _make_diff_preview():
    from truthound.ai import (
        SuiteCheckSnapshot,
        ValidationSuiteDiffCounts,
        ValidationSuiteDiffPreview,
        ValidationSuiteSnapshot,
    )

    current_check = SuiteCheckSnapshot(
        check_id="null",
        check_key="null||{}",
        validator_name="null",
        category="completeness",
        columns=[],
        params={},
        tags=["completeness"],
        rationale="Always validate completeness across discovered columns.",
        origin="current",
    )
    added_check = SuiteCheckSnapshot(
        check_id="unique|order_id|{}",
        check_key="unique|order_id|{}",
        validator_name="unique",
        category="uniqueness",
        columns=["order_id"],
        params={},
        tags=["uniqueness"],
        rationale="Order ids should remain unique.",
        origin="proposal",
    )
    return ValidationSuiteDiffPreview(
        current_suite=ValidationSuiteSnapshot(
            suite_name="truthound-auto-suite",
            check_count=1,
            schema_check_present=False,
            evidence_mode="summary",
            min_severity=None,
            checks=[current_check],
        ),
        proposed_suite=ValidationSuiteSnapshot(
            suite_name="truthound-auto-suite",
            check_count=2,
            schema_check_present=False,
            evidence_mode="summary",
            min_severity=None,
            checks=[current_check, added_check],
        ),
        added=[added_check],
        counts=ValidationSuiteDiffCounts(
            added=1,
            already_present=0,
            conflicts=0,
            rejected=0,
        ),
    )


def _make_proposal(tmp_path: Path):
    from truthound.ai import CompiledProposalCheck, InputRef, SuiteProposalArtifact

    return SuiteProposalArtifact(
        source_key="source:orders",
        input_refs=[
            InputRef(
                kind="schema_summary",
                ref="catalog:orders",
                hash="abc123",
                redacted=True,
                metadata={"columns": ["order_id", "refund_rate"]},
            )
        ],
        model_provider="openai",
        model_name="gpt-4o-mini",
        prompt_hash="prompt-hash-001",
        created_by="phase0-test",
        workspace_root=str(tmp_path),
        summary="Order validations should emphasize aggregate refund anomalies.",
        rationale="Refund rate drift is a key business risk for the source.",
        checks=[
            CompiledProposalCheck(
                check_key="unique|order_id|{}",
                validator_name="unique",
                category="uniqueness",
                columns=["order_id"],
                params={},
                rationale="Order ids should remain unique.",
            )
        ],
        risks=["False positives on seasonal spikes."],
        compile_status="ready",
        diff_preview=_make_diff_preview(),
        compiled_check_count=1,
        rejected_check_count=0,
    )


def _make_analysis(tmp_path: Path):
    from truthound._ai_contract import analysis_artifact_id_for_run
    from truthound.ai import InputRef, RunAnalysisArtifact

    run_id = "run_20260401_120000_abcd1234"
    return RunAnalysisArtifact(
        artifact_id=analysis_artifact_id_for_run(run_id),
        source_key="source:orders",
        input_refs=[
            InputRef(
                kind="run_result",
                ref=f"runs:{run_id}",
                hash="runhash001",
                redacted=True,
                metadata={"issue_count": 3},
            ),
            InputRef(
                kind="docs_artifact",
                ref=f"docs:{run_id}",
                hash="docshash001",
                redacted=True,
                metadata={"available": True},
            ),
        ],
        model_provider="openai",
        model_name="gpt-4o-mini",
        prompt_hash="prompt-hash-002",
        created_by="phase0-test",
        workspace_root=str(tmp_path),
        run_id=run_id,
        summary="Refund-related checks failed with concentrated medium severity issues.",
        evidence_refs=[f"runs:{run_id}", f"docs:{run_id}"],
        failed_checks=["refund-rate-threshold"],
        top_columns=["refund_rate"],
        recommended_next_actions=["Review refund_rate trend against weekly baseline."],
        history_window={
            "included": True,
            "history_key": "source:orders",
            "window_size": 10,
            "run_count": 5,
            "failure_count": 2,
            "success_count": 3,
            "latest_run_id": run_id,
            "recent_statuses": ["success", "failure", "failure"],
        },
    )


def test_ai_artifact_store_round_trips_and_creates_lazy_workspace(tmp_path: Path):
    from truthound.ai import AIArtifactStore, ApprovalLogEvent

    context = TruthoundContext(tmp_path)
    store = AIArtifactStore(context)

    assert not (context.workspace_dir / "ai").exists()

    proposal = _make_proposal(tmp_path)
    proposal_path = store.write_proposal(proposal)

    assert proposal_path == context.workspace_dir / "ai" / "proposals" / f"{proposal.artifact_id}.json"
    assert proposal_path.exists()
    assert (context.workspace_dir / "ai" / "analyses").is_dir()
    assert (context.workspace_dir / "ai" / "approvals").is_dir()

    loaded_proposal = store.read_proposal(proposal.artifact_id)
    assert loaded_proposal.artifact_id == proposal.artifact_id
    assert len(store.list_proposals()) == 1

    analysis = _make_analysis(tmp_path)
    analysis_path = store.write_analysis(analysis)
    assert analysis_path == context.workspace_dir / "ai" / "analyses" / f"{analysis.artifact_id}.json"
    assert store.read_analysis(analysis.artifact_id).run_id == analysis.run_id
    assert len(store.list_analyses()) == 1

    approval_log_path = store.append_approval(
        ApprovalLogEvent(
            proposal_id=proposal.artifact_id,
            action="approve",
            actor_id="operator:demo",
            actor_name="Demo Operator",
            comment="Approved for Phase 1 execution handoff.",
            diff_hash="diffhash001",
        )
    )
    assert approval_log_path.exists()
    lines = approval_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("{") and lines[0].endswith("}")
    assert not lines[0].startswith("  ")
    assert len(store.list_approval_events(proposal_id=proposal.artifact_id)) == 1


def test_phase0_redaction_blocks_pii_literals_and_row_level_prompt_content(tmp_path: Path):
    from truthound.ai import CompiledProposalCheck, InputRef, StructuredProviderRequest, SuiteProposalArtifact

    with pytest.raises(ValueError, match="Summary-only redaction rejected"):
        SuiteProposalArtifact(
            source_key="source:customers",
            input_refs=[InputRef(kind="schema_summary", ref="catalog:customers", redacted=True)],
            model_provider="openai",
            model_name="gpt-4o-mini",
            prompt_hash="hash-raw-001",
            created_by="phase0-test",
            workspace_root=str(tmp_path),
            summary="Customer alice@example.com triggered the pattern.",
            rationale="Risk explanation.",
            checks=[
                CompiledProposalCheck(
                    check_key="email|customer_email|{}",
                    validator_name="email",
                    category="string",
                    columns=["customer_email"],
                    params={},
                    rationale="Email values should follow email format.",
                )
            ],
            risks=["Operator confusion."],
            compile_status="ready",
            diff_preview=_make_diff_preview(),
            compiled_check_count=1,
            rejected_check_count=0,
        )

    with pytest.raises(ValueError, match="Summary-only redaction rejected"):
        StructuredProviderRequest(
            provider_name="openai",
            model_name="gpt-4o-mini",
            system_prompt="Summarize aggregate quality posture only.",
            user_prompt="sample row: email=alice@example.com, refund_rate=0.95",
            response_format_name="suite_proposal",
        )


def test_doctor_workspace_accepts_healthy_ai_layout(tmp_path: Path):
    from truthound.ai import AIArtifactStore

    context = TruthoundContext(tmp_path)
    store = AIArtifactStore(context)
    store.write_proposal(_make_proposal(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])

    assert result.exit_code == 0
    assert "found no structural issues" in result.output


def test_doctor_workspace_reports_ai_guardrail_failures(tmp_path: Path):
    context = TruthoundContext(tmp_path)
    ai_root = context.workspace_dir / "ai"
    proposals_dir = ai_root / "proposals"
    analyses_dir = ai_root / "analyses"
    approvals_dir = ai_root / "approvals"
    proposals_dir.mkdir(parents=True)
    analyses_dir.mkdir(parents=True)
    approvals_dir.mkdir(parents=True)

    bad_artifact_path = proposals_dir / "suite-proposal-20260401120000-abcdef.json"
    bad_artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "0",
                "artifact_id": "suite-proposal-20260401120000-fedcba",
                "artifact_type": "suite_proposal",
                "source_key": "source:orders",
                "input_refs": [],
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "prompt_hash": "hash-001",
                "compiler_version": "bad-version",
                "approval_status": "pending",
                "redaction_policy": {
                    "mode": "summary_only",
                    "raw_samples_allowed": False,
                    "pii_literals_allowed": False,
                },
                "created_at": "2026-04-01T00:00:00+00:00",
                "created_by": "phase0-test",
                "workspace_root": str(tmp_path),
                "target_type": "validation_suite",
                "summary": "alice@example.com should be checked directly.",
                "rationale": "Risk explanation.",
                "checks": [],
                "risks": [],
                "compile_status": "ready",
                "diff_preview": {},
            }
        ),
        encoding="utf-8",
    )

    outside_artifact = tmp_path / "outside-analysis.json"
    outside_artifact.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "artifact_id": "run-analysis-run_20260401_120000_abcd1234",
                "artifact_type": "run_analysis",
                "source_key": "source:orders",
                "input_refs": [],
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "prompt_hash": "hash-002",
                "compiler_version": "phase0-schema-v1",
                "approval_status": "not_required",
                "redaction_policy": {
                    "mode": "summary_only",
                    "raw_samples_allowed": False,
                    "pii_literals_allowed": False,
                },
                "created_at": "2026-04-01T00:00:00+00:00",
                "created_by": "phase0-test",
                "workspace_root": str(tmp_path),
                "run_id": "run_20260401_120000_abcd1234",
                "summary": "Aggregate summary only.",
                "evidence_refs": ["run:1"],
                "failed_checks": ["refund-rate-threshold"],
                "top_columns": ["refund_rate"],
                "recommended_next_actions": ["Review weekly trend."],
                "history_window": {"window": "7d"},
            }
        ),
        encoding="utf-8",
    )

    symlink_path = analyses_dir / "run-analysis-run_20260401_120000_abcd1234.json"
    try:
        symlink_path.symlink_to(outside_artifact)
    except OSError:
        pytest.skip("Symlinks are not available in this environment")

    (approvals_dir / "approval-log.jsonl").write_text("not-json\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace", "--format", "json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    rule_ids = {issue["rule_id"] for issue in payload["issues"]}
    assert "ai-artifact-id-mismatch" in rule_ids
    assert "ai-schema-version-invalid" in rule_ids
    assert "ai-compiler-version-invalid" in rule_ids
    assert "ai-redaction-violation" in rule_ids
    assert "ai-approval-log-invalid" in rule_ids
    assert "ai-artifact-path-escape" in rule_ids
