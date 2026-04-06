from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import truthound as th
from truthound.cli import app
from truthound.context import TruthoundContext

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "ai" / "phase4"


class FixtureProvider:
    provider_name = "fake-openai"
    api_key_env = None
    supports_structured_outputs = True
    default_model_name = "gpt-fake"

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.requests = []

    def generate_structured(self, request):
        from truthound.ai import StructuredProviderResponse

        self.requests.append(request)
        output_text = json.dumps(self.payload, ensure_ascii=False)
        return StructuredProviderResponse(
            provider_name=self.provider_name,
            model_name=request.model_name,
            output_text=output_text,
            parsed_output=self.payload,
            usage={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
            finish_reason="stop",
        )


def _load_json_fixture(name: str) -> Any:
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def _copy_orders_csv(tmp_path: Path) -> Path:
    target = tmp_path / "orders-demo.csv"
    shutil.copyfile(FIXTURE_ROOT / "orders-demo.csv", target)
    return target


def _snapshot_core_state(context: TruthoundContext) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for directory in (context.catalog_dir, context.baselines_dir, context.runs_dir, context.docs_dir):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                snapshot[str(path.relative_to(context.workspace_dir))] = path.read_text(encoding="utf-8")
    return snapshot


def _normalize_source_key(value: str) -> str:
    candidate = Path(value)
    if candidate.name:
        return candidate.name
    return value


def _normalize_input_ref(ref: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "kind": ref["kind"],
        "ref": ref["ref"],
        "redacted": ref["redacted"],
        "metadata": ref["metadata"],
    }
    if payload["ref"].startswith("schema-summary:"):
        payload["ref"] = f"schema-summary:{Path(payload['ref'].split(':', 1)[1]).name}"
    elif payload["ref"].startswith("suite-summary:"):
        payload["ref"] = f"suite-summary:{Path(payload['ref'].split(':', 1)[1]).name}"
    elif payload["ref"].startswith("runs:"):
        payload["ref"] = "runs:<run_id>"
    elif payload["ref"].startswith("docs:"):
        payload["ref"] = "docs:<run_id>"
    return payload


def _normalize_proposal_snapshot(artifact) -> dict[str, Any]:
    return {
        "schema_version": str(artifact.schema_version),
        "artifact_type": getattr(artifact.artifact_type, "value", str(artifact.artifact_type)),
        "source_key": _normalize_source_key(str(artifact.source_key)),
        "input_refs": [
            _normalize_input_ref(item.model_dump(mode="json"))
            for item in artifact.input_refs
        ],
        "model_provider": str(artifact.model_provider),
        "model_name": str(artifact.model_name),
        "compiler_version": str(artifact.compiler_version),
        "approval_status": getattr(artifact.approval_status, "value", str(artifact.approval_status)),
        "target_type": str(artifact.target_type),
        "summary": str(artifact.summary),
        "rationale": str(artifact.rationale),
        "compile_status": str(artifact.compile_status),
        "compiled_check_count": int(artifact.compiled_check_count),
        "rejected_check_count": int(artifact.rejected_check_count),
        "compiled_checks": [
            {
                "validator_name": check.validator_name,
                "category": check.category,
                "columns": list(check.columns),
                "params": dict(check.params),
            }
            for check in artifact.checks
        ],
        "diff_counts": artifact.diff_preview.counts.model_dump(mode="json"),
        "added_checks": [
            {
                "validator_name": item.validator_name,
                "columns": list(item.columns),
            }
            for item in artifact.diff_preview.added
        ],
        "already_present_validators": [
            item.validator_name for item in artifact.diff_preview.already_present
        ],
        "existing_suite_validator_names": [
            item["validator_name"]
            for item in (artifact.existing_suite_summary or {}).get("checks", [])
        ],
        "risks": list(artifact.risks),
        "compiler_errors": list(artifact.compiler_errors),
    }


def _normalize_analysis_snapshot(artifact) -> dict[str, Any]:
    input_refs = [
        _normalize_input_ref(item.model_dump(mode="json"))
        for item in artifact.input_refs
    ]
    evidence_refs = [
        "runs:<run_id>" if ref.startswith("runs:") else ref
        for ref in artifact.evidence_refs
    ]
    history_window = dict(artifact.history_window)
    if history_window.get("latest_run_id"):
        history_window["latest_run_id"] = "<run_id>"
    return {
        "schema_version": str(artifact.schema_version),
        "artifact_type": getattr(artifact.artifact_type, "value", str(artifact.artifact_type)),
        "source_key": str(artifact.source_key),
        "input_refs": input_refs,
        "model_provider": str(artifact.model_provider),
        "model_name": str(artifact.model_name),
        "compiler_version": str(artifact.compiler_version),
        "approval_status": getattr(artifact.approval_status, "value", str(artifact.approval_status)),
        "summary": str(artifact.summary),
        "evidence_refs": evidence_refs,
        "failed_checks": list(artifact.failed_checks),
        "top_columns": list(artifact.top_columns),
        "recommended_next_actions": list(artifact.recommended_next_actions),
        "history_window": history_window,
    }


def _make_ready_cli_proposal(tmp_path: Path, *, source_key: str):
    from truthound._applied_suite import canonical_check_key
    from truthound.ai import (
        CompiledProposalCheck,
        InputRef,
        SuiteCheckSnapshot,
        SuiteProposalArtifact,
        ValidationSuiteDiffCounts,
        ValidationSuiteDiffPreview,
        ValidationSuiteSnapshot,
    )

    check_key = canonical_check_key(
        validator_name="between",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 1},
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
        check_id=check_key,
        check_key=check_key,
        validator_name="between",
        category="distribution",
        columns=["refund_rate"],
        params={"min_value": 0, "max_value": 1},
        tags=["distribution"],
        rationale="Refund rate should stay between zero and one.",
        origin="proposal",
    )
    diff_preview = ValidationSuiteDiffPreview(
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
        counts=ValidationSuiteDiffCounts(added=1, already_present=0, conflicts=0, rejected=0),
    )
    return SuiteProposalArtifact(
        source_key=source_key,
        input_refs=[
            InputRef(
                kind="schema_summary",
                ref=f"schema-summary:{source_key}",
                hash="schemahash-phase4",
                redacted=True,
                metadata={"column_count": 5, "observed_count": 3},
            )
        ],
        model_provider="fake-openai",
        model_name="gpt-fake",
        prompt_hash="phase4-cli-fingerprint",
        created_by="phase4-cli-test",
        workspace_root=str(tmp_path),
        summary="CLI proposal for refund rate hardening.",
        rationale="Exercise the operator review path end to end.",
        checks=[
            CompiledProposalCheck(
                check_key=check_key,
                validator_name="between",
                category="distribution",
                columns=["refund_rate"],
                params={"min_value": 0, "max_value": 1},
                rationale="Refund rate should stay between zero and one.",
            )
        ],
        risks=[],
        compile_status="ready",
        diff_preview=diff_preview,
        compiled_check_count=1,
        rejected_check_count=0,
        compiler_errors=[],
    )


def _make_cli_analysis(tmp_path: Path, *, run_id: str, source_key: str):
    from truthound._ai_contract import analysis_artifact_id_for_run
    from truthound.ai import InputRef, RunAnalysisArtifact

    return RunAnalysisArtifact(
        artifact_id=analysis_artifact_id_for_run(run_id),
        source_key=source_key,
        input_refs=[
            InputRef(
                kind="run_result",
                ref=f"runs:{run_id}",
                hash="runhash-phase4",
                redacted=True,
                metadata={"status": "failure", "issue_count": 2, "failed_check_count": 2},
            ),
            InputRef(
                kind="history_window",
                ref=f"history-window:{source_key}",
                hash="historyhash-phase4",
                redacted=True,
                metadata={"included": True, "run_count": 1, "failure_count": 1},
            ),
        ],
        model_provider="fake-openai",
        model_name="gpt-fake",
        prompt_hash="phase4-analysis-fingerprint",
        created_by="phase4-cli-test",
        workspace_root=str(tmp_path),
        run_id=run_id,
        summary="CLI analysis highlights the duplicate identifier issue.",
        evidence_refs=[f"runs:{run_id}", f"history-window:{source_key}"],
        failed_checks=["unique", "null"],
        top_columns=["customer_id", "email"],
        recommended_next_actions=["Confirm whether customer_id is a stable business key."],
        history_window={
            "included": True,
            "history_key": source_key,
            "window_size": 10,
            "run_count": 1,
            "failure_count": 1,
            "success_count": 0,
            "latest_run_id": run_id,
            "recent_statuses": ["failure"],
        },
    )


def test_phase4_compiler_golden_snapshot_matches_normalized_fixture(tmp_path: Path) -> None:
    from truthound.ai import suggest_suite

    data_path = _copy_orders_csv(tmp_path)
    context = TruthoundContext(tmp_path)
    artifact = suggest_suite(
        prompt="Keep order ids unique and refund metrics bounded.",
        data=str(data_path),
        context=context,
        provider=FixtureProvider(_load_json_fixture("happy_path_proposal_response.json")),
    )

    expected = _load_json_fixture("expected_normalized_proposal_snapshot.json")
    assert _normalize_proposal_snapshot(artifact) == expected


def test_phase4_malformed_provider_output_is_safely_rejected_without_core_state_pollution(
    tmp_path: Path,
) -> None:
    from truthound.ai import suggest_suite

    data_path = _copy_orders_csv(tmp_path)
    context = TruthoundContext(tmp_path)
    before = _snapshot_core_state(context)

    artifact = suggest_suite(
        prompt="Return malformed output on purpose.",
        data=str(data_path),
        context=context,
        provider=FixtureProvider(_load_json_fixture("malformed_provider_output.json")),
    )

    assert str(artifact.compile_status) == "rejected"
    assert artifact.compiled_check_count == 0
    assert artifact.rejected_check_count == 0
    assert artifact.diff_preview.counts.added == 0
    assert artifact.rejected_items == []
    assert artifact.compiler_errors == ["provider_output_validation_failed"]
    assert _snapshot_core_state(context) == before


def test_phase4_unsupported_intent_and_unsafe_regex_are_safely_rejected(tmp_path: Path) -> None:
    from truthound.ai import suggest_suite

    data_path = _copy_orders_csv(tmp_path)
    context = TruthoundContext(tmp_path)
    before = _snapshot_core_state(context)

    artifact = suggest_suite(
        prompt="Keep order ids unique and reject unsafe suggestions.",
        data=str(data_path),
        context=context,
        provider=FixtureProvider(_load_json_fixture("unsupported_or_unsafe_response.json")),
    )

    assert str(artifact.compile_status) == "partial"
    assert artifact.compiled_check_count == 1
    assert artifact.rejected_check_count == 3
    assert artifact.diff_preview.counts.added == 1
    reasons = {item.reason for item in artifact.rejected_items}
    assert any("unsupported intent" in reason for reason in reasons)
    assert any("regex pattern is not allowed" in reason for reason in reasons)
    assert any(item.source == "model" for item in artifact.rejected_items)
    assert _snapshot_core_state(context) == before


def test_phase4_analysis_golden_snapshot_matches_normalized_fixture(tmp_path: Path) -> None:
    from truthound.ai import explain_run

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )
    payload = _load_json_fixture("happy_path_analysis_response.json")
    payload["evidence_refs"] = [
        f"runs:{run_result.run_id}",
        f"history-window:{run_result.metadata['context_history_key']}",
    ]

    artifact = explain_run(
        run=run_result,
        context=context,
        provider=FixtureProvider(payload),
    )

    expected = _load_json_fixture("expected_normalized_analysis_snapshot.json")
    assert _normalize_analysis_snapshot(artifact) == expected


def test_phase4_analysis_without_evidence_refs_is_rejected_and_not_persisted(tmp_path: Path) -> None:
    from truthound.ai import AIArtifactStore, explain_run
    from truthound.ai.providers import ProviderResponseError

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )

    with pytest.raises(ProviderResponseError, match="response validation failed"):
        explain_run(
            run=run_result,
            context=context,
            provider=FixtureProvider(_load_json_fixture("analysis_without_evidence_refs.json")),
        )

    assert AIArtifactStore(context).list_analyses() == []


def test_phase4_redaction_privacy_invariants_block_pii_and_sample_payloads(tmp_path: Path) -> None:
    from truthound._ai_contract import analysis_artifact_id_for_run
    from truthound.ai import (
        InputRef,
        RedactionViolationError,
        RejectedProposalItem,
        RunAnalysisArtifact,
        SuiteProposalArtifact,
    )

    context = TruthoundContext(tmp_path)

    with pytest.raises(ValidationError, match="text contains PII-like literal content"):
        SuiteProposalArtifact(
            source_key="source:orders",
            input_refs=[
                InputRef(
                    kind="schema_summary",
                    ref="schema-summary:source:orders",
                    hash="hash001",
                    redacted=True,
                    metadata={"column_count": 5, "observed_count": 3},
                )
            ],
            model_provider="fake-openai",
            model_name="gpt-fake",
            prompt_hash="phase4-redaction",
            created_by="phase4-test",
            workspace_root=str(tmp_path),
            summary="Please investigate alice@example.com anomalies.",
            rationale="This should never persist.",
            compile_status="rejected",
            rejected_items=[
                RejectedProposalItem(
                    source="compiler",
                    intent="provider_output_validation_failed",
                    columns=[],
                    params={},
                    reason="Reject unsafe payloads",
                )
            ],
        )

    with pytest.raises(ValidationError, match="row-level sample"):
        RunAnalysisArtifact(
            artifact_id=analysis_artifact_id_for_run("run-phase4"),
            source_key="dict:customer_id:email",
            input_refs=[
                InputRef(
                    kind="run_result",
                    ref="runs:run-phase4",
                    hash="runhash",
                    redacted=True,
                    metadata={"status": "failure", "issue_count": 2, "failed_check_count": 2},
                ),
                InputRef(
                    kind="history_window",
                    ref="history-window:dict:customer_id:email",
                    hash="historyhash",
                    redacted=True,
                    metadata={"included": True, "run_count": 1, "failure_count": 1},
                ),
            ],
            model_provider="fake-openai",
            model_name="gpt-fake",
            prompt_hash="phase4-redaction",
            created_by="phase4-test",
            workspace_root=str(tmp_path),
            run_id="run-phase4",
            summary="Summary only",
            evidence_refs=["runs:run-phase4"],
            failed_checks=["unique"],
            top_columns=["customer_id"],
            recommended_next_actions=["row=[customer_id=2,status=pending]"],
            history_window={
                "included": True,
                "history_key": "dict:customer_id:email",
                "window_size": 10,
                "run_count": 1,
                "failure_count": 1,
                "success_count": 0,
                "latest_run_id": "run-phase4",
                "recent_statuses": ["failure"],
            },
        )


def test_phase4_truthound_ai_cli_smoke_tracks_structured_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import truthound.ai as ai_namespace
    from truthound.ai import AIArtifactStore
    from truthound.cli_modules import ai as ai_cli_module

    context = TruthoundContext(tmp_path)
    source_path = _copy_orders_csv(tmp_path)
    source_key = context.resolve_source_key(data=str(source_path))
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )

    def fake_suggest_suite(*, prompt: str, data: Any = None, source: Any = None, context: Any = None, **_: Any):
        artifact = _make_ready_cli_proposal(tmp_path, source_key=context.resolve_source_key(data=data, source=source))
        AIArtifactStore(context).write_proposal(artifact)
        return artifact

    def fake_explain_run(*, run_id: str, context: Any = None, **_: Any):
        artifact = _make_cli_analysis(
            tmp_path,
            run_id=run_id,
            source_key=context.resolve_source_key(data=run_result.source),
        )
        AIArtifactStore(context).write_analysis(artifact)
        return artifact

    monkeypatch.setattr(ai_cli_module, "get_context", lambda: context)
    monkeypatch.setattr(ai_namespace, "suggest_suite", fake_suggest_suite)
    monkeypatch.setattr(ai_namespace, "explain_run", fake_explain_run)

    runner = CliRunner()

    suggest_result = runner.invoke(
        app,
        [
            "ai",
            "suggest-suite",
            str(source_path),
            "--prompt",
            "Propose a refund-rate bound check.",
            "--json",
        ],
    )
    assert suggest_result.exit_code == 0, suggest_result.stdout
    proposal_payload = json.loads(suggest_result.stdout)
    proposal_id = proposal_payload["artifact_id"]

    list_result = runner.invoke(app, ["ai", "proposals", "list", "--json"])
    assert list_result.exit_code == 0, list_result.stdout
    assert json.loads(list_result.stdout)[0]["artifact_id"] == proposal_id

    show_result = runner.invoke(app, ["ai", "proposals", "show", proposal_id, "--json"])
    assert show_result.exit_code == 0, show_result.stdout
    assert json.loads(show_result.stdout)["compile_status"] == "ready"

    approve_result = runner.invoke(
        app,
        [
            "ai",
            "proposals",
            "approve",
            proposal_id,
            "--actor-id",
            "user-001",
            "--actor-name",
            "Truthound Operator",
            "--comment",
            "Looks safe to apply.",
            "--json",
        ],
    )
    assert approve_result.exit_code == 0, approve_result.stdout
    assert json.loads(approve_result.stdout)["proposal"]["approval_status"] == "approved"

    apply_result = runner.invoke(
        app,
        [
            "ai",
            "proposals",
            "apply",
            proposal_id,
            "--actor-id",
            "user-001",
            "--actor-name",
            "Truthound Operator",
            "--comment",
            "Apply to the active suite.",
            "--yes",
            "--json",
        ],
    )
    assert apply_result.exit_code == 0, apply_result.stdout
    assert json.loads(apply_result.stdout)["proposal"]["approval_status"] == "applied"

    history_result = runner.invoke(app, ["ai", "proposals", "history", proposal_id, "--json"])
    assert history_result.exit_code == 0, history_result.stdout
    history = json.loads(history_result.stdout)
    assert [item["action"] for item in history] == ["apply", "approve"]

    explain_result = runner.invoke(
        app,
        [
            "ai",
            "explain-run",
            "--run-id",
            run_result.run_id,
            "--json",
        ],
    )
    assert explain_result.exit_code == 0, explain_result.stdout
    analysis_payload = json.loads(explain_result.stdout)
    analysis_id = analysis_payload["artifact_id"]

    analysis_show = runner.invoke(app, ["ai", "analyses", "show", analysis_id, "--json"])
    assert analysis_show.exit_code == 0, analysis_show.stdout
    assert json.loads(analysis_show.stdout)["artifact_id"] == analysis_id

    approval_log = (context.workspace_dir / "ai" / "approvals" / "approval-log.jsonl")
    assert approval_log.exists()
    assert len(approval_log.read_text(encoding="utf-8").splitlines()) == 2

    applied_suite = context.read_applied_suite(source_key=source_key)
    assert applied_suite is not None
    assert applied_suite.proposal_id == proposal_id
