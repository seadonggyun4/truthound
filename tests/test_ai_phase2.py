from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import truthound as th
from truthound.cli import app
from truthound.context import TruthoundContext

pytest.importorskip("pydantic")

pytestmark = pytest.mark.contract


class FakeAnalysisProvider:
    provider_name = "fake-openai"
    api_key_env = None
    supports_structured_outputs = True
    default_model_name = "gpt-fake"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.requests = []

    def generate_structured(self, request):
        from truthound.ai import StructuredProviderResponse

        self.requests.append(request)
        return StructuredProviderResponse(
            provider_name=self.provider_name,
            model_name=request.model_name,
            output_text=json.dumps(self.payload, ensure_ascii=False),
            parsed_output=self.payload,
            usage={"prompt_tokens": 7, "completion_tokens": 9, "total_tokens": 16},
            finish_reason="stop",
        )


def _snapshot_core_state(context: TruthoundContext) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for directory in (context.catalog_dir, context.baselines_dir, context.runs_dir, context.docs_dir):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                snapshot[str(path.relative_to(context.workspace_dir))] = path.read_text(encoding="utf-8")
    return snapshot


def _rewrite_persisted_run_as_legacy(run_result) -> Path:
    run_path = Path(run_result.metadata["context_run_artifact"])
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["metadata"].pop("context_source_key", None)
    payload["metadata"].pop("context_history_key", None)
    payload["metadata"].pop("context_source_fingerprint", None)
    run_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return run_path


def _make_analysis_artifact(tmp_path: Path, *, run_id: str, source_key: str):
    from truthound._ai_contract import analysis_artifact_id_for_run
    from truthound.ai import InputRef, RunAnalysisArtifact

    return RunAnalysisArtifact(
        artifact_id=analysis_artifact_id_for_run(run_id),
        source_key=source_key,
        input_refs=[
            InputRef(
                kind="run_result",
                ref=f"runs:{run_id}",
                hash="runhash001",
                redacted=True,
                metadata={"status": "failure", "issue_count": 2},
            ),
            InputRef(
                kind="history_window",
                ref=f"history-window:{source_key}",
                hash="historyhash001",
                redacted=True,
                metadata={"included": True, "run_count": 1, "failure_count": 1},
            ),
        ],
        model_provider="fake-openai",
        model_name="gpt-fake",
        prompt_hash="phase2-test-hash",
        created_by="phase2-test",
        workspace_root=str(tmp_path),
        run_id=run_id,
        summary="Operational analysis highlights duplicate keys and missing values.",
        evidence_refs=[f"runs:{run_id}", f"history-window:{source_key}"],
        failed_checks=["unique", "null"],
        top_columns=["customer_id", "email"],
        recommended_next_actions=["Confirm uniqueness expectations with the source owner."],
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


def test_explain_run_persists_analysis_and_keeps_core_state_read_only(tmp_path: Path):
    from truthound.ai import AIArtifactStore, explain_run
    from truthound._ai_contract import analysis_artifact_id_for_run

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )
    before = _snapshot_core_state(context)
    provider = FakeAnalysisProvider(
        {
            "summary": "Duplicate identifiers and missing emails should be reviewed together.",
            "recommended_next_actions": [
                "Confirm whether customer_id is a true business key.",
            ],
            "evidence_refs": [
                f"runs:{run_result.run_id}",
                f"history-window:{run_result.metadata['context_history_key']}",
            ],
        }
    )

    artifact = explain_run(
        run=run_result,
        context=context,
        provider=provider,
    )

    assert artifact.artifact_id == analysis_artifact_id_for_run(run_result.run_id)
    assert artifact.failed_checks == ["unique", "null"]
    assert artifact.top_columns == ["customer_id", "email"]
    assert artifact.history_window["included"] is True
    assert artifact.history_window["run_count"] == 1
    assert AIArtifactStore(context).read_analysis(artifact.artifact_id).run_id == run_result.run_id
    assert _snapshot_core_state(context) == before

    request = provider.requests[0]
    assert "a@example.com" not in request.user_prompt
    assert "sample_values" not in request.user_prompt
    assert "<!DOCTYPE html>" not in request.user_prompt

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])
    assert result.exit_code == 0


def test_explain_run_supports_run_id_input_and_successful_runs(tmp_path: Path):
    from truthound.ai import explain_run

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 3], "email": ["a@example.com", "b@example.com", "c@example.com"]},
        context=context,
    )
    provider = FakeAnalysisProvider(
        {
            "summary": "The latest run completed without failed checks in the current window.",
            "recommended_next_actions": ["Continue monitoring for drift in future runs."],
            "evidence_refs": [f"runs:{run_result.run_id}"],
        }
    )

    artifact = explain_run(
        run_id=run_result.run_id,
        context=context,
        include_history=False,
        provider=provider,
    )

    assert artifact.run_id == run_result.run_id
    assert artifact.failed_checks == []
    assert artifact.top_columns == []
    assert artifact.history_window["included"] is False
    assert artifact.history_window["run_count"] == 0


def test_explain_run_rejects_mismatched_run_and_run_id(tmp_path: Path):
    from truthound.ai import explain_run

    context = TruthoundContext(tmp_path)
    first_run = th.check({"customer_id": [1, 2], "email": ["a@example.com", None]}, context=context)
    second_run = th.check({"customer_id": [3, 4], "email": ["c@example.com", "d@example.com"]}, context=context)
    provider = FakeAnalysisProvider(
        {
            "summary": "This response should never be used.",
            "recommended_next_actions": ["noop"],
            "evidence_refs": [f"runs:{first_run.run_id}"],
        }
    )

    with pytest.raises(ValueError, match="same persisted run"):
        explain_run(
            run=first_run,
            run_id=second_run.run_id,
            context=context,
            provider=provider,
        )


def test_explain_run_invalid_evidence_refs_do_not_persist_artifact(tmp_path: Path):
    from truthound.ai import AIArtifactStore, explain_run
    from truthound.ai.providers import ProviderResponseError

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )
    provider = FakeAnalysisProvider(
        {
            "summary": "This analysis cites an unavailable evidence ref.",
            "recommended_next_actions": ["noop"],
            "evidence_refs": ["docs:missing-run"],
        }
    )

    with pytest.raises(ProviderResponseError, match="response validation failed"):
        explain_run(
            run=run_result,
            context=context,
            provider=provider,
        )

    assert AIArtifactStore(context).list_analyses() == []


def test_explain_run_supports_legacy_run_resolution_when_baseline_is_unambiguous(tmp_path: Path):
    from truthound.ai import explain_run

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )

    _rewrite_persisted_run_as_legacy(run_result)

    history_path = context.baselines_dir / "metric-history.json"
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    legacy_entries = next(iter(history_payload.values()))
    history_path.write_text(
        json.dumps({"dict": legacy_entries}, separators=(",", ":")),
        encoding="utf-8",
    )

    provider = FakeAnalysisProvider(
        {
            "summary": "Legacy persisted runs can still be analyzed safely.",
            "recommended_next_actions": ["Keep using the canonical run analysis flow."],
            "evidence_refs": [f"runs:{run_result.run_id}", "history-window:dict"],
        }
    )

    artifact = explain_run(
        run_id=run_result.run_id,
        context=context,
        provider=provider,
    )

    assert artifact.history_window["history_key"] == "dict"
    assert any(ref.ref.startswith("baseline-summary:") for ref in artifact.input_refs)


def test_explain_run_skips_ambiguous_legacy_baseline_candidates(tmp_path: Path):
    from truthound.ai import explain_run

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )
    _rewrite_persisted_run_as_legacy(run_result)

    baseline_index_path = context.baseline_index_path
    baseline_payload = json.loads(baseline_index_path.read_text(encoding="utf-8"))
    _, baseline_entry = next(iter(baseline_payload.items()))
    baseline_index_path.write_text(
        json.dumps(
            {
                f"{run_result.source}:orders": baseline_entry,
                f"{run_result.source}:customers": baseline_entry,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    provider = FakeAnalysisProvider(
        {
            "summary": "Legacy runs should still analyze safely even when baseline candidates are ambiguous.",
            "recommended_next_actions": ["Review the failed checks in the canonical run analysis surface."],
            "evidence_refs": [f"runs:{run_result.run_id}", "history-window:dict"],
        }
    )

    artifact = explain_run(
        run_id=run_result.run_id,
        context=context,
        provider=provider,
    )

    assert not any(ref.ref.startswith("baseline-summary:") for ref in artifact.input_refs)
    request = provider.requests[0]
    assert "baseline_summary: none" in request.user_prompt


def test_explain_run_does_not_guess_legacy_history_candidates(tmp_path: Path):
    from truthound.ai import explain_run

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )
    _rewrite_persisted_run_as_legacy(run_result)

    history_path = context.baselines_dir / "metric-history.json"
    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    legacy_entries = next(iter(history_payload.values()))
    history_path.write_text(
        json.dumps(
            {
                f"{run_result.source}:recent-a": legacy_entries,
                f"{run_result.source}:recent-b": legacy_entries,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    provider = FakeAnalysisProvider(
        {
            "summary": "Legacy history should stay empty instead of guessing a candidate key.",
            "recommended_next_actions": ["Watch future runs to rebuild canonical history under the resolved key."],
            "evidence_refs": [f"runs:{run_result.run_id}", "history-window:dict"],
        }
    )

    artifact = explain_run(
        run_id=run_result.run_id,
        context=context,
        provider=provider,
    )

    assert artifact.history_window["included"] is True
    assert artifact.history_window["history_key"] == "dict"
    assert artifact.history_window["run_count"] == 0
    assert artifact.history_window["failure_count"] == 0
    assert artifact.history_window["recent_statuses"] == []
    assert any(ref.ref == "history-window:dict" for ref in artifact.input_refs)
    assert not any(ref.ref.startswith("history-window:dict:") for ref in artifact.input_refs)
    request = provider.requests[0]
    assert "history_key dict window_size 10 run_count 0 failure_count 0" in request.user_prompt
    assert "dict:recent-a" not in request.user_prompt
    assert "dict:recent-b" not in request.user_prompt


def test_legacy_phase0_analysis_reads_and_cli_surfaces_phase2_show_list(tmp_path: Path, monkeypatch):
    import truthound.ai as ai_namespace
    from truthound.ai import AIArtifactStore

    context = TruthoundContext(tmp_path)
    run_result = th.check(
        {"customer_id": [1, 2, 2], "email": ["a@example.com", None, "c@example.com"]},
        context=context,
    )
    runner = CliRunner()

    def fake_explain_run(*args, **kwargs):
        artifact = _make_analysis_artifact(
            tmp_path,
            run_id=run_result.run_id,
            source_key=run_result.metadata["context_source_key"],
        )
        AIArtifactStore(context).write_analysis(artifact)
        return artifact

    monkeypatch.setattr(ai_namespace, "explain_run", fake_explain_run)
    monkeypatch.setattr("truthound.cli_modules.ai.get_context", lambda: context)

    explain_result = runner.invoke(
        app,
        ["ai", "explain-run", "--run-id", run_result.run_id, "--json"],
    )
    assert explain_result.exit_code == 0
    explain_payload = json.loads(explain_result.output)
    assert explain_payload["artifact_id"].startswith("run-analysis-")

    list_result = runner.invoke(app, ["ai", "analyses", "list", "--json"])
    assert list_result.exit_code == 0
    list_payload = json.loads(list_result.output)
    assert len(list_payload) == 1

    show_result = runner.invoke(
        app,
        ["ai", "analyses", "show", run_result.run_id, "--json"],
    )
    assert show_result.exit_code == 0
    show_payload = json.loads(show_result.output)
    assert show_payload["run_id"] == run_result.run_id
    assert show_payload["history_window"]["history_key"] == run_result.metadata["context_source_key"]

    list_text_result = runner.invoke(app, ["ai", "analyses", "list"])
    assert list_text_result.exit_code == 0
    assert "failed_checks=2" in list_text_result.output
    assert "top_columns=2" in list_text_result.output

    show_text_result = runner.invoke(app, ["ai", "analyses", "show", run_result.run_id])
    assert show_text_result.exit_code == 0
    assert "failed_check_count: 2" in show_text_result.output
    assert "top_column_count: 2" in show_text_result.output
    assert "evidence_ref_count: 2" in show_text_result.output


def test_legacy_phase0_analysis_artifact_is_readable_and_doctor_accepts_it(tmp_path: Path):
    from truthound._ai_contract import TRUTHOUND_AI_COMPILER_VERSION, analysis_artifact_id_for_run
    from truthound.ai import AIArtifactStore

    context = TruthoundContext(tmp_path)
    analysis_dir = context.workspace_dir / "ai" / "analyses"
    proposals_dir = context.workspace_dir / "ai" / "proposals"
    approvals_dir = context.workspace_dir / "ai" / "approvals"
    analysis_dir.mkdir(parents=True)
    proposals_dir.mkdir(parents=True)
    approvals_dir.mkdir(parents=True)
    run_id = "run_20260401_120000_abcd1234"
    artifact_id = analysis_artifact_id_for_run(run_id)

    (analysis_dir / f"{artifact_id}.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "artifact_id": artifact_id,
                "artifact_type": "run_analysis",
                "source_key": "source:orders",
                "input_refs": [
                    {
                        "kind": "run_result",
                        "ref": f"runs:{run_id}",
                        "redacted": True,
                        "metadata": {"issue_count": 2},
                    }
                ],
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "prompt_hash": "legacy-analysis-hash",
                "compiler_version": TRUTHOUND_AI_COMPILER_VERSION,
                "approval_status": "not_required",
                "approved_by": None,
                "approved_at": None,
                "redaction_policy": {
                    "mode": "summary_only",
                    "raw_samples_allowed": False,
                    "pii_literals_allowed": False,
                },
                "created_at": "2026-04-01T00:00:00+00:00",
                "created_by": "phase2-test",
                "workspace_root": str(tmp_path),
                "run_id": run_id,
                "summary": "Legacy analysis artifact.",
                "evidence_refs": [f"runs:{run_id}", f"docs:{run_id}"],
                "failed_checks": ["unique"],
                "top_columns": ["customer_id"],
                "recommended_next_actions": ["Review the duplicate-key source system."],
                "history_window": {"window": "7d", "run_count": 5},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    artifact = AIArtifactStore(context).read_analysis(artifact_id)
    assert artifact.compiler_version == TRUTHOUND_AI_COMPILER_VERSION
    assert artifact.history_window["window_size"] == 10
    assert any(item.ref == f"docs:{run_id}" for item in artifact.input_refs)

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", str(tmp_path), "--workspace"])
    assert result.exit_code == 0
    assert "found no structural issues" in result.output
