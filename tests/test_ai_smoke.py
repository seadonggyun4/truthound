from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from truthound.cli import app

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


def _make_success_artifact(root_dir: Path):
    from truthound.ai import CompiledProposalCheck, InputRef, SuiteProposalArtifact

    return SuiteProposalArtifact(
        source_key="source:smoke-orders",
        input_refs=[
            InputRef(
                kind="schema_summary",
                ref="schema-summary:source:smoke-orders",
                hash="smokehash01",
                redacted=True,
                metadata={"column_count": 4, "observed_count": 3},
            )
        ],
        model_provider="openai",
        model_name="gpt-smoke",
        prompt_hash="smoke-prompt-hash",
        created_by="smoke-test",
        workspace_root=str(root_dir),
        summary="Smoke proposal compiled aggregate quality checks.",
        rationale="Smoke prompt should return at least one executable proposal check.",
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
        risks=[],
        compile_status="ready",
        diff_preview=_make_diff_preview(),
        compiled_check_count=1,
        rejected_check_count=0,
        compiler_errors=[],
    )


def _make_parse_failure_artifact(root_dir: Path):
    from truthound.ai import InputRef, SuiteProposalArtifact

    diff_preview = _make_diff_preview()
    rejected_preview = diff_preview.model_copy(
        update={"proposed_suite": diff_preview.current_suite}
    )
    return SuiteProposalArtifact(
        source_key="source:smoke-orders",
        input_refs=[
            InputRef(
                kind="schema_summary",
                ref="schema-summary:source:smoke-orders",
                hash="smokehash02",
                redacted=True,
                metadata={"column_count": 4, "observed_count": 3},
            )
        ],
        model_provider="openai",
        model_name="gpt-smoke",
        prompt_hash="smoke-prompt-hash-failed",
        created_by="smoke-test",
        workspace_root=str(root_dir),
        summary="Smoke proposal failed to compile.",
        rationale="Provider output was malformed.",
        checks=[],
        risks=[],
        compile_status="rejected",
        diff_preview=rejected_preview,
        compiled_check_count=0,
        rejected_check_count=0,
        compiler_errors=["provider_output_validation_failed"],
    )


def _make_analysis_artifact(
    root_dir: Path,
    *,
    run_id: str = "run_20260401_120000_abcd1234",
    source_key: str = "dict",
):
    from truthound._ai_contract import analysis_artifact_id_for_run
    from truthound.ai import InputRef, RunAnalysisArtifact

    return RunAnalysisArtifact(
        artifact_id=analysis_artifact_id_for_run(run_id),
        source_key=source_key,
        input_refs=[
            InputRef(
                kind="run_result",
                ref=f"runs:{run_id}",
                hash="runhash-smoke-01",
                redacted=True,
                metadata={"status": "failure", "issue_count": 2},
            ),
            InputRef(
                kind="history_window",
                ref=f"history-window:{source_key}",
                hash="historyhash-smoke-01",
                redacted=True,
                metadata={"included": True, "run_count": 1, "failure_count": 1},
            ),
        ],
        model_provider="openai",
        model_name="gpt-smoke",
        prompt_hash="smoke-analysis-hash",
        created_by="smoke-test",
        workspace_root=str(root_dir),
        run_id=run_id,
        summary="Explain-run smoke identified duplicate keys and missing emails.",
        evidence_refs=[f"runs:{run_id}", f"history-window:{source_key}"],
        failed_checks=["unique", "null"],
        top_columns=["customer_id", "email"],
        recommended_next_actions=["Review the customer_id business key with the source owner."],
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


def test_run_openai_smoke_classifies_missing_key_and_model_as_config(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai.smoke import run_openai_smoke

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TRUTHOUND_AI_SMOKE_MODEL", raising=False)
    monkeypatch.delenv("TRUTHOUND_AI_MODEL", raising=False)

    result = run_openai_smoke()

    assert result.success is False
    assert result.failure_stage == "config"
    assert result.workspace_retained is False
    assert result.workspace_dir is None


def test_run_openai_explain_run_smoke_classifies_missing_key_and_model_as_config(
    monkeypatch: pytest.MonkeyPatch,
):
    from truthound.ai.smoke import run_openai_explain_run_smoke

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TRUTHOUND_AI_SMOKE_MODEL", raising=False)
    monkeypatch.delenv("TRUTHOUND_AI_MODEL", raising=False)

    result = run_openai_explain_run_smoke()

    assert result.success is False
    assert result.failure_stage == "config"
    assert result.workspace_retained is False
    assert result.workspace_dir is None


def test_run_openai_smoke_classifies_provider_transport_failures(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai.providers import ProviderTransportError
    from truthound.ai.smoke import run_openai_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_invoke(**kwargs):
        raise ProviderTransportError("network down")

    monkeypatch.setattr("truthound.ai.smoke._invoke_suggest_suite", fake_invoke)

    result = run_openai_smoke()

    assert result.success is False
    assert result.failure_stage == "provider"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    assert Path(result.workspace_dir).exists()
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_explain_run_smoke_classifies_prepare_failures(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai.smoke import run_openai_explain_run_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")
    monkeypatch.setattr(
        "truthound.ai.smoke._prepare_explain_run_smoke_run",
        lambda context: (_ for _ in ()).throw(RuntimeError("prep failed")),
    )

    result = run_openai_explain_run_smoke()

    assert result.success is False
    assert result.failure_stage == "prepare"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_explain_run_smoke_classifies_provider_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace

    from truthound.ai.providers import ProviderTransportError
    from truthound.ai.smoke import run_openai_explain_run_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")
    monkeypatch.setattr(
        "truthound.ai.smoke._prepare_explain_run_smoke_run",
        lambda context: SimpleNamespace(run_id="run_20260401_120000_abcd1234"),
    )

    def fake_invoke(**kwargs):
        raise ProviderTransportError("network down")

    monkeypatch.setattr("truthound.ai.smoke._invoke_explain_run", fake_invoke)

    result = run_openai_explain_run_smoke()

    assert result.success is False
    assert result.failure_stage == "provider"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    assert Path(result.workspace_dir).exists()
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_smoke_classifies_rejected_artifact_as_parse(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai import AIArtifactStore
    from truthound.ai.smoke import run_openai_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_invoke(**kwargs):
        context = kwargs["context"]
        artifact = _make_parse_failure_artifact(context.root_dir)
        AIArtifactStore(context).write_proposal(artifact)
        return artifact

    monkeypatch.setattr("truthound.ai.smoke._invoke_suggest_suite", fake_invoke)

    result = run_openai_smoke()

    assert result.success is False
    assert result.failure_stage == "parse"
    assert result.artifact_id is not None
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_explain_run_smoke_classifies_parse_failures(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from truthound.ai.providers import ProviderResponseError
    from truthound.ai.smoke import run_openai_explain_run_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")
    monkeypatch.setattr(
        "truthound.ai.smoke._prepare_explain_run_smoke_run",
        lambda context: SimpleNamespace(run_id="run_20260401_120000_abcd1234"),
    )

    def fake_invoke(**kwargs):
        raise ProviderResponseError("invalid analysis response")

    monkeypatch.setattr("truthound.ai.smoke._invoke_explain_run", fake_invoke)

    result = run_openai_explain_run_smoke()

    assert result.success is False
    assert result.failure_stage == "parse"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_smoke_classifies_persist_failures(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai.smoke import run_openai_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_invoke(**kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr("truthound.ai.smoke._invoke_suggest_suite", fake_invoke)

    result = run_openai_smoke()

    assert result.success is False
    assert result.failure_stage == "persist"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_explain_run_smoke_classifies_persist_failures(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from truthound.ai.smoke import run_openai_explain_run_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")
    monkeypatch.setattr(
        "truthound.ai.smoke._prepare_explain_run_smoke_run",
        lambda context: SimpleNamespace(run_id="run_20260401_120000_abcd1234"),
    )

    def fake_invoke(**kwargs):
        raise OSError("disk write failed")

    monkeypatch.setattr("truthound.ai.smoke._invoke_explain_run", fake_invoke)

    result = run_openai_explain_run_smoke()

    assert result.success is False
    assert result.failure_stage == "persist"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_smoke_classifies_verify_failures(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai import AIArtifactStore
    from truthound.ai.smoke import run_openai_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_invoke(**kwargs):
        context = kwargs["context"]
        artifact = _make_success_artifact(context.root_dir)
        AIArtifactStore(context).write_proposal(artifact)
        return artifact

    def fake_read(context, artifact_id):
        raise OSError("cannot read proposal")

    monkeypatch.setattr("truthound.ai.smoke._invoke_suggest_suite", fake_invoke)
    monkeypatch.setattr("truthound.ai.smoke._read_smoke_proposal", fake_read)

    result = run_openai_smoke()

    assert result.success is False
    assert result.failure_stage == "verify"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_explain_run_smoke_classifies_verify_failures(monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace

    from truthound.ai import AIArtifactStore
    from truthound.ai.smoke import run_openai_explain_run_smoke

    run_id = "run_20260401_120000_abcd1234"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")
    monkeypatch.setattr(
        "truthound.ai.smoke._prepare_explain_run_smoke_run",
        lambda context: SimpleNamespace(run_id=run_id),
    )

    def fake_invoke(**kwargs):
        context = kwargs["context"]
        artifact = _make_analysis_artifact(context.root_dir, run_id=run_id)
        AIArtifactStore(context).write_analysis(artifact)
        return artifact

    def fake_read(context, artifact_id):
        raise OSError("cannot read analysis")

    monkeypatch.setattr("truthound.ai.smoke._invoke_explain_run", fake_invoke)
    monkeypatch.setattr("truthound.ai.smoke._read_smoke_analysis", fake_read)

    result = run_openai_explain_run_smoke()

    assert result.success is False
    assert result.failure_stage == "verify"
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_smoke_cleans_up_success_workspace_by_default(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai import AIArtifactStore
    from truthound.ai.smoke import run_openai_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_invoke(**kwargs):
        context = kwargs["context"]
        artifact = _make_success_artifact(context.root_dir)
        AIArtifactStore(context).write_proposal(artifact)
        return artifact

    monkeypatch.setattr("truthound.ai.smoke._invoke_suggest_suite", fake_invoke)

    result = run_openai_smoke()

    assert result.success is True
    assert result.failure_stage is None
    assert result.workspace_retained is False
    assert result.workspace_dir is not None
    assert not Path(result.workspace_dir).exists()


def test_run_openai_explain_run_smoke_cleans_up_success_workspace_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    from truthound.ai import AIArtifactStore
    from truthound.ai.smoke import run_openai_explain_run_smoke

    run_id = "run_20260401_120000_abcd1234"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_prepare(context):
        return type("RunResultStub", (), {"run_id": run_id})()

    def fake_invoke(**kwargs):
        context = kwargs["context"]
        artifact = _make_analysis_artifact(context.root_dir, run_id=run_id)
        AIArtifactStore(context).write_analysis(artifact)
        return artifact

    monkeypatch.setattr("truthound.ai.smoke._prepare_explain_run_smoke_run", fake_prepare)
    monkeypatch.setattr("truthound.ai.smoke._invoke_explain_run", fake_invoke)

    result = run_openai_explain_run_smoke()

    assert result.success is True
    assert result.failure_stage is None
    assert result.workspace_retained is False
    assert result.workspace_dir is not None
    assert not Path(result.workspace_dir).exists()


def test_run_openai_smoke_keeps_workspace_when_requested(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai import AIArtifactStore
    from truthound.ai.smoke import run_openai_smoke

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_invoke(**kwargs):
        context = kwargs["context"]
        artifact = _make_success_artifact(context.root_dir)
        AIArtifactStore(context).write_proposal(artifact)
        return artifact

    monkeypatch.setattr("truthound.ai.smoke._invoke_suggest_suite", fake_invoke)

    result = run_openai_smoke(keep_workspace=True)

    assert result.success is True
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    assert Path(result.workspace_dir).exists()
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_run_openai_explain_run_smoke_keeps_workspace_when_requested(monkeypatch: pytest.MonkeyPatch):
    from truthound.ai import AIArtifactStore
    from truthound.ai.smoke import run_openai_explain_run_smoke

    run_id = "run_20260401_120000_abcd1234"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TRUTHOUND_AI_SMOKE_MODEL", "gpt-smoke")

    def fake_prepare(context):
        return type("RunResultStub", (), {"run_id": run_id})()

    def fake_invoke(**kwargs):
        context = kwargs["context"]
        artifact = _make_analysis_artifact(context.root_dir, run_id=run_id)
        AIArtifactStore(context).write_analysis(artifact)
        return artifact

    monkeypatch.setattr("truthound.ai.smoke._prepare_explain_run_smoke_run", fake_prepare)
    monkeypatch.setattr("truthound.ai.smoke._invoke_explain_run", fake_invoke)

    result = run_openai_explain_run_smoke(keep_workspace=True)

    assert result.success is True
    assert result.workspace_retained is True
    assert result.workspace_dir is not None
    assert Path(result.workspace_dir).exists()
    shutil.rmtree(result.workspace_dir, ignore_errors=True)


def test_ai_cli_openai_smoke_json_outputs_typed_result(monkeypatch: pytest.MonkeyPatch):
    import truthound.ai as ai_namespace

    runner = CliRunner()
    smoke_result = ai_namespace.OpenAISmokeResult(
        model_name="gpt-smoke",
        success=False,
        failure_stage="provider",
        error_message="network down",
        workspace_dir="/tmp/truthound-ai-smoke-test",
        workspace_retained=True,
    )
    monkeypatch.setattr(ai_namespace, "run_openai_smoke", lambda **kwargs: smoke_result)

    result = runner.invoke(app, ["ai", "smoke", "openai", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["failure_stage"] == "provider"
    assert payload["workspace_retained"] is True


def test_ai_cli_openai_explain_run_smoke_json_outputs_typed_result(monkeypatch: pytest.MonkeyPatch):
    import truthound.ai as ai_namespace

    runner = CliRunner()
    smoke_result = ai_namespace.OpenAIExplainRunSmokeResult(
        model_name="gpt-smoke",
        success=False,
        failure_stage="provider",
        error_message="network down",
        workspace_dir="/tmp/truthound-ai-explain-run-smoke-test",
        workspace_retained=True,
    )
    monkeypatch.setattr(ai_namespace, "run_openai_explain_run_smoke", lambda **kwargs: smoke_result)

    result = runner.invoke(app, ["ai", "smoke", "openai-explain-run", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["success"] is False
    assert payload["failure_stage"] == "provider"
    assert payload["workspace_retained"] is True
