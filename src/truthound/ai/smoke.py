"""Live OpenAI smoke runners for Truthound AI proposal generation and run analysis."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

from truthound.ai.analysis import explain_run
from truthound.ai.models import (
    CompileStatus,
    OpenAIExplainRunSmokeResult,
    OpenAISmokeResult,
    OpenAIProviderSpec,
    ProviderConfig,
)
from truthound.ai.providers import (
    ProviderConfigurationError,
    ProviderResponseError,
    ProviderTransportError,
)
from truthound.ai.store import AIArtifactStore
from truthound.ai.suggest import suggest_suite
from truthound.context import TruthoundContext

SMOKE_MODEL_ENV = "TRUTHOUND_AI_SMOKE_MODEL"
SMOKE_BASE_URL_ENV = "TRUTHOUND_AI_SMOKE_BASE_URL"
SMOKE_TIMEOUT_ENV = "TRUTHOUND_AI_SMOKE_TIMEOUT_SECONDS"
SMOKE_RUN_GATE_ENV = "TRUTHOUND_AI_RUN_LIVE_SMOKE"
SMOKE_RESULT_PATH_ENV = "TRUTHOUND_AI_SMOKE_RESULT_PATH"
SMOKE_KEEP_WORKSPACE_ENV = "TRUTHOUND_AI_SMOKE_KEEP_WORKSPACE"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_SMOKE_PROMPT = (
    "Create a conservative reviewable proposal for this small orders dataset. "
    "Keep identifiers stable, keep refund_rate bounded, and validate status with safe aggregate checks only."
)


def run_openai_smoke(
    model: str | None = None,
    *,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    keep_workspace: bool = False,
) -> OpenAISmokeResult:
    """Run a live end-to-end OpenAI smoke against a temporary proposal workspace."""

    workspace_dir: Path | None = None
    resolved_model: str | None = None
    current_stage: Literal["config", "persist", "verify"] = "config"
    try:
        resolved_model, provider_config = _resolve_smoke_provider_config(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

        workspace_dir = Path(tempfile.mkdtemp(prefix="truthound-ai-smoke-")).resolve()
        data_path = workspace_dir / "smoke_orders.csv"
        _write_smoke_csv(data_path)

        context = TruthoundContext(workspace_dir)
        current_stage = "persist"
        artifact = _invoke_suggest_suite(
            prompt=_SMOKE_PROMPT,
            data=str(data_path),
            context=context,
            provider=provider_config,
            model=resolved_model,
            sample_size=50,
        )

        proposal_path = _proposal_path(context, artifact.artifact_id)
        if not proposal_path.exists():
            return _finalize_result(
                OpenAISmokeResult(
                    model_name=resolved_model,
                    success=False,
                    artifact_id=artifact.artifact_id,
                    compile_status=str(artifact.compile_status),
                    compiled_check_count=artifact.compiled_check_count,
                    rejected_check_count=artifact.rejected_check_count,
                    proposal_path=str(proposal_path),
                    failure_stage="persist",
                    error_message="smoke proposal artifact was not persisted to disk",
                ),
                workspace_dir=workspace_dir,
                keep_workspace=keep_workspace,
            )

        current_stage = "verify"
        persisted = _read_smoke_proposal(context, artifact.artifact_id)
        artifact_failure = _classify_proposal_artifact_failure(persisted)
        if artifact_failure is not None:
            stage, message = artifact_failure
            return _finalize_result(
                OpenAISmokeResult(
                    model_name=resolved_model,
                    success=False,
                    artifact_id=persisted.artifact_id,
                    compile_status=str(persisted.compile_status),
                    compiled_check_count=persisted.compiled_check_count,
                    rejected_check_count=persisted.rejected_check_count,
                    proposal_path=str(proposal_path),
                    failure_stage=stage,
                    error_message=message,
                ),
                workspace_dir=workspace_dir,
                keep_workspace=keep_workspace,
            )

        _verify_persisted_proposal(persisted, artifact)
        return _finalize_result(
            OpenAISmokeResult(
                model_name=resolved_model,
                success=True,
                artifact_id=persisted.artifact_id,
                compile_status=str(persisted.compile_status),
                compiled_check_count=persisted.compiled_check_count,
                rejected_check_count=persisted.rejected_check_count,
                proposal_path=str(proposal_path),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except ProviderConfigurationError as exc:
        return _finalize_result(
            OpenAISmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage="config",
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except ProviderTransportError as exc:
        return _finalize_result(
            OpenAISmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage="provider",
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except ProviderResponseError as exc:
        return _finalize_result(
            OpenAISmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage="parse",
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except Exception as exc:
        return _finalize_result(
            OpenAISmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage=current_stage,
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )


def run_openai_explain_run_smoke(
    model: str | None = None,
    *,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    keep_workspace: bool = False,
) -> OpenAIExplainRunSmokeResult:
    """Run a live end-to-end OpenAI smoke against the canonical explain_run flow."""

    workspace_dir: Path | None = None
    resolved_model: str | None = None
    current_stage: Literal["config", "prepare", "persist", "verify"] = "config"
    try:
        resolved_model, provider_config = _resolve_smoke_provider_config(
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

        workspace_dir = Path(tempfile.mkdtemp(prefix="truthound-ai-explain-run-smoke-")).resolve()
        context = TruthoundContext(workspace_dir)

        current_stage = "prepare"
        run_result = _prepare_explain_run_smoke_run(context)

        current_stage = "persist"
        artifact = _invoke_explain_run(
            run_id=run_result.run_id,
            context=context,
            include_history=True,
            provider=provider_config,
            model=resolved_model,
        )

        analysis_path = _analysis_path(context, artifact.artifact_id)
        if not analysis_path.exists():
            return _finalize_result(
                OpenAIExplainRunSmokeResult(
                    model_name=resolved_model,
                    success=False,
                    run_id=run_result.run_id,
                    artifact_id=artifact.artifact_id,
                    analysis_path=str(analysis_path),
                    failed_check_count=len(artifact.failed_checks),
                    top_column_count=len(artifact.top_columns),
                    evidence_ref_count=len(artifact.evidence_refs),
                    failure_stage="persist",
                    error_message="explain-run smoke analysis artifact was not persisted to disk",
                ),
                workspace_dir=workspace_dir,
                keep_workspace=keep_workspace,
            )

        current_stage = "verify"
        persisted = _read_smoke_analysis(context, artifact.artifact_id)
        artifact_failure = _classify_analysis_artifact_failure(persisted)
        if artifact_failure is not None:
            stage, message = artifact_failure
            return _finalize_result(
                OpenAIExplainRunSmokeResult(
                    model_name=resolved_model,
                    success=False,
                    run_id=persisted.run_id,
                    artifact_id=persisted.artifact_id,
                    analysis_path=str(analysis_path),
                    failed_check_count=len(persisted.failed_checks),
                    top_column_count=len(persisted.top_columns),
                    evidence_ref_count=len(persisted.evidence_refs),
                    failure_stage=stage,
                    error_message=message,
                ),
                workspace_dir=workspace_dir,
                keep_workspace=keep_workspace,
            )

        _verify_persisted_analysis(persisted, artifact, expected_run_id=run_result.run_id)
        return _finalize_result(
            OpenAIExplainRunSmokeResult(
                model_name=resolved_model,
                success=True,
                run_id=persisted.run_id,
                artifact_id=persisted.artifact_id,
                analysis_path=str(analysis_path),
                failed_check_count=len(persisted.failed_checks),
                top_column_count=len(persisted.top_columns),
                evidence_ref_count=len(persisted.evidence_refs),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except ProviderConfigurationError as exc:
        return _finalize_result(
            OpenAIExplainRunSmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage="config",
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except ProviderTransportError as exc:
        return _finalize_result(
            OpenAIExplainRunSmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage="provider",
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except ProviderResponseError as exc:
        return _finalize_result(
            OpenAIExplainRunSmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage="parse",
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )
    except Exception as exc:
        return _finalize_result(
            OpenAIExplainRunSmokeResult(
                model_name=resolved_model,
                success=False,
                failure_stage=current_stage,
                error_message=str(exc),
            ),
            workspace_dir=workspace_dir,
            keep_workspace=keep_workspace,
        )


def _invoke_suggest_suite(**kwargs: Any):
    return suggest_suite(**kwargs)


def _invoke_explain_run(**kwargs: Any):
    return explain_run(**kwargs)


def _read_smoke_proposal(context: TruthoundContext, artifact_id: str):
    return AIArtifactStore(context).read_proposal(artifact_id)


def _read_smoke_analysis(context: TruthoundContext, artifact_id: str):
    return AIArtifactStore(context).read_analysis(artifact_id)


def _proposal_path(context: TruthoundContext, artifact_id: str) -> Path:
    return AIArtifactStore(context).layout.proposal_path(artifact_id)


def _analysis_path(context: TruthoundContext, artifact_id: str) -> Path:
    return AIArtifactStore(context).layout.analysis_path(artifact_id)


def _verify_persisted_proposal(persisted, generated) -> None:
    if persisted.artifact_id != generated.artifact_id:
        raise ValueError("persisted smoke artifact_id does not match generated artifact_id")
    if str(persisted.compile_status) != str(generated.compile_status):
        raise ValueError("persisted smoke compile_status does not match generated compile_status")
    if persisted.diff_preview.current_suite.check_count < 1:
        raise ValueError("persisted smoke current_suite snapshot is empty")
    if persisted.diff_preview.proposed_suite.check_count < 1:
        raise ValueError("persisted smoke proposed_suite snapshot is empty")


def _verify_persisted_analysis(persisted, generated, *, expected_run_id: str) -> None:
    if persisted.artifact_id != generated.artifact_id:
        raise ValueError("persisted explain-run smoke artifact_id does not match generated artifact_id")
    if persisted.run_id != generated.run_id:
        raise ValueError("persisted explain-run smoke run_id does not match generated run_id")
    if persisted.run_id != expected_run_id:
        raise ValueError("persisted explain-run smoke run_id does not match prepared smoke run")
    if not persisted.summary.strip():
        raise ValueError("persisted explain-run smoke summary is empty")
    if not persisted.evidence_refs:
        raise ValueError("persisted explain-run smoke evidence_refs are empty")
    if not persisted.failed_checks:
        raise ValueError("persisted explain-run smoke failed_checks are empty")
    if not persisted.top_columns:
        raise ValueError("persisted explain-run smoke top_columns are empty")
    if "included" not in persisted.history_window:
        raise ValueError("persisted explain-run smoke history_window is missing required fields")


def _classify_proposal_artifact_failure(artifact) -> tuple[str, str] | None:
    compile_status = str(artifact.compile_status)
    if compile_status not in {CompileStatus.READY.value, CompileStatus.PARTIAL.value}:
        if "provider_output_validation_failed" in artifact.compiler_errors:
            return "parse", "smoke proposal was rejected because provider output could not be validated"
        return "compile", "smoke proposal finished in rejected compile_status"
    if artifact.compiled_check_count < 1:
        return "compile", "smoke proposal did not compile any executable checks"
    if artifact.diff_preview.current_suite.check_count < 1:
        return "compile", "smoke proposal did not include a current_suite snapshot"
    if artifact.diff_preview.proposed_suite.check_count < 1:
        return "compile", "smoke proposal did not include a proposed_suite snapshot"
    return None


def _classify_analysis_artifact_failure(artifact) -> tuple[str, str] | None:
    if not artifact.summary.strip():
        return "verify", "explain-run smoke analysis summary is empty"
    if not artifact.evidence_refs:
        return "verify", "explain-run smoke analysis did not include evidence refs"
    if len(artifact.failed_checks) < 1:
        return "verify", "explain-run smoke analysis did not include failed checks"
    if len(artifact.top_columns) < 1:
        return "verify", "explain-run smoke analysis did not include top columns"
    if "included" not in artifact.history_window:
        return "verify", "explain-run smoke analysis did not include a history window summary"
    return None


def _prepare_explain_run_smoke_run(context: TruthoundContext):
    import truthound as th

    run_result = th.check(
        {
            "customer_id": [1, 2, 2],
            "email": ["a@example.com", None, "c@example.com"],
        },
        context=context,
    )
    required_paths = [
        context.runs_dir / f"{run_result.run_id}.json",
        context.docs_dir / f"{run_result.run_id}.html",
        context.baseline_index_path,
        context.baselines_dir / "metric-history.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "explain-run smoke preparation did not persist required run evidence: "
            + ", ".join(missing)
        )
    return run_result


def _finalize_result(
    result: Any,
    *,
    workspace_dir: Path | None,
    keep_workspace: bool,
):
    retained = workspace_dir is not None and (keep_workspace or not result.success)
    if workspace_dir is not None and not retained:
        shutil.rmtree(workspace_dir, ignore_errors=True)
    return result.model_copy(
        update={
            "workspace_dir": str(workspace_dir) if workspace_dir is not None else None,
            "workspace_retained": retained,
        }
    )


def _resolve_smoke_provider_config(
    *,
    model: str | None,
    base_url: str | None,
    timeout_seconds: float | None,
) -> tuple[str, ProviderConfig]:
    resolved_model = _resolve_smoke_model_name(model)
    resolved_base_url = _resolve_smoke_base_url(base_url)
    resolved_timeout = _resolve_smoke_timeout(timeout_seconds)
    _require_openai_api_key()

    provider_config = ProviderConfig(
        provider_name=OpenAIProviderSpec().provider_name,
        model_name=resolved_model,
        api_key_env=OpenAIProviderSpec().api_key_env,
        base_url=resolved_base_url,
        timeout_seconds=resolved_timeout,
    )
    return resolved_model, provider_config


def _resolve_smoke_model_name(explicit_model: str | None) -> str:
    for candidate in (
        explicit_model,
        os.getenv(SMOKE_MODEL_ENV),
        os.getenv("TRUTHOUND_AI_MODEL"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ProviderConfigurationError(
        "No live smoke model configured. Pass --model, set TRUTHOUND_AI_SMOKE_MODEL, "
        "or export TRUTHOUND_AI_MODEL."
    )


def _resolve_smoke_base_url(explicit_base_url: str | None) -> str | None:
    for candidate in (explicit_base_url, os.getenv(SMOKE_BASE_URL_ENV)):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _resolve_smoke_timeout(explicit_timeout: float | None) -> float:
    if explicit_timeout is not None:
        if float(explicit_timeout) <= 0:
            raise ProviderConfigurationError("Live smoke timeout_seconds must be greater than zero.")
        return float(explicit_timeout)

    env_value = os.getenv(SMOKE_TIMEOUT_ENV)
    if env_value is None or not env_value.strip():
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        resolved = float(env_value)
    except ValueError as exc:
        raise ProviderConfigurationError(
            f"Invalid {SMOKE_TIMEOUT_ENV} value {env_value!r}; expected a positive number."
        ) from exc
    if resolved <= 0:
        raise ProviderConfigurationError(
            f"Invalid {SMOKE_TIMEOUT_ENV} value {env_value!r}; expected a positive number."
        )
    return resolved


def _require_openai_api_key() -> None:
    api_key = os.getenv(OpenAIProviderSpec().api_key_env, "")
    if not api_key.strip():
        raise ProviderConfigurationError(
            "OPENAI_API_KEY is required for live OpenAI smoke verification."
        )


def _write_smoke_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "order_id,refund_rate,status,sku_code",
                "1,0.10,pending,SKU-001",
                "2,0.25,approved,SKU-002",
                "3,0.00,pending,SKU-003",
            ]
        ),
        encoding="utf-8",
    )


__all__ = [
    "SMOKE_BASE_URL_ENV",
    "SMOKE_KEEP_WORKSPACE_ENV",
    "SMOKE_MODEL_ENV",
    "SMOKE_RESULT_PATH_ENV",
    "SMOKE_RUN_GATE_ENV",
    "SMOKE_TIMEOUT_ENV",
    "run_openai_explain_run_smoke",
    "run_openai_smoke",
]
