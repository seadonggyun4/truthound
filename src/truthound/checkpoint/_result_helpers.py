"""Private helpers for top-level checkpoint result handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from truthound.core.results import ValidationRunResult

if TYPE_CHECKING:
    from collections.abc import Mapping

CheckpointOutcome = Literal["success", "warning", "failure"]


def merge_checkpoint_metadata(
    run_result: ValidationRunResult,
    checkpoint_metadata: Mapping[str, Any],
    tags: Mapping[str, str],
    duration_ms: float,
) -> dict[str, Any]:
    """Merge runtime and checkpoint metadata into the persisted checkpoint view."""
    return {
        **run_result.metadata,
        **dict(checkpoint_metadata),
        "tags": dict(tags),
        "execution_time_ms": duration_ms,
    }


def build_checkpoint_validation_run(
    run_result: ValidationRunResult,
    *,
    run_id: str,
    data_asset: str,
    checkpoint_metadata: Mapping[str, Any],
    tags: Mapping[str, str],
    duration_ms: float,
) -> ValidationRunResult:
    """Rebuild a validation run with checkpoint-specific metadata and identity."""
    return ValidationRunResult(
        run_id=run_id,
        run_time=run_result.run_time,
        suite_name=run_result.suite_name,
        source=data_asset,
        row_count=run_result.row_count,
        column_count=run_result.column_count,
        result_format=run_result.result_format,
        execution_mode=run_result.execution_mode,
        planned_execution_mode=run_result.planned_execution_mode,
        checks=run_result.checks,
        issues=run_result.issues,
        execution_issues=run_result.execution_issues,
        metadata=merge_checkpoint_metadata(
            run_result,
            checkpoint_metadata,
            tags,
            duration_ms,
        ),
    )


def derive_checkpoint_outcome(
    run_result: ValidationRunResult,
    *,
    fail_on_critical: bool,
    fail_on_high: bool,
) -> CheckpointOutcome:
    """Derive checkpoint outcome from the compatibility statistics view."""
    from truthound.checkpoint.adapters import checkpoint_validation_view

    statistics = checkpoint_validation_view(run_result).statistics
    if (
        statistics.critical_issues > 0
        and fail_on_critical
    ) or (
        statistics.high_issues > 0
        and fail_on_high
    ):
        return "failure"
    if statistics.total_issues > 0:
        return "warning"
    return "success"
