"""Adapter helpers for checkpoint result boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from truthound.core.results import ValidationRunResult
from truthound.reporters.presentation import (
    LegacyStatisticsView,
    LegacyStatusView,
    LegacyValidatorResultView,
    build_run_presentation,
)
from truthound.validators.base import ValidationIssue


@dataclass(frozen=True)
class CheckpointValidationView:
    """Compatibility view exposed during the checkpoint migration window."""

    run_id: str
    run_time: datetime
    data_asset: str
    status: LegacyStatusView
    success: bool
    results: tuple[LegacyValidatorResultView, ...]
    statistics: LegacyStatisticsView
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    suite_name: str | None = None
    runtime_environment: dict[str, Any] = field(default_factory=dict)


def ensure_validation_run_result(report: Any) -> ValidationRunResult:
    """Extract the canonical validation run result from a legacy report."""
    run_result = getattr(report, "validation_run", None)
    if isinstance(run_result, ValidationRunResult):
        return run_result

    from truthound.reporters.adapters import report_to_validation_run_result

    return report_to_validation_run_result(report)


def checkpoint_validation_view(
    run_result: ValidationRunResult,
) -> CheckpointValidationView:
    """Build the compatibility view used by legacy checkpoint actions."""
    presentation = build_run_presentation(run_result)
    legacy = presentation.to_legacy_view()
    return CheckpointValidationView(
        run_id=legacy.run_id,
        run_time=legacy.run_time,
        data_asset=legacy.data_asset,
        status=legacy.status,
        success=legacy.success,
        results=legacy.results,
        statistics=legacy.statistics,
        issues=tuple(run_result.issues),
        tags=dict(legacy.tags),
        metadata=dict(legacy.metadata),
        suite_name=legacy.suite_name,
        runtime_environment=dict(legacy.runtime_environment),
    )


def validation_run_to_persistence_result(run_result: ValidationRunResult) -> Any:
    """Convert a runtime validation result into the persistence DTO."""
    from truthound.stores.results import ValidationResult

    return ValidationResult.from_validation_run_result(run_result)


def validation_run_to_persistence_dict(run_result: ValidationRunResult) -> dict[str, Any]:
    """Convert a runtime validation result into the persisted wire format."""
    return validation_run_to_persistence_result(run_result).to_dict()


def validation_run_from_checkpoint_dict(data: dict[str, Any]) -> ValidationRunResult | None:
    """Restore the canonical result model from serialized checkpoint data."""
    payload = data.get("validation_run")
    if isinstance(payload, dict):
        return ValidationRunResult.from_dict(payload)

    legacy_payload = data.get("validation_result")
    if isinstance(legacy_payload, dict):
        from truthound.stores.results import ValidationResult

        return ValidationResult.from_dict(legacy_payload).to_validation_run_result()

    return None
