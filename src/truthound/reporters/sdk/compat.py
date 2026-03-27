"""Compatibility helpers for the reporter SDK.

The SDK now treats ``ValidationRunResult`` as the canonical input contract.
Legacy report and persistence DTO inputs are normalized here so templates and
scaffolding can consume the shared presentation model without importing
storage-only result types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from truthound.reporters.adapters import canonicalize_validation_run_result
from truthound.reporters.base import ReporterContext
from truthound.reporters.presentation import (
    LegacyValidationResultView,
    RunPresentation,
    build_run_presentation,
)

if TYPE_CHECKING:
    from truthound.core.results import ValidationRunResult


def to_validation_run_result(data: Any) -> ValidationRunResult:
    """Normalize supported reporter inputs into ``ValidationRunResult``."""
    return canonicalize_validation_run_result(data, warn_legacy=True)


def build_sdk_presentation(
    data: Any,
    *,
    context: ReporterContext | None = None,
    title: str | None = None,
    max_sample_values: int = 5,
) -> RunPresentation:
    """Build the shared presentation model for SDK helpers."""
    resolved_context = context or ReporterContext(title=title or "Truthound Validation Report")
    run_result = to_validation_run_result(data)
    return build_run_presentation(
        run_result,
        title=title or resolved_context.title,
        max_sample_values=max_sample_values,
    )


def build_sdk_legacy_view(
    data: Any,
    *,
    context: ReporterContext | None = None,
    title: str | None = None,
    max_sample_values: int = 5,
) -> LegacyValidationResultView:
    """Build the legacy-shaped compatibility view from canonical input."""
    return build_sdk_presentation(
        data,
        context=context,
        title=title,
        max_sample_values=max_sample_values,
    ).to_legacy_view()
