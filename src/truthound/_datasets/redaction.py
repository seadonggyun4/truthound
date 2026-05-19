"""Redaction boundary for private dataset repository artifacts."""

from __future__ import annotations

from typing import Any

from truthound._redaction import (
    RedactionViolation,
    RedactionViolationError,
    SummaryOnlyRedactor,
)


def assert_dataset_artifact_safe(
    payload: Any,
    *,
    label: str = "dataset artifact",
) -> None:
    """Reject dataset artifact payloads that leak raw rows or PII-like values."""

    SummaryOnlyRedactor().assert_safe(payload, label=label)


__all__ = [
    "RedactionViolation",
    "RedactionViolationError",
    "SummaryOnlyRedactor",
    "assert_dataset_artifact_safe",
]
