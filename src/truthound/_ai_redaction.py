"""Shared redaction helpers for Truthound AI payloads and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import polars as pl

from truthound.audit.core import mask_sensitive_value
from truthound.maskers import mask_data
from truthound.scanners import PII_PATTERNS, scan_pii

FORBIDDEN_FIELD_MARKERS = (
    "sample",
    "samples",
    "sample_values",
    "raw_rows",
    "raw_row",
    "row_samples",
    "row_sample",
    "record_samples",
    "record_sample",
    "example_rows",
    "example_row",
)
FORBIDDEN_CONTEXT_MARKERS = ("sample", "samples", "row", "rows", "record", "records", "example")


@dataclass(frozen=True)
class RedactionViolation:
    """A redaction violation found during payload inspection."""

    path: str
    message: str


class RedactionViolationError(ValueError):
    """Raised when a payload violates the summary-only redaction contract."""

    def __init__(self, violations: list[RedactionViolation], *, label: str = "payload") -> None:
        self.violations = violations
        joined = "; ".join(f"{item.path}: {item.message}" for item in violations)
        super().__init__(f"Summary-only redaction rejected {label}: {joined}")


class SummaryOnlyRedactor:
    """Reject payloads that include row-level or PII-like outbound content."""

    def inspect_payload(self, payload: Any, *, path: str = "$") -> list[RedactionViolation]:
        violations: list[RedactionViolation] = []

        if isinstance(payload, dict):
            for key, value in payload.items():
                child_path = f"{path}.{key}"
                if self._key_looks_forbidden(str(key)):
                    preview = self._masked_preview(value)
                    violations.append(
                        RedactionViolation(
                            path=child_path,
                            message=(
                                "field suggests row-level samples, which are not allowed "
                                f"under summary-only redaction (preview={preview})"
                            ),
                        )
                    )
                    continue
                violations.extend(self.inspect_payload(value, path=child_path))
            return violations

        if isinstance(payload, list):
            if self._path_looks_forbidden(path):
                violations.append(
                    RedactionViolation(
                        path=path,
                        message=(
                            "list-style sample payloads are not allowed under summary-only "
                            f"redaction (preview={self._masked_preview(payload)})"
                        ),
                    )
                )
                return violations
            for index, item in enumerate(payload):
                violations.extend(self.inspect_payload(item, path=f"{path}[{index}]"))
            return violations

        if isinstance(payload, tuple):
            for index, item in enumerate(payload):
                violations.extend(self.inspect_payload(item, path=f"{path}[{index}]"))
            return violations

        if isinstance(payload, str):
            violations.extend(self.inspect_text(payload, path=path))

        return violations

    def inspect_text(self, text: str, *, path: str) -> list[RedactionViolation]:
        stripped = text.strip()
        if not stripped:
            return []

        findings = self._scan_text_for_pii(stripped)
        if findings:
            pii_types = ", ".join(sorted({str(item.get("pii_type", "unknown")) for item in findings}))
            return [
                RedactionViolation(
                    path=path,
                    message=(
                        "text contains PII-like literal content "
                        f"(types={pii_types}, preview={mask_sensitive_value(stripped)})"
                    ),
                )
            ]

        lowered = stripped.lower()
        if self._path_looks_forbidden(path) or any(marker in lowered for marker in FORBIDDEN_CONTEXT_MARKERS):
            if any(token in stripped for token in ("[", "]", "{", "}", "=", "\n")):
                return [
                    RedactionViolation(
                        path=path,
                        message=(
                            "text looks like a row-level sample or raw excerpt, which is "
                            "forbidden under summary-only redaction"
                        ),
                    )
                ]

        return []

    def assert_safe(self, payload: Any, *, label: str = "payload") -> None:
        violations = self.inspect_payload(payload)
        if violations:
            raise RedactionViolationError(violations, label=label)

    def _key_looks_forbidden(self, key: str) -> bool:
        lowered = key.lower()
        return lowered in FORBIDDEN_FIELD_MARKERS or any(
            marker in lowered for marker in ("sample_values", "raw_rows", "example_rows")
        )

    def _path_looks_forbidden(self, path: str) -> bool:
        lowered = path.lower()
        return any(f".{marker}" in lowered or lowered.endswith(marker) for marker in FORBIDDEN_FIELD_MARKERS)

    def _scan_text_for_pii(self, text: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        try:
            frame = pl.DataFrame({"value": [text]})
            findings.extend(scan_pii(frame.lazy()))
        except Exception:
            pass

        token_candidates = [
            candidate.strip("\"'()[]{}")
            for candidate in re.split(r"[\s,;|]+", text)
            if candidate.strip("\"'()[]{}")
        ]
        for pii_pattern in PII_PATTERNS:
            for candidate in token_candidates:
                if pii_pattern.pattern.match(candidate):
                    findings.append(
                        {
                            "column": "_inline_",
                            "pii_type": pii_pattern.pii_type.value,
                            "count": 1,
                            "confidence": pii_pattern.confidence_base,
                        }
                    )
                    break
        return findings

    def _masked_preview(self, value: Any) -> str:
        try:
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                masked = mask_data(pl.DataFrame(value).lazy())
                return masked.head(1).write_json()
            if isinstance(value, dict):
                masked = mask_data(pl.DataFrame([value]).lazy())
                return masked.head(1).write_json()
        except Exception:
            pass
        return mask_sensitive_value(value)


__all__ = [
    "RedactionViolation",
    "RedactionViolationError",
    "SummaryOnlyRedactor",
]
