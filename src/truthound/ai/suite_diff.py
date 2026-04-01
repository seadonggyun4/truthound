"""Formal ValidationSuite snapshot and diff helpers for AI proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl

from truthound._applied_suite import canonical_check_key, normalize_json_value
from truthound.ai.models import (
    CompiledProposalCheck,
    RejectedProposalItem,
    SuiteCheckSnapshot,
    ValidationSuiteConflict,
    ValidationSuiteDiffCounts,
    ValidationSuiteDiffPreview,
    ValidationSuiteSnapshot,
)
from truthound.schema import Schema, learn

if TYPE_CHECKING:
    from truthound.core.suite import CheckSpec, ValidationSuite


@dataclass(frozen=True)
class FormalSuiteDiffResult:
    """Resolved proposal diff and the materialized compiled checks it accepted."""

    compiled_checks: list[CompiledProposalCheck]
    rejected_items: list[RejectedProposalItem]
    diff_preview: ValidationSuiteDiffPreview
def build_current_validation_suite(
    *,
    observed_df: pl.DataFrame,
    baseline_schema: Schema | None,
) -> "ValidationSuite":
    from truthound.core.suite import ValidationSuite

    transient_schema = baseline_schema
    if transient_schema is None:
        transient_schema = learn(data=observed_df)

    return ValidationSuite.from_legacy(
        context=None,
        validators=None,
        schema=transient_schema,
        auto_schema=False,
        data=observed_df,
        source=None,
    )


def build_existing_suite_summary(snapshot: ValidationSuiteSnapshot) -> dict[str, Any]:
    return {
        "suite_name": snapshot.suite_name,
        "check_count": snapshot.check_count,
        "checks": [
            {
                "check_key": check.check_key,
                "validator_name": check.validator_name,
                "category": check.category,
                "columns": list(check.columns),
                "params": dict(check.params),
            }
            for check in snapshot.checks
        ],
    }


def snapshot_validation_suite(suite: "ValidationSuite") -> ValidationSuiteSnapshot:
    checks = [snapshot_check_spec(spec) for spec in suite.checks]
    evidence_mode = getattr(suite.evidence_policy.result_format.format, "value", None)
    min_severity = getattr(suite.severity_policy.min_severity, "value", None)
    return ValidationSuiteSnapshot(
        suite_name=suite.name,
        check_count=len(checks),
        schema_check_present=any(check.validator_name == "schema" for check in checks),
        evidence_mode=str(evidence_mode) if evidence_mode is not None else "summary",
        min_severity=str(min_severity) if min_severity is not None else None,
        checks=checks,
    )


def snapshot_check_spec(spec: "CheckSpec") -> SuiteCheckSnapshot:
    metadata = spec.metadata if isinstance(spec.metadata, dict) else {}
    raw_config = metadata.get("config", {}) if isinstance(metadata.get("config", {}), dict) else {}
    normalized_config = normalize_json_value(raw_config)
    columns = [str(item) for item in normalized_config.get("columns", ()) or ()]
    params = {
        str(key): value
        for key, value in normalized_config.items()
        if key != "columns"
    }
    rationale = str(metadata.get("rationale") or metadata.get("auto_reason") or "")
    origin = _resolve_origin(metadata)
    return SuiteCheckSnapshot(
        check_id=str(spec.id),
        check_key=canonical_check_key(
            validator_name=spec.name,
            columns=columns,
            params=params,
        ),
        validator_name=spec.name,
        category=spec.category,
        columns=columns,
        params=params,
        tags=[str(item) for item in spec.tags],
        rationale=rationale,
        origin=origin,
    )


def build_formal_suite_diff(
    *,
    current_suite: "ValidationSuite",
    compiled_checks: list[CompiledProposalCheck],
    rejected_items: list[RejectedProposalItem],
) -> FormalSuiteDiffResult:
    current_snapshot = snapshot_validation_suite(current_suite)
    current_by_key = {check.check_key: check for check in current_snapshot.checks}
    current_by_base = {
        _base_key(check.validator_name, tuple(check.columns)): check
        for check in current_snapshot.checks
    }

    accepted_checks: list[CompiledProposalCheck] = []
    added_specs: list["CheckSpec"] = []
    added_snapshots: list[SuiteCheckSnapshot] = []
    already_present: list[SuiteCheckSnapshot] = []
    conflicts: list[ValidationSuiteConflict] = []
    materialization_rejections: list[RejectedProposalItem] = []

    for check in compiled_checks:
        try:
            spec = check.to_check_spec()
            spec.build_validator()
            snapshot = snapshot_check_spec(spec)
        except Exception as exc:
            materialization_rejections.append(
                RejectedProposalItem(
                    source="compiler",
                    intent=check.validator_name,
                    columns=list(check.columns),
                    params=dict(check.params),
                    reason=f"compiled check could not materialize as CheckSpec: {exc}",
                    rationale=check.rationale or None,
                )
            )
            continue

        accepted_checks.append(check)
        if snapshot.check_key in current_by_key:
            already_present.append(snapshot)
            continue

        base_key = _base_key(snapshot.validator_name, tuple(snapshot.columns))
        existing = current_by_base.get(base_key)
        if existing is not None:
            conflicts.append(
                ValidationSuiteConflict(
                    proposed=snapshot,
                    existing=existing,
                )
            )
            continue

        added_specs.append(spec)
        added_snapshots.append(snapshot)

    proposed_suite = merge_validation_suites(
        current_suite=current_suite,
        added_specs=added_specs,
    )
    proposed_snapshot = snapshot_validation_suite(proposed_suite)
    all_rejected = [*rejected_items, *materialization_rejections]
    return FormalSuiteDiffResult(
        compiled_checks=accepted_checks,
        rejected_items=all_rejected,
        diff_preview=ValidationSuiteDiffPreview(
            current_suite=current_snapshot,
            proposed_suite=proposed_snapshot,
            added=added_snapshots,
            already_present=already_present,
            conflicts=conflicts,
            rejected=all_rejected,
            counts=ValidationSuiteDiffCounts(
                added=len(added_snapshots),
                already_present=len(already_present),
                conflicts=len(conflicts),
                rejected=len(all_rejected),
            ),
        ),
    )


def merge_validation_suites(
    *,
    current_suite: "ValidationSuite",
    added_specs: list["CheckSpec"],
) -> "ValidationSuite":
    from truthound.core.suite import ValidationSuite

    return ValidationSuite(
        name=current_suite.name,
        checks=tuple([*current_suite.checks, *added_specs]),
        evidence_policy=current_suite.evidence_policy,
        severity_policy=current_suite.severity_policy,
        schema_spec=current_suite.schema_spec,
        metadata={
            **current_suite.metadata,
            "proposal_candidate": True,
        },
    )


def _base_key(validator_name: str, columns: tuple[str, ...]) -> str:
    return f"{validator_name}|{','.join(sorted(columns))}"
def _resolve_origin(metadata: dict[str, Any]) -> str:
    explicit = metadata.get("origin")
    if isinstance(explicit, str) and explicit:
        return explicit
    if metadata.get("proposal_check_key"):
        return "proposal"
    return "current"


__all__ = [
    "FormalSuiteDiffResult",
    "build_current_validation_suite",
    "build_existing_suite_summary",
    "build_formal_suite_diff",
    "canonical_check_key",
    "merge_validation_suites",
    "snapshot_check_spec",
    "snapshot_validation_suite",
]
