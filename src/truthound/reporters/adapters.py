"""Adapters for reporter-facing result contracts.

This module canonicalizes persisted validation DTOs into the Truthound 3.0
``ValidationRunResult`` model. Legacy ``Report`` objects are no longer a
supported runtime input.
"""

from __future__ import annotations

from typing import Any
from warnings import warn

from truthound.core.execution_modes import coarse_planned_execution_mode
from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.types import ResultFormat, Severity
from truthound.validators.base import ValidationIssue


def canonicalize_validation_run_result(
    data: Any,
    *,
    warn_legacy: bool = False,
) -> ValidationRunResult:
    """Normalize supported reporter inputs into ``ValidationRunResult``."""
    from truthound.stores.results import ValidationResult as StoredValidationResult

    if isinstance(data, ValidationRunResult):
        return data

    if isinstance(data, StoredValidationResult):
        return stored_validation_result_to_validation_run_result(
            data,
            warn_legacy=warn_legacy,
        )

    raise TypeError(
        "Reporters expect ValidationRunResult as the canonical input in Truthound 3.0. "
        f"Received unsupported type: {type(data).__name__}"
    )


def stored_validation_result_to_validation_run_result(
    result: Any,
    *,
    warn_legacy: bool = False,
) -> ValidationRunResult:
    """Convert the persisted ValidationResult DTO into ``ValidationRunResult``."""
    if warn_legacy:
        warn(
            "Passing truthound.stores.results.ValidationResult directly to reporters "
            "is deprecated. Convert it to ValidationRunResult instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    issues: list[ValidationIssue] = []
    checks: list[CheckResult] = []

    for validator_result in result.results:
        issue_tuple: tuple[ValidationIssue, ...] = ()
        if not validator_result.success:
            issue = ValidationIssue(
                column=validator_result.column or "_table_",
                issue_type=validator_result.issue_type or validator_result.validator_name,
                count=validator_result.count,
                severity=_coerce_severity(validator_result.severity),
                details=validator_result.message,
                expected=validator_result.details.get("expected"),
                actual=validator_result.details.get("actual"),
                sample_values=validator_result.details.get("sample_values"),
                validator_name=validator_result.validator_name,
            )
            issues.append(issue)
            issue_tuple = (issue,)

        checks.append(
            CheckResult(
                name=validator_result.validator_name,
                category=str(validator_result.details.get("category", "legacy")),
                success=validator_result.success,
                issue_count=len(issue_tuple),
                issues=issue_tuple,
                metadata=dict(validator_result.details),
            )
        )

    metadata = dict(result.metadata)
    validation_run_meta = metadata.pop("_validation_run_contract", {})
    if isinstance(validation_run_meta, dict) and validation_run_meta.get("checks"):
        issues_by_name: dict[str, list[ValidationIssue]] = {}
        for issue in issues:
            issues_by_name.setdefault(issue.validator_name or issue.issue_type, []).append(issue)

        checks = [
            CheckResult(
                name=str(check_data["name"]),
                category=str(check_data.get("category", "legacy")),
                success=bool(check_data.get("success", False)),
                issue_count=int(check_data.get("issue_count", len(issues_by_name.get(str(check_data["name"]), [])))),
                issues=tuple(issues_by_name.get(str(check_data["name"]), [])),
                metadata=dict(check_data.get("metadata", {})),
            )
            for check_data in validation_run_meta.get("checks", [])
        ]

    if result.tags:
        metadata["tags"] = dict(result.tags)
    if result.runtime_environment:
        metadata["runtime_environment"] = dict(result.runtime_environment)
    execution_issues = ()
    if isinstance(validation_run_meta, dict):
        execution_issues = tuple(
            ExecutionIssue(
                check_name=str(issue.get("check_name", "")),
                message=str(issue.get("message", "")),
                exception_type=issue.get("exception_type"),
                failure_category=issue.get("failure_category"),
                retry_count=int(issue.get("retry_count", 0)),
            )
            for issue in validation_run_meta.get("execution_issues", [])
        )
    execution_mode = str(metadata.pop("_execution_mode", "sequential"))
    planned_execution_mode = metadata.pop("_planned_execution_mode", None)

    return ValidationRunResult(
        run_id=result.run_id,
        run_time=result.run_time,
        suite_name=result.suite_name or "stored-validation-result",
        source=result.data_asset,
        row_count=result.statistics.total_rows,
        column_count=result.statistics.total_columns,
        result_format=ResultFormat(str(metadata.pop("_result_format", ResultFormat.SUMMARY.value))),
        execution_mode=execution_mode,
        planned_execution_mode=planned_execution_mode or coarse_planned_execution_mode(execution_mode),
        checks=tuple(checks),
        issues=tuple(issues),
        execution_issues=execution_issues,
        metadata=metadata,
    )


def _coerce_severity(value: str | Severity | None) -> Severity:
    if isinstance(value, Severity):
        return value
    if value is None:
        return Severity.LOW
    return Severity(value.lower())
