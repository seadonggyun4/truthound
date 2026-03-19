"""Adapters for reporter-facing result contracts.

This module canonicalizes legacy report objects and persisted validation DTOs
into the Truthound 2.0 ``ValidationRunResult`` model.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from warnings import warn

from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.report import Report
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

    if isinstance(data, Report):
        return report_to_validation_run_result(data, warn_legacy=warn_legacy)

    if isinstance(data, StoredValidationResult):
        return stored_validation_result_to_validation_run_result(
            data,
            warn_legacy=warn_legacy,
        )

    raise TypeError(
        "Reporters expect ValidationRunResult as the canonical input. "
        f"Received unsupported type: {type(data).__name__}"
    )


def report_to_validation_run_result(
    report: Report,
    *,
    warn_legacy: bool = False,
) -> ValidationRunResult:
    """Convert a legacy ``Report`` into ``ValidationRunResult``."""
    existing = getattr(report, "validation_run", None)
    if isinstance(existing, ValidationRunResult):
        return existing

    if warn_legacy:
        warn(
            "Passing truthound.report.Report directly to reporters is deprecated. "
            "Pass report.validation_run or a ValidationRunResult instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    issues = tuple(report.get_sorted_issues() if hasattr(report, "get_sorted_issues") else report.issues)
    checks = _build_checks_from_issues(issues)
    metadata: dict[str, Any] = {
        "legacy_report": True,
        "report_success": getattr(report, "success", not bool(issues)),
    }
    statistics = getattr(report, "statistics", None)
    if statistics is not None and hasattr(statistics, "to_dict"):
        metadata["statistics"] = statistics.to_dict()
    exception_summary = getattr(report, "exception_summary", None)
    if exception_summary is not None and hasattr(exception_summary, "to_dict"):
        metadata["exception_summary"] = exception_summary.to_dict()

    return ValidationRunResult(
        suite_name="legacy-report",
        source=report.source,
        row_count=report.row_count,
        column_count=report.column_count,
        result_format=report.result_format,
        execution_mode="legacy",
        checks=checks,
        issues=issues,
        metadata=metadata,
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

    return ValidationRunResult(
        run_id=result.run_id,
        run_time=result.run_time,
        suite_name=result.suite_name or "stored-validation-result",
        source=result.data_asset,
        row_count=result.statistics.total_rows,
        column_count=result.statistics.total_columns,
        result_format=ResultFormat(str(metadata.pop("_result_format", ResultFormat.SUMMARY.value))),
        execution_mode=str(metadata.pop("_execution_mode", "stored")),
        checks=tuple(checks),
        issues=tuple(issues),
        execution_issues=execution_issues,
        metadata=metadata,
    )


def _build_checks_from_issues(
    issues: tuple[ValidationIssue, ...],
) -> tuple[CheckResult, ...]:
    by_name: "OrderedDict[str, list[ValidationIssue]]" = OrderedDict()
    for issue in issues:
        check_name = issue.validator_name or issue.issue_type
        by_name.setdefault(check_name, []).append(issue)

    checks: list[CheckResult] = []
    for name, grouped in by_name.items():
        checks.append(
            CheckResult(
                name=name,
                category="legacy",
                success=False,
                issue_count=len(grouped),
                issues=tuple(grouped),
            )
        )

    return tuple(checks)


def _coerce_severity(value: str | Severity | None) -> Severity:
    if isinstance(value, Severity):
        return value
    if value is None:
        return Severity.LOW
    return Severity(value.lower())
