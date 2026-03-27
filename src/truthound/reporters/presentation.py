"""Shared presentation model for validation reporters and data docs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from truthound.types import Severity

if TYPE_CHECKING:
    from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
    from truthound.validators.base import ValidationIssue

_SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}


@dataclass(frozen=True)
class IssuePresentation:
    validator_name: str
    issue_type: str
    column: str | None
    count: int
    severity: str
    message: str | None
    expected: Any = None
    actual: Any = None
    sample_values: tuple[Any, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)

    def to_legacy_issue_dict(self) -> dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "success": False,
            "column": self.column,
            "issue_type": self.issue_type,
            "count": self.count,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
            "execution_time_ms": 0.0,
        }


@dataclass(frozen=True)
class LegacyStatusView:
    value: str


@dataclass(frozen=True)
class LegacyStatisticsView:
    total_validators: int
    passed_validators: int
    failed_validators: int
    error_validators: int
    total_rows: int
    total_columns: int
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    execution_time_ms: float

    @property
    def pass_rate(self) -> float:
        if self.total_validators == 0:
            return 1.0
        return self.passed_validators / self.total_validators

    @property
    def issue_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.total_issues / self.total_rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_validators": self.total_validators,
            "passed_validators": self.passed_validators,
            "failed_validators": self.failed_validators,
            "error_validators": self.error_validators,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
            "execution_time_ms": self.execution_time_ms,
            "pass_rate": self.pass_rate,
            "issue_rate": self.issue_rate,
        }


@dataclass(frozen=True)
class LegacyValidatorResultView:
    validator_name: str
    success: bool
    column: str | None = None
    issue_type: str | None = None
    count: int = 0
    severity: str | None = None
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator_name": self.validator_name,
            "success": self.success,
            "column": self.column,
            "issue_type": self.issue_type,
            "count": self.count,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass(frozen=True)
class LegacyValidationResultView:
    run_id: str
    run_time: datetime
    data_asset: str
    status: LegacyStatusView
    success: bool
    results: tuple[LegacyValidatorResultView, ...]
    statistics: LegacyStatisticsView
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    suite_name: str | None = None
    runtime_environment: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckPresentation:
    name: str
    category: str
    success: bool
    issue_count: int
    top_severity: str | None
    columns: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionIssuePresentation:
    check_name: str
    message: str
    exception_type: str | None = None
    failure_category: str | None = None
    retry_count: int = 0


@dataclass(frozen=True)
class RunSummary:
    total_checks: int
    passed_checks: int
    failed_checks: int
    total_rows: int
    total_columns: int
    total_issues: int
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    total_execution_issues: int = 0
    execution_time_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total_checks == 0:
            return 1.0
        return self.passed_checks / self.total_checks

    @property
    def issue_rate(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.total_issues / self.total_rows

    def to_legacy_statistics_dict(self) -> dict[str, Any]:
        return {
            "total_validators": self.total_checks,
            "passed_validators": self.passed_checks,
            "failed_validators": self.failed_checks,
            "error_validators": self.total_execution_issues,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "high_issues": self.high_issues,
            "medium_issues": self.medium_issues,
            "low_issues": self.low_issues,
            "execution_time_ms": self.execution_time_ms,
            "pass_rate": self.pass_rate,
            "issue_rate": self.issue_rate,
        }


@dataclass(frozen=True)
class RunPresentation:
    title: str
    run_id: str
    run_time: datetime
    suite_name: str
    source: str
    success: bool
    status: str
    result_format: str
    execution_mode: str
    planned_execution_mode: str
    summary: RunSummary
    checks: tuple[CheckPresentation, ...]
    issues: tuple[IssuePresentation, ...]
    execution_issues: tuple[ExecutionIssuePresentation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_links: tuple[str, ...] = field(default_factory=tuple)
    issue_counts_by_severity: dict[str, int] = field(default_factory=dict)
    issue_counts_by_column: dict[str, int] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_legacy_view(self) -> LegacyValidationResultView:
        """Expose a compatibility view for legacy renderers."""
        stats = LegacyStatisticsView(
            total_validators=self.summary.total_checks,
            passed_validators=self.summary.passed_checks,
            failed_validators=self.summary.failed_checks,
            error_validators=self.summary.total_execution_issues,
            total_rows=self.summary.total_rows,
            total_columns=self.summary.total_columns,
            total_issues=self.summary.total_issues,
            critical_issues=self.summary.critical_issues,
            high_issues=self.summary.high_issues,
            medium_issues=self.summary.medium_issues,
            low_issues=self.summary.low_issues,
            execution_time_ms=self.summary.execution_time_ms,
        )
        result_rows = tuple(
            LegacyValidatorResultView(
                validator_name=issue.validator_name,
                success=False,
                column=issue.column,
                issue_type=issue.issue_type,
                count=issue.count,
                severity=issue.severity,
                message=issue.message,
                details=dict(issue.details),
            )
            for issue in self.issues
        )
        return LegacyValidationResultView(
            run_id=self.run_id,
            run_time=self.run_time,
            data_asset=self.source,
            status=LegacyStatusView(self.status),
            success=self.success,
            results=result_rows,
            statistics=stats,
            tags=dict(self.metadata.get("tags", {})),
            metadata=dict(self.metadata),
            suite_name=self.suite_name,
            runtime_environment=dict(self.metadata.get("runtime_environment", {})),
        )


def build_run_presentation(
    run_result: ValidationRunResult,
    *,
    title: str = "Truthound Validation Report",
    max_sample_values: int = 5,
) -> RunPresentation:
    issues = tuple(
        _issue_to_presentation(issue, max_sample_values=max_sample_values)
        for issue in _sort_issues(run_result.issues)
    )
    checks = tuple(_check_to_presentation(check) for check in run_result.checks)
    execution_issues = tuple(
        _execution_issue_to_presentation(issue)
        for issue in run_result.execution_issues
    )
    severity_counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }
    column_counts: dict[str, int] = {}
    for issue in issues:
        severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
        column_key = issue.column or "_table_"
        column_counts[column_key] = column_counts.get(column_key, 0) + 1

    passed_checks = sum(1 for check in run_result.checks if check.success)
    failed_checks = len(run_result.checks) - passed_checks
    execution_time_ms = _extract_execution_time_ms(run_result.metadata)
    summary = RunSummary(
        total_checks=len(run_result.checks),
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        total_rows=run_result.row_count,
        total_columns=run_result.column_count,
        total_issues=len(issues),
        critical_issues=severity_counts["critical"],
        high_issues=severity_counts["high"],
        medium_issues=severity_counts["medium"],
        low_issues=severity_counts["low"],
        total_execution_issues=len(execution_issues),
        execution_time_ms=execution_time_ms,
    )
    metadata = dict(run_result.metadata)
    artifact_links = tuple(metadata.get("artifact_links", ()))

    return RunPresentation(
        title=title,
        run_id=run_result.run_id,
        run_time=run_result.run_time,
        suite_name=run_result.suite_name,
        source=run_result.source,
        success=not run_result.has_failures,
        status="success" if not run_result.has_failures else "failure",
        result_format=run_result.result_format.value,
        execution_mode=run_result.execution_mode,
        planned_execution_mode=run_result.planned_execution_mode or run_result.execution_mode,
        summary=summary,
        checks=checks,
        issues=issues,
        execution_issues=execution_issues,
        metadata=metadata,
        artifact_links=artifact_links,
        issue_counts_by_severity=severity_counts,
        issue_counts_by_column=column_counts,
    )


def _sort_issues(issues: tuple[ValidationIssue, ...]) -> list[ValidationIssue]:
    return sorted(
        issues,
        key=lambda issue: (
            _SEVERITY_ORDER.get(issue.severity, 99),
            issue.column or "",
            issue.issue_type,
        ),
    )


def _issue_to_presentation(
    issue: ValidationIssue,
    *,
    max_sample_values: int,
) -> IssuePresentation:
    details: dict[str, Any] = {}
    if issue.expected is not None:
        details["expected"] = issue.expected
    if issue.actual is not None:
        details["actual"] = issue.actual
    if issue.sample_values:
        details["sample_values"] = list((issue.sample_values or [])[:max_sample_values])

    return IssuePresentation(
        validator_name=issue.validator_name or issue.issue_type,
        issue_type=issue.issue_type,
        column=issue.column,
        count=issue.count,
        severity=issue.severity.value,
        message=issue.details,
        expected=issue.expected,
        actual=issue.actual,
        sample_values=tuple((issue.sample_values or [])[:max_sample_values]),
        details=details,
    )


def _check_to_presentation(check: CheckResult) -> CheckPresentation:
    issue_columns = tuple(
        column
        for column in dict.fromkeys(issue.column for issue in check.issues if issue.column)
        if column is not None
    )
    top_severity = None
    if check.issues:
        top_severity = min(
            check.issues,
            key=lambda issue: _SEVERITY_ORDER.get(issue.severity, 99),
        ).severity.value
    return CheckPresentation(
        name=check.name,
        category=check.category,
        success=check.success,
        issue_count=check.issue_count,
        top_severity=top_severity,
        columns=issue_columns,
        metadata=dict(check.metadata),
    )


def _execution_issue_to_presentation(
    issue: ExecutionIssue,
) -> ExecutionIssuePresentation:
    return ExecutionIssuePresentation(
        check_name=issue.check_name,
        message=issue.message,
        exception_type=issue.exception_type,
        failure_category=issue.failure_category,
        retry_count=issue.retry_count,
    )


def _extract_execution_time_ms(metadata: dict[str, Any]) -> float:
    direct = metadata.get("execution_time_ms")
    if isinstance(direct, (int, float)):
        return float(direct)

    statistics = metadata.get("statistics")
    if isinstance(statistics, dict):
        nested = statistics.get("execution_time_ms")
        if isinstance(nested, (int, float)):
            return float(nested)
    return 0.0
