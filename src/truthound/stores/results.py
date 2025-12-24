"""Validation result types for storage.

This module defines the data structures used to represent validation results
that can be persisted and queried through the store system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from truthound.report import Report
    from truthound.validators.base import ValidationIssue


class ResultStatus(str, Enum):
    """Status of a validation run."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    WARNING = "warning"

    def __str__(self) -> str:
        return self.value


@dataclass
class ValidatorResult:
    """Result from a single validator.

    Attributes:
        validator_name: Name of the validator that ran.
        success: Whether the validation passed.
        column: Column that was validated (if applicable).
        issue_type: Type of issue found (if any).
        count: Number of failing rows/items.
        severity: Severity of the issue.
        message: Human-readable result message.
        details: Additional details or context.
        execution_time_ms: Time taken to run this validator.
    """

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
        """Convert to dictionary for serialization."""
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidatorResult":
        """Create from dictionary."""
        return cls(
            validator_name=data["validator_name"],
            success=data["success"],
            column=data.get("column"),
            issue_type=data.get("issue_type"),
            count=data.get("count", 0),
            severity=data.get("severity"),
            message=data.get("message"),
            details=data.get("details", {}),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )

    @classmethod
    def from_issue(cls, issue: "ValidationIssue") -> "ValidatorResult":
        """Create from a ValidationIssue."""
        return cls(
            validator_name=issue.issue_type,
            success=False,
            column=issue.column,
            issue_type=issue.issue_type,
            count=issue.count,
            severity=issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity),
            message=issue.details,
            details={
                "expected": issue.expected,
                "actual": issue.actual,
                "sample_values": issue.sample_values,
            },
        )


@dataclass
class ResultStatistics:
    """Statistics about a validation run.

    Attributes:
        total_validators: Total number of validators run.
        passed_validators: Number of validators that passed.
        failed_validators: Number of validators that failed.
        error_validators: Number of validators that errored.
        total_rows: Total rows in the dataset.
        total_columns: Total columns in the dataset.
        total_issues: Total number of issues found.
        critical_issues: Number of critical severity issues.
        high_issues: Number of high severity issues.
        medium_issues: Number of medium severity issues.
        low_issues: Number of low severity issues.
        execution_time_ms: Total execution time in milliseconds.
    """

    total_validators: int = 0
    passed_validators: int = 0
    failed_validators: int = 0
    error_validators: int = 0
    total_rows: int = 0
    total_columns: int = 0
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    execution_time_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Calculate the validator pass rate."""
        if self.total_validators == 0:
            return 1.0
        return self.passed_validators / self.total_validators

    @property
    def issue_rate(self) -> float:
        """Calculate issue rate (issues per row)."""
        if self.total_rows == 0:
            return 0.0
        return self.total_issues / self.total_rows

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResultStatistics":
        """Create from dictionary."""
        return cls(
            total_validators=data.get("total_validators", 0),
            passed_validators=data.get("passed_validators", 0),
            failed_validators=data.get("failed_validators", 0),
            error_validators=data.get("error_validators", 0),
            total_rows=data.get("total_rows", 0),
            total_columns=data.get("total_columns", 0),
            total_issues=data.get("total_issues", 0),
            critical_issues=data.get("critical_issues", 0),
            high_issues=data.get("high_issues", 0),
            medium_issues=data.get("medium_issues", 0),
            low_issues=data.get("low_issues", 0),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )


@dataclass
class ValidationResult:
    """Complete result of a validation run.

    This is the primary data structure that gets persisted to stores.
    It contains all information about a validation run including results,
    statistics, and metadata.

    Attributes:
        run_id: Unique identifier for this run.
        run_time: When the validation was run.
        data_asset: Name/path of the data being validated.
        status: Overall status of the validation.
        results: Individual validator results.
        statistics: Aggregated statistics.
        tags: Key-value tags for filtering/grouping.
        metadata: Additional metadata about the run.
        suite_name: Name of the expectation suite used (if any).
        runtime_environment: Information about the runtime environment.
    """

    run_id: str
    run_time: datetime
    data_asset: str
    status: ResultStatus
    results: list[ValidatorResult] = field(default_factory=list)
    statistics: ResultStatistics = field(default_factory=ResultStatistics)
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    suite_name: str | None = None
    runtime_environment: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Unique identifier for this result."""
        return self.run_id

    @property
    def success(self) -> bool:
        """Check if the validation was successful."""
        return self.status == ResultStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_time": self.run_time.isoformat(),
            "data_asset": self.data_asset,
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "statistics": self.statistics.to_dict(),
            "tags": self.tags,
            "metadata": self.metadata,
            "suite_name": self.suite_name,
            "runtime_environment": self.runtime_environment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            run_time=datetime.fromisoformat(data["run_time"]),
            data_asset=data["data_asset"],
            status=ResultStatus(data["status"]),
            results=[ValidatorResult.from_dict(r) for r in data.get("results", [])],
            statistics=ResultStatistics.from_dict(data.get("statistics", {})),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
            suite_name=data.get("suite_name"),
            runtime_environment=data.get("runtime_environment", {}),
        )

    @classmethod
    def from_report(
        cls,
        report: "Report",
        data_asset: str,
        run_id: str | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        execution_time_ms: float = 0.0,
    ) -> "ValidationResult":
        """Create a ValidationResult from a Report.

        Args:
            report: The Report to convert.
            data_asset: Name/path of the data being validated.
            run_id: Optional run ID (generated if not provided).
            tags: Optional tags for the result.
            metadata: Optional additional metadata.
            execution_time_ms: Time taken for validation.

        Returns:
            A new ValidationResult instance.
        """
        # Convert issues to validator results
        results: list[ValidatorResult] = []
        for issue in report.issues:
            results.append(ValidatorResult.from_issue(issue))

        # Calculate statistics
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for issue in report.issues:
            severity_str = issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
            if severity_str in severity_counts:
                severity_counts[severity_str] += 1

        statistics = ResultStatistics(
            total_validators=len(results),
            passed_validators=0,  # Will be calculated if we have pass info
            failed_validators=len(results),
            total_rows=report.row_count,
            total_columns=report.column_count,
            total_issues=len(report.issues),
            critical_issues=severity_counts["critical"],
            high_issues=severity_counts["high"],
            medium_issues=severity_counts["medium"],
            low_issues=severity_counts["low"],
            execution_time_ms=execution_time_ms,
        )

        # Determine status
        if report.has_critical:
            status = ResultStatus.FAILURE
        elif report.has_issues:
            status = ResultStatus.WARNING
        else:
            status = ResultStatus.SUCCESS

        return cls(
            run_id=run_id or cls._generate_run_id(),
            run_time=datetime.now(),
            data_asset=data_asset,
            status=status,
            results=results,
            statistics=statistics,
            tags=tags or {},
            metadata=metadata or {"source": report.source},
        )

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid4().hex[:8]
        return f"run_{timestamp}_{unique_id}"

    def get_failed_columns(self) -> set[str]:
        """Get the set of columns that had failures."""
        return {r.column for r in self.results if not r.success and r.column}

    def get_issues_by_severity(self, severity: str) -> list[ValidatorResult]:
        """Get issues filtered by severity."""
        return [r for r in self.results if not r.success and r.severity == severity]

    def summary(self) -> str:
        """Get a human-readable summary of the result."""
        lines = [
            f"Validation Result: {self.status.value.upper()}",
            f"Run ID: {self.run_id}",
            f"Data Asset: {self.data_asset}",
            f"Run Time: {self.run_time.isoformat()}",
            f"Issues: {self.statistics.total_issues} "
            f"(Critical: {self.statistics.critical_issues}, "
            f"High: {self.statistics.high_issues}, "
            f"Medium: {self.statistics.medium_issues}, "
            f"Low: {self.statistics.low_issues})",
        ]
        return "\n".join(lines)
