"""Domain models for checkpoint analytics.

This module defines domain-specific models for checkpoint execution
metrics and analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CheckpointExecution:
    """Record of a single checkpoint execution.

    Attributes:
        checkpoint_name: Name of the checkpoint.
        run_id: Unique run identifier.
        started_at: When execution started.
        completed_at: When execution completed.
        success: Whether execution succeeded.
        duration_ms: Execution duration in milliseconds.
        issue_count: Number of validation issues found.
        critical_count: Number of critical issues.
        high_count: Number of high severity issues.
        medium_count: Number of medium severity issues.
        low_count: Number of low severity issues.
        data_asset: Name of the data asset validated.
        row_count: Number of rows processed.
        error: Error message if failed.
        tags: Execution tags.
        metadata: Additional metadata.
    """

    checkpoint_name: str
    run_id: str
    started_at: datetime
    completed_at: datetime
    success: bool
    duration_ms: float
    issue_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    data_asset: str = ""
    row_count: int = 0
    error: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """Get execution status."""
        if not self.success:
            return "failed"
        if self.critical_count > 0:
            return "critical"
        if self.high_count > 0:
            return "warning"
        return "success"

    @property
    def has_issues(self) -> bool:
        """Check if execution found any issues."""
        return self.issue_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_name": self.checkpoint_name,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "success": self.success,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "issue_count": self.issue_count,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "data_asset": self.data_asset,
            "row_count": self.row_count,
            "error": self.error,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointExecution":
        """Create from dictionary."""
        return cls(
            checkpoint_name=data["checkpoint_name"],
            run_id=data["run_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            success=data["success"],
            duration_ms=data["duration_ms"],
            issue_count=data.get("issue_count", 0),
            critical_count=data.get("critical_count", 0),
            high_count=data.get("high_count", 0),
            medium_count=data.get("medium_count", 0),
            low_count=data.get("low_count", 0),
            data_asset=data.get("data_asset", ""),
            row_count=data.get("row_count", 0),
            error=data.get("error"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionMetrics:
    """Aggregated execution metrics for a checkpoint.

    Attributes:
        checkpoint_name: Name of the checkpoint.
        period_start: Start of the analysis period.
        period_end: End of the analysis period.
        total_runs: Total number of runs.
        successful_runs: Number of successful runs.
        failed_runs: Number of failed runs.
        total_issues: Total issues found across all runs.
        total_critical: Total critical issues.
        total_high: Total high severity issues.
        avg_duration_ms: Average duration.
        min_duration_ms: Minimum duration.
        max_duration_ms: Maximum duration.
        p50_duration_ms: 50th percentile duration.
        p95_duration_ms: 95th percentile duration.
        p99_duration_ms: 99th percentile duration.
        total_rows_processed: Total rows processed.
    """

    checkpoint_name: str
    period_start: datetime
    period_end: datetime
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_issues: int = 0
    total_critical: int = 0
    total_high: int = 0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    p50_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    p99_duration_ms: float = 0.0
    total_rows_processed: int = 0

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total_runs == 0:
            return 1.0
        return self.successful_runs / self.total_runs

    @property
    def failure_rate(self) -> float:
        """Get failure rate."""
        return 1.0 - self.success_rate

    @property
    def avg_issues_per_run(self) -> float:
        """Get average issues per run."""
        if self.total_runs == 0:
            return 0.0
        return self.total_issues / self.total_runs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_name": self.checkpoint_name,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "total_issues": self.total_issues,
            "total_critical": self.total_critical,
            "total_high": self.total_high,
            "avg_issues_per_run": self.avg_issues_per_run,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "p50_duration_ms": self.p50_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "total_rows_processed": self.total_rows_processed,
        }


@dataclass
class SuccessRateMetrics:
    """Success rate metrics over time.

    Attributes:
        checkpoint_name: Name of the checkpoint.
        timestamp: Timestamp of the data point.
        window_runs: Runs in the window.
        window_successes: Successes in the window.
        success_rate: Success rate for the window.
        rolling_avg: Rolling average success rate.
    """

    checkpoint_name: str
    timestamp: datetime
    window_runs: int
    window_successes: int
    success_rate: float
    rolling_avg: float = 0.0

    @property
    def window_failures(self) -> int:
        """Get failures in the window."""
        return self.window_runs - self.window_successes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_name": self.checkpoint_name,
            "timestamp": self.timestamp.isoformat(),
            "window_runs": self.window_runs,
            "window_successes": self.window_successes,
            "window_failures": self.window_failures,
            "success_rate": self.success_rate,
            "rolling_avg": self.rolling_avg,
        }


@dataclass
class DurationMetrics:
    """Duration metrics over time.

    Attributes:
        checkpoint_name: Name of the checkpoint.
        timestamp: Timestamp of the data point.
        sample_count: Number of samples.
        avg_ms: Average duration.
        min_ms: Minimum duration.
        max_ms: Maximum duration.
        p50_ms: 50th percentile.
        p95_ms: 95th percentile.
        p99_ms: 99th percentile.
        std_dev_ms: Standard deviation.
    """

    checkpoint_name: str
    timestamp: datetime
    sample_count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float = 0.0

    @property
    def range_ms(self) -> float:
        """Get duration range."""
        return self.max_ms - self.min_ms

    @property
    def coefficient_of_variation(self) -> float:
        """Get coefficient of variation."""
        if self.avg_ms == 0:
            return 0.0
        return self.std_dev_ms / self.avg_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_name": self.checkpoint_name,
            "timestamp": self.timestamp.isoformat(),
            "sample_count": self.sample_count,
            "avg_ms": self.avg_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "range_ms": self.range_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "std_dev_ms": self.std_dev_ms,
            "coefficient_of_variation": self.coefficient_of_variation,
        }
