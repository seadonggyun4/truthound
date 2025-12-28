"""Core protocols and data classes for Historical Trend Analysis.

This module defines the foundational abstractions for the analytics system,
following Protocol-First design for loose coupling and extensibility.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)


# =============================================================================
# Exceptions
# =============================================================================


class AnalyticsError(Exception):
    """Base exception for analytics errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class StoreError(AnalyticsError):
    """Raised when time series store operations fail."""

    def __init__(
        self,
        message: str,
        store_name: str,
        operation: str | None = None,
    ) -> None:
        super().__init__(message, {"store": store_name, "operation": operation})
        self.store_name = store_name
        self.operation = operation


class AnalyzerError(AnalyticsError):
    """Raised when analysis operations fail."""

    def __init__(
        self,
        message: str,
        analyzer_name: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(message, {"analyzer": analyzer_name, "reason": reason})
        self.analyzer_name = analyzer_name
        self.reason = reason


# =============================================================================
# Enums
# =============================================================================


class TimeGranularity(str, Enum):
    """Time granularity for aggregation."""

    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"

    def __str__(self) -> str:
        return self.value

    def to_timedelta(self) -> timedelta:
        """Convert to approximate timedelta."""
        mapping = {
            TimeGranularity.SECOND: timedelta(seconds=1),
            TimeGranularity.MINUTE: timedelta(minutes=1),
            TimeGranularity.HOUR: timedelta(hours=1),
            TimeGranularity.DAY: timedelta(days=1),
            TimeGranularity.WEEK: timedelta(weeks=1),
            TimeGranularity.MONTH: timedelta(days=30),
            TimeGranularity.QUARTER: timedelta(days=90),
            TimeGranularity.YEAR: timedelta(days=365),
        }
        return mapping.get(self, timedelta(days=1))


class TrendDirection(str, Enum):
    """Direction of a trend."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"

    def __str__(self) -> str:
        return self.value


class AnomalyType(str, Enum):
    """Type of anomaly detected."""

    SPIKE = "spike"
    DROP = "drop"
    SHIFT = "shift"
    OUTLIER = "outlier"
    MISSING = "missing"

    def __str__(self) -> str:
        return self.value


class AggregationFunction(str, Enum):
    """Aggregation functions for time series data."""

    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    STDDEV = "stddev"
    PERCENTILE_50 = "p50"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class TimeSeriesPoint:
    """A single point in a time series.

    Attributes:
        timestamp: Point timestamp.
        value: Numeric value.
        labels: Optional labels for the point.
        metadata: Optional metadata.
    """

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeSeriesPoint":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            value=data["value"],
            labels=data.get("labels", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TrendResult:
    """Result of trend analysis.

    Attributes:
        direction: Trend direction.
        slope: Rate of change (per time unit).
        confidence: Confidence in the trend (0.0 to 1.0).
        change_percent: Percentage change over period.
        period_start: Analysis period start.
        period_end: Analysis period end.
        r_squared: R-squared value of linear fit.
        sample_count: Number of samples analyzed.
    """

    direction: TrendDirection
    slope: float
    confidence: float
    change_percent: float
    period_start: datetime
    period_end: datetime
    r_squared: float = 0.0
    sample_count: int = 0

    @property
    def is_significant(self) -> bool:
        """Check if trend is statistically significant."""
        return self.confidence > 0.8 and self.r_squared > 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": self.direction.value,
            "slope": self.slope,
            "confidence": self.confidence,
            "change_percent": self.change_percent,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "r_squared": self.r_squared,
            "sample_count": self.sample_count,
            "is_significant": self.is_significant,
        }


@dataclass
class AnomalyResult:
    """Result of anomaly detection.

    Attributes:
        timestamp: When the anomaly occurred.
        anomaly_type: Type of anomaly.
        value: Observed value.
        expected_value: Expected value.
        deviation: Deviation from expected.
        severity: Anomaly severity (0.0 to 1.0).
        labels: Associated labels.
    """

    timestamp: datetime
    anomaly_type: AnomalyType
    value: float
    expected_value: float
    deviation: float
    severity: float = 0.5
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def deviation_percent(self) -> float:
        """Get deviation as percentage."""
        if self.expected_value == 0:
            return float("inf") if self.value != 0 else 0
        return abs(self.value - self.expected_value) / abs(self.expected_value) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type.value,
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "deviation_percent": self.deviation_percent,
            "severity": self.severity,
            "labels": self.labels,
        }


@dataclass
class ForecastResult:
    """Result of time series forecasting.

    Attributes:
        predictions: List of predicted points.
        confidence_lower: Lower confidence bound.
        confidence_upper: Upper confidence bound.
        method: Forecasting method used.
        horizon: Forecast horizon.
    """

    predictions: list[TimeSeriesPoint]
    confidence_lower: list[float]
    confidence_upper: list[float]
    method: str
    horizon: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "method": self.method,
            "horizon": self.horizon,
        }


@dataclass
class AnalysisSummary:
    """Summary of checkpoint execution analysis.

    Attributes:
        checkpoint_name: Checkpoint being analyzed.
        period: Analysis time period.
        total_runs: Total number of runs.
        success_rate: Success rate (0.0 to 1.0).
        avg_duration_ms: Average execution time.
        p50_duration_ms: 50th percentile duration.
        p95_duration_ms: 95th percentile duration.
        p99_duration_ms: 99th percentile duration.
        trend: Trend analysis result.
        anomalies: Detected anomalies.
    """

    checkpoint_name: str
    period: timedelta
    total_runs: int
    success_rate: float
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    trend: TrendResult | None = None
    anomalies: list[AnomalyResult] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Get failure rate."""
        return 1.0 - self.success_rate

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_name": self.checkpoint_name,
            "period_seconds": self.period.total_seconds(),
            "total_runs": self.total_runs,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "p50_duration_ms": self.p50_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "p99_duration_ms": self.p99_duration_ms,
            "trend": self.trend.to_dict() if self.trend else None,
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


# =============================================================================
# Protocols
# =============================================================================


T = TypeVar("T")


@runtime_checkable
class TimeSeriesStoreProtocol(Protocol):
    """Protocol for time series data storage.

    Stores support writing individual points and batches,
    querying time ranges, and aggregating data.
    """

    @property
    def name(self) -> str:
        """Get store name."""
        ...

    async def connect(self) -> None:
        """Connect to the storage backend."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        ...

    async def write(self, metric_name: str, point: TimeSeriesPoint) -> None:
        """Write a single point.

        Args:
            metric_name: Name of the metric.
            point: Point to write.

        Raises:
            StoreError: If write fails.
        """
        ...

    async def write_batch(self, metric_name: str, points: list[TimeSeriesPoint]) -> None:
        """Write multiple points.

        Args:
            metric_name: Name of the metric.
            points: Points to write.

        Raises:
            StoreError: If write fails.
        """
        ...

    async def query(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        granularity: TimeGranularity | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Query points in a time range.

        Args:
            metric_name: Name of the metric.
            start: Start time (inclusive).
            end: End time (inclusive).
            granularity: Optional aggregation granularity.
            labels: Optional label filter.

        Returns:
            List of points in the range.

        Raises:
            StoreError: If query fails.
        """
        ...

    async def aggregate(
        self,
        metric_name: str,
        start: datetime,
        end: datetime,
        aggregation: str,
        granularity: TimeGranularity,
        labels: dict[str, str] | None = None,
    ) -> list[TimeSeriesPoint]:
        """Aggregate points over time buckets.

        Args:
            metric_name: Name of the metric.
            start: Start time.
            end: End time.
            aggregation: Aggregation function ("sum", "avg", "min", "max", "count").
            granularity: Time bucket granularity.
            labels: Optional label filter.

        Returns:
            List of aggregated points.

        Raises:
            StoreError: If aggregation fails.
        """
        ...

    async def delete(
        self,
        metric_name: str,
        before: datetime | None = None,
        labels: dict[str, str] | None = None,
    ) -> int:
        """Delete points.

        Args:
            metric_name: Name of the metric.
            before: Delete points before this time.
            labels: Optional label filter.

        Returns:
            Number of points deleted.

        Raises:
            StoreError: If delete fails.
        """
        ...


@runtime_checkable
class TrendAnalyzerProtocol(Protocol):
    """Protocol for trend analysis.

    Analyzers process time series data to detect trends,
    anomalies, and make forecasts.
    """

    @property
    def name(self) -> str:
        """Get analyzer name."""
        ...

    def analyze(self, points: list[TimeSeriesPoint]) -> TrendResult:
        """Analyze trend in time series.

        Args:
            points: Time series points.

        Returns:
            Trend analysis result.

        Raises:
            AnalyzerError: If analysis fails.
        """
        ...

    def detect_anomalies(
        self,
        points: list[TimeSeriesPoint],
        threshold: float = 2.0,
    ) -> list[AnomalyResult]:
        """Detect anomalies in time series.

        Args:
            points: Time series points.
            threshold: Detection threshold.

        Returns:
            List of detected anomalies.

        Raises:
            AnalyzerError: If detection fails.
        """
        ...

    def forecast(
        self,
        points: list[TimeSeriesPoint],
        periods: int,
    ) -> ForecastResult:
        """Forecast future values.

        Args:
            points: Historical time series points.
            periods: Number of periods to forecast.

        Returns:
            Forecast result.

        Raises:
            AnalyzerError: If forecasting fails.
        """
        ...


# =============================================================================
# Type Aliases
# =============================================================================

PointT = TypeVar("PointT", bound=TimeSeriesPoint)
StoreT = TypeVar("StoreT", bound=TimeSeriesStoreProtocol)
AnalyzerT = TypeVar("AnalyzerT", bound=TrendAnalyzerProtocol)
