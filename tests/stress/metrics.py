"""Metrics collection for stress testing.

Provides detailed metrics collection and analysis for stress tests.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import logging

logger = logging.getLogger(__name__)


@dataclass
class LatencyBucket:
    """Histogram bucket for latency tracking.

    Attributes:
        le: Less than or equal bound.
        count: Number of samples in this bucket.
    """

    le: float  # Less than or equal
    count: int = 0


class LatencyHistogram:
    """Histogram for tracking latency distribution.

    Provides percentile calculations and histogram buckets
    compatible with Prometheus histogram format.

    Example:
        >>> histogram = LatencyHistogram()
        >>> histogram.observe(10.5)
        >>> histogram.observe(25.3)
        >>> print(f"P99: {histogram.percentile(99):.2f}ms")
    """

    # Default bucket boundaries in milliseconds
    DEFAULT_BUCKETS = [
        1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500,
        750, 1000, 1500, 2000, 3000, 5000, 10000, float("inf"),
    ]

    def __init__(
        self,
        buckets: list[float] | None = None,
        name: str = "latency",
    ) -> None:
        """Initialize latency histogram.

        Args:
            buckets: Bucket boundaries (uses defaults if None).
            name: Histogram name.
        """
        self._name = name
        self._bucket_bounds = sorted(buckets or self.DEFAULT_BUCKETS)
        self._buckets = [LatencyBucket(le=b) for b in self._bucket_bounds]
        self._samples: list[float] = []
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get histogram name."""
        return self._name

    def observe(self, value: float) -> None:
        """Record a latency observation.

        Args:
            value: Latency value in milliseconds.
        """
        with self._lock:
            self._samples.append(value)
            self._sum += value
            self._count += 1

            # Update buckets
            for bucket in self._buckets:
                if value <= bucket.le:
                    bucket.count += 1

    def percentile(self, p: float) -> float:
        """Calculate a percentile.

        Args:
            p: Percentile (0-100).

        Returns:
            Latency at the given percentile.
        """
        with self._lock:
            if not self._samples:
                return 0.0

            sorted_samples = sorted(self._samples)
            index = int((p / 100) * (len(sorted_samples) - 1))
            return sorted_samples[index]

    @property
    def p50(self) -> float:
        """Get P50 (median) latency."""
        return self.percentile(50)

    @property
    def p75(self) -> float:
        """Get P75 latency."""
        return self.percentile(75)

    @property
    def p90(self) -> float:
        """Get P90 latency."""
        return self.percentile(90)

    @property
    def p95(self) -> float:
        """Get P95 latency."""
        return self.percentile(95)

    @property
    def p99(self) -> float:
        """Get P99 latency."""
        return self.percentile(99)

    @property
    def p999(self) -> float:
        """Get P99.9 latency."""
        return self.percentile(99.9)

    @property
    def mean(self) -> float:
        """Get mean latency."""
        with self._lock:
            if self._count == 0:
                return 0.0
            return self._sum / self._count

    @property
    def min(self) -> float:
        """Get minimum latency."""
        with self._lock:
            return min(self._samples) if self._samples else 0.0

    @property
    def max(self) -> float:
        """Get maximum latency."""
        with self._lock:
            return max(self._samples) if self._samples else 0.0

    @property
    def stddev(self) -> float:
        """Get standard deviation."""
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            mean = self._sum / self._count
            variance = sum((x - mean) ** 2 for x in self._samples) / len(self._samples)
            return math.sqrt(variance)

    @property
    def count(self) -> int:
        """Get total count."""
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        """Get sum of all values."""
        with self._lock:
            return self._sum

    def get_buckets(self) -> list[dict[str, Any]]:
        """Get histogram buckets.

        Returns:
            List of bucket dictionaries.
        """
        with self._lock:
            return [{"le": b.le, "count": b.count} for b in self._buckets]

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self._samples.clear()
            self._sum = 0.0
            self._count = 0
            for bucket in self._buckets:
                bucket.count = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self._name,
            "count": self.count,
            "sum": self.sum,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "stddev": self.stddev,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "p999": self.p999,
            "buckets": self.get_buckets(),
        }


class ThroughputTracker:
    """Track throughput over time.

    Provides real-time and historical throughput metrics.

    Example:
        >>> tracker = ThroughputTracker(window_seconds=60)
        >>> tracker.record_operation()
        >>> tracker.record_operation()
        >>> print(f"Throughput: {tracker.current_rate:.2f} ops/sec")
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        bucket_size_seconds: float = 1.0,
        name: str = "throughput",
    ) -> None:
        """Initialize throughput tracker.

        Args:
            window_seconds: Sliding window size.
            bucket_size_seconds: Size of each bucket.
            name: Tracker name.
        """
        self._name = name
        self._window_seconds = window_seconds
        self._bucket_size = bucket_size_seconds
        self._num_buckets = int(window_seconds / bucket_size_seconds)

        self._buckets: deque[int] = deque(maxlen=self._num_buckets)
        self._current_bucket = 0
        self._current_bucket_time = time.monotonic()
        self._total_count = 0
        self._lock = threading.Lock()

        # Initialize buckets
        for _ in range(self._num_buckets):
            self._buckets.append(0)

    @property
    def name(self) -> str:
        """Get tracker name."""
        return self._name

    def record_operation(self, count: int = 1) -> None:
        """Record one or more operations.

        Args:
            count: Number of operations to record.
        """
        with self._lock:
            self._maybe_rotate_bucket()
            self._current_bucket += count
            self._total_count += count

    def _maybe_rotate_bucket(self) -> None:
        """Rotate to a new bucket if needed."""
        now = time.monotonic()
        elapsed = now - self._current_bucket_time

        if elapsed >= self._bucket_size:
            # Calculate how many buckets to advance
            buckets_to_advance = int(elapsed / self._bucket_size)

            # Add completed buckets
            self._buckets.append(self._current_bucket)
            self._current_bucket = 0

            # Add empty buckets for any skipped time
            for _ in range(min(buckets_to_advance - 1, self._num_buckets)):
                self._buckets.append(0)

            self._current_bucket_time = now

    @property
    def current_rate(self) -> float:
        """Get current throughput rate (ops/sec)."""
        with self._lock:
            self._maybe_rotate_bucket()

            # Sum recent buckets
            total = sum(self._buckets) + self._current_bucket
            return total / self._window_seconds

    @property
    def total_count(self) -> int:
        """Get total operation count."""
        with self._lock:
            return self._total_count

    def get_history(self) -> list[int]:
        """Get bucket history.

        Returns:
            List of operation counts per bucket.
        """
        with self._lock:
            return list(self._buckets) + [self._current_bucket]

    def reset(self) -> None:
        """Reset tracker."""
        with self._lock:
            self._buckets.clear()
            for _ in range(self._num_buckets):
                self._buckets.append(0)
            self._current_bucket = 0
            self._current_bucket_time = time.monotonic()
            self._total_count = 0


@dataclass
class MetricSample:
    """A timestamped metric sample."""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)


class StressMetricsCollector:
    """Comprehensive metrics collector for stress tests.

    Aggregates latency, throughput, success/failure rates,
    and custom metrics.

    Example:
        >>> collector = StressMetricsCollector()
        >>> collector.start()
        >>>
        >>> # Record metrics
        >>> collector.record_operation(success=True, latency_ms=15.5)
        >>>
        >>> # Get summary
        >>> summary = collector.get_summary()
    """

    def __init__(
        self,
        name: str = "stress_test",
        latency_buckets: list[float] | None = None,
    ) -> None:
        """Initialize metrics collector.

        Args:
            name: Collector name.
            latency_buckets: Custom latency histogram buckets.
        """
        self._name = name
        self._started = False
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None

        # Core metrics
        self._latency_histogram = LatencyHistogram(
            buckets=latency_buckets,
            name=f"{name}_latency",
        )
        self._throughput_tracker = ThroughputTracker(name=f"{name}_throughput")

        # Counters
        self._success_count = 0
        self._failure_count = 0
        self._timeout_count = 0
        self._error_count = 0

        # Error tracking
        self._errors: list[str] = []
        self._max_errors = 1000

        # Time series for trending
        self._samples: list[MetricSample] = []
        self._sample_interval_seconds = 1.0
        self._last_sample_time: float = 0.0

        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get collector name."""
        return self._name

    def start(self) -> None:
        """Start metrics collection."""
        with self._lock:
            self._started = True
            self._start_time = datetime.now()
            self._last_sample_time = time.monotonic()

    def stop(self) -> None:
        """Stop metrics collection."""
        with self._lock:
            self._started = False
            self._end_time = datetime.now()

    def record_operation(
        self,
        success: bool,
        latency_ms: float,
        error: str | None = None,
        is_timeout: bool = False,
    ) -> None:
        """Record an operation result.

        Args:
            success: Whether operation succeeded.
            latency_ms: Operation latency in milliseconds.
            error: Error message if failed.
            is_timeout: Whether failure was a timeout.
        """
        with self._lock:
            # Record latency
            self._latency_histogram.observe(latency_ms)

            # Record throughput
            self._throughput_tracker.record_operation()

            # Update counters
            if success:
                self._success_count += 1
            else:
                self._failure_count += 1
                if is_timeout:
                    self._timeout_count += 1
                if error:
                    self._error_count += 1
                    if len(self._errors) < self._max_errors:
                        self._errors.append(error)

            # Maybe take a sample
            self._maybe_sample()

    def _maybe_sample(self) -> None:
        """Take a time series sample if interval has passed."""
        now = time.monotonic()
        if now - self._last_sample_time >= self._sample_interval_seconds:
            self._samples.append(MetricSample(
                timestamp=datetime.now(),
                value=self._throughput_tracker.current_rate,
                labels={"metric": "throughput"},
            ))
            self._last_sample_time = now

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0-1.0)."""
        with self._lock:
            total = self._success_count + self._failure_count
            if total == 0:
                return 1.0
            return self._success_count / total

    @property
    def failure_rate(self) -> float:
        """Get failure rate (0.0-1.0)."""
        return 1.0 - self.success_rate

    @property
    def total_operations(self) -> int:
        """Get total operation count."""
        with self._lock:
            return self._success_count + self._failure_count

    @property
    def duration_seconds(self) -> float:
        """Get test duration in seconds."""
        with self._lock:
            if self._start_time is None:
                return 0.0
            end = self._end_time or datetime.now()
            return (end - self._start_time).total_seconds()

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary.

        Returns:
            Summary dictionary.
        """
        with self._lock:
            return {
                "name": self._name,
                "started": self._started,
                "start_time": self._start_time.isoformat() if self._start_time else None,
                "end_time": self._end_time.isoformat() if self._end_time else None,
                "duration_seconds": self.duration_seconds,
                "operations": {
                    "total": self.total_operations,
                    "successful": self._success_count,
                    "failed": self._failure_count,
                    "timeouts": self._timeout_count,
                    "errors": self._error_count,
                },
                "rates": {
                    "success_rate": self.success_rate,
                    "failure_rate": self.failure_rate,
                    "throughput_per_second": self._throughput_tracker.current_rate,
                },
                "latency": self._latency_histogram.to_dict(),
                "recent_errors": self._errors[-10:],
            }

    def get_time_series(self) -> list[dict[str, Any]]:
        """Get time series samples.

        Returns:
            List of sample dictionaries.
        """
        with self._lock:
            return [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "value": s.value,
                    "labels": s.labels,
                }
                for s in self._samples
            ]

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latency_histogram.reset()
            self._throughput_tracker.reset()
            self._success_count = 0
            self._failure_count = 0
            self._timeout_count = 0
            self._error_count = 0
            self._errors.clear()
            self._samples.clear()
            self._start_time = None
            self._end_time = None
            self._started = False
