"""Failure detection strategies for Circuit Breaker.

This module provides different strategies for determining when
a circuit should trip (open):

- ConsecutiveFailureDetector: Trip after N consecutive failures
- PercentageFailureDetector: Trip when failure rate exceeds threshold
- TimeWindowFailureDetector: Trip based on failures within time window
- CompositeFailureDetector: Combine multiple strategies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Protocol, runtime_checkable

from truthound.checkpoint.circuitbreaker.core import (
    CircuitBreakerConfig,
    FailureDetectionStrategy,
)


@runtime_checkable
class FailureDetector(Protocol):
    """Protocol for failure detection strategies."""

    def record_success(self) -> None:
        """Record a successful call."""
        ...

    def record_failure(self, exception: Exception | None = None) -> None:
        """Record a failed call."""
        ...

    def should_trip(self) -> bool:
        """Determine if circuit should trip (open)."""
        ...

    def reset(self) -> None:
        """Reset detector state."""
        ...

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        ...

    @property
    def success_count(self) -> int:
        """Current success count."""
        ...


class BaseFailureDetector(ABC):
    """Base class for failure detectors with common functionality."""

    def __init__(self):
        self._lock = RLock()
        self._failures = 0
        self._successes = 0

    @abstractmethod
    def record_success(self) -> None:
        """Record a successful call."""
        pass

    @abstractmethod
    def record_failure(self, exception: Exception | None = None) -> None:
        """Record a failed call."""
        pass

    @abstractmethod
    def should_trip(self) -> bool:
        """Determine if circuit should trip."""
        pass

    def reset(self) -> None:
        """Reset detector state."""
        with self._lock:
            self._failures = 0
            self._successes = 0

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        return self._failures

    @property
    def success_count(self) -> int:
        """Current success count."""
        return self._successes


class ConsecutiveFailureDetector(BaseFailureDetector):
    """Trips after N consecutive failures.

    This is the simplest detection strategy. The circuit trips when
    a specified number of consecutive failures occur. A single success
    resets the counter.

    Example:
        >>> detector = ConsecutiveFailureDetector(threshold=3)
        >>> detector.record_failure()  # count: 1
        >>> detector.record_failure()  # count: 2
        >>> detector.should_trip()      # False
        >>> detector.record_failure()  # count: 3
        >>> detector.should_trip()      # True
    """

    def __init__(self, threshold: int = 5):
        """Initialize detector.

        Args:
            threshold: Number of consecutive failures to trip
        """
        super().__init__()
        self._threshold = threshold
        self._consecutive_failures = 0

    def record_success(self) -> None:
        """Record success and reset consecutive failures."""
        with self._lock:
            self._successes += 1
            self._consecutive_failures = 0

    def record_failure(self, exception: Exception | None = None) -> None:
        """Record failure and increment consecutive counter."""
        with self._lock:
            self._failures += 1
            self._consecutive_failures += 1

    def should_trip(self) -> bool:
        """Trip if consecutive failures >= threshold."""
        with self._lock:
            return self._consecutive_failures >= self._threshold

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            super().reset()
            self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        """Current consecutive failure count."""
        return self._consecutive_failures


class PercentageFailureDetector(BaseFailureDetector):
    """Trips when failure percentage exceeds threshold.

    This strategy considers the overall failure rate rather than
    consecutive failures. Useful for services with occasional
    transient errors.

    Example:
        >>> detector = PercentageFailureDetector(
        ...     threshold=0.5,  # 50%
        ...     min_calls=10,
        ... )
        >>> for _ in range(6):
        ...     detector.record_failure()
        >>> for _ in range(4):
        ...     detector.record_success()
        >>> detector.should_trip()  # True (60% failure rate)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_calls: int = 10,
        window_size: int | None = None,
    ):
        """Initialize detector.

        Args:
            threshold: Failure rate (0.0-1.0) to trip
            min_calls: Minimum calls before percentage applies
            window_size: If set, only consider last N calls
        """
        super().__init__()
        self._threshold = threshold
        self._min_calls = min_calls
        self._window_size = window_size
        self._calls: deque[bool] = deque(maxlen=window_size)

    def record_success(self) -> None:
        """Record success."""
        with self._lock:
            self._successes += 1
            self._calls.append(True)

    def record_failure(self, exception: Exception | None = None) -> None:
        """Record failure."""
        with self._lock:
            self._failures += 1
            self._calls.append(False)

    def should_trip(self) -> bool:
        """Trip if failure rate >= threshold and min calls met."""
        with self._lock:
            total = len(self._calls) if self._window_size else (self._successes + self._failures)
            if total < self._min_calls:
                return False

            if self._window_size:
                failures = sum(1 for success in self._calls if not success)
                rate = failures / total
            else:
                rate = self._failures / total if total > 0 else 0.0

            return rate >= self._threshold

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            super().reset()
            self._calls.clear()

    @property
    def failure_rate(self) -> float:
        """Current failure rate."""
        with self._lock:
            if self._window_size:
                total = len(self._calls)
                if total == 0:
                    return 0.0
                failures = sum(1 for success in self._calls if not success)
                return failures / total
            else:
                total = self._successes + self._failures
                return self._failures / total if total > 0 else 0.0


@dataclass
class TimestampedCall:
    """A call with timestamp for time-window tracking."""

    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    exception_type: str | None = None


class TimeWindowFailureDetector(BaseFailureDetector):
    """Trips based on failures within a sliding time window.

    This strategy only considers recent calls, allowing the circuit
    to recover naturally as old failures age out of the window.

    Example:
        >>> detector = TimeWindowFailureDetector(
        ...     threshold=5,
        ...     window_seconds=60.0,
        ... )
        >>> # 5 failures within 60 seconds will trip
    """

    def __init__(
        self,
        threshold: int = 5,
        window_seconds: float = 60.0,
        use_percentage: bool = False,
        percentage_threshold: float = 0.5,
        min_calls: int = 10,
    ):
        """Initialize detector.

        Args:
            threshold: Number of failures to trip (absolute mode)
            window_seconds: Time window size in seconds
            use_percentage: Use percentage mode instead of absolute
            percentage_threshold: Failure rate to trip (percentage mode)
            min_calls: Minimum calls for percentage mode
        """
        super().__init__()
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._use_percentage = use_percentage
        self._percentage_threshold = percentage_threshold
        self._min_calls = min_calls
        self._calls: list[TimestampedCall] = []

    def _cleanup_old_calls(self) -> None:
        """Remove calls outside the time window."""
        cutoff = datetime.now() - timedelta(seconds=self._window_seconds)
        self._calls = [c for c in self._calls if c.timestamp >= cutoff]

    def record_success(self) -> None:
        """Record success."""
        with self._lock:
            self._successes += 1
            self._calls.append(TimestampedCall(success=True))
            self._cleanup_old_calls()

    def record_failure(self, exception: Exception | None = None) -> None:
        """Record failure."""
        with self._lock:
            self._failures += 1
            exc_type = type(exception).__name__ if exception else None
            self._calls.append(TimestampedCall(success=False, exception_type=exc_type))
            self._cleanup_old_calls()

    def should_trip(self) -> bool:
        """Trip based on failures in time window."""
        with self._lock:
            self._cleanup_old_calls()

            failures_in_window = sum(1 for c in self._calls if not c.success)
            total_in_window = len(self._calls)

            if self._use_percentage:
                if total_in_window < self._min_calls:
                    return False
                rate = failures_in_window / total_in_window
                return rate >= self._percentage_threshold
            else:
                return failures_in_window >= self._threshold

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            super().reset()
            self._calls.clear()

    @property
    def calls_in_window(self) -> int:
        """Number of calls in current window."""
        with self._lock:
            self._cleanup_old_calls()
            return len(self._calls)

    @property
    def failures_in_window(self) -> int:
        """Number of failures in current window."""
        with self._lock:
            self._cleanup_old_calls()
            return sum(1 for c in self._calls if not c.success)


class CompositeFailureDetector(BaseFailureDetector):
    """Combines multiple detection strategies.

    This detector can use either AND or OR logic to combine
    multiple detection strategies.

    Example:
        >>> detector = CompositeFailureDetector(
        ...     detectors=[
        ...         ConsecutiveFailureDetector(threshold=3),
        ...         PercentageFailureDetector(threshold=0.5),
        ...     ],
        ...     require_all=False,  # OR logic
        ... )
    """

    def __init__(
        self,
        detectors: list[FailureDetector],
        require_all: bool = False,
    ):
        """Initialize composite detector.

        Args:
            detectors: List of detectors to combine
            require_all: If True, all must trip (AND). If False, any trips (OR)
        """
        super().__init__()
        self._detectors = detectors
        self._require_all = require_all

    def record_success(self) -> None:
        """Record success on all detectors."""
        with self._lock:
            self._successes += 1
            for detector in self._detectors:
                detector.record_success()

    def record_failure(self, exception: Exception | None = None) -> None:
        """Record failure on all detectors."""
        with self._lock:
            self._failures += 1
            for detector in self._detectors:
                detector.record_failure(exception)

    def should_trip(self) -> bool:
        """Trip based on combined detector results."""
        with self._lock:
            if self._require_all:
                return all(d.should_trip() for d in self._detectors)
            else:
                return any(d.should_trip() for d in self._detectors)

    def reset(self) -> None:
        """Reset all detectors."""
        with self._lock:
            super().reset()
            for detector in self._detectors:
                detector.reset()

    @property
    def detector_states(self) -> list[dict]:
        """Get state of all detectors."""
        return [
            {
                "type": type(d).__name__,
                "should_trip": d.should_trip(),
                "failure_count": d.failure_count,
                "success_count": d.success_count,
            }
            for d in self._detectors
        ]


def create_detector(config: CircuitBreakerConfig) -> FailureDetector:
    """Factory function to create appropriate detector from config.

    Args:
        config: Circuit breaker configuration

    Returns:
        Appropriate failure detector based on config.detection_strategy
    """
    strategy = config.detection_strategy

    if strategy == FailureDetectionStrategy.CONSECUTIVE:
        return ConsecutiveFailureDetector(threshold=config.failure_threshold)

    elif strategy == FailureDetectionStrategy.PERCENTAGE:
        return PercentageFailureDetector(
            threshold=config.failure_rate_threshold,
            min_calls=config.min_calls_in_window,
        )

    elif strategy == FailureDetectionStrategy.TIME_WINDOW:
        return TimeWindowFailureDetector(
            threshold=config.failure_threshold,
            window_seconds=config.time_window_seconds,
        )

    elif strategy == FailureDetectionStrategy.COMPOSITE:
        # Create a composite with both consecutive and percentage
        return CompositeFailureDetector(
            detectors=[
                ConsecutiveFailureDetector(threshold=config.failure_threshold),
                PercentageFailureDetector(
                    threshold=config.failure_rate_threshold,
                    min_calls=config.min_calls_in_window,
                ),
            ],
            require_all=False,
        )

    else:
        # Default to consecutive
        return ConsecutiveFailureDetector(threshold=config.failure_threshold)
