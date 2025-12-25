"""Core types and configuration for Circuit Breaker pattern.

This module defines:
- Circuit states (CLOSED, OPEN, HALF_OPEN)
- Configuration dataclasses
- Exception types
- Metrics and event types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class CircuitState(str, Enum):
    """Circuit breaker states.

    State transitions:
        CLOSED -> OPEN: When failure threshold is exceeded
        OPEN -> HALF_OPEN: After recovery timeout expires
        HALF_OPEN -> CLOSED: When test calls succeed
        HALF_OPEN -> OPEN: When test calls fail
    """

    CLOSED = "closed"
    """Normal operation - requests pass through."""

    OPEN = "open"
    """Circuit tripped - requests fail immediately."""

    HALF_OPEN = "half_open"
    """Testing recovery - limited requests allowed."""


class FailureDetectionStrategy(str, Enum):
    """Strategy for detecting failures."""

    CONSECUTIVE = "consecutive"
    """Trip after N consecutive failures."""

    PERCENTAGE = "percentage"
    """Trip when failure percentage exceeds threshold."""

    TIME_WINDOW = "time_window"
    """Trip based on failures within a time window."""

    COMPOSITE = "composite"
    """Combine multiple detection strategies."""


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        half_open_max_calls: Max calls allowed in half-open state
        success_threshold: Successes needed to close from half-open
        detection_strategy: Strategy for detecting failures
        failure_rate_threshold: Failure rate to trip (for percentage strategy)
        time_window_seconds: Window size for time-based detection
        min_calls_in_window: Minimum calls before percentage applies
        excluded_exceptions: Exception types that don't count as failures
        included_exceptions: Only these exceptions count as failures (if set)
        fallback: Optional fallback function when circuit is open
        on_state_change: Callback for state changes
        on_failure: Callback for failures
        on_success: Callback for successes
    """

    # Thresholds
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2

    # Detection strategy
    detection_strategy: FailureDetectionStrategy = FailureDetectionStrategy.CONSECUTIVE
    failure_rate_threshold: float = 0.5
    time_window_seconds: float = 60.0
    min_calls_in_window: int = 10

    # Exception filtering
    excluded_exceptions: tuple[type[Exception], ...] = ()
    included_exceptions: tuple[type[Exception], ...] | None = None

    # Callbacks
    fallback: Callable[..., Any] | None = None
    on_state_change: Callable[[StateChangeEvent], None] | None = None
    on_failure: Callable[[Exception, CircuitBreakerMetrics], None] | None = None
    on_success: Callable[[Any, CircuitBreakerMetrics], None] | None = None

    def should_count_exception(self, exc: Exception) -> bool:
        """Determine if an exception should count as a failure."""
        if self.excluded_exceptions and isinstance(exc, self.excluded_exceptions):
            return False
        if self.included_exceptions is not None:
            return isinstance(exc, self.included_exceptions)
        return True


class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors."""

    def __init__(
        self,
        message: str,
        breaker_name: str | None = None,
        state: CircuitState | None = None,
    ):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.state = state


class CircuitOpenError(CircuitBreakerError):
    """Raised when attempting to call through an open circuit."""

    def __init__(
        self,
        breaker_name: str,
        remaining_time: float | None = None,
    ):
        message = f"Circuit '{breaker_name}' is open"
        if remaining_time is not None:
            message += f", recovery in {remaining_time:.1f}s"
        super().__init__(message, breaker_name, CircuitState.OPEN)
        self.remaining_time = remaining_time


class CircuitHalfOpenError(CircuitBreakerError):
    """Raised when half-open circuit has reached max test calls."""

    def __init__(self, breaker_name: str, max_calls: int):
        message = f"Circuit '{breaker_name}' is half-open and at max test calls ({max_calls})"
        super().__init__(message, breaker_name, CircuitState.HALF_OPEN)
        self.max_calls = max_calls


@dataclass
class CallResult:
    """Result of a call through the circuit breaker.

    Attributes:
        success: Whether the call succeeded
        result: The return value if successful
        exception: The exception if failed
        duration_ms: Call duration in milliseconds
        timestamp: When the call was made
        from_fallback: Whether result came from fallback
    """

    success: bool
    result: Any = None
    exception: Exception | None = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    from_fallback: bool = False


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring.

    Attributes:
        name: Breaker name
        state: Current state
        total_calls: Total number of calls
        successful_calls: Number of successful calls
        failed_calls: Number of failed calls
        rejected_calls: Calls rejected due to open circuit
        consecutive_failures: Current consecutive failure count
        consecutive_successes: Current consecutive success count
        failure_rate: Current failure rate (0.0 - 1.0)
        last_failure_time: Time of last failure
        last_success_time: Time of last success
        last_state_change_time: Time of last state change
        time_in_current_state_ms: Time spent in current state
        average_response_time_ms: Average call duration
    """

    name: str
    state: CircuitState
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    failure_rate: float = 0.0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    last_state_change_time: datetime | None = None
    time_in_current_state_ms: float = 0.0
    average_response_time_ms: float = 0.0

    @property
    def is_healthy(self) -> bool:
        """Check if circuit is in healthy state."""
        return self.state == CircuitState.CLOSED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "failure_rate": self.failure_rate,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "last_state_change_time": self.last_state_change_time.isoformat() if self.last_state_change_time else None,
            "time_in_current_state_ms": self.time_in_current_state_ms,
            "average_response_time_ms": self.average_response_time_ms,
            "is_healthy": self.is_healthy,
        }


@dataclass
class StateChangeEvent:
    """Event emitted when circuit state changes.

    Attributes:
        breaker_name: Name of the circuit breaker
        from_state: Previous state
        to_state: New state
        timestamp: When the change occurred
        reason: Reason for state change
        metrics: Metrics at time of change
    """

    breaker_name: str
    from_state: CircuitState
    to_state: CircuitState
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""
    metrics: CircuitBreakerMetrics | None = None

    def __str__(self) -> str:
        return (
            f"CircuitBreaker '{self.breaker_name}' state change: "
            f"{self.from_state.value} -> {self.to_state.value}"
            f"{f' ({self.reason})' if self.reason else ''}"
        )
