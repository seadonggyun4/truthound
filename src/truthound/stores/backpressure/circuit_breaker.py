"""Circuit breaker pattern for backpressure protection.

This module implements the circuit breaker pattern to prevent
cascading failures in streaming systems.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitBreakerState(str, Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Failures before opening circuit.
        success_threshold: Successes in half-open to close.
        timeout_seconds: Time in open state before half-open.
        half_open_max_calls: Max calls allowed in half-open.
        failure_rate_threshold: Failure rate % to trigger open.
        slow_call_threshold_ms: Latency considered slow.
        slow_call_rate_threshold: Slow call rate % to trigger open.
        window_size: Number of calls to track.
        wait_duration_in_open: Wait time in open state.
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3
    failure_rate_threshold: float = 50.0
    slow_call_threshold_ms: float = 1000.0
    slow_call_rate_threshold: float = 50.0
    window_size: int = 100
    wait_duration_in_open: float = 60.0

    def validate(self) -> None:
        """Validate configuration values."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0 <= self.failure_rate_threshold <= 100:
            raise ValueError("failure_rate_threshold must be between 0 and 100")
        if not 0 <= self.slow_call_rate_threshold <= 100:
            raise ValueError("slow_call_rate_threshold must be between 0 and 100")


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker.

    Attributes:
        total_calls: Total number of calls.
        successful_calls: Number of successful calls.
        failed_calls: Number of failed calls.
        rejected_calls: Number of rejected calls (circuit open).
        slow_calls: Number of slow calls.
        state_transitions: Number of state transitions.
        last_failure_time: Time of last failure.
        last_success_time: Time of last success.
        current_state: Current circuit state.
        opened_at: When circuit opened.
        closed_at: When circuit closed.
    """

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0
    state_transitions: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    current_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    opened_at: datetime | None = None
    closed_at: datetime | None = None
    call_latencies: list[float] = field(default_factory=list)

    def record_success(self, latency_ms: float) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success_time = datetime.now()
        self._add_latency(latency_ms)

    def record_failure(self, latency_ms: float) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure_time = datetime.now()
        self._add_latency(latency_ms)

    def record_rejection(self) -> None:
        """Record a rejected call."""
        self.rejected_calls += 1

    def record_slow_call(self) -> None:
        """Record a slow call."""
        self.slow_calls += 1

    def record_state_change(self, new_state: CircuitBreakerState) -> None:
        """Record a state transition."""
        self.state_transitions += 1
        self.current_state = new_state
        if new_state == CircuitBreakerState.OPEN:
            self.opened_at = datetime.now()
        elif new_state == CircuitBreakerState.CLOSED:
            self.closed_at = datetime.now()

    def _add_latency(self, latency_ms: float, max_samples: int = 100) -> None:
        """Add latency sample."""
        self.call_latencies.append(latency_ms)
        if len(self.call_latencies) > max_samples:
            self.call_latencies.pop(0)

    def get_failure_rate(self) -> float:
        """Get failure rate percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100

    def get_slow_call_rate(self) -> float:
        """Get slow call rate percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.slow_calls / self.total_calls) * 100

    def get_average_latency(self) -> float:
        """Get average latency in ms."""
        if not self.call_latencies:
            return 0.0
        return sum(self.call_latencies) / len(self.call_latencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "slow_calls": self.slow_calls,
            "state_transitions": self.state_transitions,
            "failure_rate": self.get_failure_rate(),
            "slow_call_rate": self.get_slow_call_rate(),
            "average_latency_ms": self.get_average_latency(),
            "current_state": self.current_state.value,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
        }


class CircuitBreakerError(Exception):
    """Raised when circuit breaker rejects a call."""

    def __init__(self, state: CircuitBreakerState, message: str = "") -> None:
        self.state = state
        super().__init__(message or f"Circuit breaker is {state.value}")


class CircuitBreaker:
    """Circuit breaker for protecting downstream systems.

    Prevents cascading failures by tracking failures and temporarily
    stopping calls to failing services.

    Example:
        >>> cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        >>>
        >>> async def call_service():
        ...     async with cb:
        ...         return await external_service()
        >>>
        >>> try:
        ...     result = await call_service()
        ... except CircuitBreakerError:
        ...     result = fallback_value()
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration.
        """
        self._config = config or CircuitBreakerConfig()
        self._config.validate()
        self._state = CircuitBreakerState.CLOSED
        self._metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()

        # Tracking
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._opened_at: float = 0.0
        self._call_start: float = 0.0

        # Sliding window for failure rate
        self._call_results: list[bool] = []  # True = success, False = failure

    @property
    def state(self) -> CircuitBreakerState:
        """Get current state."""
        return self._state

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get metrics."""
        return self._metrics

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitBreakerState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self._state == CircuitBreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state == CircuitBreakerState.HALF_OPEN

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """Transition to new state."""
        if self._state == new_state:
            return

        self._state = new_state
        self._metrics.record_state_change(new_state)

        if new_state == CircuitBreakerState.OPEN:
            self._opened_at = time.monotonic()
            self._half_open_calls = 0
        elif new_state == CircuitBreakerState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitBreakerState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._call_results.clear()

    def _should_attempt_reset(self) -> bool:
        """Check if should try to reset (transition to half-open)."""
        if self._state != CircuitBreakerState.OPEN:
            return False

        elapsed = time.monotonic() - self._opened_at
        return elapsed >= self._config.timeout_seconds

    def _check_failure_rate(self) -> bool:
        """Check if failure rate exceeds threshold."""
        if len(self._call_results) < self._config.window_size // 2:
            # Not enough data
            return False

        failures = self._call_results.count(False)
        failure_rate = (failures / len(self._call_results)) * 100
        return failure_rate >= self._config.failure_rate_threshold

    def _record_result(self, success: bool, latency_ms: float) -> None:
        """Record call result."""
        # Add to sliding window
        self._call_results.append(success)
        if len(self._call_results) > self._config.window_size:
            self._call_results.pop(0)

        # Check for slow call
        if latency_ms >= self._config.slow_call_threshold_ms:
            self._metrics.record_slow_call()

        if success:
            self._metrics.record_success(latency_ms)
            self._on_success()
        else:
            self._metrics.record_failure(latency_ms)
            self._on_failure()

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                self._transition_to(CircuitBreakerState.CLOSED)
        elif self._state == CircuitBreakerState.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Any failure in half-open opens circuit
            self._transition_to(CircuitBreakerState.OPEN)
        elif self._state == CircuitBreakerState.CLOSED:
            # Check if should open
            if self._failure_count >= self._config.failure_threshold:
                self._transition_to(CircuitBreakerState.OPEN)
            elif self._check_failure_rate():
                self._transition_to(CircuitBreakerState.OPEN)

    async def _acquire(self) -> bool:
        """Acquire permission to make a call.

        Returns:
            True if allowed, raises CircuitBreakerError otherwise.
        """
        async with self._lock:
            # Check if should reset
            if self._should_attempt_reset():
                self._transition_to(CircuitBreakerState.HALF_OPEN)

            if self._state == CircuitBreakerState.OPEN:
                self._metrics.record_rejection()
                raise CircuitBreakerError(self._state)

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._metrics.record_rejection()
                    raise CircuitBreakerError(self._state, "Half-open call limit reached")
                self._half_open_calls += 1

            self._call_start = time.monotonic()
            return True

    async def _release(self, success: bool) -> None:
        """Release after call completes."""
        latency_ms = (time.monotonic() - self._call_start) * 1000
        async with self._lock:
            self._record_result(success, latency_ms)

    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        fallback: Callable[..., T] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args: Positional arguments for func.
            fallback: Optional fallback function if circuit is open.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of func or fallback.

        Raises:
            CircuitBreakerError: If circuit is open and no fallback.
        """
        try:
            await self._acquire()
        except CircuitBreakerError:
            if fallback:
                return fallback(*args, **kwargs)
            raise

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self._release(success=True)
            return result
        except Exception:
            await self._release(success=False)
            raise

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        await self._acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        success = exc_type is None
        await self._release(success)

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._transition_to(CircuitBreakerState.CLOSED)
        self._failure_count = 0
        self._success_count = 0
        self._call_results.clear()

    def force_open(self) -> None:
        """Force circuit to open state."""
        self._transition_to(CircuitBreakerState.OPEN)

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "metrics": self._metrics.to_dict(),
            "time_in_current_state": (
                time.monotonic() - self._opened_at
                if self._state == CircuitBreakerState.OPEN
                else None
            ),
        }
