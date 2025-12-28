"""Circuit breaker pattern for resilient timeout handling.

This module implements a full-featured circuit breaker pattern:
- Closed: Normal operation
- Open: Fast-fail without execution
- Half-Open: Test if service recovered

The circuit breaker prevents cascading failures by failing fast
when a service is experiencing issues.

Example:
    from truthound.validators.timeout.advanced.circuit_breaker import (
        CircuitBreaker,
        with_circuit_breaker,
    )

    # Create circuit breaker
    breaker = CircuitBreaker(
        name="validation_service",
        failure_threshold=5,
        recovery_timeout=30,
    )

    # Use directly
    result = breaker.execute(validate, data)

    # Or use decorator
    @with_circuit_breaker(failure_threshold=3)
    def validate(data):
        return check(data)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Fast-failing
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(str, Enum):
    """Types of failures tracked."""

    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    SLOW_RESPONSE = "slow_response"
    REJECTED = "rejected"


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Failures before opening circuit
        success_threshold: Successes in half-open to close
        recovery_timeout: Seconds before testing recovery
        half_open_max_calls: Max concurrent calls in half-open
        slow_call_threshold_ms: Threshold for slow calls
        slow_call_rate_threshold: Ratio of slow calls to open
        failure_rate_threshold: Ratio of failures to open (0.0-1.0)
        minimum_calls: Minimum calls before evaluating rates
        window_size: Time window for rate calculations
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    slow_call_threshold_ms: float = 5000.0
    slow_call_rate_threshold: float = 0.5
    failure_rate_threshold: float = 0.5
    minimum_calls: int = 10
    window_size: float = 60.0


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker.

    Attributes:
        state: Current circuit state
        failure_count: Number of failures
        success_count: Number of successes
        total_calls: Total calls attempted
        rejected_calls: Calls rejected due to open circuit
        slow_calls: Number of slow calls
        last_failure_time: Time of last failure
        last_state_change: Time of last state change
        time_in_current_state: Seconds in current state
    """

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    total_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0
    last_failure_time: datetime | None = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def time_in_current_state(self) -> float:
        """Get seconds in current state."""
        return (datetime.now(timezone.utc) - self.last_state_change).total_seconds()

    @property
    def failure_rate(self) -> float:
        """Get failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failure_count / self.total_calls

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "rejected_calls": self.rejected_calls,
            "slow_calls": self.slow_calls,
            "failure_rate": self.failure_rate,
            "success_rate": self.success_rate,
            "time_in_current_state": self.time_in_current_state,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


class HalfOpenPolicy(ABC):
    """Policy for handling half-open state."""

    @abstractmethod
    def should_allow_call(self, metrics: CircuitMetrics) -> bool:
        """Check if a call should be allowed in half-open state.

        Args:
            metrics: Current metrics

        Returns:
            True if call should be allowed
        """
        pass


class CountBasedHalfOpenPolicy(HalfOpenPolicy):
    """Allow fixed number of calls in half-open state."""

    def __init__(self, max_calls: int = 3):
        """Initialize policy.

        Args:
            max_calls: Maximum calls to allow
        """
        self.max_calls = max_calls
        self._current_calls = 0
        self._lock = threading.Lock()

    def should_allow_call(self, metrics: CircuitMetrics) -> bool:
        """Check if call should be allowed."""
        with self._lock:
            if self._current_calls >= self.max_calls:
                return False
            self._current_calls += 1
            return True

    def reset(self) -> None:
        """Reset the counter."""
        with self._lock:
            self._current_calls = 0


class TimeBasedHalfOpenPolicy(HalfOpenPolicy):
    """Allow calls based on time since half-open started."""

    def __init__(self, ramp_up_seconds: float = 10.0):
        """Initialize policy.

        Args:
            ramp_up_seconds: Seconds over which to gradually increase allowed calls
        """
        self.ramp_up_seconds = ramp_up_seconds
        self._half_open_start: datetime | None = None

    def should_allow_call(self, metrics: CircuitMetrics) -> bool:
        """Check if call should be allowed based on time."""
        if self._half_open_start is None:
            self._half_open_start = datetime.now(timezone.utc)

        elapsed = (datetime.now(timezone.utc) - self._half_open_start).total_seconds()
        # Increase allowed rate over time
        allowed_rate = min(1.0, elapsed / self.ramp_up_seconds)

        import random
        return random.random() < allowed_rate

    def reset(self) -> None:
        """Reset the timer."""
        self._half_open_start = None


class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting calls."""

    def __init__(self, name: str, remaining_seconds: float):
        super().__init__(f"Circuit '{name}' is open, retry in {remaining_seconds:.1f}s")
        self.name = name
        self.remaining_seconds = remaining_seconds


@dataclass
class CallRecord:
    """Record of a call attempt."""

    timestamp: datetime
    success: bool
    duration_ms: float
    failure_type: FailureType | None = None


class CircuitBreaker(Generic[T]):
    """Full-featured circuit breaker implementation.

    Implements the circuit breaker pattern with:
    - Configurable failure thresholds
    - Time-based recovery
    - Half-open state with controlled testing
    - Metrics and monitoring

    Example:
        breaker = CircuitBreaker("my_service")

        try:
            result = breaker.execute(risky_operation)
        except CircuitOpenError:
            # Use fallback
            result = fallback_operation()
    """

    def __init__(
        self,
        name: str,
        config: CircuitConfig | None = None,
        half_open_policy: HalfOpenPolicy | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
        fallback: Callable[[], T] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration
            half_open_policy: Policy for half-open state
            on_state_change: Callback for state changes
            fallback: Fallback function when circuit is open
        """
        self.name = name
        self.config = config or CircuitConfig()
        self._half_open_policy = half_open_policy or CountBasedHalfOpenPolicy(
            self.config.half_open_max_calls
        )
        self._on_state_change = on_state_change
        self._fallback = fallback

        self._state = CircuitState.CLOSED
        self._metrics = CircuitMetrics()
        self._opened_at: datetime | None = None
        self._call_history: list[CallRecord] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def metrics(self) -> CircuitMetrics:
        """Get current metrics."""
        with self._lock:
            self._metrics.state = self._state
            return CircuitMetrics(
                state=self._state,
                failure_count=self._metrics.failure_count,
                success_count=self._metrics.success_count,
                total_calls=self._metrics.total_calls,
                rejected_calls=self._metrics.rejected_calls,
                slow_calls=self._metrics.slow_calls,
                last_failure_time=self._metrics.last_failure_time,
                last_state_change=self._metrics.last_state_change,
            )

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._opened_at:
                elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                if elapsed >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state.

        Args:
            new_state: New state
        """
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._metrics.last_state_change = datetime.now(timezone.utc)

        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now(timezone.utc)
        elif new_state == CircuitState.HALF_OPEN:
            if hasattr(self._half_open_policy, 'reset'):
                self._half_open_policy.reset()
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._metrics.failure_count = 0
            self._metrics.success_count = 0

        logger.info(f"Circuit '{self.name}' transitioned from {old_state.value} to {new_state.value}")

        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

    def _should_allow_call(self) -> bool:
        """Check if a call should be allowed.

        Returns:
            True if call should be allowed
        """
        self._check_state_transition()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            return False

        # Half-open
        return self._half_open_policy.should_allow_call(self._metrics)

    def _record_success(self, duration_ms: float) -> None:
        """Record a successful call.

        Args:
            duration_ms: Call duration
        """
        self._metrics.success_count += 1
        self._metrics.total_calls += 1

        is_slow = duration_ms > self.config.slow_call_threshold_ms
        if is_slow:
            self._metrics.slow_calls += 1

        self._call_history.append(CallRecord(
            timestamp=datetime.now(timezone.utc),
            success=True,
            duration_ms=duration_ms,
        ))
        self._trim_history()

        # Check state transition
        if self._state == CircuitState.HALF_OPEN:
            if self._metrics.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, failure_type: FailureType = FailureType.EXCEPTION) -> None:
        """Record a failed call.

        Args:
            failure_type: Type of failure
        """
        self._metrics.failure_count += 1
        self._metrics.total_calls += 1
        self._metrics.last_failure_time = datetime.now(timezone.utc)

        self._call_history.append(CallRecord(
            timestamp=datetime.now(timezone.utc),
            success=False,
            duration_ms=0,
            failure_type=failure_type,
        ))
        self._trim_history()

        # Check state transition
        if self._state == CircuitState.CLOSED:
            if self._should_open():
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)

    def _should_open(self) -> bool:
        """Check if circuit should open.

        Returns:
            True if circuit should open
        """
        # Check failure count threshold
        if self._metrics.failure_count >= self.config.failure_threshold:
            return True

        # Check failure rate threshold
        if self._metrics.total_calls >= self.config.minimum_calls:
            if self._metrics.failure_rate >= self.config.failure_rate_threshold:
                return True

        return False

    def _trim_history(self) -> None:
        """Trim call history to window size."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.config.window_size)
        self._call_history = [r for r in self._call_history if r.timestamp > cutoff]

    def execute(
        self,
        operation: Callable[[], T],
        fallback: Callable[[], T] | None = None,
    ) -> T:
        """Execute an operation with circuit breaker protection.

        Args:
            operation: Operation to execute
            fallback: Optional fallback function

        Returns:
            Operation result

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        with self._lock:
            if not self._should_allow_call():
                self._metrics.rejected_calls += 1

                fb = fallback or self._fallback
                if fb:
                    return fb()

                remaining = 0.0
                if self._opened_at:
                    elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                    remaining = max(0.0, self.config.recovery_timeout - elapsed)

                raise CircuitOpenError(self.name, remaining)

        start = time.time()

        try:
            result = operation()
            duration_ms = (time.time() - start) * 1000

            with self._lock:
                self._record_success(duration_ms)

            return result

        except Exception as e:
            with self._lock:
                self._record_failure(FailureType.EXCEPTION)
            raise

    async def execute_async(
        self,
        operation: Callable[[], T],
        fallback: Callable[[], T] | None = None,
    ) -> T:
        """Execute an async operation with circuit breaker protection.

        Args:
            operation: Operation to execute
            fallback: Optional fallback function

        Returns:
            Operation result
        """
        with self._lock:
            if not self._should_allow_call():
                self._metrics.rejected_calls += 1

                fb = fallback or self._fallback
                if fb:
                    if asyncio.iscoroutinefunction(fb):
                        return await fb()
                    return fb()

                remaining = 0.0
                if self._opened_at:
                    elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                    remaining = max(0.0, self.config.recovery_timeout - elapsed)

                raise CircuitOpenError(self.name, remaining)

        start = time.time()

        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, operation)

            duration_ms = (time.time() - start) * 1000

            with self._lock:
                self._record_success(duration_ms)

            return result

        except Exception as e:
            with self._lock:
                self._record_failure(FailureType.EXCEPTION)
            raise

    @contextmanager
    def protect(self):
        """Context manager for circuit breaker protection.

        Yields:
            None

        Raises:
            CircuitOpenError: If circuit is open

        Example:
            with breaker.protect():
                risky_operation()
        """
        with self._lock:
            if not self._should_allow_call():
                self._metrics.rejected_calls += 1

                remaining = 0.0
                if self._opened_at:
                    elapsed = (datetime.now(timezone.utc) - self._opened_at).total_seconds()
                    remaining = max(0.0, self.config.recovery_timeout - elapsed)

                raise CircuitOpenError(self.name, remaining)

        start = time.time()

        try:
            yield
            duration_ms = (time.time() - start) * 1000
            with self._lock:
                self._record_success(duration_ms)

        except Exception:
            with self._lock:
                self._record_failure(FailureType.EXCEPTION)
            raise

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._metrics = CircuitMetrics()
            self._call_history.clear()
            if hasattr(self._half_open_policy, 'reset'):
                self._half_open_policy.reset()

    def force_open(self) -> None:
        """Force circuit to open state."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Force circuit to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)


def with_circuit_breaker(
    name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    fallback: Callable[..., T] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add circuit breaker to a function.

    Args:
        name: Circuit breaker name (default: function name)
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before recovery
        fallback: Fallback function

    Returns:
        Decorated function

    Example:
        @with_circuit_breaker(failure_threshold=3)
        def risky_operation():
            ...
    """
    config = CircuitConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        breaker_name = name or func.__name__
        breaker: CircuitBreaker[T] = CircuitBreaker(breaker_name, config)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            fb = (lambda: fallback(*args, **kwargs)) if fallback else None
            return breaker.execute(lambda: func(*args, **kwargs), fb)

        # Attach breaker for inspection
        wrapper.circuit_breaker = breaker  # type: ignore

        return wrapper

    return decorator


def create_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    **kwargs: Any,
) -> CircuitBreaker[Any]:
    """Create a circuit breaker with common defaults.

    Args:
        name: Circuit breaker name
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before recovery
        **kwargs: Additional config options

    Returns:
        CircuitBreaker
    """
    config = CircuitConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs,
    )
    return CircuitBreaker(name, config)
