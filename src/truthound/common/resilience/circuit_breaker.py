"""Unified CircuitBreaker implementation.

This module provides a thread-safe circuit breaker that can be used
across all Truthound components, replacing duplicated implementations
in profiler, stores, and realtime modules.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

from truthound.common.resilience.config import CircuitBreakerConfig
from truthound.common.resilience.protocols import CircuitBreakerProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, name: str, state: CircuitState, remaining_seconds: float = 0.0):
        self.circuit_name = name
        self.state = state
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Circuit breaker '{name}' is {state.value}. "
            f"Retry after {remaining_seconds:.1f}s"
        )


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker observability."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    slow_calls: int = 0
    state_transitions: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    last_state_change_time: datetime | None = None
    current_state: CircuitState = CircuitState.CLOSED
    call_durations: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record_success(self, duration_ms: float) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.call_durations.append(duration_ms)

    def record_failure(self, duration_ms: float = 0.0) -> None:
        """Record a failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()
        if duration_ms > 0:
            self.call_durations.append(duration_ms)

    def record_rejection(self) -> None:
        """Record a rejected call."""
        self.rejected_calls += 1

    def record_slow_call(self) -> None:
        """Record a slow call."""
        self.slow_calls += 1

    def record_state_change(self, new_state: CircuitState) -> None:
        """Record a state transition."""
        self.state_transitions += 1
        self.current_state = new_state
        self.last_state_change_time = datetime.now()

    def get_failure_rate(self) -> float:
        """Get failure rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.failed_calls / self.total_calls) * 100

    def get_slow_call_rate(self) -> float:
        """Get slow call rate as percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.slow_calls / self.total_calls) * 100

    def get_average_duration(self) -> float:
        """Get average call duration in ms."""
        if not self.call_durations:
            return 0.0
        return sum(self.call_durations) / len(self.call_durations)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "slow_calls": self.slow_calls,
            "state_transitions": self.state_transitions,
            "failure_rate": self.get_failure_rate(),
            "slow_call_rate": self.get_slow_call_rate(),
            "average_duration_ms": self.get_average_duration(),
            "current_state": self.current_state.value,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat() if self.last_success_time else None
            ),
        }


class CircuitBreaker(CircuitBreakerProtocol):
    """Thread-safe circuit breaker implementation.

    The circuit breaker prevents cascade failures by:
    1. CLOSED: Normal operation, tracking failures
    2. OPEN: Rejecting requests after too many failures
    3. HALF_OPEN: Testing if service has recovered

    Example:
        cb = CircuitBreaker("my-service")

        # As decorator
        @cb
        def risky_call():
            return external_service.call()

        # As context manager
        with cb.protect():
            result = external_service.call()

        # Manual state management
        if cb.allow_request():
            try:
                result = external_service.call()
                cb.record_success()
            except Exception as e:
                cb.record_failure(e)
                raise
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Unique name for this circuit breaker.
            config: Configuration options.
            on_state_change: Callback when state changes (old_state, new_state).
        """
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()
        self._metrics = CircuitBreakerMetrics()

        # Timing
        self._opened_at: float | None = None
        self._half_open_calls = 0

        # Sliding window for rate-based decisions
        self._recent_calls: deque[tuple[float, bool]] = deque(maxlen=self._config.window_size)

    @property
    def name(self) -> str:
        """Get circuit breaker name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            return self._state

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get configuration."""
        return self._config

    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get metrics."""
        return self._metrics

    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED

    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self.state == CircuitState.HALF_OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            # HALF_OPEN: allow limited requests
            if self._half_open_calls < self._config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            now = time.time()
            duration_ms = 0.0
            self._recent_calls.append((now, True))
            self._metrics.record_success(duration_ms)

            if self._state == CircuitState.HALF_OPEN:
                if self._metrics.consecutive_successes >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""
        with self._lock:
            # Check if exception is excluded
            if error and isinstance(error, self._config.excluded_exceptions):
                return

            now = time.time()
            self._recent_calls.append((now, False))
            self._metrics.record_failure()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._should_trip():
                    self._transition_to(CircuitState.OPEN)

    def record_duration(self, duration_ms: float) -> None:
        """Record call duration for slow call detection."""
        if not self._config.record_slow_calls:
            return

        with self._lock:
            if duration_ms >= self._config.slow_call_threshold_ms:
                self._metrics.record_slow_call()

                # Check if slow call rate should trip circuit
                if self._metrics.get_slow_call_rate() >= self._config.slow_call_rate_threshold:
                    if self._state == CircuitState.CLOSED:
                        self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._half_open_calls = 0
            self._recent_calls.clear()
            self._metrics = CircuitBreakerMetrics()

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics as dictionary."""
        return self._metrics.to_dict()

    def get_remaining_timeout(self) -> float:
        """Get remaining time until half-open transition."""
        with self._lock:
            if self._state != CircuitState.OPEN or self._opened_at is None:
                return 0.0
            elapsed = time.time() - self._opened_at
            remaining = self._config.timeout_seconds - elapsed
            return max(0.0, remaining)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._opened_at is None:
            return True
        elapsed = time.time() - self._opened_at
        return elapsed >= self._config.timeout_seconds

    def _should_trip(self) -> bool:
        """Check if circuit should trip to open."""
        # Check consecutive failures
        if self._metrics.consecutive_failures >= self._config.failure_threshold:
            return True

        # Check failure rate
        if self._metrics.get_failure_rate() >= self._config.failure_rate_threshold:
            return True

        return False

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._metrics.record_state_change(new_state)

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._opened_at = None
            self._metrics.consecutive_failures = 0

        logger.info(
            f"Circuit breaker '{self._name}' transitioned from {old_state.value} to {new_state.value}"
        )

        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception as e:
                logger.warning(f"Error in state change callback: {e}")

    @contextmanager
    def protect(self) -> Generator[None, None, None]:
        """Context manager for protected execution.

        Example:
            with circuit_breaker.protect():
                result = risky_operation()
        """
        if not self.allow_request():
            self._metrics.record_rejection()
            raise CircuitOpenError(
                self._name,
                self._state,
                self.get_remaining_timeout(),
            )

        start_time = time.time()
        try:
            yield
            duration_ms = (time.time() - start_time) * 1000
            self.record_success()
            self.record_duration(duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.record_failure(e)
            self.record_duration(duration_ms)
            raise

    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for protected execution.

        Example:
            @circuit_breaker
            def risky_operation():
                return external_service.call()
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            with self.protect():
                return func(*args, **kwargs)
        return wrapper


class CircuitBreakerRegistry:
    """Registry for managing circuit breakers.

    Provides a centralized way to create and access circuit breakers
    across the application.

    Example:
        registry = CircuitBreakerRegistry()

        # Get or create circuit breaker
        cb = registry.get_or_create("database", CircuitBreakerConfig.for_database())

        # Use the same circuit breaker elsewhere
        same_cb = registry.get("database")
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config,
                    on_state_change=on_state_change,
                )
            return self._breakers[name]

    def register(self, breaker: CircuitBreaker) -> None:
        """Register an existing circuit breaker."""
        with self._lock:
            self._breakers[breaker.name] = breaker

    def unregister(self, name: str) -> CircuitBreaker | None:
        """Unregister and return a circuit breaker."""
        with self._lock:
            return self._breakers.pop(name, None)

    def reset(self, name: str) -> bool:
        """Reset a circuit breaker by name."""
        with self._lock:
            if name in self._breakers:
                self._breakers[name].reset()
                return True
            return False

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }

    def list_names(self) -> list[str]:
        """List all registered circuit breaker names."""
        with self._lock:
            return list(self._breakers.keys())

    def __len__(self) -> int:
        """Get number of registered circuit breakers."""
        return len(self._breakers)

    def __contains__(self, name: str) -> bool:
        """Check if circuit breaker exists."""
        return name in self._breakers


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    config: CircuitBreakerConfig | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry.

    This is a convenience function for accessing the global registry.

    Example:
        cb = get_circuit_breaker("my-service")

        @cb
        def risky_operation():
            return external_service.call()
    """
    return circuit_breaker_registry.get_or_create(name, config)
