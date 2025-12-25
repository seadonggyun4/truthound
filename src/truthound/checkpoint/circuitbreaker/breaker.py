"""Circuit Breaker implementation with state machine.

This module provides the core CircuitBreaker class that wraps
calls to external services and manages state transitions.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from threading import RLock
from typing import Any, Callable, Generator, TypeVar

from truthound.checkpoint.circuitbreaker.core import (
    CallResult,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitOpenError,
    CircuitHalfOpenError,
    CircuitState,
    StateChangeEvent,
)
from truthound.checkpoint.circuitbreaker.detection import (
    ConsecutiveFailureDetector,
    FailureDetector,
    create_detector,
)

T = TypeVar("T")


class CircuitBreakerStateMachine:
    """State machine for circuit breaker transitions.

    Manages state transitions and enforces state machine rules:
    - CLOSED -> OPEN: When detector.should_trip() returns True
    - OPEN -> HALF_OPEN: After recovery_timeout expires
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
        detector: FailureDetector | None = None,
    ):
        """Initialize state machine.

        Args:
            name: Breaker name for identification
            config: Configuration for behavior
            detector: Failure detector (created from config if not provided)
        """
        self._name = name
        self._config = config
        self._detector = detector or create_detector(config)
        self._lock = RLock()

        # State
        self._state = CircuitState.CLOSED
        self._state_changed_at = datetime.now()
        self._opened_at: datetime | None = None

        # Half-open state tracking
        self._half_open_calls = 0
        self._half_open_successes = 0

        # Metrics
        self._total_calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._rejected_calls = 0
        self._total_duration_ms = 0.0

        # Event listeners
        self._listeners: list[Callable[[StateChangeEvent], None]] = []
        if config.on_state_change:
            self._listeners.append(config.on_state_change)

    @property
    def name(self) -> str:
        """Get breaker name."""
        return self._name

    @property
    def state(self) -> CircuitState:
        """Get current state (may trigger transition check)."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self.state == CircuitState.HALF_OPEN

    def _check_state_transition(self) -> None:
        """Check and perform any automatic state transitions."""
        if self._state == CircuitState.OPEN and self._opened_at:
            elapsed = (datetime.now() - self._opened_at).total_seconds()
            if elapsed >= self._config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN, "Recovery timeout expired")

    def _transition_to(self, new_state: CircuitState, reason: str = "") -> None:
        """Transition to a new state.

        Args:
            new_state: State to transition to
            reason: Reason for transition
        """
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._state_changed_at = datetime.now()

        # State-specific setup
        if new_state == CircuitState.OPEN:
            self._opened_at = datetime.now()
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
        elif new_state == CircuitState.CLOSED:
            self._detector.reset()
            self._opened_at = None

        # Emit event
        event = StateChangeEvent(
            breaker_name=self._name,
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            metrics=self.get_metrics(),
        )

        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # Don't let listener errors affect circuit

    def can_execute(self) -> bool:
        """Check if a call can be executed.

        Returns:
            True if call is allowed, False otherwise
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                return self._half_open_calls < self._config.half_open_max_calls

            return False

    def record_success(self, duration_ms: float = 0.0) -> None:
        """Record a successful call.

        Args:
            duration_ms: Call duration in milliseconds
        """
        with self._lock:
            self._total_calls += 1
            self._successful_calls += 1
            self._total_duration_ms += duration_ms
            self._detector.record_success()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self._config.success_threshold:
                    self._transition_to(
                        CircuitState.CLOSED,
                        f"Success threshold reached ({self._half_open_successes})"
                    )

            if self._config.on_success:
                self._config.on_success(None, self.get_metrics())

    def record_failure(self, exception: Exception, duration_ms: float = 0.0) -> None:
        """Record a failed call.

        Args:
            exception: The exception that occurred
            duration_ms: Call duration in milliseconds
        """
        with self._lock:
            self._total_calls += 1
            self._failed_calls += 1
            self._total_duration_ms += duration_ms

            # Check if exception should count as failure
            if not self._config.should_count_exception(exception):
                return

            self._detector.record_failure(exception)

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN, f"Failure in half-open: {type(exception).__name__}")
            elif self._state == CircuitState.CLOSED and self._detector.should_trip():
                self._transition_to(CircuitState.OPEN, f"Failure threshold reached: {type(exception).__name__}")

            if self._config.on_failure:
                self._config.on_failure(exception, self.get_metrics())

    def record_rejection(self) -> None:
        """Record a rejected call (circuit open)."""
        with self._lock:
            self._rejected_calls += 1

    def get_remaining_timeout(self) -> float | None:
        """Get remaining time until recovery attempt.

        Returns:
            Seconds until half-open transition, or None if not open
        """
        with self._lock:
            if self._state != CircuitState.OPEN or not self._opened_at:
                return None

            elapsed = (datetime.now() - self._opened_at).total_seconds()
            remaining = self._config.recovery_timeout - elapsed
            return max(0, remaining)

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics.

        Returns:
            CircuitBreakerMetrics with current state
        """
        with self._lock:
            total = self._successful_calls + self._failed_calls
            avg_duration = self._total_duration_ms / total if total > 0 else 0.0
            failure_rate = self._failed_calls / total if total > 0 else 0.0

            time_in_state = (datetime.now() - self._state_changed_at).total_seconds() * 1000

            return CircuitBreakerMetrics(
                name=self._name,
                state=self._state,
                total_calls=self._total_calls,
                successful_calls=self._successful_calls,
                failed_calls=self._failed_calls,
                rejected_calls=self._rejected_calls,
                consecutive_failures=getattr(self._detector, "consecutive_failures", 0),
                consecutive_successes=self._half_open_successes if self._state == CircuitState.HALF_OPEN else 0,
                failure_rate=failure_rate,
                last_failure_time=None,  # Could track if needed
                last_success_time=None,
                last_state_change_time=self._state_changed_at,
                time_in_current_state_ms=time_in_state,
                average_response_time_ms=avg_duration,
            )

    def reset(self) -> None:
        """Force reset to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED, "Manual reset")
            self._total_calls = 0
            self._successful_calls = 0
            self._failed_calls = 0
            self._rejected_calls = 0
            self._total_duration_ms = 0.0

    def add_listener(self, listener: Callable[[StateChangeEvent], None]) -> None:
        """Add state change listener.

        Args:
            listener: Callback for state changes
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[StateChangeEvent], None]) -> None:
        """Remove state change listener.

        Args:
            listener: Callback to remove
        """
        if listener in self._listeners:
            self._listeners.remove(listener)


class CircuitBreaker:
    """Circuit Breaker for wrapping unreliable operations.

    Provides protection against cascading failures by monitoring
    calls and "tripping" when failure thresholds are exceeded.

    Example:
        >>> breaker = CircuitBreaker("external_api")
        >>>
        >>> # Execute with protection
        >>> result = breaker.call(lambda: api.get_data())
        >>>
        >>> # Or use context manager
        >>> with breaker:
        ...     response = api.get_data()
        >>>
        >>> # Check state
        >>> if breaker.is_open:
        ...     print("Circuit is open, using fallback")
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        detector: FailureDetector | None = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Unique name for this breaker
            config: Configuration (uses defaults if not provided)
            detector: Custom failure detector (created from config if not provided)
        """
        self._config = config or CircuitBreakerConfig()
        self._state_machine = CircuitBreakerStateMachine(
            name=name,
            config=self._config,
            detector=detector,
        )

    @property
    def name(self) -> str:
        """Get breaker name."""
        return self._state_machine.name

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state_machine.state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state_machine.is_closed

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state_machine.is_open

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self._state_machine.is_half_open

    @property
    def config(self) -> CircuitBreakerConfig:
        """Get configuration."""
        return self._config

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitOpenError: If circuit is open
            CircuitHalfOpenError: If half-open at max calls
            Exception: Re-raised from func
        """
        # Check if execution is allowed
        if not self._state_machine.can_execute():
            self._state_machine.record_rejection()

            if self._state_machine.state == CircuitState.OPEN:
                remaining = self._state_machine.get_remaining_timeout()

                # Try fallback
                if self._config.fallback:
                    return self._config.fallback(*args, **kwargs)

                raise CircuitOpenError(self.name, remaining)
            else:
                raise CircuitHalfOpenError(
                    self.name,
                    self._config.half_open_max_calls
                )

        # Execute the call
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._state_machine.record_success(duration_ms)
            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._state_machine.record_failure(e, duration_ms)
            raise

    def call_with_result(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> CallResult:
        """Execute function and return CallResult instead of raising.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            CallResult with success/failure details
        """
        start_time = time.perf_counter()

        try:
            result = self.call(func, *args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            return CallResult(
                success=True,
                result=result,
                duration_ms=duration_ms,
            )

        except (CircuitOpenError, CircuitHalfOpenError) as e:
            # Circuit is open, try fallback
            if self._config.fallback:
                try:
                    fallback_result = self._config.fallback(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    return CallResult(
                        success=True,
                        result=fallback_result,
                        duration_ms=duration_ms,
                        from_fallback=True,
                    )
                except Exception as fallback_error:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    return CallResult(
                        success=False,
                        exception=fallback_error,
                        duration_ms=duration_ms,
                        from_fallback=True,
                    )

            duration_ms = (time.perf_counter() - start_time) * 1000
            return CallResult(
                success=False,
                exception=e,
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return CallResult(
                success=False,
                exception=e,
                duration_ms=duration_ms,
            )

    @contextmanager
    def __call__(self) -> Generator[CircuitBreaker, None, None]:
        """Context manager for circuit breaker.

        Example:
            >>> with breaker():
            ...     result = external_call()

        Yields:
            Self for chaining
        """
        if not self._state_machine.can_execute():
            self._state_machine.record_rejection()

            if self._state_machine.state == CircuitState.OPEN:
                raise CircuitOpenError(self.name, self._state_machine.get_remaining_timeout())
            else:
                raise CircuitHalfOpenError(self.name, self._config.half_open_max_calls)

        start_time = time.perf_counter()
        try:
            yield self
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._state_machine.record_success(duration_ms)

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._state_machine.record_failure(e, duration_ms)
            raise

    def __enter__(self) -> CircuitBreaker:
        """Enter context manager."""
        if not self._state_machine.can_execute():
            self._state_machine.record_rejection()

            if self._state_machine.state == CircuitState.OPEN:
                raise CircuitOpenError(self.name, self._state_machine.get_remaining_timeout())
            else:
                raise CircuitHalfOpenError(self.name, self._config.half_open_max_calls)

        self._context_start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context manager."""
        duration_ms = (time.perf_counter() - self._context_start_time) * 1000

        if exc_type is None:
            self._state_machine.record_success(duration_ms)
        elif exc_val is not None:
            self._state_machine.record_failure(exc_val, duration_ms)

        return False  # Don't suppress exceptions

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        return self._state_machine.get_metrics()

    def reset(self) -> None:
        """Force reset to closed state."""
        self._state_machine.reset()

    def add_listener(self, listener: Callable[[StateChangeEvent], None]) -> None:
        """Add state change listener."""
        self._state_machine.add_listener(listener)

    def remove_listener(self, listener: Callable[[StateChangeEvent], None]) -> None:
        """Remove state change listener."""
        self._state_machine.remove_listener(listener)
