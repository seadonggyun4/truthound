"""Circuit Breaker Middleware and Decorators.

This module provides decorators and middleware for easily integrating
circuit breaker protection into functions and action classes.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar, ParamSpec, overload

from truthound.checkpoint.circuitbreaker.core import (
    CircuitBreakerConfig,
    CircuitOpenError,
    FailureDetectionStrategy,
)
from truthound.checkpoint.circuitbreaker.breaker import CircuitBreaker
from truthound.checkpoint.circuitbreaker.registry import get_registry
from truthound.checkpoint.circuitbreaker.detection import FailureDetector

P = ParamSpec("P")
T = TypeVar("T")


class CircuitBreakerMiddleware:
    """Middleware for applying circuit breaker to action execution.

    This middleware wraps action execution with circuit breaker protection,
    automatically handling failures and state transitions.

    Example:
        >>> middleware = CircuitBreakerMiddleware(
        ...     breaker_name="action_executor",
        ...     config=CircuitBreakerConfig(failure_threshold=3),
        ... )
        >>>
        >>> # Wrap execution
        >>> result = middleware.execute(lambda: action.run(context))
    """

    def __init__(
        self,
        breaker_name: str,
        config: CircuitBreakerConfig | None = None,
        breaker: CircuitBreaker | None = None,
        register_globally: bool = True,
    ):
        """Initialize middleware.

        Args:
            breaker_name: Name for the circuit breaker
            config: Configuration (uses defaults if not provided)
            breaker: Existing breaker to use (creates new if not provided)
            register_globally: Register breaker in global registry
        """
        if breaker:
            self._breaker = breaker
        else:
            self._breaker = CircuitBreaker(breaker_name, config)

        if register_globally:
            registry = get_registry()
            if breaker_name not in registry:
                registry.register_breaker(self._breaker)

    @property
    def breaker(self) -> CircuitBreaker:
        """Get the underlying circuit breaker."""
        return self._breaker

    def execute(self, func: Callable[[], T]) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute

        Returns:
            Result of function

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Re-raised from func
        """
        return self._breaker.call(func)

    def execute_with_fallback(
        self,
        func: Callable[[], T],
        fallback: Callable[[], T],
    ) -> T:
        """Execute with fallback on circuit open.

        Args:
            func: Primary function
            fallback: Fallback function if circuit is open

        Returns:
            Result from func or fallback
        """
        try:
            return self._breaker.call(func)
        except CircuitOpenError:
            return fallback()

    def is_available(self) -> bool:
        """Check if circuit is available for calls."""
        return self._breaker.is_closed or self._breaker.is_half_open

    def get_status(self) -> dict[str, Any]:
        """Get middleware status."""
        metrics = self._breaker.get_metrics()
        return {
            "breaker_name": self._breaker.name,
            "state": metrics.state.value,
            "is_available": self.is_available(),
            "metrics": metrics.to_dict(),
        }


@overload
def circuit_breaker(
    func: Callable[P, T],
) -> Callable[P, T]: ...


@overload
def circuit_breaker(
    *,
    name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 3,
    success_threshold: int = 2,
    detection_strategy: FailureDetectionStrategy = FailureDetectionStrategy.CONSECUTIVE,
    failure_rate_threshold: float = 0.5,
    excluded_exceptions: tuple[type[Exception], ...] = (),
    fallback: Callable[..., Any] | None = None,
    register_globally: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def circuit_breaker(
    func: Callable[P, T] | None = None,
    *,
    name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 3,
    success_threshold: int = 2,
    detection_strategy: FailureDetectionStrategy = FailureDetectionStrategy.CONSECUTIVE,
    failure_rate_threshold: float = 0.5,
    excluded_exceptions: tuple[type[Exception], ...] = (),
    fallback: Callable[..., Any] | None = None,
    register_globally: bool = True,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to wrap function with circuit breaker protection.

    Can be used with or without arguments:

    Example:
        >>> @circuit_breaker
        ... def simple_api_call():
        ...     return api.get_data()
        >>>
        >>> @circuit_breaker(
        ...     name="external_api",
        ...     failure_threshold=3,
        ...     recovery_timeout=60.0,
        ... )
        ... def configured_call():
        ...     return api.get_data()
        >>>
        >>> @circuit_breaker(
        ...     fallback=lambda: {"status": "unavailable"},
        ... )
        ... def call_with_fallback():
        ...     return api.get_data()

    Args:
        func: Function to wrap (when used without parentheses)
        name: Breaker name (defaults to function name)
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before half-open
        half_open_max_calls: Test calls in half-open
        success_threshold: Successes to close
        detection_strategy: How to detect failures
        failure_rate_threshold: Rate for percentage strategy
        excluded_exceptions: Exceptions that don't count as failures
        fallback: Fallback function when circuit is open
        register_globally: Register in global registry

    Returns:
        Decorated function or decorator
    """
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        breaker_name = name or fn.__name__

        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            success_threshold=success_threshold,
            detection_strategy=detection_strategy,
            failure_rate_threshold=failure_rate_threshold,
            excluded_exceptions=excluded_exceptions,
            fallback=fallback,
        )

        breaker = CircuitBreaker(breaker_name, config)

        if register_globally:
            registry = get_registry()
            if breaker_name not in registry:
                registry.register_breaker(breaker)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return breaker.call(fn, *args, **kwargs)

        # Attach breaker for inspection
        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def with_circuit_breaker(
    breaker: CircuitBreaker | str,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to use existing circuit breaker.

    Example:
        >>> from truthound.checkpoint.circuitbreaker import CircuitBreaker
        >>>
        >>> api_breaker = CircuitBreaker("api", config)
        >>>
        >>> @with_circuit_breaker(api_breaker)
        ... def get_user(user_id: int):
        ...     return api.get_user(user_id)
        >>>
        >>> @with_circuit_breaker(api_breaker)
        ... def get_products():
        ...     return api.get_products()

    Or use registered breaker by name:

        >>> @with_circuit_breaker("api")
        ... def get_orders():
        ...     return api.get_orders()

    Args:
        breaker: CircuitBreaker instance or name of registered breaker

    Returns:
        Decorator function
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Resolve breaker
            if isinstance(breaker, str):
                actual_breaker = get_registry().get(breaker)
                if actual_breaker is None:
                    raise ValueError(f"Circuit breaker '{breaker}' not found in registry")
            else:
                actual_breaker = breaker

            return actual_breaker.call(func, *args, **kwargs)

        return wrapper
    return decorator


class CircuitBreakerGroup:
    """Group of related circuit breakers for coordinated management.

    Useful when multiple operations share failure characteristics
    or should be managed together.

    Example:
        >>> group = CircuitBreakerGroup(
        ...     name="payment_api",
        ...     operations=["charge", "refund", "validate"],
        ...     shared_config=CircuitBreakerConfig(failure_threshold=3),
        ... )
        >>>
        >>> # Execute with specific operation
        >>> result = group.execute("charge", lambda: payment.charge(amount))
        >>>
        >>> # Check group health
        >>> if group.is_healthy():
        ...     print("All payment operations available")
    """

    def __init__(
        self,
        name: str,
        operations: list[str],
        shared_config: CircuitBreakerConfig | None = None,
        register_globally: bool = True,
    ):
        """Initialize group.

        Args:
            name: Group name prefix
            operations: List of operation names
            shared_config: Config shared by all breakers
            register_globally: Register breakers globally
        """
        self._name = name
        self._config = shared_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

        for operation in operations:
            breaker_name = f"{name}.{operation}"
            breaker = CircuitBreaker(breaker_name, self._config)
            self._breakers[operation] = breaker

            if register_globally:
                registry = get_registry()
                if breaker_name not in registry:
                    registry.register_breaker(breaker)

    @property
    def name(self) -> str:
        """Get group name."""
        return self._name

    def get_breaker(self, operation: str) -> CircuitBreaker:
        """Get breaker for specific operation.

        Args:
            operation: Operation name

        Returns:
            CircuitBreaker for the operation

        Raises:
            KeyError: If operation not found
        """
        return self._breakers[operation]

    def execute(self, operation: str, func: Callable[[], T]) -> T:
        """Execute function with operation's breaker.

        Args:
            operation: Operation name
            func: Function to execute

        Returns:
            Result of function
        """
        return self._breakers[operation].call(func)

    def is_healthy(self) -> bool:
        """Check if all operations are healthy."""
        return all(b.is_closed for b in self._breakers.values())

    def get_unhealthy_operations(self) -> list[str]:
        """Get list of unhealthy operations."""
        return [
            op for op, breaker in self._breakers.items()
            if not breaker.is_closed
        ]

    def reset_all(self) -> None:
        """Reset all breakers in group."""
        for breaker in self._breakers.values():
            breaker.reset()

    def reset(self, operation: str) -> None:
        """Reset specific operation's breaker."""
        self._breakers[operation].reset()

    def get_status(self) -> dict[str, Any]:
        """Get status of all operations."""
        return {
            "group_name": self._name,
            "is_healthy": self.is_healthy(),
            "operations": {
                op: {
                    "state": breaker.state.value,
                    "metrics": breaker.get_metrics().to_dict(),
                }
                for op, breaker in self._breakers.items()
            },
        }
