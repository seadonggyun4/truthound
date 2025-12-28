"""Composite resilience patterns.

This module provides a fluent builder for combining multiple
resilience patterns into a single wrapper.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from truthound.common.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from truthound.common.resilience.retry import RetryPolicy, RetryConfig
from truthound.common.resilience.bulkhead import SemaphoreBulkhead, BulkheadConfig
from truthound.common.resilience.rate_limiter import (
    TokenBucketRateLimiter,
    RateLimiterConfig,
)
from truthound.common.resilience.protocols import (
    CircuitBreakerProtocol,
    RetryPolicyProtocol,
    BulkheadProtocol,
    RateLimiterProtocol,
)

logger = logging.getLogger(__name__)

R = TypeVar("R")


class ResilientWrapper:
    """Wrapper that applies multiple resilience patterns.

    Patterns are applied in order:
    1. Rate limiter (if configured)
    2. Bulkhead (if configured)
    3. Circuit breaker (if configured)
    4. Retry (if configured)
    5. Function execution

    Example:
        wrapper = ResilientWrapper(
            name="my-service",
            circuit_breaker=CircuitBreaker("cb"),
            retry=RetryPolicy(),
        )

        result = wrapper.execute(risky_function, arg1, arg2)
    """

    def __init__(
        self,
        name: str,
        circuit_breaker: CircuitBreakerProtocol | None = None,
        retry: RetryPolicyProtocol | None = None,
        bulkhead: BulkheadProtocol | None = None,
        rate_limiter: RateLimiterProtocol | None = None,
    ):
        """Initialize resilient wrapper.

        Args:
            name: Unique name for this wrapper.
            circuit_breaker: Optional circuit breaker.
            retry: Optional retry policy.
            bulkhead: Optional bulkhead.
            rate_limiter: Optional rate limiter.
        """
        self._name = name
        self._circuit_breaker = circuit_breaker
        self._retry = retry
        self._bulkhead = bulkhead
        self._rate_limiter = rate_limiter

    @property
    def name(self) -> str:
        """Get wrapper name."""
        return self._name

    def execute(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Execute function with all configured resilience patterns."""
        def wrapped() -> R:
            return func(*args, **kwargs)

        # Build execution chain from inside out using default arguments
        # to avoid late binding closure issues
        execution = wrapped

        # Innermost: circuit breaker
        if self._circuit_breaker:
            def with_cb(cb: CircuitBreakerProtocol = self._circuit_breaker, inner: Callable[[], R] = execution) -> R:
                with cb.protect():
                    return inner()
            execution = with_cb

        # Wrap with retry
        if self._retry:
            def with_retry(retry: RetryPolicyProtocol = self._retry, inner: Callable[[], R] = execution) -> R:
                return retry.execute(inner)
            execution = with_retry

        # Wrap with bulkhead
        if self._bulkhead:
            def with_bulkhead(bulkhead: BulkheadProtocol = self._bulkhead, inner: Callable[[], R] = execution) -> R:
                with bulkhead.limit():
                    return inner()
            execution = with_bulkhead

        # Outermost: rate limiter
        if self._rate_limiter:
            def with_limiter(limiter: RateLimiterProtocol = self._rate_limiter, inner: Callable[[], R] = execution) -> R:
                with limiter.limit():
                    return inner()
            execution = with_limiter

        return execution()

    @contextmanager
    def protect(self) -> Generator[None, None, None]:
        """Context manager for protected execution."""
        # Apply patterns in order
        cm_stack: list[Any] = []

        if self._rate_limiter:
            cm_stack.append(self._rate_limiter.limit())
        if self._bulkhead:
            cm_stack.append(self._bulkhead.limit())
        if self._circuit_breaker:
            cm_stack.append(self._circuit_breaker.protect())

        # Enter all context managers
        for cm in cm_stack:
            cm.__enter__()

        try:
            yield
        except Exception as e:
            # Exit with exception
            for cm in reversed(cm_stack):
                try:
                    cm.__exit__(type(e), e, e.__traceback__)
                except Exception:
                    pass
            raise
        else:
            # Exit normally
            for cm in reversed(cm_stack):
                cm.__exit__(None, None, None)

    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for resilient execution."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            return self.execute(func, *args, **kwargs)
        return wrapper

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics from all components."""
        metrics: dict[str, Any] = {"name": self._name}

        if self._circuit_breaker:
            metrics["circuit_breaker"] = self._circuit_breaker.get_metrics()
        if self._retry:
            metrics["retry"] = self._retry.get_metrics()
        if self._bulkhead:
            metrics["bulkhead"] = self._bulkhead.get_metrics()
        if self._rate_limiter:
            metrics["rate_limiter"] = self._rate_limiter.get_metrics()

        return metrics

    def reset(self) -> None:
        """Reset all components."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()
        if self._retry:
            self._retry.reset()
        if self._bulkhead:
            self._bulkhead.reset()
        if self._rate_limiter:
            self._rate_limiter.reset()


class ResilienceBuilder:
    """Fluent builder for creating resilient wrappers.

    Example:
        wrapper = (
            ResilienceBuilder("my-service")
            .with_circuit_breaker(CircuitBreakerConfig.aggressive())
            .with_retry(RetryConfig.exponential())
            .with_bulkhead(BulkheadConfig(max_concurrent=10))
            .with_rate_limit(RateLimiterConfig.per_second(100))
            .build()
        )

        @wrapper
        def risky_operation():
            return external_service.call()
    """

    def __init__(self, name: str):
        """Initialize builder.

        Args:
            name: Unique name for the wrapper.
        """
        self._name = name
        self._circuit_breaker: CircuitBreakerProtocol | None = None
        self._retry: RetryPolicyProtocol | None = None
        self._bulkhead: BulkheadProtocol | None = None
        self._rate_limiter: RateLimiterProtocol | None = None

    def with_circuit_breaker(
        self,
        config: CircuitBreakerConfig | None = None,
        instance: CircuitBreakerProtocol | None = None,
    ) -> "ResilienceBuilder":
        """Add circuit breaker to the wrapper.

        Args:
            config: Configuration for a new circuit breaker.
            instance: Existing circuit breaker instance.
        """
        if instance:
            self._circuit_breaker = instance
        else:
            self._circuit_breaker = CircuitBreaker(
                f"{self._name}-cb",
                config or CircuitBreakerConfig(),
            )
        return self

    def with_retry(
        self,
        config: RetryConfig | None = None,
        instance: RetryPolicyProtocol | None = None,
    ) -> "ResilienceBuilder":
        """Add retry policy to the wrapper.

        Args:
            config: Configuration for a new retry policy.
            instance: Existing retry policy instance.
        """
        if instance:
            self._retry = instance
        else:
            self._retry = RetryPolicy(config or RetryConfig())
        return self

    def with_bulkhead(
        self,
        config: BulkheadConfig | None = None,
        instance: BulkheadProtocol | None = None,
    ) -> "ResilienceBuilder":
        """Add bulkhead to the wrapper.

        Args:
            config: Configuration for a new bulkhead.
            instance: Existing bulkhead instance.
        """
        if instance:
            self._bulkhead = instance
        else:
            self._bulkhead = SemaphoreBulkhead(
                f"{self._name}-bulkhead",
                config or BulkheadConfig(),
            )
        return self

    def with_rate_limit(
        self,
        config: RateLimiterConfig | None = None,
        instance: RateLimiterProtocol | None = None,
    ) -> "ResilienceBuilder":
        """Add rate limiter to the wrapper.

        Args:
            config: Configuration for a new rate limiter.
            instance: Existing rate limiter instance.
        """
        if instance:
            self._rate_limiter = instance
        else:
            self._rate_limiter = TokenBucketRateLimiter(
                f"{self._name}-limiter",
                config or RateLimiterConfig(),
            )
        return self

    def build(self) -> ResilientWrapper:
        """Build the resilient wrapper."""
        return ResilientWrapper(
            name=self._name,
            circuit_breaker=self._circuit_breaker,
            retry=self._retry,
            bulkhead=self._bulkhead,
            rate_limiter=self._rate_limiter,
        )

    @classmethod
    def for_database(cls, name: str) -> ResilientWrapper:
        """Create a wrapper optimized for database operations."""
        return (
            cls(name)
            .with_circuit_breaker(CircuitBreakerConfig.for_database())
            .with_retry(RetryConfig.quick())
            .with_bulkhead(BulkheadConfig.for_database())
            .build()
        )

    @classmethod
    def for_external_api(cls, name: str) -> ResilientWrapper:
        """Create a wrapper optimized for external API calls."""
        return (
            cls(name)
            .with_circuit_breaker(CircuitBreakerConfig.for_external_api())
            .with_retry(RetryConfig.exponential())
            .with_rate_limit(RateLimiterConfig.per_second(100))
            .build()
        )

    @classmethod
    def simple(cls, name: str) -> ResilientWrapper:
        """Create a simple wrapper with default settings."""
        return (
            cls(name)
            .with_circuit_breaker()
            .with_retry()
            .build()
        )
