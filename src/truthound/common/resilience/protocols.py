"""Protocol definitions for resilience patterns.

This module defines the interfaces that all resilience components must implement,
enabling easy composition and substitution of implementations.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Generator, Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")
R = TypeVar("R")


@runtime_checkable
class ResilienceProtocol(Protocol):
    """Base protocol for all resilience patterns."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this resilience component."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the component to its initial state."""
        ...

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics for observability."""
        ...


@runtime_checkable
class CircuitBreakerProtocol(ResilienceProtocol, Protocol):
    """Protocol for circuit breaker implementations."""

    @abstractmethod
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        ...

    @abstractmethod
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting requests)."""
        ...

    @abstractmethod
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        ...

    @abstractmethod
    def record_success(self) -> None:
        """Record a successful call."""
        ...

    @abstractmethod
    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed call."""
        ...

    @abstractmethod
    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        ...

    @abstractmethod
    @contextmanager
    def protect(self) -> Generator[None, None, None]:
        """Context manager for protected execution."""
        ...

    @abstractmethod
    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for protected execution."""
        ...


@runtime_checkable
class RetryPolicyProtocol(ResilienceProtocol, Protocol):
    """Protocol for retry policy implementations."""

    @abstractmethod
    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Check if operation should be retried."""
        ...

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry in seconds."""
        ...

    @abstractmethod
    def execute(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Execute function with retry policy."""
        ...

    @abstractmethod
    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for retryable execution."""
        ...


@runtime_checkable
class BulkheadProtocol(ResilienceProtocol, Protocol):
    """Protocol for bulkhead (resource isolation) implementations."""

    @abstractmethod
    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a slot, returns False if bulkhead is full."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release a slot."""
        ...

    @abstractmethod
    def available_slots(self) -> int:
        """Get number of available slots."""
        ...

    @abstractmethod
    @contextmanager
    def limit(self, timeout: float | None = None) -> Generator[None, None, None]:
        """Context manager for bulkhead-limited execution."""
        ...

    @abstractmethod
    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for bulkhead-limited execution."""
        ...


@runtime_checkable
class RateLimiterProtocol(ResilienceProtocol, Protocol):
    """Protocol for rate limiter implementations."""

    @abstractmethod
    def acquire(self, permits: int = 1, timeout: float | None = None) -> bool:
        """Acquire permits, returns False if rate limit exceeded."""
        ...

    @abstractmethod
    def try_acquire(self, permits: int = 1) -> bool:
        """Try to acquire permits without blocking."""
        ...

    @abstractmethod
    def get_wait_time(self, permits: int = 1) -> float:
        """Get time to wait before permits are available."""
        ...

    @abstractmethod
    @contextmanager
    def limit(self, permits: int = 1) -> Generator[None, None, None]:
        """Context manager for rate-limited execution."""
        ...

    @abstractmethod
    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for rate-limited execution."""
        ...


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Protocol for health check implementations."""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the component is healthy."""
        ...

    @abstractmethod
    def get_health_status(self) -> dict[str, Any]:
        """Get detailed health status."""
        ...

    @abstractmethod
    def perform_check(self) -> bool:
        """Perform an active health check."""
        ...


class CompositeResilienceProtocol(ResilienceProtocol, Protocol):
    """Protocol for composite resilience patterns."""

    @abstractmethod
    def add_circuit_breaker(self, cb: CircuitBreakerProtocol) -> "CompositeResilienceProtocol":
        """Add circuit breaker to the chain."""
        ...

    @abstractmethod
    def add_retry(self, retry: RetryPolicyProtocol) -> "CompositeResilienceProtocol":
        """Add retry policy to the chain."""
        ...

    @abstractmethod
    def add_bulkhead(self, bulkhead: BulkheadProtocol) -> "CompositeResilienceProtocol":
        """Add bulkhead to the chain."""
        ...

    @abstractmethod
    def add_rate_limiter(self, limiter: RateLimiterProtocol) -> "CompositeResilienceProtocol":
        """Add rate limiter to the chain."""
        ...

    @abstractmethod
    def execute(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Execute with all configured resilience patterns."""
        ...
