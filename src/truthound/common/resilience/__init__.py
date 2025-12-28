"""Unified resilience patterns for Truthound.

This module provides fault-tolerance patterns that can be used across
all Truthound components (profiler, stores, realtime, checkpoint).

Key patterns:
- CircuitBreaker: Prevents cascade failures
- RetryPolicy: Configurable retry with exponential backoff
- Bulkhead: Resource isolation
- RateLimiter: Request rate limiting

Example:
    from truthound.common.resilience import (
        CircuitBreaker,
        CircuitBreakerConfig,
        RetryPolicy,
        RetryConfig,
    )

    # Using circuit breaker as decorator
    cb = CircuitBreaker("my-service")

    @cb
    def risky_operation():
        return external_service.call()

    # Using circuit breaker as context manager
    with cb.protect():
        result = external_service.call()

    # Using retry policy
    retry = RetryPolicy(RetryConfig.exponential())

    @retry
    def flaky_operation():
        return unreliable_service.call()
"""

from truthound.common.resilience.protocols import (
    ResilienceProtocol,
    CircuitBreakerProtocol,
    RetryPolicyProtocol,
    BulkheadProtocol,
    RateLimiterProtocol,
    HealthCheckProtocol,
)
from truthound.common.resilience.config import (
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    RateLimiterConfig,
)
from truthound.common.resilience.circuit_breaker import (
    CircuitState,
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    circuit_breaker_registry,
    get_circuit_breaker,
)
from truthound.common.resilience.retry import (
    RetryPolicy,
    RetryExhaustedError,
    BackoffStrategy,
    ExponentialBackoff,
    LinearBackoff,
    ConstantBackoff,
    JitteredBackoff,
)
from truthound.common.resilience.bulkhead import (
    Bulkhead,
    BulkheadFullError,
    SemaphoreBulkhead,
    ThreadPoolBulkhead,
)
from truthound.common.resilience.rate_limiter import (
    RateLimiter,
    RateLimitExceededError,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    FixedWindowRateLimiter,
)
from truthound.common.resilience.composite import (
    ResilienceBuilder,
    ResilientWrapper,
)

__all__ = [
    # Protocols
    "ResilienceProtocol",
    "CircuitBreakerProtocol",
    "RetryPolicyProtocol",
    "BulkheadProtocol",
    "RateLimiterProtocol",
    "HealthCheckProtocol",
    # Config
    "CircuitBreakerConfig",
    "RetryConfig",
    "BulkheadConfig",
    "RateLimiterConfig",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "circuit_breaker_registry",
    "get_circuit_breaker",
    # Retry
    "RetryPolicy",
    "RetryExhaustedError",
    "BackoffStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "JitteredBackoff",
    # Bulkhead
    "Bulkhead",
    "BulkheadFullError",
    "SemaphoreBulkhead",
    "ThreadPoolBulkhead",
    # Rate Limiter
    "RateLimiter",
    "RateLimitExceededError",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "FixedWindowRateLimiter",
    # Composite
    "ResilienceBuilder",
    "ResilientWrapper",
]
