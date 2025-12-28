"""Configuration classes for resilience patterns.

This module provides dataclass-based configuration for all resilience patterns,
with factory methods for common presets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit.
        success_threshold: Number of successes in half-open to close circuit.
        timeout_seconds: Time in open state before transitioning to half-open.
        half_open_max_calls: Maximum calls allowed in half-open state.
        failure_rate_threshold: Failure rate percentage to trigger open.
        slow_call_threshold_ms: Latency threshold for slow calls.
        slow_call_rate_threshold: Slow call rate percentage to trigger open.
        window_size: Number of calls to track for rate calculations.
        excluded_exceptions: Exceptions that don't count as failures.
        record_slow_calls: Whether to track slow calls.
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3
    failure_rate_threshold: float = 50.0
    slow_call_threshold_ms: float = 1000.0
    slow_call_rate_threshold: float = 50.0
    window_size: int = 100
    excluded_exceptions: tuple[type[Exception], ...] = ()
    record_slow_calls: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if not 0 <= self.failure_rate_threshold <= 100:
            raise ValueError("failure_rate_threshold must be between 0 and 100")
        if not 0 <= self.slow_call_rate_threshold <= 100:
            raise ValueError("slow_call_rate_threshold must be between 0 and 100")

    @classmethod
    def aggressive(cls) -> "CircuitBreakerConfig":
        """Aggressive config - opens quickly, recovers slowly."""
        return cls(
            failure_threshold=3,
            success_threshold=3,
            timeout_seconds=60.0,
            failure_rate_threshold=30.0,
        )

    @classmethod
    def lenient(cls) -> "CircuitBreakerConfig":
        """Lenient config - tolerates more failures."""
        return cls(
            failure_threshold=10,
            success_threshold=1,
            timeout_seconds=15.0,
            failure_rate_threshold=80.0,
        )

    @classmethod
    def disabled(cls) -> "CircuitBreakerConfig":
        """Effectively disabled circuit breaker."""
        return cls(
            failure_threshold=1_000_000,
            timeout_seconds=0.1,
        )

    @classmethod
    def for_database(cls) -> "CircuitBreakerConfig":
        """Optimized for database connections."""
        return cls(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=30.0,
            slow_call_threshold_ms=5000.0,
        )

    @classmethod
    def for_external_api(cls) -> "CircuitBreakerConfig":
        """Optimized for external API calls."""
        return cls(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=60.0,
            slow_call_threshold_ms=2000.0,
        )


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry policy.

    Attributes:
        max_attempts: Maximum number of attempts (1 = no retry).
        base_delay: Base delay in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff.
        jitter: Whether to add random jitter to delays.
        jitter_factor: Maximum jitter as a fraction (0.0-1.0).
        retryable_exceptions: Exceptions that trigger retry.
        non_retryable_exceptions: Exceptions that should not be retried.
    """

    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.5
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )
    non_retryable_exceptions: tuple[type[Exception], ...] = (
        ValueError,
        TypeError,
        KeyError,
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.exponential_base < 1:
            raise ValueError("exponential_base must be >= 1")
        if not 0 <= self.jitter_factor <= 1:
            raise ValueError("jitter_factor must be between 0 and 1")

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)."""
        import random

        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay

    def is_retryable(self, error: Exception) -> bool:
        """Check if the error should trigger a retry."""
        if isinstance(error, self.non_retryable_exceptions):
            return False
        if self.retryable_exceptions:
            return isinstance(error, self.retryable_exceptions)
        return True

    @classmethod
    def no_retry(cls) -> "RetryConfig":
        """No retry - fail immediately."""
        return cls(max_attempts=1)

    @classmethod
    def quick(cls) -> "RetryConfig":
        """Quick retry for transient failures."""
        return cls(
            max_attempts=3,
            base_delay=0.05,
            max_delay=1.0,
        )

    @classmethod
    def persistent(cls) -> "RetryConfig":
        """Persistent retry for important operations."""
        return cls(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
        )

    @classmethod
    def exponential(cls) -> "RetryConfig":
        """Standard exponential backoff."""
        return cls(
            max_attempts=4,
            base_delay=0.1,
            max_delay=10.0,
            exponential_base=2.0,
        )


@dataclass(frozen=True)
class BulkheadConfig:
    """Configuration for bulkhead (resource isolation).

    Attributes:
        max_concurrent: Maximum concurrent executions.
        max_wait_time: Maximum time to wait for a slot in seconds.
        fairness: Whether to use fair ordering for waiting requests.
    """

    max_concurrent: int = 10
    max_wait_time: float = 0.0
    fairness: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if self.max_wait_time < 0:
            raise ValueError("max_wait_time must be non-negative")

    @classmethod
    def small(cls) -> "BulkheadConfig":
        """Small bulkhead for limited resources."""
        return cls(max_concurrent=5)

    @classmethod
    def medium(cls) -> "BulkheadConfig":
        """Medium bulkhead for moderate concurrency."""
        return cls(max_concurrent=20)

    @classmethod
    def large(cls) -> "BulkheadConfig":
        """Large bulkhead for high concurrency."""
        return cls(max_concurrent=50)

    @classmethod
    def for_database(cls) -> "BulkheadConfig":
        """Optimized for database connection pools."""
        return cls(max_concurrent=10, max_wait_time=5.0)


@dataclass(frozen=True)
class RateLimiterConfig:
    """Configuration for rate limiter.

    Attributes:
        rate: Number of permits per period.
        period_seconds: Period duration in seconds.
        burst_size: Maximum burst size (for token bucket).
        algorithm: Rate limiting algorithm ('token_bucket', 'sliding_window', 'fixed_window').
    """

    rate: int = 100
    period_seconds: float = 1.0
    burst_size: int | None = None
    algorithm: str = "token_bucket"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        if self.period_seconds <= 0:
            raise ValueError("period_seconds must be positive")
        if self.burst_size is not None and self.burst_size < 0:
            raise ValueError("burst_size must be non-negative")
        valid_algorithms = {"token_bucket", "sliding_window", "fixed_window"}
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"algorithm must be one of {valid_algorithms}")

    @property
    def effective_burst_size(self) -> int:
        """Get effective burst size (defaults to rate if not set)."""
        return self.burst_size if self.burst_size is not None else self.rate

    @classmethod
    def per_second(cls, rate: int, burst: int | None = None) -> "RateLimiterConfig":
        """Create rate limiter for N requests per second."""
        return cls(rate=rate, period_seconds=1.0, burst_size=burst)

    @classmethod
    def per_minute(cls, rate: int, burst: int | None = None) -> "RateLimiterConfig":
        """Create rate limiter for N requests per minute."""
        return cls(rate=rate, period_seconds=60.0, burst_size=burst)

    @classmethod
    def per_hour(cls, rate: int, burst: int | None = None) -> "RateLimiterConfig":
        """Create rate limiter for N requests per hour."""
        return cls(rate=rate, period_seconds=3600.0, burst_size=burst)
