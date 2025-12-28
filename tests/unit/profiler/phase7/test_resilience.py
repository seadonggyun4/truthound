"""Tests for unified resilience patterns."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock, patch

import pytest

from truthound.common.resilience import (
    # Config
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    RateLimiterConfig,
    # Circuit Breaker
    CircuitState,
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    get_circuit_breaker,
    # Retry
    RetryPolicy,
    RetryExhaustedError,
    ExponentialBackoff,
    LinearBackoff,
    ConstantBackoff,
    JitteredBackoff,
    # Bulkhead
    SemaphoreBulkhead,
    BulkheadFullError,
    # Rate Limiter
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    FixedWindowRateLimiter,
    RateLimitExceededError,
    # Composite
    ResilienceBuilder,
    ResilientWrapper,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 30.0

    def test_aggressive_preset(self) -> None:
        """Test aggressive preset."""
        config = CircuitBreakerConfig.aggressive()

        assert config.failure_threshold == 3
        assert config.success_threshold == 3
        assert config.timeout_seconds == 60.0

    def test_lenient_preset(self) -> None:
        """Test lenient preset."""
        config = CircuitBreakerConfig.lenient()

        assert config.failure_threshold == 10
        assert config.success_threshold == 1

    def test_validation(self) -> None:
        """Test configuration validation."""
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)

        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_rate_threshold=101)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_is_closed(self) -> None:
        """Test that circuit starts closed."""
        cb = CircuitBreaker("test")
        assert cb.is_closed()
        assert not cb.is_open()

    def test_opens_after_failures(self) -> None:
        """Test that circuit opens after enough failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.is_open()

    def test_rejects_requests_when_open(self) -> None:
        """Test that open circuit rejects requests."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.is_open()
        assert not cb.allow_request()

    def test_half_open_after_timeout(self) -> None:
        """Test transition to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.is_open()

        # Wait for timeout
        time.sleep(0.15)

        assert cb.allow_request()  # This should trigger half-open
        assert cb.is_half_open()

    def test_closes_after_successful_half_open(self) -> None:
        """Test circuit closes after successful half-open calls."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        time.sleep(0.15)

        # First success
        cb.allow_request()
        cb.record_success()

        # Second success - should close
        cb.record_success()
        assert cb.is_closed()

    def test_decorator_usage(self) -> None:
        """Test using circuit breaker as decorator."""
        cb = CircuitBreaker("test")
        call_count = 0

        @cb
        def my_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = my_function()

        assert result == "success"
        assert call_count == 1

    def test_context_manager_usage(self) -> None:
        """Test using circuit breaker as context manager."""
        cb = CircuitBreaker("test")

        with cb.protect():
            result = "success"

        assert result == "success"

    def test_context_manager_records_failure(self) -> None:
        """Test that context manager records failures."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        with pytest.raises(ValueError):
            with cb.protect():
                raise ValueError("test error")

        assert cb.is_open()

    def test_raises_when_open(self) -> None:
        """Test that protect raises when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()

        with pytest.raises(CircuitOpenError):
            with cb.protect():
                pass

    def test_metrics(self) -> None:
        """Test metrics collection."""
        cb = CircuitBreaker("test")

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        metrics = cb.get_metrics()

        assert metrics["total_calls"] == 3
        assert metrics["successful_calls"] == 2
        assert metrics["failed_calls"] == 1

    def test_reset(self) -> None:
        """Test resetting circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.is_open()

        cb.reset()
        assert cb.is_closed()


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create(self) -> None:
        """Test getting or creating a circuit breaker."""
        registry = CircuitBreakerRegistry()

        cb = registry.get_or_create("test")
        assert cb is not None
        assert cb.name == "test"

        # Same name returns same instance
        cb2 = registry.get_or_create("test")
        assert cb is cb2

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent circuit breaker."""
        registry = CircuitBreakerRegistry()

        assert registry.get("nonexistent") is None

    def test_reset_all(self) -> None:
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("test1", CircuitBreakerConfig(failure_threshold=1))
        cb2 = registry.get_or_create("test2", CircuitBreakerConfig(failure_threshold=1))

        cb1.record_failure()
        cb2.record_failure()

        registry.reset_all()

        assert cb1.is_closed()
        assert cb2.is_closed()


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_successful_execution(self) -> None:
        """Test successful execution without retry."""
        policy = RetryPolicy()

        result = policy.execute(lambda: "success")

        assert result == "success"

    def test_retry_on_failure(self) -> None:
        """Test retrying on failure."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)

        attempt = 0

        def flaky_function():
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                raise ConnectionError("transient error")
            return "success"

        result = policy.execute(flaky_function)

        assert result == "success"
        assert attempt == 3

    def test_exhausted_retries(self) -> None:
        """Test that RetryExhaustedError is raised after max attempts."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        policy = RetryPolicy(config)

        def always_fails():
            raise ConnectionError("permanent error")

        with pytest.raises(RetryExhaustedError) as exc_info:
            policy.execute(always_fails)

        assert exc_info.value.attempts == 2

    def test_non_retryable_exception(self) -> None:
        """Test that non-retryable exceptions are not retried."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=(ConnectionError,),
            base_delay=0.01,
        )
        policy = RetryPolicy(config)

        attempt = 0

        def fails_with_value_error():
            nonlocal attempt
            attempt += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            policy.execute(fails_with_value_error)

        assert attempt == 1  # Should not retry

    def test_decorator_usage(self) -> None:
        """Test using retry policy as decorator."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        policy = RetryPolicy(config)

        @policy
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"


class TestBackoffStrategies:
    """Tests for backoff strategies."""

    def test_exponential_backoff(self) -> None:
        """Test exponential backoff calculation."""
        backoff = ExponentialBackoff(base_delay=0.1, multiplier=2.0)

        assert backoff.get_delay(0) == 0.1
        assert backoff.get_delay(1) == 0.2
        assert backoff.get_delay(2) == 0.4

    def test_linear_backoff(self) -> None:
        """Test linear backoff calculation."""
        backoff = LinearBackoff(base_delay=0.1, increment=0.5)

        assert backoff.get_delay(0) == 0.1
        assert backoff.get_delay(1) == 0.6
        assert backoff.get_delay(2) == 1.1

    def test_constant_backoff(self) -> None:
        """Test constant backoff."""
        backoff = ConstantBackoff(delay=1.0)

        assert backoff.get_delay(0) == 1.0
        assert backoff.get_delay(5) == 1.0

    def test_jittered_backoff(self) -> None:
        """Test jittered backoff adds randomness."""
        base = ConstantBackoff(delay=1.0)
        backoff = JitteredBackoff(base=base, jitter_factor=0.5)

        delays = [backoff.get_delay(0) for _ in range(10)]

        # Delays should vary
        assert len(set(delays)) > 1


class TestBulkhead:
    """Tests for bulkhead implementations."""

    def test_semaphore_bulkhead_limits_concurrency(self) -> None:
        """Test that semaphore bulkhead limits concurrency."""
        config = BulkheadConfig(max_concurrent=2)
        bulkhead = SemaphoreBulkhead("test", config)

        acquired = []

        def try_acquire():
            result = bulkhead.acquire(timeout=0.1)
            acquired.append(result)
            if result:
                time.sleep(0.2)
                bulkhead.release()

        threads = [threading.Thread(target=try_acquire) for _ in range(4)]
        for t in threads:
            t.start()

        # Give some time for first batch
        time.sleep(0.05)

        # Only 2 should have acquired initially
        initial_acquired = sum(1 for r in acquired if r)
        assert initial_acquired <= 2

        for t in threads:
            t.join()

    def test_bulkhead_decorator(self) -> None:
        """Test using bulkhead as decorator."""
        config = BulkheadConfig(max_concurrent=2)
        bulkhead = SemaphoreBulkhead("test", config)

        @bulkhead
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_bulkhead_raises_when_full(self) -> None:
        """Test that bulkhead raises when full."""
        config = BulkheadConfig(max_concurrent=1, max_wait_time=0)
        bulkhead = SemaphoreBulkhead("test", config)

        # Acquire the only slot
        bulkhead.acquire()

        with pytest.raises(BulkheadFullError):
            with bulkhead.limit():
                pass

        bulkhead.release()


class TestRateLimiter:
    """Tests for rate limiter implementations."""

    def test_token_bucket_allows_within_limit(self) -> None:
        """Test that token bucket allows requests within limit."""
        config = RateLimiterConfig(rate=10, period_seconds=1.0)
        limiter = TokenBucketRateLimiter("test", config)

        # Should allow requests
        for _ in range(10):
            assert limiter.try_acquire()

    def test_token_bucket_rejects_over_limit(self) -> None:
        """Test that token bucket rejects over limit."""
        config = RateLimiterConfig(rate=2, period_seconds=1.0)
        limiter = TokenBucketRateLimiter("test", config)

        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert not limiter.try_acquire()  # Should fail

    def test_rate_limiter_decorator(self) -> None:
        """Test using rate limiter as decorator."""
        config = RateLimiterConfig(rate=10)
        limiter = TokenBucketRateLimiter("test", config)

        @limiter
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_fixed_window_resets(self) -> None:
        """Test that fixed window resets after period."""
        config = RateLimiterConfig(rate=2, period_seconds=0.1)
        limiter = FixedWindowRateLimiter("test", config)

        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert not limiter.try_acquire()

        # Wait for window reset
        time.sleep(0.15)

        assert limiter.try_acquire()


class TestResilienceBuilder:
    """Tests for ResilienceBuilder."""

    def test_build_simple_wrapper(self) -> None:
        """Test building a simple wrapper."""
        wrapper = ResilienceBuilder("test").build()

        assert wrapper.name == "test"

    def test_build_with_circuit_breaker(self) -> None:
        """Test building wrapper with circuit breaker."""
        wrapper = (
            ResilienceBuilder("test")
            .with_circuit_breaker()
            .build()
        )

        result = wrapper.execute(lambda: "success")
        assert result == "success"

    def test_build_with_retry(self) -> None:
        """Test building wrapper with retry."""
        wrapper = (
            ResilienceBuilder("test")
            .with_retry(RetryConfig(max_attempts=2, base_delay=0.01))
            .build()
        )

        result = wrapper.execute(lambda: "success")
        assert result == "success"

    def test_build_full_chain(self) -> None:
        """Test building full resilience chain."""
        wrapper = (
            ResilienceBuilder("test")
            .with_rate_limit()
            .with_bulkhead()
            .with_circuit_breaker()
            .with_retry()
            .build()
        )

        result = wrapper.execute(lambda: "success")
        assert result == "success"

    def test_preset_for_database(self) -> None:
        """Test database preset."""
        wrapper = ResilienceBuilder.for_database("db")

        result = wrapper.execute(lambda: "success")
        assert result == "success"

    def test_wrapper_as_decorator(self) -> None:
        """Test using wrapper as decorator."""
        wrapper = ResilienceBuilder.simple("test")

        @wrapper
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_wrapper_metrics(self) -> None:
        """Test wrapper metrics collection."""
        wrapper = (
            ResilienceBuilder("test")
            .with_circuit_breaker()
            .with_retry()
            .build()
        )

        wrapper.execute(lambda: "success")

        metrics = wrapper.get_metrics()
        assert "circuit_breaker" in metrics
        assert "retry" in metrics
