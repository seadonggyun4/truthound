"""Tests for cache resilience and fallback patterns.

This module tests:
- Circuit breaker behavior
- Retry logic with exponential backoff
- Health monitoring
- Resilient cache backend with fallback
- Fallback chain
- Factory functions
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound.profiler.caching import (
    CacheBackend,
    CacheEntry,
    MemoryCacheBackend,
    FileCacheBackend,
)
from truthound.profiler.base import TableProfile
from truthound.profiler.resilience import (
    # States and types
    CircuitState,
    BackendHealth,
    FailureType,
    # Configuration
    CircuitBreakerConfig,
    RetryConfig,
    HealthCheckConfig,
    ResilienceConfig,
    # Circuit breaker
    CircuitBreaker,
    CircuitOpenError,
    # Retry logic
    RetryPolicy,
    # Health monitoring
    HealthMonitor,
    # Resilient backends
    ResilientCacheBackend,
    FallbackChain,
    # Factory functions
    create_resilient_redis_backend,
    create_high_availability_cache,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class FailingCacheBackend(CacheBackend):
    """A cache backend that fails on demand for testing."""

    def __init__(self, fail_count: int = 0, fail_exception: type[Exception] = ConnectionError):
        self.fail_count = fail_count
        self.fail_exception = fail_exception
        self._calls = 0
        self._storage: dict[str, CacheEntry] = {}

    def _maybe_fail(self) -> None:
        self._calls += 1
        if self._calls <= self.fail_count:
            raise self.fail_exception(f"Simulated failure #{self._calls}")

    def get(self, key: str) -> CacheEntry | None:
        self._maybe_fail()
        return self._storage.get(key)

    def set(self, key: str, entry: CacheEntry, ttl: timedelta | None = None) -> None:
        self._maybe_fail()
        self._storage[key] = entry

    def delete(self, key: str) -> bool:
        self._maybe_fail()
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    def clear(self) -> int:
        self._maybe_fail()
        count = len(self._storage)
        self._storage.clear()
        return count

    def exists(self, key: str) -> bool:
        self._maybe_fail()
        return key in self._storage

    def get_stats(self) -> dict[str, Any]:
        return {"type": "failing", "calls": self._calls}


class CountingCacheBackend(CacheBackend):
    """A cache backend that counts operations for testing."""

    def __init__(self):
        self._storage: dict[str, CacheEntry] = {}
        self.get_count = 0
        self.set_count = 0
        self.delete_count = 0
        self.exists_count = 0

    def get(self, key: str) -> CacheEntry | None:
        self.get_count += 1
        return self._storage.get(key)

    def set(self, key: str, entry: CacheEntry, ttl: timedelta | None = None) -> None:
        self.set_count += 1
        self._storage[key] = entry

    def delete(self, key: str) -> bool:
        self.delete_count += 1
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    def clear(self) -> int:
        count = len(self._storage)
        self._storage.clear()
        return count

    def exists(self, key: str) -> bool:
        self.exists_count += 1
        return key in self._storage

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "counting",
            "get_count": self.get_count,
            "set_count": self.set_count,
        }


@pytest.fixture
def sample_profile() -> TableProfile:
    """Create a sample table profile for testing."""
    return TableProfile(
        name="test-table",
        row_count=100,
        column_count=5,
    )


@pytest.fixture
def sample_entry(sample_profile: TableProfile) -> CacheEntry:
    """Create a sample cache entry for testing."""
    return CacheEntry(
        profile=sample_profile,
        created_at=datetime.now(),
    )


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in closed state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_success_keeps_circuit_closed(self):
        """Successful operations keep circuit closed."""
        breaker = CircuitBreaker()
        for _ in range(10):
            breaker.record_success()
        assert breaker.is_closed

    def test_circuit_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        for i in range(3):
            breaker.record_failure(ConnectionError(f"Failure {i+1}"))

        assert breaker.is_open
        assert not breaker.is_closed

    def test_circuit_rejects_when_open(self):
        """Open circuit rejects execution."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)
        breaker.record_failure(ConnectionError("Failure"))

        assert not breaker.can_execute()

    def test_circuit_recovers_to_half_open(self):
        """Circuit transitions to half-open after recovery timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure(ConnectionError("Failure"))
        assert breaker.is_open

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        """Half-open circuit closes after successful operations."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            recovery_timeout=0.1,
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure(ConnectionError("Failure"))

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        breaker.record_success()
        assert breaker.is_closed

    def test_half_open_opens_on_failure(self):
        """Half-open circuit opens again on failure."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
        )
        breaker = CircuitBreaker(config)
        breaker.record_failure(ConnectionError("Failure"))

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_failure(ConnectionError("Another failure"))
        assert breaker.is_open

    def test_excluded_exceptions_dont_trigger_circuit(self):
        """Excluded exceptions don't count toward failure threshold."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker(config)

        # ValueError is excluded
        for _ in range(5):
            breaker.record_failure(ValueError("Excluded"))
        assert breaker.is_closed

        # But ConnectionError isn't
        breaker.record_failure(ConnectionError("Not excluded"))
        breaker.record_failure(ConnectionError("Not excluded"))
        assert breaker.is_open

    def test_protect_decorator(self):
        """Protect decorator wraps function with circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        call_count = 0

        @breaker.protect
        def operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("Failure")
            return "success"

        # First two calls fail, opening the circuit
        with pytest.raises(ConnectionError):
            operation()
        with pytest.raises(ConnectionError):
            operation()

        # Next call is rejected by circuit
        with pytest.raises(CircuitOpenError):
            operation()

    def test_reset_circuit(self):
        """Manual reset closes the circuit."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)
        breaker.record_failure(ConnectionError("Failure"))
        assert breaker.is_open

        breaker.reset()
        assert breaker.is_closed

    def test_get_stats(self):
        """Statistics are tracked correctly."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker(config)

        breaker.record_success()
        breaker.record_success()
        breaker.record_failure(ConnectionError("Fail"))

        stats = breaker.get_stats()
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 3
        assert stats["total_failures"] == 1
        assert stats["recent_failures"] == 1

    def test_preset_configs(self):
        """Preset configurations work correctly."""
        aggressive = CircuitBreakerConfig.aggressive()
        assert aggressive.failure_threshold == 3
        assert aggressive.recovery_timeout == 60.0

        lenient = CircuitBreakerConfig.lenient()
        assert lenient.failure_threshold == 10
        assert lenient.recovery_timeout == 15.0

        disabled = CircuitBreakerConfig.disabled()
        assert disabled.failure_threshold == 1000000


# =============================================================================
# Retry Policy Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_no_retry_on_success(self):
        """Successful operations don't retry."""
        policy = RetryPolicy()
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = policy.execute_with_retry(operation)
        assert result == "success"
        assert call_count == 1

    def test_retry_on_retryable_exception(self):
        """Retries on retryable exceptions."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        policy = RetryPolicy(config)
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        result = policy.execute_with_retry(operation)
        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retryable_exception(self):
        """Doesn't retry on non-retryable exceptions."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=(ConnectionError,),
        )
        policy = RetryPolicy(config)

        def operation():
            raise ValueError("Non-retryable")

        with pytest.raises(ValueError):
            policy.execute_with_retry(operation)

    def test_max_attempts_respected(self):
        """Stops retrying after max attempts."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            policy.execute_with_retry(operation)

        assert call_count == 3

    def test_exponential_backoff_delay(self):
        """Delay increases exponentially."""
        config = RetryConfig(
            max_attempts=4,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_max_delay_cap(self):
        """Delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=10.0,
            jitter=False,
        )

        # 1 * 10^5 = 100000, but capped at 5
        assert config.calculate_delay(5) == 5.0

    def test_jitter_adds_randomness(self):
        """Jitter adds randomness to delays."""
        config = RetryConfig(
            base_delay=1.0,
            jitter=True,
        )

        delays = [config.calculate_delay(0) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1

    def test_retry_decorator(self):
        """Retry decorator works correctly."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)
        call_count = 0

        @policy.retry
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Flaky")
            return "success"

        result = flaky_operation()
        assert result == "success"
        assert call_count == 2

    def test_get_stats(self):
        """Statistics are tracked correctly."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        policy = RetryPolicy(config)
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Flaky")
            return "success"

        policy.execute_with_retry(operation)

        stats = policy.get_stats()
        assert stats["total_attempts"] == 2
        assert stats["total_retries"] == 1

    def test_preset_configs(self):
        """Preset configurations work correctly."""
        no_retry = RetryConfig.no_retry()
        assert no_retry.max_attempts == 1

        quick = RetryConfig.quick()
        assert quick.max_attempts == 3
        assert quick.base_delay == 0.05

        persistent = RetryConfig.persistent()
        assert persistent.max_attempts == 5


# =============================================================================
# Health Monitor Tests
# =============================================================================


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    def test_initial_health_is_unknown(self):
        """Health starts as unknown."""
        backend = MemoryCacheBackend()
        monitor = HealthMonitor(backend)
        assert monitor.health == BackendHealth.UNKNOWN

    def test_health_check_success(self):
        """Successful health check updates status."""
        backend = MemoryCacheBackend()
        config = HealthCheckConfig(healthy_threshold=1)
        monitor = HealthMonitor(backend, config)

        result = monitor.check_health()
        assert result is True
        assert monitor.health == BackendHealth.HEALTHY

    def test_health_check_failure(self):
        """Failed health check updates status."""
        backend = FailingCacheBackend(fail_count=10)
        config = HealthCheckConfig(unhealthy_threshold=1)
        monitor = HealthMonitor(backend, config)

        result = monitor.check_health()
        assert result is False
        assert monitor.health == BackendHealth.UNHEALTHY

    def test_degraded_state(self):
        """Monitors degraded state correctly."""
        backend = FailingCacheBackend(fail_count=1)
        config = HealthCheckConfig(unhealthy_threshold=3)
        monitor = HealthMonitor(backend, config)

        # First check fails but doesn't reach unhealthy threshold
        result = monitor.check_health()
        assert result is False
        assert monitor.health == BackendHealth.DEGRADED

    def test_recovery_from_unhealthy(self):
        """Monitors recovery from unhealthy state."""
        backend = FailingCacheBackend(fail_count=3)
        config = HealthCheckConfig(
            unhealthy_threshold=1,
            healthy_threshold=2,
        )
        monitor = HealthMonitor(backend, config)

        # Fail to unhealthy
        monitor.check_health()
        monitor.check_health()
        monitor.check_health()
        assert monitor.health == BackendHealth.UNHEALTHY

        # Next checks succeed
        monitor.check_health()  # First success -> degraded
        assert monitor.health == BackendHealth.DEGRADED

        monitor.check_health()  # Second success -> healthy
        assert monitor.health == BackendHealth.HEALTHY

    def test_background_monitoring(self):
        """Background monitoring runs correctly."""
        backend = MemoryCacheBackend()
        config = HealthCheckConfig(
            check_interval=0.05,
            healthy_threshold=1,
            enabled=True,
        )
        monitor = HealthMonitor(backend, config)

        monitor.start()
        time.sleep(0.2)  # Allow a few checks
        monitor.stop()

        assert monitor.health == BackendHealth.HEALTHY
        stats = monitor.get_stats()
        assert stats["total_checks"] >= 2

    def test_disabled_monitoring(self):
        """Disabled monitoring doesn't start."""
        backend = MemoryCacheBackend()
        config = HealthCheckConfig(enabled=False)
        monitor = HealthMonitor(backend, config)

        monitor.start()
        assert not monitor._running

    def test_get_stats(self):
        """Statistics are tracked correctly."""
        backend = MemoryCacheBackend()
        config = HealthCheckConfig(healthy_threshold=1)
        monitor = HealthMonitor(backend, config, name="test-backend")

        monitor.check_health()

        stats = monitor.get_stats()
        assert stats["name"] == "test-backend"
        assert stats["health"] == "healthy"
        assert stats["total_checks"] == 1
        assert stats["consecutive_successes"] == 1


# =============================================================================
# Resilient Cache Backend Tests
# =============================================================================


class TestResilientCacheBackend:
    """Tests for ResilientCacheBackend."""

    def test_uses_primary_when_healthy(self, sample_entry):
        """Uses primary backend when healthy."""
        primary = CountingCacheBackend()
        fallback = CountingCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )

        cache.set("key", sample_entry)
        cache.get("key")

        assert primary.set_count == 1
        assert primary.get_count == 1
        assert fallback.set_count == 0
        assert fallback.get_count == 0

        cache.shutdown()

    def test_falls_back_on_primary_failure(self, sample_entry):
        """Falls back to secondary when primary fails."""
        primary = FailingCacheBackend(fail_count=10)
        fallback = CountingCacheBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            retry=RetryConfig(max_attempts=1),
            health_check=HealthCheckConfig(enabled=False),
            fallback_on_error=True,
        )

        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=config,
        )

        cache.set("key", sample_entry)

        # Primary failed, fallback used
        assert fallback.set_count == 1

        cache.shutdown()

    def test_circuit_breaker_integration(self, sample_entry):
        """Circuit breaker prevents calls to failing primary."""
        primary = FailingCacheBackend(fail_count=100)
        fallback = CountingCacheBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=60.0,
            ),
            retry=RetryConfig(max_attempts=1),
            health_check=HealthCheckConfig(enabled=False),
        )

        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=config,
        )

        # Trigger failures to open circuit
        cache.set("key1", sample_entry)
        cache.set("key2", sample_entry)

        # Now circuit is open, primary shouldn't be called
        initial_calls = primary._calls
        cache.set("key3", sample_entry)
        cache.set("key4", sample_entry)

        # Primary calls shouldn't increase (circuit is open)
        assert primary._calls == initial_calls

        cache.shutdown()

    def test_retry_integration(self, sample_entry):
        """Retry logic retries transient failures."""
        primary = FailingCacheBackend(fail_count=2)
        fallback = CountingCacheBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=10),
            retry=RetryConfig(
                max_attempts=3,
                base_delay=0.01,
            ),
            health_check=HealthCheckConfig(enabled=False),
        )

        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=config,
        )

        cache.set("key", sample_entry)

        # Retry succeeded on 3rd attempt
        assert primary._calls == 3
        assert fallback.set_count == 0

        cache.shutdown()

    def test_get_returns_none_on_miss(self):
        """Get returns None for missing keys."""
        primary = MemoryCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )

        result = cache.get("nonexistent")
        assert result is None

        cache.shutdown()

    def test_exists_operation(self, sample_entry):
        """Exists operation works correctly."""
        primary = MemoryCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )

        assert not cache.exists("key")
        cache.set("key", sample_entry)
        assert cache.exists("key")

        cache.shutdown()

    def test_delete_operation(self, sample_entry):
        """Delete operation works correctly."""
        primary = MemoryCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )

        cache.set("key", sample_entry)
        assert cache.exists("key")

        result = cache.delete("key")
        assert result is True
        assert not cache.exists("key")

        cache.shutdown()

    def test_clear_operation(self, sample_entry):
        """Clear operation works on both backends."""
        primary = MemoryCacheBackend()
        fallback = MemoryCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )

        cache.set("key", sample_entry)
        fallback.set("fallback-key", sample_entry, None)

        count = cache.clear()
        assert count >= 1

        cache.shutdown()

    def test_get_stats(self, sample_entry):
        """Statistics are tracked comprehensively."""
        primary = CountingCacheBackend()
        fallback = CountingCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            name="test-cache",
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )

        cache.set("key", sample_entry)
        cache.get("key")

        stats = cache.get_stats()
        assert stats["type"] == "resilient"
        assert stats["name"] == "test-cache"
        assert stats["primary_calls"] == 2
        assert stats["fallback_calls"] == 0
        assert "circuit_breaker" in stats
        assert "retry" in stats

        cache.shutdown()

    def test_force_primary_check(self):
        """Can force health check on primary."""
        primary = MemoryCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(
                    enabled=False,
                    healthy_threshold=1,  # Only need 1 success
                ),
            ),
        )

        result = cache.force_primary_check()
        assert result is True
        # After successful health check, primary should be healthy
        assert cache.is_primary_healthy()

        cache.shutdown()

    def test_reset_circuit(self, sample_entry):
        """Can manually reset circuit breaker."""
        primary = FailingCacheBackend(fail_count=100)
        fallback = MemoryCacheBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(failure_threshold=1),
            retry=RetryConfig(max_attempts=1),
            health_check=HealthCheckConfig(enabled=False),
        )

        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=config,
        )

        # Open circuit
        cache.set("key", sample_entry)

        stats = cache.get_stats()
        assert stats["circuit_breaker"]["state"] == "open"

        # Reset circuit
        cache.reset_circuit()

        stats = cache.get_stats()
        assert stats["circuit_breaker"]["state"] == "closed"

        cache.shutdown()

    def test_preset_configs(self):
        """Preset resilience configurations work."""
        default = ResilienceConfig.default()
        assert default.fallback_on_error is True

        ha = ResilienceConfig.high_availability()
        assert ha.retry.max_attempts == 5

        low_latency = ResilienceConfig.low_latency()
        assert low_latency.retry.max_attempts == 1


# =============================================================================
# Fallback Chain Tests
# =============================================================================


class TestFallbackChain:
    """Tests for FallbackChain."""

    def test_uses_first_available_backend(self, sample_entry):
        """Uses first working backend."""
        backend1 = CountingCacheBackend()
        backend2 = CountingCacheBackend()
        chain = FallbackChain([backend1, backend2])

        chain.set("key", sample_entry)

        assert backend1.set_count == 1
        assert backend2.set_count == 0

    def test_falls_back_on_failure(self, sample_entry):
        """Falls back when first backend fails."""
        failing = FailingCacheBackend(fail_count=10)
        working = CountingCacheBackend()
        chain = FallbackChain([failing, working])

        chain.set("key", sample_entry)

        assert working.set_count == 1

    def test_falls_back_through_multiple_failures(self, sample_entry):
        """Falls back through multiple failing backends."""
        failing1 = FailingCacheBackend(fail_count=10)
        failing2 = FailingCacheBackend(fail_count=10)
        working = CountingCacheBackend()
        chain = FallbackChain([failing1, failing2, working])

        chain.set("key", sample_entry)

        assert working.set_count == 1

    def test_tracks_backend_health(self, sample_entry):
        """Tracks which backends are working."""
        failing = FailingCacheBackend(fail_count=10)
        working = CountingCacheBackend()
        chain = FallbackChain([failing, working])

        chain.set("key", sample_entry)

        stats = chain.get_stats()
        assert stats["backend_health"][0] is False
        assert stats["backend_health"][1] is True

    def test_skips_unhealthy_backends(self, sample_entry):
        """Skips backends marked unhealthy."""
        failing = FailingCacheBackend(fail_count=10)
        working = CountingCacheBackend()
        chain = FallbackChain([failing, working])

        # First call marks failing as unhealthy
        chain.set("key1", sample_entry)
        initial_failing_calls = failing._calls

        # Second call should skip failing
        chain.set("key2", sample_entry)

        # Failing backend shouldn't have been called again
        assert failing._calls == initial_failing_calls

    def test_get_returns_none_when_all_fail(self):
        """Get returns None when all backends fail."""
        failing1 = FailingCacheBackend(fail_count=10)
        failing2 = FailingCacheBackend(fail_count=10)
        chain = FallbackChain([failing1, failing2])

        result = chain.get("key")
        assert result is None

    def test_exists_returns_false_when_all_fail(self):
        """Exists returns False when all backends fail."""
        failing1 = FailingCacheBackend(fail_count=10)
        failing2 = FailingCacheBackend(fail_count=10)
        chain = FallbackChain([failing1, failing2])

        result = chain.exists("key")
        assert result is False

    def test_delete_returns_false_when_all_fail(self):
        """Delete returns False when all backends fail."""
        failing1 = FailingCacheBackend(fail_count=10)
        failing2 = FailingCacheBackend(fail_count=10)
        chain = FallbackChain([failing1, failing2])

        result = chain.delete("key")
        assert result is False

    def test_clear_clears_all_backends(self, sample_entry):
        """Clear attempts to clear all backends."""
        backend1 = MemoryCacheBackend()
        backend2 = MemoryCacheBackend()
        chain = FallbackChain([backend1, backend2])

        backend1.set("key1", sample_entry, None)
        backend2.set("key2", sample_entry, None)

        count = chain.clear()
        assert count == 2

    def test_requires_at_least_one_backend(self):
        """Raises error with no backends."""
        with pytest.raises(ValueError, match="At least one backend"):
            FallbackChain([])

    def test_get_stats(self, sample_entry):
        """Statistics are tracked correctly."""
        backend1 = MemoryCacheBackend()
        backend2 = MemoryCacheBackend()
        chain = FallbackChain([backend1, backend2])

        chain.set("key", sample_entry)

        stats = chain.get_stats()
        assert stats["type"] == "fallback_chain"
        assert stats["backend_count"] == 2
        assert stats["calls_per_backend"][0] == 1
        assert stats["calls_per_backend"][1] == 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_resilient_redis_backend_without_redis(self):
        """Factory returns memory backend when Redis unavailable."""
        with patch.dict("sys.modules", {"redis": None}):
            # Force reimport to trigger ImportError handling
            cache = create_resilient_redis_backend(
                host="localhost",
                fallback_to_memory=True,
            )
            # Should fall back gracefully
            assert cache is not None

    def test_create_resilient_redis_backend_with_file_fallback(self, tmp_path):
        """Factory can include file fallback."""
        cache_dir = str(tmp_path / "cache")
        cache = create_resilient_redis_backend(
            host="localhost",
            fallback_to_memory=True,
            fallback_to_file=True,
            file_cache_dir=cache_dir,
        )
        assert cache is not None

    def test_create_high_availability_cache(self):
        """High availability factory creates chain."""
        cache = create_high_availability_cache(
            primary_host="localhost",
            secondary_host="localhost",
        )
        assert cache is not None
        # Should be a fallback chain with multiple backends
        stats = cache.get_stats()
        assert stats["type"] == "fallback_chain"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_circuit_breaker_thread_safety(self):
        """Circuit breaker is thread-safe."""
        breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=100))
        errors = []

        def worker():
            try:
                for _ in range(100):
                    breaker.record_success()
                    breaker.record_failure(ConnectionError("Test"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = breaker.get_stats()
        assert stats["total_calls"] == 2000  # 10 threads * 100 iterations * 2 ops

    def test_resilient_cache_thread_safety(self, sample_entry):
        """Resilient cache is thread-safe."""
        primary = MemoryCacheBackend()
        cache = ResilientCacheBackend(
            primary=primary,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(50):
                    key = f"key-{worker_id}-{i}"
                    profile = TableProfile(
                        name=f"table-{worker_id}-{i}",
                        row_count=i,
                        column_count=worker_id,
                    )
                    entry = CacheEntry(
                        profile=profile,
                        created_at=datetime.now(),
                    )
                    cache.set(key, entry)
                    cache.get(key)
                    cache.exists(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        cache.shutdown()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete scenarios."""

    def test_complete_failover_scenario(self, sample_entry):
        """Test complete failover from primary to fallback."""
        # Primary fails for first 3 operations only
        primary = FailingCacheBackend(fail_count=3)
        fallback = MemoryCacheBackend()

        config = ResilienceConfig(
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=0.1,
                success_threshold=2,  # 2 successes to close
            ),
            retry=RetryConfig(max_attempts=1),
            health_check=HealthCheckConfig(enabled=False),
        )

        cache = ResilientCacheBackend(
            primary=primary,
            fallback=fallback,
            config=config,
        )

        # First 3 operations fail, triggering circuit open
        for i in range(3):
            cache.set(f"key{i}", sample_entry)

        stats = cache.get_stats()
        assert stats["circuit_breaker"]["state"] == "open"

        # Next operations go to fallback (circuit is open)
        cache.set("fallback-key", sample_entry)
        assert fallback.exists("fallback-key")

        # Wait for circuit to half-open
        time.sleep(0.15)

        # Do an operation - this will try primary (which now works)
        # and record success in half-open state
        cache.set("recovery-key-1", sample_entry)

        stats = cache.get_stats()
        # After one success in half-open, still half-open (need 2 successes)
        state = stats["circuit_breaker"]["state"]
        # Could be half_open (one success) or closed (if enough successes)
        assert state in ["half_open", "closed"], f"Expected half_open or closed, got {state}"

        # Do another operation - should succeed and close circuit
        cache.set("recovery-key-2", sample_entry)

        # Circuit should be closed after 2 successes in half_open
        stats = cache.get_stats()
        assert stats["circuit_breaker"]["state"] == "closed"

        cache.shutdown()

    def test_profile_cache_with_resilience(self, sample_profile, tmp_path):
        """Test using resilient backend with ProfileCache."""
        from truthound.profiler.caching import ProfileCache, CacheKey

        cache_dir = str(tmp_path / "cache")

        # Create resilient backend
        fallback = FileCacheBackend(cache_dir=cache_dir)
        resilient_backend = ResilientCacheBackend(
            primary=MemoryCacheBackend(),
            fallback=fallback,
            config=ResilienceConfig(
                health_check=HealthCheckConfig(enabled=False),
            ),
        )

        # ProfileCache takes backend as first arg (can be string or CacheBackend)
        cache = ProfileCache(
            backend=resilient_backend,
            default_ttl=timedelta(seconds=300),
        )

        # Create a proper cache key
        key = CacheKey(key="test-key")

        # Set and get through resilient cache (ProfileCache.set takes TableProfile, not CacheEntry)
        cache.set(key, sample_profile)
        result = cache.get(key)

        # ProfileCache.get returns TableProfile directly
        assert result is not None
        assert result.name == sample_profile.name

        resilient_backend.shutdown()
