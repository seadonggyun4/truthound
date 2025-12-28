"""Tests for circuit breaker."""

from __future__ import annotations

import asyncio
import pytest

from truthound.stores.backpressure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerMetrics,
    CircuitBreakerState,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.timeout_seconds == 30.0
        assert config.half_open_max_calls == 3
        assert config.failure_rate_threshold == 50.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=10.0,
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.timeout_seconds == 10.0

    def test_validate_failure_threshold(self) -> None:
        """Test failure threshold validation."""
        with pytest.raises(ValueError, match="failure_threshold"):
            config = CircuitBreakerConfig(failure_threshold=0)
            config.validate()

    def test_validate_success_threshold(self) -> None:
        """Test success threshold validation."""
        with pytest.raises(ValueError, match="success_threshold"):
            config = CircuitBreakerConfig(success_threshold=0)
            config.validate()

    def test_validate_timeout(self) -> None:
        """Test timeout validation."""
        with pytest.raises(ValueError, match="timeout_seconds"):
            config = CircuitBreakerConfig(timeout_seconds=0)
            config.validate()

    def test_validate_failure_rate(self) -> None:
        """Test failure rate threshold validation."""
        with pytest.raises(ValueError, match="failure_rate_threshold"):
            config = CircuitBreakerConfig(failure_rate_threshold=150.0)
            config.validate()


class TestCircuitBreakerMetrics:
    """Tests for CircuitBreakerMetrics."""

    def test_default_values(self) -> None:
        """Test default metrics values."""
        metrics = CircuitBreakerMetrics()
        assert metrics.total_calls == 0
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 0
        assert metrics.rejected_calls == 0
        assert metrics.current_state == CircuitBreakerState.CLOSED

    def test_record_success(self) -> None:
        """Test recording successful call."""
        metrics = CircuitBreakerMetrics()
        metrics.record_success(50.0)
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0
        assert metrics.last_success_time is not None

    def test_record_failure(self) -> None:
        """Test recording failed call."""
        metrics = CircuitBreakerMetrics()
        metrics.record_failure(100.0)
        assert metrics.total_calls == 1
        assert metrics.successful_calls == 0
        assert metrics.failed_calls == 1
        assert metrics.last_failure_time is not None

    def test_record_rejection(self) -> None:
        """Test recording rejected call."""
        metrics = CircuitBreakerMetrics()
        metrics.record_rejection()
        assert metrics.rejected_calls == 1

    def test_record_slow_call(self) -> None:
        """Test recording slow call."""
        metrics = CircuitBreakerMetrics()
        metrics.record_slow_call()
        assert metrics.slow_calls == 1

    def test_get_failure_rate(self) -> None:
        """Test failure rate calculation."""
        metrics = CircuitBreakerMetrics()
        assert metrics.get_failure_rate() == 0.0

        metrics.record_success(50.0)
        metrics.record_failure(50.0)
        assert metrics.get_failure_rate() == 50.0

        metrics.record_success(50.0)
        metrics.record_success(50.0)
        assert metrics.get_failure_rate() == 25.0

    def test_get_average_latency(self) -> None:
        """Test average latency calculation."""
        metrics = CircuitBreakerMetrics()
        assert metrics.get_average_latency() == 0.0

        metrics.record_success(50.0)
        metrics.record_success(100.0)
        metrics.record_success(150.0)
        assert metrics.get_average_latency() == 100.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = CircuitBreakerMetrics()
        metrics.record_success(50.0)
        metrics.record_failure(100.0)

        d = metrics.to_dict()
        assert d["total_calls"] == 2
        assert d["successful_calls"] == 1
        assert d["failed_calls"] == 1
        assert d["failure_rate"] == 50.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        cb = CircuitBreaker()
        assert cb.is_closed
        assert not cb.is_open
        assert not cb.is_half_open

    def test_state_properties(self) -> None:
        """Test state property methods."""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.is_half_open is False

    @pytest.mark.asyncio
    async def test_successful_calls_stay_closed(self) -> None:
        """Test successful calls keep circuit closed."""
        cb = CircuitBreaker()

        async def success():
            return "ok"

        for _ in range(10):
            result = await cb.call(success)
            assert result == "ok"

        assert cb.is_closed
        assert cb.metrics.successful_calls == 10

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self) -> None:
        """Test failures open the circuit."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        async def failure():
            raise ValueError("fail")

        for _ in range(3):
            with pytest.raises(ValueError):
                await cb.call(failure)

        assert cb.is_open
        assert cb.metrics.failed_calls == 3

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self) -> None:
        """Test open circuit rejects calls."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)

        async def failure():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(failure)

        assert cb.is_open

        # Further calls should be rejected
        async def success():
            return "ok"

        with pytest.raises(CircuitBreakerError):
            await cb.call(success)

        assert cb.metrics.rejected_calls == 1

    @pytest.mark.asyncio
    async def test_fallback_on_open(self) -> None:
        """Test fallback function when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)

        async def failure():
            raise ValueError("fail")

        def fallback():
            return "fallback"

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(failure)

        # Call with fallback
        result = await cb.call(failure, fallback=fallback)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self) -> None:
        """Test circuit goes to half-open after timeout."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,  # Short timeout for testing
        )
        cb = CircuitBreaker(config)

        async def failure():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(failure)

        assert cb.is_open

        # Wait for timeout
        await asyncio.sleep(0.15)

        async def success():
            return "ok"

        # Should transition to half-open and allow call
        result = await cb.call(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self) -> None:
        """Test successful calls in half-open close circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config)

        async def failure():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(failure)

        await asyncio.sleep(0.15)

        async def success():
            return "ok"

        # Enough successes to close
        await cb.call(success)
        await cb.call(success)

        assert cb.is_closed

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self) -> None:
        """Test failure in half-open reopens circuit."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            timeout_seconds=0.1,
        )
        cb = CircuitBreaker(config)

        async def failure():
            raise ValueError("fail")

        # Open the circuit
        with pytest.raises(ValueError):
            await cb.call(failure)

        await asyncio.sleep(0.15)

        # Failure in half-open
        with pytest.raises(ValueError):
            await cb.call(failure)

        assert cb.is_open

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager usage."""
        cb = CircuitBreaker()

        async with cb:
            pass  # Success

        assert cb.metrics.successful_calls == 1

        try:
            async with cb:
                raise ValueError("fail")
        except ValueError:
            pass

        assert cb.metrics.failed_calls == 1

    def test_reset(self) -> None:
        """Test circuit reset."""
        cb = CircuitBreaker()
        cb.force_open()
        assert cb.is_open

        cb.reset()
        assert cb.is_closed

    def test_force_open(self) -> None:
        """Test forcing circuit open."""
        cb = CircuitBreaker()
        assert cb.is_closed

        cb.force_open()
        assert cb.is_open

    def test_get_stats(self) -> None:
        """Test getting statistics."""
        cb = CircuitBreaker()
        stats = cb.get_stats()

        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert "metrics" in stats

    @pytest.mark.asyncio
    async def test_sync_function_support(self) -> None:
        """Test calling sync functions."""
        cb = CircuitBreaker()

        def sync_success():
            return "sync ok"

        result = await cb.call(sync_success)
        assert result == "sync ok"

    @pytest.mark.asyncio
    async def test_slow_call_tracking(self) -> None:
        """Test slow call tracking."""
        import time

        config = CircuitBreakerConfig(slow_call_threshold_ms=50.0)
        cb = CircuitBreaker(config)

        async def slow_call():
            time.sleep(0.1)  # 100ms
            return "ok"

        await cb.call(slow_call)
        assert cb.metrics.slow_calls == 1
