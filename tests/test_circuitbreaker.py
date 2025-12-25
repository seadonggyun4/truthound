"""Tests for Circuit Breaker pattern implementation."""

from __future__ import annotations

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Any

from truthound.checkpoint.circuitbreaker import (
    # Core types
    CircuitState,
    FailureDetectionStrategy,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitOpenError,
    CircuitHalfOpenError,
    CallResult,
    CircuitBreakerMetrics,
    StateChangeEvent,
    # Detection
    FailureDetector,
    ConsecutiveFailureDetector,
    PercentageFailureDetector,
    TimeWindowFailureDetector,
    CompositeFailureDetector,
    create_detector,
    # Breaker
    CircuitBreaker,
    CircuitBreakerStateMachine,
    # Registry
    CircuitBreakerRegistry,
    get_registry,
    get_breaker,
    register_breaker,
    # Middleware
    CircuitBreakerMiddleware,
    circuit_breaker,
    with_circuit_breaker,
)


# =============================================================================
# Core Types Tests
# =============================================================================


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states(self):
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_values(self):
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0
        assert config.half_open_max_calls == 3
        assert config.success_threshold == 2
        assert config.detection_strategy == FailureDetectionStrategy.CONSECUTIVE

    def test_custom_values(self):
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            half_open_max_calls=5,
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 60.0
        assert config.half_open_max_calls == 5

    def test_should_count_exception_default(self):
        config = CircuitBreakerConfig()
        assert config.should_count_exception(ValueError("test"))
        assert config.should_count_exception(RuntimeError("test"))

    def test_should_count_exception_excluded(self):
        config = CircuitBreakerConfig(excluded_exceptions=(ValueError,))
        assert not config.should_count_exception(ValueError("test"))
        assert config.should_count_exception(RuntimeError("test"))

    def test_should_count_exception_included(self):
        config = CircuitBreakerConfig(included_exceptions=(ValueError,))
        assert config.should_count_exception(ValueError("test"))
        assert not config.should_count_exception(RuntimeError("test"))


class TestCircuitBreakerMetrics:
    """Tests for CircuitBreakerMetrics."""

    def test_default_values(self):
        metrics = CircuitBreakerMetrics(name="test", state=CircuitState.CLOSED)
        assert metrics.name == "test"
        assert metrics.state == CircuitState.CLOSED
        assert metrics.total_calls == 0
        assert metrics.is_healthy

    def test_is_healthy(self):
        closed_metrics = CircuitBreakerMetrics(name="test", state=CircuitState.CLOSED)
        assert closed_metrics.is_healthy

        open_metrics = CircuitBreakerMetrics(name="test", state=CircuitState.OPEN)
        assert not open_metrics.is_healthy

        half_open_metrics = CircuitBreakerMetrics(name="test", state=CircuitState.HALF_OPEN)
        assert not half_open_metrics.is_healthy

    def test_to_dict(self):
        metrics = CircuitBreakerMetrics(
            name="test",
            state=CircuitState.CLOSED,
            total_calls=100,
            successful_calls=90,
            failed_calls=10,
        )
        data = metrics.to_dict()
        assert data["name"] == "test"
        assert data["state"] == "closed"
        assert data["total_calls"] == 100
        assert data["is_healthy"] is True


class TestStateChangeEvent:
    """Tests for StateChangeEvent."""

    def test_creation(self):
        event = StateChangeEvent(
            breaker_name="test",
            from_state=CircuitState.CLOSED,
            to_state=CircuitState.OPEN,
            reason="Failure threshold exceeded",
        )
        assert event.breaker_name == "test"
        assert event.from_state == CircuitState.CLOSED
        assert event.to_state == CircuitState.OPEN
        assert "Failure threshold" in event.reason

    def test_str(self):
        event = StateChangeEvent(
            breaker_name="test",
            from_state=CircuitState.CLOSED,
            to_state=CircuitState.OPEN,
            reason="Test reason",
        )
        result = str(event)
        assert "test" in result
        assert "closed" in result
        assert "open" in result


# =============================================================================
# Failure Detection Tests
# =============================================================================


class TestConsecutiveFailureDetector:
    """Tests for ConsecutiveFailureDetector."""

    def test_should_trip_at_threshold(self):
        detector = ConsecutiveFailureDetector(threshold=3)

        detector.record_failure()
        assert not detector.should_trip()

        detector.record_failure()
        assert not detector.should_trip()

        detector.record_failure()
        assert detector.should_trip()

    def test_success_resets_counter(self):
        detector = ConsecutiveFailureDetector(threshold=3)

        detector.record_failure()
        detector.record_failure()
        detector.record_success()

        assert not detector.should_trip()
        assert detector.consecutive_failures == 0

    def test_reset(self):
        detector = ConsecutiveFailureDetector(threshold=3)
        detector.record_failure()
        detector.record_failure()

        detector.reset()

        assert detector.consecutive_failures == 0
        assert detector.failure_count == 0


class TestPercentageFailureDetector:
    """Tests for PercentageFailureDetector."""

    def test_min_calls_required(self):
        detector = PercentageFailureDetector(threshold=0.5, min_calls=10)

        # 100% failures but below min calls
        for _ in range(5):
            detector.record_failure()

        assert not detector.should_trip()

    def test_should_trip_above_threshold(self):
        detector = PercentageFailureDetector(threshold=0.5, min_calls=10)

        for _ in range(6):
            detector.record_failure()
        for _ in range(4):
            detector.record_success()

        # 60% failure rate > 50% threshold
        assert detector.should_trip()

    def test_should_not_trip_below_threshold(self):
        detector = PercentageFailureDetector(threshold=0.5, min_calls=10)

        for _ in range(4):
            detector.record_failure()
        for _ in range(6):
            detector.record_success()

        # 40% failure rate < 50% threshold
        assert not detector.should_trip()

    def test_failure_rate(self):
        detector = PercentageFailureDetector(threshold=0.5, min_calls=5)

        for _ in range(3):
            detector.record_failure()
        for _ in range(2):
            detector.record_success()

        assert abs(detector.failure_rate - 0.6) < 0.01

    def test_sliding_window(self):
        detector = PercentageFailureDetector(
            threshold=0.5,
            min_calls=5,
            window_size=5,
        )

        # Fill window with failures
        for _ in range(5):
            detector.record_failure()

        assert detector.should_trip()

        # Add successes to push out failures
        for _ in range(5):
            detector.record_success()

        assert not detector.should_trip()


class TestTimeWindowFailureDetector:
    """Tests for TimeWindowFailureDetector."""

    def test_should_trip_in_window(self):
        detector = TimeWindowFailureDetector(
            threshold=3,
            window_seconds=60.0,
        )

        detector.record_failure()
        detector.record_failure()
        assert not detector.should_trip()

        detector.record_failure()
        assert detector.should_trip()

    def test_failures_expire(self):
        detector = TimeWindowFailureDetector(
            threshold=3,
            window_seconds=0.1,  # 100ms window
        )

        detector.record_failure()
        detector.record_failure()
        detector.record_failure()

        assert detector.should_trip()

        time.sleep(0.15)  # Wait for window to expire

        assert not detector.should_trip()

    def test_percentage_mode(self):
        detector = TimeWindowFailureDetector(
            threshold=5,
            window_seconds=60.0,
            use_percentage=True,
            percentage_threshold=0.5,
            min_calls=10,
        )

        for _ in range(6):
            detector.record_failure()
        for _ in range(4):
            detector.record_success()

        assert detector.should_trip()


class TestCompositeFailureDetector:
    """Tests for CompositeFailureDetector."""

    def test_or_logic(self):
        detector = CompositeFailureDetector(
            detectors=[
                ConsecutiveFailureDetector(threshold=3),
                PercentageFailureDetector(threshold=0.8, min_calls=5),
            ],
            require_all=False,
        )

        # Trip consecutive detector
        for _ in range(3):
            detector.record_failure()

        assert detector.should_trip()

    def test_and_logic(self):
        detector = CompositeFailureDetector(
            detectors=[
                ConsecutiveFailureDetector(threshold=3),
                PercentageFailureDetector(threshold=0.5, min_calls=5),
            ],
            require_all=True,
        )

        # Only consecutive would trip
        for _ in range(3):
            detector.record_failure()

        # Not enough calls for percentage
        assert not detector.should_trip()

        # Add more to satisfy both
        for _ in range(2):
            detector.record_failure()

        assert detector.should_trip()

    def test_detector_states(self):
        detector = CompositeFailureDetector(
            detectors=[
                ConsecutiveFailureDetector(threshold=3),
            ],
            require_all=False,
        )

        states = detector.detector_states
        assert len(states) == 1
        assert states[0]["type"] == "ConsecutiveFailureDetector"


class TestCreateDetector:
    """Tests for create_detector factory function."""

    def test_consecutive_strategy(self):
        config = CircuitBreakerConfig(
            detection_strategy=FailureDetectionStrategy.CONSECUTIVE,
            failure_threshold=5,
        )
        detector = create_detector(config)
        assert isinstance(detector, ConsecutiveFailureDetector)

    def test_percentage_strategy(self):
        config = CircuitBreakerConfig(
            detection_strategy=FailureDetectionStrategy.PERCENTAGE,
        )
        detector = create_detector(config)
        assert isinstance(detector, PercentageFailureDetector)

    def test_time_window_strategy(self):
        config = CircuitBreakerConfig(
            detection_strategy=FailureDetectionStrategy.TIME_WINDOW,
        )
        detector = create_detector(config)
        assert isinstance(detector, TimeWindowFailureDetector)

    def test_composite_strategy(self):
        config = CircuitBreakerConfig(
            detection_strategy=FailureDetectionStrategy.COMPOSITE,
        )
        detector = create_detector(config)
        assert isinstance(detector, CompositeFailureDetector)


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        return CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=0.1,  # Fast recovery for tests
                half_open_max_calls=2,
                success_threshold=2,
            ),
        )

    def test_initial_state(self, breaker: CircuitBreaker):
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    def test_successful_call(self, breaker: CircuitBreaker):
        result = breaker.call(lambda: "success")
        assert result == "success"

        metrics = breaker.get_metrics()
        assert metrics.successful_calls == 1
        assert metrics.failed_calls == 0

    def test_failed_call(self, breaker: CircuitBreaker):
        def failing():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            breaker.call(failing)

        metrics = breaker.get_metrics()
        assert metrics.failed_calls == 1

    def test_trips_to_open(self, breaker: CircuitBreaker):
        def failing():
            raise RuntimeError("test")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        assert breaker.is_open

    def test_open_circuit_raises(self, breaker: CircuitBreaker):
        def failing():
            raise RuntimeError("test")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        # Next call should fail fast
        with pytest.raises(CircuitOpenError) as exc_info:
            breaker.call(lambda: "should not execute")

        assert "test" in exc_info.value.breaker_name

    def test_recovery_to_half_open(self, breaker: CircuitBreaker):
        def failing():
            raise RuntimeError("test")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        assert breaker.is_open

        # Wait for recovery timeout
        time.sleep(0.15)

        # Accessing state triggers transition check
        assert breaker.is_half_open

    def test_half_open_to_closed(self, breaker: CircuitBreaker):
        def failing():
            raise RuntimeError("test")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        time.sleep(0.15)
        assert breaker.is_half_open

        # Successful calls should close
        breaker.call(lambda: "success")
        breaker.call(lambda: "success")

        assert breaker.is_closed

    def test_half_open_to_open(self, breaker: CircuitBreaker):
        def failing():
            raise RuntimeError("test")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        time.sleep(0.15)
        assert breaker.is_half_open

        # Failure should reopen
        with pytest.raises(RuntimeError):
            breaker.call(failing)

        assert breaker.is_open

    def test_half_open_max_calls(self, breaker: CircuitBreaker):
        def failing():
            raise RuntimeError("test")

        # Trip and wait for half-open
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        time.sleep(0.15)

        # Use up allowed calls (they succeed but don't close yet)
        breaker.call(lambda: "1")
        breaker.call(lambda: "2")

        # Should be closed now after 2 successes
        assert breaker.is_closed

    def test_context_manager(self, breaker: CircuitBreaker):
        with breaker:
            result = "inside"

        assert result == "inside"
        assert breaker.get_metrics().successful_calls == 1

    def test_context_manager_exception(self, breaker: CircuitBreaker):
        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("test")

        assert breaker.get_metrics().failed_calls == 1

    def test_call_with_result_success(self, breaker: CircuitBreaker):
        result = breaker.call_with_result(lambda: "success")
        assert result.success
        assert result.result == "success"
        assert result.exception is None

    def test_call_with_result_failure(self, breaker: CircuitBreaker):
        result = breaker.call_with_result(lambda: (_ for _ in ()).throw(ValueError("test")))
        assert not result.success
        assert isinstance(result.exception, ValueError)

    def test_fallback(self):
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=1,
                fallback=lambda: "fallback_value",
            ),
        )

        # Trip the breaker
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("test")))

        # Should use fallback
        result = breaker.call(lambda: "should not execute")
        assert result == "fallback_value"

    def test_excluded_exceptions(self):
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(
                failure_threshold=2,
                excluded_exceptions=(ValueError,),
            ),
        )

        # ValueError should not count
        for _ in range(5):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("ignored")))

        assert breaker.is_closed

    def test_reset(self, breaker: CircuitBreaker):
        def failing():
            raise RuntimeError("test")

        # Trip the breaker
        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        assert breaker.is_open

        breaker.reset()

        assert breaker.is_closed
        assert breaker.get_metrics().total_calls == 0

    def test_state_change_listener(self, breaker: CircuitBreaker):
        events: list[StateChangeEvent] = []
        breaker.add_listener(lambda e: events.append(e))

        def failing():
            raise RuntimeError("test")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(failing)

        assert len(events) == 1
        assert events[0].from_state == CircuitState.CLOSED
        assert events[0].to_state == CircuitState.OPEN


class TestCircuitBreakerStateMachine:
    """Tests for CircuitBreakerStateMachine."""

    def test_can_execute_closed(self):
        sm = CircuitBreakerStateMachine("test", CircuitBreakerConfig())
        assert sm.can_execute()

    def test_can_execute_open(self):
        sm = CircuitBreakerStateMachine(
            "test",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0),
        )
        sm.record_failure(RuntimeError("test"))
        assert not sm.can_execute()

    def test_remaining_timeout(self):
        sm = CircuitBreakerStateMachine(
            "test",
            CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0),
        )
        sm.record_failure(RuntimeError("test"))

        remaining = sm.get_remaining_timeout()
        assert remaining is not None
        assert remaining > 55  # Should be close to 60


# =============================================================================
# Registry Tests
# =============================================================================


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    @pytest.fixture
    def registry(self) -> CircuitBreakerRegistry:
        reg = CircuitBreakerRegistry()
        yield reg
        reg.clear()

    def test_register(self, registry: CircuitBreakerRegistry):
        breaker = registry.register("test")
        assert breaker.name == "test"
        assert "test" in registry

    def test_register_duplicate_raises(self, registry: CircuitBreakerRegistry):
        registry.register("test")
        with pytest.raises(ValueError):
            registry.register("test")

    def test_register_replace(self, registry: CircuitBreakerRegistry):
        breaker1 = registry.register("test")
        breaker2 = registry.register("test", replace=True)
        assert breaker1 is not breaker2

    def test_get(self, registry: CircuitBreakerRegistry):
        registry.register("test")
        breaker = registry.get("test")
        assert breaker is not None
        assert breaker.name == "test"

    def test_get_nonexistent(self, registry: CircuitBreakerRegistry):
        assert registry.get("nonexistent") is None

    def test_get_or_create(self, registry: CircuitBreakerRegistry):
        breaker1 = registry.get_or_create("test")
        breaker2 = registry.get_or_create("test")
        assert breaker1 is breaker2

    def test_unregister(self, registry: CircuitBreakerRegistry):
        registry.register("test")
        breaker = registry.unregister("test")
        assert breaker is not None
        assert "test" not in registry

    def test_get_all(self, registry: CircuitBreakerRegistry):
        registry.register("test1")
        registry.register("test2")

        all_breakers = registry.get_all()
        assert len(all_breakers) == 2

    def test_get_all_metrics(self, registry: CircuitBreakerRegistry):
        registry.register("test1")
        registry.register("test2")

        metrics = registry.get_all_metrics()
        assert "test1" in metrics
        assert "test2" in metrics

    def test_get_health_status(self, registry: CircuitBreakerRegistry):
        b1 = registry.register(
            "test1",
            CircuitBreakerConfig(failure_threshold=1),
        )
        registry.register("test2")

        # Trip test1
        with pytest.raises(RuntimeError):
            b1.call(lambda: (_ for _ in ()).throw(RuntimeError("test")))

        health = registry.get_health_status()
        assert not health["test1"]
        assert health["test2"]

    def test_get_open_breakers(self, registry: CircuitBreakerRegistry):
        b1 = registry.register(
            "test1",
            CircuitBreakerConfig(failure_threshold=1),
        )
        registry.register("test2")

        with pytest.raises(RuntimeError):
            b1.call(lambda: (_ for _ in ()).throw(RuntimeError("test")))

        open_breakers = registry.get_open_breakers()
        assert "test1" in open_breakers
        assert "test2" not in open_breakers

    def test_reset_all(self, registry: CircuitBreakerRegistry):
        b1 = registry.register(
            "test1",
            CircuitBreakerConfig(failure_threshold=1),
        )
        b2 = registry.register(
            "test2",
            CircuitBreakerConfig(failure_threshold=1),
        )

        with pytest.raises(RuntimeError):
            b1.call(lambda: (_ for _ in ()).throw(RuntimeError("test")))
        with pytest.raises(RuntimeError):
            b2.call(lambda: (_ for _ in ()).throw(RuntimeError("test")))

        registry.reset_all()

        assert b1.is_closed
        assert b2.is_closed

    def test_global_listener(self, registry: CircuitBreakerRegistry):
        events: list[StateChangeEvent] = []
        registry.add_global_listener(lambda e: events.append(e))

        b1 = registry.register(
            "test1",
            CircuitBreakerConfig(failure_threshold=1),
        )

        with pytest.raises(RuntimeError):
            b1.call(lambda: (_ for _ in ()).throw(RuntimeError("test")))

        assert len(events) == 1

    def test_iteration(self, registry: CircuitBreakerRegistry):
        registry.register("test1")
        registry.register("test2")

        names = list(registry)
        assert "test1" in names
        assert "test2" in names

    def test_len(self, registry: CircuitBreakerRegistry):
        registry.register("test1")
        registry.register("test2")
        assert len(registry) == 2

    def test_to_dict(self, registry: CircuitBreakerRegistry):
        registry.register("test1")
        registry.register("test2")

        data = registry.to_dict()
        assert data["total_count"] == 2
        assert data["healthy_count"] == 2
        assert "test1" in data["breakers"]


class TestRegistryConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CircuitBreakerRegistry.reset_instance()
        yield
        CircuitBreakerRegistry.reset_instance()

    def test_get_registry(self):
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2

    def test_register_breaker(self):
        breaker = register_breaker("test")
        assert breaker.name == "test"
        assert get_breaker("test") is breaker

    def test_get_breaker(self):
        register_breaker("test")
        breaker = get_breaker("test")
        assert breaker is not None
        assert breaker.name == "test"


# =============================================================================
# Middleware Tests
# =============================================================================


class TestCircuitBreakerMiddleware:
    """Tests for CircuitBreakerMiddleware."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CircuitBreakerRegistry.reset_instance()
        yield
        CircuitBreakerRegistry.reset_instance()

    def test_execute_success(self):
        middleware = CircuitBreakerMiddleware("test")
        result = middleware.execute(lambda: "success")
        assert result == "success"

    def test_execute_failure(self):
        middleware = CircuitBreakerMiddleware(
            "test",
            CircuitBreakerConfig(failure_threshold=1),
        )

        with pytest.raises(RuntimeError):
            middleware.execute(lambda: (_ for _ in ()).throw(RuntimeError("test")))

    def test_execute_with_fallback(self):
        middleware = CircuitBreakerMiddleware(
            "test",
            CircuitBreakerConfig(failure_threshold=1),
        )

        with pytest.raises(RuntimeError):
            middleware.execute(lambda: (_ for _ in ()).throw(RuntimeError("test")))

        result = middleware.execute_with_fallback(
            lambda: "primary",
            lambda: "fallback",
        )
        assert result == "fallback"

    def test_is_available(self):
        middleware = CircuitBreakerMiddleware(
            "test",
            CircuitBreakerConfig(failure_threshold=1),
        )

        assert middleware.is_available()

        with pytest.raises(RuntimeError):
            middleware.execute(lambda: (_ for _ in ()).throw(RuntimeError("test")))

        assert not middleware.is_available()

    def test_get_status(self):
        middleware = CircuitBreakerMiddleware("test")
        status = middleware.get_status()

        assert status["breaker_name"] == "test"
        assert status["state"] == "closed"
        assert status["is_available"]


class TestCircuitBreakerDecorator:
    """Tests for @circuit_breaker decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CircuitBreakerRegistry.reset_instance()
        yield
        CircuitBreakerRegistry.reset_instance()

    def test_decorator_without_args(self):
        @circuit_breaker
        def my_func():
            return "success"

        result = my_func()
        assert result == "success"

    def test_decorator_with_args(self):
        @circuit_breaker(name="custom_name", failure_threshold=3)
        def my_func():
            return "success"

        result = my_func()
        assert result == "success"

        breaker = get_breaker("custom_name")
        assert breaker is not None

    def test_decorator_trips(self):
        @circuit_breaker(failure_threshold=2)
        def failing_func():
            raise RuntimeError("test")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                failing_func()

        with pytest.raises(CircuitOpenError):
            failing_func()

    def test_decorator_with_fallback(self):
        @circuit_breaker(
            failure_threshold=1,
            fallback=lambda: "fallback_value",
        )
        def my_func():
            raise RuntimeError("test")

        with pytest.raises(RuntimeError):
            my_func()

        result = my_func()
        assert result == "fallback_value"

    def test_decorator_preserves_function(self):
        @circuit_breaker
        def my_func():
            """My docstring."""
            return "success"

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_attached_breaker(self):
        @circuit_breaker(name="attached")
        def my_func():
            return "success"

        assert hasattr(my_func, "_circuit_breaker")
        assert my_func._circuit_breaker.name == "attached"


class TestWithCircuitBreakerDecorator:
    """Tests for @with_circuit_breaker decorator."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CircuitBreakerRegistry.reset_instance()
        yield
        CircuitBreakerRegistry.reset_instance()

    def test_with_breaker_instance(self):
        breaker = CircuitBreaker("shared")

        @with_circuit_breaker(breaker)
        def func1():
            return "func1"

        @with_circuit_breaker(breaker)
        def func2():
            return "func2"

        assert func1() == "func1"
        assert func2() == "func2"

        metrics = breaker.get_metrics()
        assert metrics.successful_calls == 2

    def test_with_breaker_name(self):
        register_breaker("shared")

        @with_circuit_breaker("shared")
        def my_func():
            return "success"

        result = my_func()
        assert result == "success"

    def test_with_nonexistent_name_raises(self):
        @with_circuit_breaker("nonexistent")
        def my_func():
            return "success"

        with pytest.raises(ValueError, match="not found"):
            my_func()


# =============================================================================
# Integration Tests
# =============================================================================


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        CircuitBreakerRegistry.reset_instance()
        yield
        CircuitBreakerRegistry.reset_instance()

    def test_full_lifecycle(self):
        """Test complete circuit breaker lifecycle."""
        events: list[StateChangeEvent] = []

        breaker = CircuitBreaker(
            "lifecycle_test",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=0.1,
                success_threshold=2,
                on_state_change=lambda e: events.append(e),
            ),
        )

        # 1. Start closed
        assert breaker.state == CircuitState.CLOSED

        # 2. Successful calls
        for _ in range(5):
            breaker.call(lambda: "ok")
        assert breaker.is_closed

        # 3. Trip with failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        assert breaker.is_open
        assert len(events) == 1
        assert events[0].to_state == CircuitState.OPEN

        # 4. Rejected calls
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "blocked")

        # 5. Wait for half-open
        time.sleep(0.15)
        assert breaker.is_half_open
        assert len(events) == 2

        # 6. Recover with successes
        breaker.call(lambda: "ok")
        breaker.call(lambda: "ok")

        assert breaker.is_closed
        assert len(events) == 3

    def test_concurrent_access(self):
        """Test thread-safe concurrent access."""
        breaker = CircuitBreaker(
            "concurrent_test",
            CircuitBreakerConfig(failure_threshold=100),
        )

        success_count = [0]
        error_count = [0]
        lock = threading.Lock()

        def worker():
            for _ in range(100):
                try:
                    breaker.call(lambda: "ok")
                    with lock:
                        success_count[0] += 1
                except Exception:
                    with lock:
                        error_count[0] += 1

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count[0] == 500
        assert error_count[0] == 0

        metrics = breaker.get_metrics()
        assert metrics.successful_calls == 500

    def test_multiple_breakers(self):
        """Test multiple independent breakers."""
        api_breaker = CircuitBreaker(
            "api",
            CircuitBreakerConfig(failure_threshold=2),
        )
        db_breaker = CircuitBreaker(
            "db",
            CircuitBreakerConfig(failure_threshold=3),
        )

        # Trip API breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                api_breaker.call(lambda: (_ for _ in ()).throw(RuntimeError()))

        # DB should still work
        assert api_breaker.is_open
        assert db_breaker.is_closed

        result = db_breaker.call(lambda: "db_ok")
        assert result == "db_ok"

    def test_metrics_accuracy(self):
        """Test metrics are accurate."""
        breaker = CircuitBreaker(
            "metrics_test",
            CircuitBreakerConfig(failure_threshold=5),
        )

        # 10 successes
        for _ in range(10):
            breaker.call(lambda: "ok")

        # 3 failures
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError()))

        metrics = breaker.get_metrics()
        assert metrics.total_calls == 13
        assert metrics.successful_calls == 10
        assert metrics.failed_calls == 3
        assert abs(metrics.failure_rate - (3/13)) < 0.01
