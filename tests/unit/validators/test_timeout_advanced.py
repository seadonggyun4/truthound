"""Tests for advanced timeout management module.

This module tests all 8 advanced timeout features:
1. OpenTelemetry integration (telemetry.py)
2. Performance prediction (prediction.py)
3. Adaptive sampling (sampling.py)
4. Priority-based execution (priority.py)
5. Retry/rollback policies (retry.py)
6. SLA monitoring (sla.py)
7. Redis integration (redis_backend.py)
8. Circuit breaker (circuit_breaker.py)
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest


# =============================================================================
# Tests for OpenTelemetry integration (telemetry.py)
# =============================================================================

class TestTelemetry:
    """Tests for OpenTelemetry integration."""

    def test_span_context_creation(self):
        """Test span context creation."""
        from truthound.validators.timeout.advanced.telemetry import SpanContext

        ctx = SpanContext.create_root()
        assert ctx.trace_id
        assert ctx.span_id
        assert ctx.parent_span_id is None

    def test_span_context_child(self):
        """Test child span context creation."""
        from truthound.validators.timeout.advanced.telemetry import SpanContext

        parent = SpanContext.create_root()
        child = SpanContext.create_child(parent)

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id

    def test_span_context_w3c_format(self):
        """Test W3C traceparent format."""
        from truthound.validators.timeout.advanced.telemetry import SpanContext

        ctx = SpanContext.create_root()
        traceparent = ctx.to_w3c_traceparent()

        assert traceparent.startswith("00-")
        parts = traceparent.split("-")
        assert len(parts) == 4

        # Parse it back
        parsed = SpanContext.from_w3c_traceparent(traceparent)
        assert parsed.trace_id == ctx.trace_id
        assert parsed.span_id == ctx.span_id

    def test_tracing_span(self):
        """Test tracing span operations."""
        from truthound.validators.timeout.advanced.telemetry import (
            TracingSpan,
            SpanContext,
            SpanStatus,
        )

        ctx = SpanContext.create_root()
        span = TracingSpan(name="test_operation", context=ctx)

        span.set_attribute("key", "value")
        assert span.attributes["key"] == "value"

        span.add_event("checkpoint", {"step": 1})
        assert len(span.events) == 1

        span.set_ok()
        assert span.status == SpanStatus.OK

        span.end()
        assert span.end_time is not None
        assert span.duration_ms is not None

    def test_telemetry_provider(self):
        """Test telemetry provider."""
        from truthound.validators.timeout.advanced.telemetry import (
            TelemetryProvider,
            TracingConfig,
        )

        config = TracingConfig(
            service_name="test_service",
            exporter_type="console",
            enabled=True,
        )
        provider = TelemetryProvider(config)

        with provider.trace("test_operation") as span:
            span.set_attribute("test", True)

        # Verify metrics recorded
        metrics = provider.get_metrics()
        assert "histograms" in metrics

    def test_metrics_collector(self):
        """Test metrics collector."""
        from truthound.validators.timeout.advanced.telemetry import MetricsCollector

        collector = MetricsCollector()

        collector.increment("requests", 1, {"path": "/api"})
        collector.increment("requests", 1, {"path": "/api"})
        collector.record_histogram("latency", 100.0)
        collector.record_histogram("latency", 150.0)
        collector.set_gauge("connections", 10)

        metrics = collector.get_metrics()
        assert metrics["counters"]["requests{path=/api}"] == 2
        assert metrics["histograms"]["latency"]["count"] == 2
        assert metrics["gauges"]["connections"] == 10

    def test_trace_operation_decorator(self):
        """Test trace operation decorator."""
        from truthound.validators.timeout.advanced.telemetry import trace_operation

        @trace_operation("decorated_operation")
        def my_function():
            return 42

        result = my_function()
        assert result == 42


# =============================================================================
# Tests for Performance Prediction (prediction.py)
# =============================================================================

class TestPrediction:
    """Tests for performance prediction."""

    def test_execution_history(self):
        """Test execution history."""
        from truthound.validators.timeout.advanced.prediction import (
            ExecutionHistory,
            ExecutionRecord,
        )

        history = ExecutionHistory(max_size=100)

        for i in range(10):
            record = ExecutionRecord(
                operation="test",
                duration_ms=100.0 + i * 10,
                features={"rows": 1000},
            )
            history.add(record)

        assert len(history) == 10
        durations = history.get_durations()
        assert len(durations) == 10
        assert durations[0] == 100.0

    def test_moving_average_model(self):
        """Test moving average prediction model."""
        from truthound.validators.timeout.advanced.prediction import (
            MovingAverageModel,
            ExecutionHistory,
            ExecutionRecord,
        )

        model = MovingAverageModel(window_size=5)
        history = ExecutionHistory()

        for duration in [100, 110, 90, 105, 95]:
            history.add(ExecutionRecord(operation="test", duration_ms=duration))

        result = model.predict(history)
        assert 95 <= result.estimated_ms <= 105

    def test_exponential_smoothing_model(self):
        """Test exponential smoothing model."""
        from truthound.validators.timeout.advanced.prediction import (
            ExponentialSmoothingModel,
            ExecutionHistory,
            ExecutionRecord,
        )

        model = ExponentialSmoothingModel(alpha=0.3)
        history = ExecutionHistory()

        for duration in [100, 120, 110, 130, 115]:
            history.add(ExecutionRecord(operation="test", duration_ms=duration))

        result = model.predict(history)
        assert result.estimated_ms > 0
        assert result.confidence > 0

    def test_performance_predictor(self):
        """Test performance predictor."""
        from truthound.validators.timeout.advanced.prediction import PerformancePredictor

        predictor = PerformancePredictor()

        # Record some executions
        for i in range(20):
            predictor.record("validate", 100.0 + i, {"rows": 10000})

        # Predict
        result = predictor.predict("validate", {"rows": 10000})

        assert result.operation == "validate"
        assert result.estimated_ms > 0
        assert result.sample_count > 0
        assert result.model_used == "ensemble"

    def test_prediction_result_timeout(self):
        """Test suggested timeout from prediction."""
        from truthound.validators.timeout.advanced.prediction import PredictionResult

        result = PredictionResult(
            operation="test",
            estimated_ms=100.0,
            p95_ms=150.0,
        )

        # P95 * 1.2 = 180
        assert result.suggested_timeout_ms == 180.0


# =============================================================================
# Tests for Adaptive Sampling (sampling.py)
# =============================================================================

class TestSampling:
    """Tests for adaptive sampling."""

    def test_uniform_sampling(self):
        """Test uniform random sampling."""
        from truthound.validators.timeout.advanced.sampling import UniformSampling

        data = list(range(1000))
        sampler = UniformSampling(seed=42)

        result = sampler.sample(data, 100)

        assert result.sample_size == 100
        assert result.original_size == 1000
        assert result.sampling_ratio == 0.1
        assert len(result.data) == 100

    def test_stratified_sampling(self):
        """Test stratified sampling."""
        from truthound.validators.timeout.advanced.sampling import StratifiedSampling

        data = [{"type": "A"} for _ in range(500)] + [{"type": "B"} for _ in range(500)]
        sampler = StratifiedSampling(key_fn=lambda x: x["type"], seed=42)

        result = sampler.sample(data, 100)

        assert result.sample_size <= 100
        # Should have samples from both strata
        types = [d["type"] for d in result.data]
        assert "A" in types
        assert "B" in types

    def test_reservoir_sampling(self):
        """Test reservoir sampling."""
        from truthound.validators.timeout.advanced.sampling import ReservoirSampling

        data = list(range(10000))
        sampler = ReservoirSampling(seed=42)

        result = sampler.sample(data, 100)

        assert result.sample_size == 100
        assert len(result.data) == 100

    def test_adaptive_sampler(self):
        """Test adaptive sampler."""
        from truthound.validators.timeout.advanced.sampling import (
            AdaptiveSampler,
            DataCharacteristics,
        )

        sampler = AdaptiveSampler()

        # Calculate sample size
        size = sampler.calculate_size(
            total_rows=1000000,
            time_budget_seconds=5.0,
            characteristics=DataCharacteristics(
                row_count=1000000,
                estimated_processing_time_per_row_ms=0.01,
            ),
        )

        assert size > 0
        assert size <= 100000

    def test_calculate_sample_size(self):
        """Test module-level calculate_sample_size function."""
        from truthound.validators.timeout.advanced.sampling import calculate_sample_size

        size = calculate_sample_size(
            total_rows=100000,
            time_budget_seconds=10.0,
            min_sample=100,
            max_sample=10000,
        )

        assert 100 <= size <= 10000


# =============================================================================
# Tests for Priority-Based Execution (priority.py)
# =============================================================================

class TestPriority:
    """Tests for priority-based execution."""

    def test_priority_queue(self):
        """Test priority queue."""
        from truthound.validators.timeout.advanced.priority import (
            PriorityQueue,
            PriorityItem,
            ValidationPriority,
        )

        queue: PriorityQueue[int] = PriorityQueue()

        queue.push(PriorityItem.create("low", ValidationPriority.LOW, lambda: 1))
        queue.push(PriorityItem.create("critical", ValidationPriority.CRITICAL, lambda: 2))
        queue.push(PriorityItem.create("high", ValidationPriority.HIGH, lambda: 3))

        assert len(queue) == 3

        # Should get critical first
        item = queue.pop()
        assert item is not None
        assert item.name == "critical"

        item = queue.pop()
        assert item is not None
        assert item.name == "high"

    def test_priority_executor(self):
        """Test priority executor."""
        from truthound.validators.timeout.advanced.priority import (
            PriorityExecutor,
            ValidationPriority,
        )

        executor: PriorityExecutor[int] = PriorityExecutor()

        results = []

        executor.add("critical", ValidationPriority.CRITICAL, lambda: results.append(1) or 1)
        executor.add("low", ValidationPriority.LOW, lambda: results.append(3) or 3)
        executor.add("high", ValidationPriority.HIGH, lambda: results.append(2) or 2)

        execution_results = executor.execute_all(deadline_seconds=10)

        # Critical should execute first
        assert results[0] == 1
        assert len(execution_results) == 3
        assert all(r.success for r in execution_results)

    def test_execute_by_priority(self):
        """Test execute_by_priority convenience function."""
        from truthound.validators.timeout.advanced.priority import (
            execute_by_priority,
            ValidationPriority,
        )

        operations = [
            ("low", ValidationPriority.LOW, lambda: "low"),
            ("critical", ValidationPriority.CRITICAL, lambda: "critical"),
        ]

        results = execute_by_priority(operations, deadline_seconds=5)

        assert len(results) == 2
        # First result should be critical
        assert results[0].name == "critical"

    def test_priority_time_pressure_skip(self):
        """Test skipping low priority under time pressure."""
        from truthound.validators.timeout.advanced.priority import (
            PriorityExecutor,
            PriorityConfig,
            ValidationPriority,
        )

        config = PriorityConfig(
            skip_low_priority_under_pressure=True,
            time_pressure_threshold=0.9,  # High threshold for testing
            min_priority_to_execute=ValidationPriority.CRITICAL,
        )
        executor: PriorityExecutor[int] = PriorityExecutor(config)

        executor.add("low", ValidationPriority.LOW, lambda: (time.sleep(0.1) or 1))
        executor.add("critical", ValidationPriority.CRITICAL, lambda: 2)

        results = executor.execute_all(deadline_seconds=0.05)

        # Critical should succeed, low might be skipped
        critical_result = next(r for r in results if r.name == "critical")
        assert critical_result.success


# =============================================================================
# Tests for Retry/Rollback (retry.py)
# =============================================================================

class TestRetry:
    """Tests for retry and rollback policies."""

    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        from truthound.validators.timeout.advanced.retry import ExponentialBackoff

        backoff = ExponentialBackoff(base_ms=100, max_ms=10000, multiplier=2, jitter_ratio=0)

        assert backoff.get_delay_ms(0) == 100
        assert backoff.get_delay_ms(1) == 200
        assert backoff.get_delay_ms(2) == 400
        assert backoff.get_delay_ms(10) == 10000  # Capped

    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        from truthound.validators.timeout.advanced.retry import LinearBackoff

        backoff = LinearBackoff(base_ms=100, increment_ms=50, max_ms=500)

        assert backoff.get_delay_ms(0) == 100
        assert backoff.get_delay_ms(1) == 150
        assert backoff.get_delay_ms(2) == 200
        assert backoff.get_delay_ms(100) == 500  # Capped

    def test_retry_policy_success(self):
        """Test retry policy with successful operation."""
        from truthound.validators.timeout.advanced.retry import RetryPolicy

        policy = RetryPolicy(max_attempts=3)

        result = policy.execute(lambda: 42)

        assert result.success
        assert result.value == 42
        assert result.attempts == 1

    def test_retry_policy_retry(self):
        """Test retry policy with retries."""
        from truthound.validators.timeout.advanced.retry import (
            RetryPolicy,
            ConstantBackoff,
        )

        attempts = [0]

        def flaky_operation():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Temporary error")
            return "success"

        policy = RetryPolicy(
            max_attempts=5,
            backoff=ConstantBackoff(10),  # Fast for testing
        )

        result = policy.execute(flaky_operation)

        assert result.success
        assert result.value == "success"
        assert result.attempts == 3

    def test_retry_policy_exhausted(self):
        """Test retry policy when exhausted."""
        from truthound.validators.timeout.advanced.retry import (
            RetryPolicy,
            ConstantBackoff,
        )

        def always_fails():
            raise ValueError("Always fails")

        policy = RetryPolicy(
            max_attempts=3,
            backoff=ConstantBackoff(10),
        )

        result = policy.execute(always_fails)

        assert not result.success
        assert result.attempts == 3
        assert "Always fails" in result.final_error

    def test_with_retry_decorator(self):
        """Test with_retry decorator."""
        from truthound.validators.timeout.advanced.retry import with_retry

        attempts = [0]

        @with_retry(max_attempts=3)
        def flaky_function():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ValueError("Retry")
            return "ok"

        result = flaky_function()
        assert result == "ok"
        assert attempts[0] == 2

    def test_rollback_manager(self):
        """Test rollback manager."""
        from truthound.validators.timeout.advanced.retry import RollbackManager

        rolled_back = []

        with RollbackManager() as rollback:
            rollback.register("action1", lambda: rolled_back.append(1))
            rollback.register("action2", lambda: rolled_back.append(2))
            rollback.commit()

        # No rollback since committed
        assert rolled_back == []

    def test_rollback_manager_on_exception(self):
        """Test rollback manager on exception."""
        from truthound.validators.timeout.advanced.retry import RollbackManager

        rolled_back = []

        try:
            with RollbackManager() as rollback:
                rollback.register("action1", lambda: rolled_back.append(1))
                rollback.register("action2", lambda: rolled_back.append(2))
                raise ValueError("Error")
        except ValueError:
            pass

        # Rollback in reverse order
        assert rolled_back == [2, 1]


# =============================================================================
# Tests for SLA Monitoring (sla.py)
# =============================================================================

class TestSLA:
    """Tests for SLA monitoring."""

    def test_sla_definition(self):
        """Test SLA definition creation."""
        from truthound.validators.timeout.advanced.sla import SLADefinition

        sla = SLADefinition(
            name="api_latency",
            target_ms=100.0,
            target_percentile=0.99,
            max_violation_rate=0.01,
        )

        assert sla.name == "api_latency"
        assert sla.target_ms == 100.0

    def test_sla_monitor_record(self):
        """Test SLA monitor recording."""
        from truthound.validators.timeout.advanced.sla import SLAMonitor, SLADefinition

        monitor = SLAMonitor()
        monitor.register_sla(SLADefinition(
            name="test_sla",
            target_ms=100.0,
            max_violation_rate=0.1,
        ))

        # Record some good measurements
        for _ in range(10):
            monitor.record("test_sla", 50.0)

        metrics = monitor.get_metrics("test_sla")
        assert metrics.total_requests == 10
        assert metrics.violations == 0

    def test_sla_violation_detection(self):
        """Test SLA violation detection."""
        from truthound.validators.timeout.advanced.sla import SLAMonitor, SLADefinition

        monitor = SLAMonitor()
        monitor.register_sla(SLADefinition(
            name="test_sla",
            target_ms=100.0,
        ))

        # Record a violation
        violation = monitor.record("test_sla", 150.0)

        assert violation is not None
        assert violation.actual_ms == 150.0
        assert violation.target_ms == 100.0

    def test_sla_compliance_check(self):
        """Test SLA compliance checking."""
        from truthound.validators.timeout.advanced.sla import SLAMonitor, SLADefinition

        monitor = SLAMonitor()
        monitor.register_sla(SLADefinition(
            name="test_sla",
            target_ms=100.0,
            max_violation_rate=0.5,
        ))

        # All good - should be compliant
        for _ in range(20):
            monitor.record("test_sla", 50.0)

        assert monitor.check_compliance("test_sla")

    def test_sla_metrics_percentiles(self):
        """Test SLA metrics percentile calculation."""
        from truthound.validators.timeout.advanced.sla import SLAMonitor, SLADefinition

        monitor = SLAMonitor()
        monitor.register_sla(SLADefinition(
            name="test_sla",
            target_ms=200.0,
        ))

        # Record range of values
        for i in range(100):
            monitor.record("test_sla", float(i))

        metrics = monitor.get_metrics("test_sla")
        assert metrics.p50_ms is not None
        assert metrics.p90_ms is not None
        assert metrics.p99_ms is not None


# =============================================================================
# Tests for Redis Integration (redis_backend.py)
# =============================================================================

class TestRedisBackend:
    """Tests for Redis backend (using in-memory mock)."""

    @pytest.mark.asyncio
    async def test_in_memory_backend(self):
        """Test in-memory Redis backend."""
        from truthound.validators.timeout.advanced.redis_backend import InMemoryRedisBackend

        backend = InMemoryRedisBackend()

        # Set and get
        await backend.set("key1", "value1")
        assert await backend.get("key1") == "value1"

        # Delete
        await backend.delete("key1")
        assert await backend.get("key1") is None

        # Set with NX (only if not exists)
        await backend.set("key2", "value2", nx=True)
        success = await backend.set("key2", "new_value", nx=True)
        assert not success
        assert await backend.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_in_memory_expiration(self):
        """Test in-memory backend expiration."""
        from truthound.validators.timeout.advanced.redis_backend import InMemoryRedisBackend

        backend = InMemoryRedisBackend()

        # Set with short expiration
        await backend.set("key", "value", ex=1)
        assert await backend.get("key") == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)
        assert await backend.get("key") is None

    @pytest.mark.asyncio
    async def test_redis_coordinator(self):
        """Test Redis coordinator with in-memory backend."""
        from truthound.validators.timeout.advanced.redis_backend import (
            RedisCoordinator,
            InMemoryRedisBackend,
        )

        backend = InMemoryRedisBackend()
        coordinator = RedisCoordinator(backend=backend)

        await coordinator.start()

        try:
            # Test lock
            async with coordinator.lock("test_lock") as lock:
                assert lock.acquired

            # Test deadline
            deadline = await coordinator.create_deadline(60, "test_operation")
            assert deadline.remaining_seconds > 0

            retrieved = await coordinator.get_deadline(deadline.deadline_id)
            assert retrieved is not None
            assert retrieved.deadline_id == deadline.deadline_id

        finally:
            await coordinator.stop()

    @pytest.mark.asyncio
    async def test_redis_distributed_lock(self):
        """Test distributed locking."""
        from truthound.validators.timeout.advanced.redis_backend import (
            RedisCoordinator,
            InMemoryRedisBackend,
        )

        backend = InMemoryRedisBackend()
        coordinator1 = RedisCoordinator(backend=backend)
        coordinator2 = RedisCoordinator(backend=backend)

        await coordinator1.start()
        await coordinator2.start()

        try:
            # First coordinator acquires lock
            async with coordinator1.lock("shared_lock") as lock1:
                assert lock1.acquired

                # Second coordinator should fail to acquire
                async with coordinator2.lock("shared_lock", blocking=False) as lock2:
                    assert not lock2.acquired

        finally:
            await coordinator1.stop()
            await coordinator2.stop()


# =============================================================================
# Tests for Circuit Breaker (circuit_breaker.py)
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    def test_circuit_breaker_closed(self):
        """Test circuit breaker in closed state."""
        from truthound.validators.timeout.advanced.circuit_breaker import (
            CircuitBreaker,
            CircuitState,
        )

        breaker: CircuitBreaker[int] = CircuitBreaker("test")

        result = breaker.execute(lambda: 42)
        assert result == 42
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures."""
        from truthound.validators.timeout.advanced.circuit_breaker import (
            CircuitBreaker,
            CircuitConfig,
            CircuitState,
        )

        config = CircuitConfig(failure_threshold=3)
        breaker: CircuitBreaker[int] = CircuitBreaker("test", config)

        # Cause failures
        for _ in range(3):
            try:
                breaker.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_rejects_when_open(self):
        """Test circuit breaker rejects calls when open."""
        from truthound.validators.timeout.advanced.circuit_breaker import (
            CircuitBreaker,
            CircuitConfig,
            CircuitOpenError,
        )

        config = CircuitConfig(failure_threshold=1, recovery_timeout=60)
        breaker: CircuitBreaker[int] = CircuitBreaker("test", config)

        # Open the circuit
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        # Should reject
        with pytest.raises(CircuitOpenError):
            breaker.execute(lambda: 42)

    def test_circuit_breaker_fallback(self):
        """Test circuit breaker with fallback."""
        from truthound.validators.timeout.advanced.circuit_breaker import (
            CircuitBreaker,
            CircuitConfig,
        )

        config = CircuitConfig(failure_threshold=1)
        breaker: CircuitBreaker[int] = CircuitBreaker(
            "test",
            config,
            fallback=lambda: 0,
        )

        # Open the circuit
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        # Should use fallback
        result = breaker.execute(lambda: 42)
        assert result == 0

    def test_circuit_breaker_half_open(self):
        """Test circuit breaker half-open state."""
        from truthound.validators.timeout.advanced.circuit_breaker import (
            CircuitBreaker,
            CircuitConfig,
            CircuitState,
        )

        config = CircuitConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=1,
        )
        breaker: CircuitBreaker[int] = CircuitBreaker("test", config)

        # Open the circuit
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery
        time.sleep(0.15)

        # Should be half-open now
        result = breaker.execute(lambda: 42)
        assert result == 42
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics."""
        from truthound.validators.timeout.advanced.circuit_breaker import CircuitBreaker

        breaker: CircuitBreaker[int] = CircuitBreaker("test")

        for _ in range(5):
            breaker.execute(lambda: 42)

        metrics = breaker.metrics
        assert metrics.success_count == 5
        assert metrics.failure_count == 0
        assert metrics.success_rate == 1.0

    def test_with_circuit_breaker_decorator(self):
        """Test with_circuit_breaker decorator."""
        from truthound.validators.timeout.advanced.circuit_breaker import with_circuit_breaker

        @with_circuit_breaker(failure_threshold=5)
        def protected_function():
            return "ok"

        result = protected_function()
        assert result == "ok"

        # Check that breaker is accessible
        assert hasattr(protected_function, "circuit_breaker")

    def test_circuit_breaker_force_open(self):
        """Test forcing circuit open."""
        from truthound.validators.timeout.advanced.circuit_breaker import (
            CircuitBreaker,
            CircuitState,
            CircuitOpenError,
        )

        breaker: CircuitBreaker[int] = CircuitBreaker("test")
        breaker.force_open()

        assert breaker.state == CircuitState.OPEN

        with pytest.raises(CircuitOpenError):
            breaker.execute(lambda: 42)

    def test_circuit_breaker_reset(self):
        """Test resetting circuit breaker."""
        from truthound.validators.timeout.advanced.circuit_breaker import (
            CircuitBreaker,
            CircuitConfig,
            CircuitState,
        )

        config = CircuitConfig(failure_threshold=1)
        breaker: CircuitBreaker[int] = CircuitBreaker("test", config)

        # Open the circuit
        try:
            breaker.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

        result = breaker.execute(lambda: 42)
        assert result == 42


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_telemetry_with_circuit_breaker(self):
        """Test telemetry tracking circuit breaker operations."""
        from truthound.validators.timeout.advanced.telemetry import TelemetryProvider
        from truthound.validators.timeout.advanced.circuit_breaker import CircuitBreaker

        provider = TelemetryProvider()
        breaker: CircuitBreaker[int] = CircuitBreaker("test")

        with provider.trace("validation_with_breaker") as span:
            result = breaker.execute(lambda: 42)
            span.set_attribute("result", result)

        metrics = provider.get_metrics()
        assert "histograms" in metrics

    def test_retry_with_prediction(self):
        """Test retry with performance prediction."""
        from truthound.validators.timeout.advanced.retry import RetryPolicy, ConstantBackoff
        from truthound.validators.timeout.advanced.prediction import PerformancePredictor

        predictor = PerformancePredictor()
        policy = RetryPolicy(max_attempts=3, backoff=ConstantBackoff(10))

        def operation():
            start = time.time()
            result = 42
            duration = (time.time() - start) * 1000
            predictor.record("operation", duration)
            return result

        result = policy.execute(operation)
        assert result.success

        # Should have recorded execution
        prediction = predictor.predict("operation")
        assert prediction.sample_count > 0

    def test_priority_with_sla(self):
        """Test priority execution with SLA monitoring."""
        from truthound.validators.timeout.advanced.priority import (
            PriorityExecutor,
            ValidationPriority,
        )
        from truthound.validators.timeout.advanced.sla import SLAMonitor, SLADefinition

        monitor = SLAMonitor()
        monitor.register_sla(SLADefinition(
            name="validation_latency",
            target_ms=100.0,
        ))

        executor: PriorityExecutor[int] = PriorityExecutor()

        def tracked_operation():
            result = 42
            monitor.record("validation_latency", 50.0)
            return result

        executor.add("validate", ValidationPriority.HIGH, tracked_operation)
        results = executor.execute_all()

        metrics = monitor.get_metrics("validation_latency")
        assert metrics.total_requests == 1

    @pytest.mark.asyncio
    async def test_redis_with_circuit_breaker(self):
        """Test Redis coordination with circuit breaker."""
        from truthound.validators.timeout.advanced.redis_backend import (
            RedisCoordinator,
            InMemoryRedisBackend,
        )
        from truthound.validators.timeout.advanced.circuit_breaker import CircuitBreaker

        backend = InMemoryRedisBackend()
        coordinator = RedisCoordinator(backend=backend)
        breaker: CircuitBreaker[bool] = CircuitBreaker("redis_operations")

        await coordinator.start()

        try:
            # Test lock acquisition
            async with coordinator.lock("resource") as lock:
                assert lock.acquired

            # Test circuit breaker with sync operation
            result = breaker.execute(lambda: True)
            assert result

            # Test async execution through breaker
            async_result = await breaker.execute_async(lambda: True)
            assert async_result

        finally:
            await coordinator.stop()


# =============================================================================
# Module Import Test
# =============================================================================

class TestModuleImports:
    """Test that all module exports are accessible."""

    def test_all_imports(self):
        """Test all exports from advanced module."""
        from truthound.validators.timeout.advanced import (
            # Telemetry
            TelemetryProvider,
            TracingConfig,
            create_tracer,
            # Prediction
            PerformancePredictor,
            predict_execution_time,
            # Sampling
            AdaptiveSampler,
            calculate_sample_size,
            # Priority
            PriorityExecutor,
            ValidationPriority,
            # Retry
            RetryPolicy,
            with_retry,
            # SLA
            SLAMonitor,
            SLADefinition,
            # Redis
            RedisCoordinator,
            create_redis_coordinator,
            # Circuit Breaker
            CircuitBreaker,
            with_circuit_breaker,
        )

        # All imports should work
        assert TelemetryProvider is not None
        assert PerformancePredictor is not None
        assert AdaptiveSampler is not None
        assert PriorityExecutor is not None
        assert RetryPolicy is not None
        assert SLAMonitor is not None
        assert RedisCoordinator is not None
        assert CircuitBreaker is not None
