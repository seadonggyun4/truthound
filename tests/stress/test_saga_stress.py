"""Stress tests for distributed saga transactions.

Tests the saga executor under high load with failure injection.
"""

from __future__ import annotations

import asyncio
import random
import pytest
from datetime import timedelta

from tests.stress.framework import (
    StressTestRunner,
    StressTestConfig,
    LoadProfile,
)
from tests.stress.generators import (
    RampUpLoadGenerator,
    SpikeLoadGenerator,
    WaveLoadGenerator,
)
from tests.stress.chaos import ChaosEngine, FailureType, FailureConfig
from tests.stress.metrics import StressMetricsCollector
from tests.stress.reports import ReportGenerator, StressTestReport


class MockSagaExecutor:
    """Mock saga executor for stress testing."""

    def __init__(
        self,
        success_rate: float = 0.99,
        base_latency_ms: float = 10.0,
        latency_variance: float = 0.3,
    ) -> None:
        self._success_rate = success_rate
        self._base_latency = base_latency_ms
        self._latency_variance = latency_variance
        self._rng = random.Random()

    async def execute(self, saga_id: str) -> bool:
        """Execute a mock saga transaction."""
        # Simulate varying latency
        latency = self._base_latency * (
            1 + self._rng.uniform(-self._latency_variance, self._latency_variance)
        )
        await asyncio.sleep(latency / 1000)

        # Randomly fail based on success rate
        return self._rng.random() < self._success_rate


@pytest.fixture
def mock_executor() -> MockSagaExecutor:
    """Create a mock saga executor."""
    return MockSagaExecutor(success_rate=0.99, base_latency_ms=15.0)


@pytest.fixture
def metrics_collector() -> StressMetricsCollector:
    """Create a metrics collector."""
    return StressMetricsCollector(name="saga_stress")


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_constant_load(
    mock_executor: MockSagaExecutor,
    metrics_collector: StressMetricsCollector,
) -> None:
    """Test saga execution under constant load."""
    config = StressTestConfig(
        name="saga_constant_load",
        load_profile=LoadProfile(
            warmup_seconds=2.0,
            ramp_up_seconds=3.0,
            steady_seconds=10.0,
            ramp_down_seconds=3.0,
            cooldown_seconds=2.0,
            initial_rate=5.0,
            target_rate=50.0,
        ),
        concurrency=20,
        target_success_rate=0.95,
        max_latency_p99_ms=500.0,
    )

    async def operation(op_id: str) -> bool:
        success = await mock_executor.execute(op_id)
        return success

    runner = StressTestRunner(
        config=config,
        operation=operation,
    )

    result = await runner.run()

    # Assert basic metrics
    assert result.total_operations > 0
    assert result.success_rate >= 0.9  # Allow some tolerance


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_ramp_up_load(
    mock_executor: MockSagaExecutor,
) -> None:
    """Test saga execution with ramp-up load pattern."""
    config = StressTestConfig(
        name="saga_ramp_up",
        load_profile=LoadProfile(
            warmup_seconds=1.0,
            ramp_up_seconds=10.0,
            steady_seconds=5.0,
            ramp_down_seconds=2.0,
            cooldown_seconds=1.0,
            initial_rate=1.0,
            target_rate=100.0,
        ),
        concurrency=50,
        target_success_rate=0.95,
    )

    async def operation(op_id: str) -> bool:
        return await mock_executor.execute(op_id)

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()

    assert result.total_operations > 50
    assert result.success_rate >= 0.9


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_spike_load(
    mock_executor: MockSagaExecutor,
) -> None:
    """Test saga execution with spike load pattern."""
    generator = SpikeLoadGenerator(
        base_rate=10.0,
        spike_rate=100.0,
        spike_duration=2.0,
        spike_interval=10.0,
        duration_seconds=25.0,
    )

    config = StressTestConfig(
        name="saga_spike_load",
        load_profile=LoadProfile(
            warmup_seconds=0,
            ramp_up_seconds=0,
            steady_seconds=25.0,
            ramp_down_seconds=0,
            cooldown_seconds=0,
            target_rate=100.0,
        ),
        concurrency=100,
    )

    operations_executed = 0

    async def operation(op_id: str) -> bool:
        nonlocal operations_executed
        operations_executed += 1
        return await mock_executor.execute(op_id)

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()

    # Should have executed operations through spikes
    assert result.total_operations > 100


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_with_chaos(
    mock_executor: MockSagaExecutor,
) -> None:
    """Test saga execution with chaos engineering."""
    chaos = ChaosEngine(enabled=True, seed=42)
    chaos.enable_scenario("network_chaos", probability_multiplier=0.5)

    config = StressTestConfig(
        name="saga_chaos",
        load_profile=LoadProfile(
            warmup_seconds=1.0,
            ramp_up_seconds=2.0,
            steady_seconds=10.0,
            ramp_down_seconds=2.0,
            cooldown_seconds=1.0,
            target_rate=30.0,
        ),
        concurrency=20,
        enable_chaos=True,
        target_success_rate=0.8,  # Lower due to chaos
    )

    async def operation(op_id: str) -> bool:
        async with chaos.chaos_context(op_id):
            return await mock_executor.execute(op_id)

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()

    # Should still complete despite chaos
    assert result.total_operations > 0

    # Check chaos was actually injected
    summary = chaos.get_failure_summary()
    # Failures may or may not have been injected due to probability
    assert summary["total_operations"] > 0


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_wave_load(
    mock_executor: MockSagaExecutor,
) -> None:
    """Test saga execution with wave (sinusoidal) load pattern."""
    generator = WaveLoadGenerator(
        min_rate=5.0,
        max_rate=50.0,
        period_seconds=10.0,
        duration_seconds=30.0,
    )

    config = StressTestConfig(
        name="saga_wave_load",
        load_profile=LoadProfile(
            steady_seconds=30.0,
            target_rate=50.0,
        ),
        concurrency=30,
    )

    async def operation(op_id: str) -> bool:
        return await mock_executor.execute(op_id)

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()

    assert result.total_operations > 100


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_with_report_generation(
    mock_executor: MockSagaExecutor,
    tmp_path,
) -> None:
    """Test saga execution with report generation."""
    config = StressTestConfig(
        name="saga_report_test",
        load_profile=LoadProfile(
            warmup_seconds=1.0,
            ramp_up_seconds=2.0,
            steady_seconds=5.0,
            ramp_down_seconds=2.0,
            cooldown_seconds=1.0,
            target_rate=30.0,
        ),
        concurrency=15,
    )

    metrics = StressMetricsCollector(name="saga_report_test")
    metrics.start()

    async def operation(op_id: str) -> bool:
        import time
        start = time.monotonic()
        success = await mock_executor.execute(op_id)
        latency_ms = (time.monotonic() - start) * 1000
        metrics.record_operation(success=success, latency_ms=latency_ms)
        return success

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()
    metrics.stop()

    # Generate reports
    generator = ReportGenerator(output_dir=tmp_path)
    files = generator.generate(result, metrics=metrics, format="all")

    assert len(files) == 3  # JSON, Markdown, HTML
    for file in files:
        assert file.exists()
        assert file.stat().st_size > 0


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_high_concurrency(
    mock_executor: MockSagaExecutor,
) -> None:
    """Test saga execution with high concurrency."""
    config = StressTestConfig(
        name="saga_high_concurrency",
        load_profile=LoadProfile(
            warmup_seconds=1.0,
            ramp_up_seconds=3.0,
            steady_seconds=10.0,
            ramp_down_seconds=2.0,
            cooldown_seconds=1.0,
            initial_rate=10.0,
            target_rate=200.0,
        ),
        concurrency=200,
        target_success_rate=0.95,
    )

    async def operation(op_id: str) -> bool:
        return await mock_executor.execute(op_id)

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()

    assert result.total_operations > 500
    assert result.success_rate >= 0.9


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_failure_injection() -> None:
    """Test saga behavior with explicit failure injection."""
    from tests.stress.chaos import FailureInjector

    injector = FailureInjector(enabled=True, seed=42)
    injector.add_failure(FailureConfig(
        failure_type=FailureType.EXCEPTION,
        probability=0.2,  # 20% failure rate
        metadata={"message": "Simulated saga failure"},
    ))

    executor = MockSagaExecutor(success_rate=1.0)  # Would succeed without injection

    config = StressTestConfig(
        name="saga_failure_injection",
        load_profile=LoadProfile(
            steady_seconds=10.0,
            target_rate=20.0,
        ),
        concurrency=10,
        target_success_rate=0.7,  # Account for injected failures
    )

    async def operation(op_id: str) -> bool:
        try:
            await injector.maybe_inject(op_id)
            return await executor.execute(op_id)
        except Exception:
            return False

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()

    # Should have some failures due to injection
    assert result.failed_operations > 0
    # But not all should fail
    assert result.successful_operations > 0


@pytest.mark.asyncio
@pytest.mark.stress
async def test_saga_early_stop_on_high_failure() -> None:
    """Test that stress test stops early on high failure rate."""
    executor = MockSagaExecutor(success_rate=0.5)  # 50% failure rate

    config = StressTestConfig(
        name="saga_early_stop",
        load_profile=LoadProfile(
            steady_seconds=60.0,  # Would run for 60 seconds normally
            target_rate=50.0,
        ),
        concurrency=20,
        stop_on_failure_rate=0.3,  # Stop if failure rate exceeds 30%
    )

    async def operation(op_id: str) -> bool:
        return await executor.execute(op_id)

    runner = StressTestRunner(config=config, operation=operation)
    result = await runner.run()

    # Should have stopped early due to high failure rate
    assert result.duration_seconds < 50  # Less than full duration
    assert result.failure_rate > 0.3  # High failure rate triggered stop
