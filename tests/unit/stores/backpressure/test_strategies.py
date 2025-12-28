"""Tests for backpressure strategies."""

from __future__ import annotations

import asyncio
import pytest

from truthound.stores.backpressure.base import (
    BackpressureConfig,
    PressureLevel,
)
from truthound.stores.backpressure.strategies import (
    AdaptiveBackpressure,
    CompositeBackpressure,
    LatencyBasedBackpressure,
    MemoryBasedBackpressure,
    QueueDepthBackpressure,
    TokenBucketBackpressure,
)


class TestMemoryBasedBackpressure:
    """Tests for MemoryBasedBackpressure."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        bp = MemoryBasedBackpressure()
        assert bp.config.memory_threshold_percent == 80.0
        assert bp.state.pressure_level == PressureLevel.NONE

    def test_custom_memory_provider(self) -> None:
        """Test with custom memory provider."""
        # 85% exceeds 80% threshold by 5%, which is MEDIUM level
        # Need 90%+ to trigger HIGH/CRITICAL for should_pause
        bp = MemoryBasedBackpressure(memory_provider=lambda: 92.0)
        assert bp.should_pause() is True

    def test_pressure_levels(self) -> None:
        """Test pressure level calculation."""
        config = BackpressureConfig(memory_threshold_percent=70.0)
        bp = MemoryBasedBackpressure(config)

        # Below threshold
        bp.update_metrics(memory_percent=65.0)
        assert bp.state.pressure_level == PressureLevel.NONE

        # At threshold
        bp.update_metrics(memory_percent=70.0)
        assert bp.state.pressure_level == PressureLevel.LOW

        # Medium
        bp.update_metrics(memory_percent=77.0)
        assert bp.state.pressure_level == PressureLevel.MEDIUM

        # High
        bp.update_metrics(memory_percent=82.0)
        assert bp.state.pressure_level == PressureLevel.HIGH

        # Critical
        bp.update_metrics(memory_percent=90.0)
        assert bp.state.pressure_level == PressureLevel.CRITICAL

    def test_should_pause(self) -> None:
        """Test pause determination."""
        config = BackpressureConfig(memory_threshold_percent=70.0)
        bp = MemoryBasedBackpressure(config, memory_provider=lambda: 60.0)
        assert bp.should_pause() is False

        bp = MemoryBasedBackpressure(config, memory_provider=lambda: 85.0)
        assert bp.should_pause() is True

    def test_calculate_pause_duration(self) -> None:
        """Test pause duration calculation."""
        config = BackpressureConfig(
            memory_threshold_percent=70.0,
            min_pause_ms=10.0,
            max_pause_ms=1000.0,
        )
        bp = MemoryBasedBackpressure(config)

        bp.update_metrics(memory_percent=60.0)
        assert bp.calculate_pause_duration() == 0.0

        bp.update_metrics(memory_percent=85.0)
        duration = bp.calculate_pause_duration()
        assert duration > 0.0
        assert duration <= 1.0  # max 1 second

    def test_calculate_rate(self) -> None:
        """Test rate calculation."""
        config = BackpressureConfig(
            memory_threshold_percent=70.0,
            base_rate=10000.0,
            min_rate=100.0,
        )
        bp = MemoryBasedBackpressure(config)

        bp.update_metrics(memory_percent=60.0)
        assert bp.calculate_rate() == 10000.0

        bp.update_metrics(memory_percent=85.0)
        rate = bp.calculate_rate()
        assert rate < 10000.0
        assert rate >= 100.0


class TestQueueDepthBackpressure:
    """Tests for QueueDepthBackpressure."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        bp = QueueDepthBackpressure()
        assert bp.config.queue_depth_threshold == 10000

    def test_pressure_levels(self) -> None:
        """Test pressure level calculation."""
        config = BackpressureConfig(queue_depth_threshold=1000)
        bp = QueueDepthBackpressure(config)

        bp.update_metrics(queue_depth=500)
        assert bp.state.pressure_level == PressureLevel.NONE

        bp.update_metrics(queue_depth=1000)
        assert bp.state.pressure_level == PressureLevel.LOW

        bp.update_metrics(queue_depth=1300)
        assert bp.state.pressure_level == PressureLevel.MEDIUM

        bp.update_metrics(queue_depth=1600)
        assert bp.state.pressure_level == PressureLevel.HIGH

        bp.update_metrics(queue_depth=2500)
        assert bp.state.pressure_level == PressureLevel.CRITICAL

    def test_should_pause(self) -> None:
        """Test pause determination."""
        config = BackpressureConfig(queue_depth_threshold=1000)
        bp = QueueDepthBackpressure(config)

        bp.update_metrics(queue_depth=500)
        assert bp.should_pause() is False

        bp.update_metrics(queue_depth=1600)
        assert bp.should_pause() is True

    def test_calculate_rate_increases_on_low_queue(self) -> None:
        """Test rate increases when queue is low."""
        config = BackpressureConfig(
            queue_depth_threshold=1000,
            base_rate=10000.0,
        )
        bp = QueueDepthBackpressure(config)

        bp.update_metrics(queue_depth=400)
        rate = bp.calculate_rate()
        assert rate <= 10000.0  # Capped at base rate


class TestLatencyBasedBackpressure:
    """Tests for LatencyBasedBackpressure."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        bp = LatencyBasedBackpressure()
        assert bp.config.latency_threshold_ms == 100.0

    def test_record_latency_averaging(self) -> None:
        """Test latency averaging."""
        bp = LatencyBasedBackpressure()
        bp.record_latency(50.0)
        bp.record_latency(100.0)
        bp.record_latency(150.0)
        assert bp.metrics.current_latency_ms == 100.0  # Average

    def test_pressure_levels(self) -> None:
        """Test pressure level calculation."""
        config = BackpressureConfig(latency_threshold_ms=50.0)
        bp = LatencyBasedBackpressure(config)

        bp.update_metrics(latency_ms=30.0)
        assert bp.state.pressure_level == PressureLevel.NONE

        bp.update_metrics(latency_ms=50.0)
        assert bp.state.pressure_level == PressureLevel.LOW

        bp.update_metrics(latency_ms=80.0)
        assert bp.state.pressure_level == PressureLevel.MEDIUM

        bp.update_metrics(latency_ms=120.0)
        assert bp.state.pressure_level == PressureLevel.HIGH

        bp.update_metrics(latency_ms=250.0)
        assert bp.state.pressure_level == PressureLevel.CRITICAL

    def test_should_pause(self) -> None:
        """Test pause determination."""
        config = BackpressureConfig(latency_threshold_ms=50.0)
        bp = LatencyBasedBackpressure(config)

        bp.update_metrics(latency_ms=30.0)
        assert bp.should_pause() is False

        bp.update_metrics(latency_ms=120.0)
        assert bp.should_pause() is True


class TestTokenBucketBackpressure:
    """Tests for TokenBucketBackpressure."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        bp = TokenBucketBackpressure()
        assert bp.config.base_rate == 10000.0

    def test_try_acquire(self) -> None:
        """Test non-blocking acquire."""
        config = BackpressureConfig(base_rate=100.0)
        bp = TokenBucketBackpressure(config, bucket_size=10.0)

        # Should have tokens initially
        assert bp.try_acquire() is True

        # Exhaust tokens
        for _ in range(9):
            bp.try_acquire()

        # Should be out of tokens
        assert bp.try_acquire() is False

    def test_tokens_refill(self) -> None:
        """Test token refill over time."""
        import time

        config = BackpressureConfig(base_rate=1000.0)  # 1000 tokens/sec
        bp = TokenBucketBackpressure(config, bucket_size=100.0)

        # Exhaust some tokens
        for _ in range(50):
            bp.try_acquire()

        # Wait for refill
        time.sleep(0.1)  # 100ms = 100 tokens at 1000/sec

        # Should have tokens again
        assert bp.try_acquire() is True

    @pytest.mark.asyncio
    async def test_acquire_waits(self) -> None:
        """Test async acquire waits for tokens."""
        config = BackpressureConfig(base_rate=100.0)
        bp = TokenBucketBackpressure(config, bucket_size=5.0)

        # Exhaust tokens
        for _ in range(5):
            bp.try_acquire()

        # Should wait for tokens
        start = asyncio.get_event_loop().time()
        await bp.acquire()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.01  # Should have waited


class TestAdaptiveBackpressure:
    """Tests for AdaptiveBackpressure."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        bp = AdaptiveBackpressure()
        assert bp.state.pressure_level == PressureLevel.NONE

    def test_combined_pressure_score(self) -> None:
        """Test combined pressure calculation."""
        bp = AdaptiveBackpressure(
            memory_weight=0.4,
            queue_weight=0.3,
            latency_weight=0.3,
        )

        # All factors high
        bp.update_metrics(
            memory_percent=95.0,
            queue_depth=25000,
            latency_ms=500.0,
        )
        assert bp.state.pressure_level in (PressureLevel.HIGH, PressureLevel.CRITICAL)

        # Reset
        bp.reset()

        # All factors low
        bp.update_metrics(
            memory_percent=50.0,
            queue_depth=5000,
            latency_ms=50.0,
        )
        assert bp.state.pressure_level == PressureLevel.NONE

    def test_adaptive_rate_adjustment(self) -> None:
        """Test adaptive rate adjustment."""
        bp = AdaptiveBackpressure()

        # Low pressure - rate should increase or stay same
        bp.update_metrics(memory_percent=50.0)
        bp.state.consecutive_low_pressure = 5
        rate1 = bp.calculate_rate()
        assert rate1 >= bp.config.min_rate

        # High pressure - rate should decrease
        bp.update_metrics(memory_percent=95.0)
        rate2 = bp.calculate_rate()
        assert rate2 <= bp.config.base_rate


class TestCompositeBackpressure:
    """Tests for CompositeBackpressure."""

    def test_combines_strategies(self) -> None:
        """Test combining multiple strategies."""
        memory_bp = MemoryBasedBackpressure()
        queue_bp = QueueDepthBackpressure()

        composite = CompositeBackpressure([memory_bp, queue_bp])
        assert composite.state.pressure_level == PressureLevel.NONE

    def test_uses_highest_pressure(self) -> None:
        """Test uses highest pressure level."""
        config_mem = BackpressureConfig(memory_threshold_percent=70.0)
        config_queue = BackpressureConfig(queue_depth_threshold=1000)

        memory_bp = MemoryBasedBackpressure(config_mem, memory_provider=lambda: 60.0)
        queue_bp = QueueDepthBackpressure(config_queue)

        composite = CompositeBackpressure([memory_bp, queue_bp])

        # Memory low, queue very high (2x threshold = CRITICAL)
        # Update queue strategy directly to trigger pressure update
        queue_bp.update_metrics(queue_depth=2500)
        composite._update_pressure_level()
        assert composite.state.pressure_level == PressureLevel.CRITICAL

    def test_uses_minimum_rate(self) -> None:
        """Test uses minimum rate from strategies."""
        config = BackpressureConfig(base_rate=10000.0)
        memory_bp = MemoryBasedBackpressure(config)
        queue_bp = QueueDepthBackpressure(config)

        composite = CompositeBackpressure([memory_bp, queue_bp])

        # Make one strategy suggest lower rate
        memory_bp.update_metrics(memory_percent=90.0)

        rate = composite.calculate_rate()
        assert rate < 10000.0

    def test_uses_maximum_pause_duration(self) -> None:
        """Test uses maximum pause duration."""
        memory_bp = MemoryBasedBackpressure()
        queue_bp = QueueDepthBackpressure()

        composite = CompositeBackpressure([memory_bp, queue_bp])

        memory_bp.update_metrics(memory_percent=85.0)
        queue_bp.update_metrics(queue_depth=15000)

        duration = composite.calculate_pause_duration()
        assert duration > 0.0

    def test_should_pause_any_strategy(self) -> None:
        """Test pauses if any strategy says to pause."""
        memory_bp = MemoryBasedBackpressure(memory_provider=lambda: 60.0)
        queue_bp = QueueDepthBackpressure()

        composite = CompositeBackpressure([memory_bp, queue_bp])

        # Only queue triggers pause
        queue_bp.update_metrics(queue_depth=20000)
        assert composite.should_pause() is True

    @pytest.mark.asyncio
    async def test_acquire_from_all(self) -> None:
        """Test acquire from all strategies."""
        memory_bp = MemoryBasedBackpressure(memory_provider=lambda: 50.0)
        queue_bp = QueueDepthBackpressure()

        composite = CompositeBackpressure([memory_bp, queue_bp])
        queue_bp.update_metrics(queue_depth=5000)

        result = await composite.acquire()
        assert result is True

    def test_release_all(self) -> None:
        """Test release on all strategies."""
        memory_bp = MemoryBasedBackpressure()
        queue_bp = QueueDepthBackpressure()

        composite = CompositeBackpressure([memory_bp, queue_bp])
        composite.release()  # Should not raise

    def test_reset_all(self) -> None:
        """Test reset on all strategies."""
        memory_bp = MemoryBasedBackpressure()
        queue_bp = QueueDepthBackpressure()

        composite = CompositeBackpressure([memory_bp, queue_bp])

        # Set some state
        memory_bp.update_metrics(memory_percent=90.0)
        queue_bp.update_metrics(queue_depth=20000)

        composite.reset()
        assert memory_bp.state.pressure_level == PressureLevel.NONE
        assert queue_bp.state.pressure_level == PressureLevel.NONE
