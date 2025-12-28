"""Tests for backpressure base module."""

from __future__ import annotations

import pytest
from datetime import datetime

from truthound.stores.backpressure.base import (
    BackpressureConfig,
    BackpressureMetrics,
    BackpressureState,
    PressureLevel,
)


class TestBackpressureConfig:
    """Tests for BackpressureConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BackpressureConfig()
        assert config.enabled is True
        assert config.memory_threshold_percent == 80.0
        assert config.queue_depth_threshold == 10000
        assert config.latency_threshold_ms == 100.0
        assert config.min_pause_ms == 10.0
        assert config.max_pause_ms == 5000.0
        assert config.base_rate == 10000.0
        assert config.min_rate == 100.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BackpressureConfig(
            memory_threshold_percent=70.0,
            queue_depth_threshold=5000,
            latency_threshold_ms=50.0,
        )
        assert config.memory_threshold_percent == 70.0
        assert config.queue_depth_threshold == 5000
        assert config.latency_threshold_ms == 50.0

    def test_validate_memory_threshold(self) -> None:
        """Test memory threshold validation."""
        with pytest.raises(ValueError, match="memory_threshold_percent"):
            config = BackpressureConfig(memory_threshold_percent=150.0)
            config.validate()

        with pytest.raises(ValueError, match="memory_threshold_percent"):
            config = BackpressureConfig(memory_threshold_percent=-10.0)
            config.validate()

    def test_validate_queue_depth(self) -> None:
        """Test queue depth validation."""
        with pytest.raises(ValueError, match="queue_depth_threshold"):
            config = BackpressureConfig(queue_depth_threshold=0)
            config.validate()

    def test_validate_latency_threshold(self) -> None:
        """Test latency threshold validation."""
        with pytest.raises(ValueError, match="latency_threshold_ms"):
            config = BackpressureConfig(latency_threshold_ms=-5.0)
            config.validate()

    def test_validate_pause_times(self) -> None:
        """Test pause time validation."""
        with pytest.raises(ValueError, match="min_pause_ms"):
            config = BackpressureConfig(min_pause_ms=-10.0)
            config.validate()

        with pytest.raises(ValueError, match="max_pause_ms"):
            config = BackpressureConfig(min_pause_ms=100.0, max_pause_ms=50.0)
            config.validate()

    def test_validate_rates(self) -> None:
        """Test rate validation."""
        with pytest.raises(ValueError, match="base_rate"):
            config = BackpressureConfig(base_rate=0)
            config.validate()

        with pytest.raises(ValueError, match="min_rate"):
            config = BackpressureConfig(min_rate=0)
            config.validate()

        with pytest.raises(ValueError, match="min_rate must be <= base_rate"):
            config = BackpressureConfig(min_rate=20000.0, base_rate=10000.0)
            config.validate()

    def test_validate_recovery_rate(self) -> None:
        """Test recovery rate validation."""
        with pytest.raises(ValueError, match="recovery_rate"):
            config = BackpressureConfig(recovery_rate=0)
            config.validate()

        with pytest.raises(ValueError, match="recovery_rate"):
            config = BackpressureConfig(recovery_rate=1.5)
            config.validate()


class TestBackpressureMetrics:
    """Tests for BackpressureMetrics."""

    def test_default_values(self) -> None:
        """Test default metrics values."""
        metrics = BackpressureMetrics()
        assert metrics.current_memory_percent == 0.0
        assert metrics.current_queue_depth == 0
        assert metrics.current_latency_ms == 0.0
        assert metrics.current_rate == 0.0
        assert metrics.pressure_level == PressureLevel.NONE
        assert metrics.pause_count == 0
        assert metrics.total_pause_time_ms == 0.0
        assert metrics.throttled_ops == 0
        assert metrics.dropped_ops == 0

    def test_update_memory(self) -> None:
        """Test memory update."""
        metrics = BackpressureMetrics()
        old_update = metrics.last_update
        metrics.update_memory(75.5)
        assert metrics.current_memory_percent == 75.5
        assert metrics.last_update >= old_update

    def test_update_queue_depth(self) -> None:
        """Test queue depth update."""
        metrics = BackpressureMetrics()
        metrics.update_queue_depth(5000)
        assert metrics.current_queue_depth == 5000

    def test_update_latency(self) -> None:
        """Test latency update."""
        metrics = BackpressureMetrics()
        metrics.update_latency(50.0)
        assert metrics.current_latency_ms == 50.0

    def test_record_pause(self) -> None:
        """Test pause recording."""
        metrics = BackpressureMetrics()
        metrics.record_pause(100.0)
        assert metrics.pause_count == 1
        assert metrics.total_pause_time_ms == 100.0

        metrics.record_pause(50.0)
        assert metrics.pause_count == 2
        assert metrics.total_pause_time_ms == 150.0

    def test_record_throttle(self) -> None:
        """Test throttle recording."""
        metrics = BackpressureMetrics()
        metrics.record_throttle()
        assert metrics.throttled_ops == 1

    def test_record_drop(self) -> None:
        """Test drop recording."""
        metrics = BackpressureMetrics()
        metrics.record_drop()
        assert metrics.dropped_ops == 1

    def test_add_sample(self) -> None:
        """Test sample window."""
        metrics = BackpressureMetrics()
        for i in range(5):
            metrics.add_sample(float(i))
        assert len(metrics.window_samples) == 5
        assert metrics.window_samples == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_add_sample_max_window(self) -> None:
        """Test sample window max size."""
        metrics = BackpressureMetrics()
        for i in range(150):
            metrics.add_sample(float(i), max_samples=100)
        assert len(metrics.window_samples) == 100
        assert metrics.window_samples[0] == 50.0  # First 50 were evicted

    def test_get_average_sample(self) -> None:
        """Test average sample calculation."""
        metrics = BackpressureMetrics()
        assert metrics.get_average_sample() == 0.0

        metrics.add_sample(10.0)
        metrics.add_sample(20.0)
        metrics.add_sample(30.0)
        assert metrics.get_average_sample() == 20.0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        metrics = BackpressureMetrics()
        metrics.update_memory(70.0)
        metrics.update_queue_depth(5000)
        metrics.record_pause(100.0)

        d = metrics.to_dict()
        assert d["current_memory_percent"] == 70.0
        assert d["current_queue_depth"] == 5000
        assert d["pause_count"] == 1
        assert d["pressure_level"] == "none"
        assert "last_update" in d


class TestBackpressureState:
    """Tests for BackpressureState."""

    def test_default_values(self) -> None:
        """Test default state values."""
        state = BackpressureState()
        assert state.is_paused is False
        assert state.current_rate == 10000.0
        assert state.pressure_level == PressureLevel.NONE
        assert state.consecutive_high_pressure == 0
        assert state.consecutive_low_pressure == 0

    def test_set_paused(self) -> None:
        """Test setting pause state."""
        state = BackpressureState()
        old_time = state.last_adjustment
        state.set_paused(True)
        assert state.is_paused is True
        assert state.last_adjustment >= old_time

        state.set_paused(False)
        assert state.is_paused is False

    def test_update_pressure_high(self) -> None:
        """Test updating pressure level to high."""
        state = BackpressureState()
        state.update_pressure(PressureLevel.HIGH)
        assert state.pressure_level == PressureLevel.HIGH
        assert state.consecutive_high_pressure == 1
        assert state.consecutive_low_pressure == 0

        state.update_pressure(PressureLevel.CRITICAL)
        assert state.consecutive_high_pressure == 2

    def test_update_pressure_none(self) -> None:
        """Test updating pressure level to none."""
        state = BackpressureState()
        state.update_pressure(PressureLevel.NONE)
        assert state.consecutive_low_pressure == 1
        assert state.consecutive_high_pressure == 0

        state.update_pressure(PressureLevel.NONE)
        assert state.consecutive_low_pressure == 2

    def test_update_pressure_medium_resets(self) -> None:
        """Test medium pressure resets consecutive counts."""
        state = BackpressureState()
        state.consecutive_high_pressure = 3
        state.consecutive_low_pressure = 2

        state.update_pressure(PressureLevel.MEDIUM)
        assert state.consecutive_high_pressure == 0
        assert state.consecutive_low_pressure == 0

    def test_adjust_rate(self) -> None:
        """Test rate adjustment."""
        state = BackpressureState()
        state.adjust_rate(5000.0)
        assert state.current_rate == 5000.0


class TestPressureLevel:
    """Tests for PressureLevel enum."""

    def test_values(self) -> None:
        """Test pressure level values."""
        assert PressureLevel.NONE.value == "none"
        assert PressureLevel.LOW.value == "low"
        assert PressureLevel.MEDIUM.value == "medium"
        assert PressureLevel.HIGH.value == "high"
        assert PressureLevel.CRITICAL.value == "critical"

    def test_from_string(self) -> None:
        """Test creating from string."""
        assert PressureLevel("none") == PressureLevel.NONE
        assert PressureLevel("critical") == PressureLevel.CRITICAL
