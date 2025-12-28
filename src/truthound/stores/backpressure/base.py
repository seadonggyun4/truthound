"""Base classes and protocols for backpressure management.

This module defines the abstract interfaces and data structures for
backpressure strategies used in streaming stores.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class PressureLevel(str, Enum):
    """Pressure levels indicating system load."""

    NONE = "none"  # No pressure, full speed
    LOW = "low"  # Minor pressure, slight throttling
    MEDIUM = "medium"  # Moderate pressure, significant throttling
    HIGH = "high"  # High pressure, aggressive throttling
    CRITICAL = "critical"  # Critical pressure, pause operations


@dataclass
class BackpressureConfig:
    """Configuration for backpressure management.

    Attributes:
        enabled: Whether backpressure is enabled.
        memory_threshold_percent: Memory usage threshold (0-100).
        queue_depth_threshold: Maximum queue depth before throttling.
        latency_threshold_ms: Latency threshold in milliseconds.
        min_pause_ms: Minimum pause duration in milliseconds.
        max_pause_ms: Maximum pause duration in milliseconds.
        base_rate: Base operations per second.
        min_rate: Minimum operations per second.
        adaptive_window_size: Window size for adaptive calculations.
        recovery_rate: Rate multiplier for recovery (0-1).
        sampling_interval_ms: Metrics sampling interval.
        pressure_decay_factor: Factor for pressure decay over time.
    """

    enabled: bool = True
    memory_threshold_percent: float = 80.0
    queue_depth_threshold: int = 10000
    latency_threshold_ms: float = 100.0
    min_pause_ms: float = 10.0
    max_pause_ms: float = 5000.0
    base_rate: float = 10000.0  # ops/sec
    min_rate: float = 100.0  # ops/sec
    adaptive_window_size: int = 100
    recovery_rate: float = 0.1
    sampling_interval_ms: float = 100.0
    pressure_decay_factor: float = 0.95

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 <= self.memory_threshold_percent <= 100:
            raise ValueError("memory_threshold_percent must be between 0 and 100")
        if self.queue_depth_threshold <= 0:
            raise ValueError("queue_depth_threshold must be positive")
        if self.latency_threshold_ms <= 0:
            raise ValueError("latency_threshold_ms must be positive")
        if self.min_pause_ms < 0:
            raise ValueError("min_pause_ms must be non-negative")
        if self.max_pause_ms < self.min_pause_ms:
            raise ValueError("max_pause_ms must be >= min_pause_ms")
        if self.base_rate <= 0:
            raise ValueError("base_rate must be positive")
        if self.min_rate <= 0:
            raise ValueError("min_rate must be positive")
        if self.min_rate > self.base_rate:
            raise ValueError("min_rate must be <= base_rate")
        if not 0 < self.recovery_rate <= 1:
            raise ValueError("recovery_rate must be between 0 and 1")
        if not 0 < self.pressure_decay_factor < 1:
            raise ValueError("pressure_decay_factor must be between 0 and 1")


@dataclass
class BackpressureMetrics:
    """Metrics for backpressure monitoring.

    Attributes:
        current_memory_percent: Current memory usage percentage.
        current_queue_depth: Current queue depth.
        current_latency_ms: Current operation latency in ms.
        current_rate: Current operations per second.
        pressure_level: Current pressure level.
        pause_count: Number of pauses triggered.
        total_pause_time_ms: Total time spent paused.
        throttled_ops: Number of throttled operations.
        dropped_ops: Number of dropped operations.
        last_update: Last metrics update time.
        window_samples: Recent samples for adaptive calculation.
    """

    current_memory_percent: float = 0.0
    current_queue_depth: int = 0
    current_latency_ms: float = 0.0
    current_rate: float = 0.0
    pressure_level: PressureLevel = PressureLevel.NONE
    pause_count: int = 0
    total_pause_time_ms: float = 0.0
    throttled_ops: int = 0
    dropped_ops: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    window_samples: list[float] = field(default_factory=list)

    def update_memory(self, percent: float) -> None:
        """Update memory usage."""
        self.current_memory_percent = percent
        self.last_update = datetime.now()

    def update_queue_depth(self, depth: int) -> None:
        """Update queue depth."""
        self.current_queue_depth = depth
        self.last_update = datetime.now()

    def update_latency(self, latency_ms: float) -> None:
        """Update latency."""
        self.current_latency_ms = latency_ms
        self.last_update = datetime.now()

    def record_pause(self, duration_ms: float) -> None:
        """Record a pause event."""
        self.pause_count += 1
        self.total_pause_time_ms += duration_ms

    def record_throttle(self) -> None:
        """Record a throttled operation."""
        self.throttled_ops += 1

    def record_drop(self) -> None:
        """Record a dropped operation."""
        self.dropped_ops += 1

    def add_sample(self, value: float, max_samples: int = 100) -> None:
        """Add a sample to the window."""
        self.window_samples.append(value)
        if len(self.window_samples) > max_samples:
            self.window_samples.pop(0)

    def get_average_sample(self) -> float:
        """Get average of window samples."""
        if not self.window_samples:
            return 0.0
        return sum(self.window_samples) / len(self.window_samples)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_memory_percent": self.current_memory_percent,
            "current_queue_depth": self.current_queue_depth,
            "current_latency_ms": self.current_latency_ms,
            "current_rate": self.current_rate,
            "pressure_level": self.pressure_level.value,
            "pause_count": self.pause_count,
            "total_pause_time_ms": self.total_pause_time_ms,
            "throttled_ops": self.throttled_ops,
            "dropped_ops": self.dropped_ops,
            "last_update": self.last_update.isoformat(),
        }


@dataclass
class BackpressureState:
    """Current state of backpressure control.

    Attributes:
        is_paused: Whether operations are paused.
        current_rate: Current allowed rate.
        pressure_level: Current pressure level.
        last_adjustment: Last rate adjustment time.
        consecutive_high_pressure: Count of consecutive high pressure readings.
        consecutive_low_pressure: Count of consecutive low pressure readings.
    """

    is_paused: bool = False
    current_rate: float = 10000.0
    pressure_level: PressureLevel = PressureLevel.NONE
    last_adjustment: datetime = field(default_factory=datetime.now)
    consecutive_high_pressure: int = 0
    consecutive_low_pressure: int = 0

    def set_paused(self, paused: bool) -> None:
        """Set pause state."""
        self.is_paused = paused
        self.last_adjustment = datetime.now()

    def update_pressure(self, level: PressureLevel) -> None:
        """Update pressure level and track consecutive readings."""
        if level in (PressureLevel.HIGH, PressureLevel.CRITICAL):
            self.consecutive_high_pressure += 1
            self.consecutive_low_pressure = 0
        elif level == PressureLevel.NONE:
            self.consecutive_low_pressure += 1
            self.consecutive_high_pressure = 0
        else:
            self.consecutive_high_pressure = 0
            self.consecutive_low_pressure = 0
        self.pressure_level = level
        self.last_adjustment = datetime.now()

    def adjust_rate(self, new_rate: float) -> None:
        """Adjust the current rate."""
        self.current_rate = new_rate
        self.last_adjustment = datetime.now()


@runtime_checkable
class BackpressureStrategy(Protocol):
    """Protocol for backpressure strategies."""

    @property
    def config(self) -> BackpressureConfig:
        """Get the configuration."""
        ...

    @property
    def metrics(self) -> BackpressureMetrics:
        """Get current metrics."""
        ...

    @property
    def state(self) -> BackpressureState:
        """Get current state."""
        ...

    def should_pause(self) -> bool:
        """Check if operations should pause."""
        ...

    def calculate_pause_duration(self) -> float:
        """Calculate pause duration in seconds."""
        ...

    def calculate_rate(self) -> float:
        """Calculate allowed rate (ops/sec)."""
        ...

    def update_metrics(
        self,
        memory_percent: float | None = None,
        queue_depth: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Update metrics with new values."""
        ...

    async def acquire(self) -> bool:
        """Acquire permission to proceed (async)."""
        ...

    def release(self) -> None:
        """Release after operation completes."""
        ...

    def reset(self) -> None:
        """Reset to initial state."""
        ...


class BaseBackpressure(ABC):
    """Abstract base class for backpressure implementations.

    Provides common functionality for all backpressure strategies.
    """

    def __init__(self, config: BackpressureConfig | None = None) -> None:
        """Initialize backpressure strategy.

        Args:
            config: Backpressure configuration.
        """
        self._config = config or BackpressureConfig()
        self._config.validate()
        self._metrics = BackpressureMetrics()
        self._state = BackpressureState(current_rate=self._config.base_rate)
        self._lock = asyncio.Lock()
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def config(self) -> BackpressureConfig:
        """Get the configuration."""
        return self._config

    @property
    def metrics(self) -> BackpressureMetrics:
        """Get current metrics."""
        return self._metrics

    @property
    def state(self) -> BackpressureState:
        """Get current state."""
        return self._state

    def update_metrics(
        self,
        memory_percent: float | None = None,
        queue_depth: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Update metrics with new values."""
        if memory_percent is not None:
            self._metrics.update_memory(memory_percent)
        if queue_depth is not None:
            self._metrics.update_queue_depth(queue_depth)
        if latency_ms is not None:
            self._metrics.update_latency(latency_ms)

        # Recalculate pressure level
        self._update_pressure_level()

    @abstractmethod
    def _update_pressure_level(self) -> None:
        """Update pressure level based on current metrics."""
        pass

    @abstractmethod
    def should_pause(self) -> bool:
        """Check if operations should pause."""
        pass

    @abstractmethod
    def calculate_pause_duration(self) -> float:
        """Calculate pause duration in seconds."""
        pass

    @abstractmethod
    def calculate_rate(self) -> float:
        """Calculate allowed rate (ops/sec)."""
        pass

    async def acquire(self) -> bool:
        """Acquire permission to proceed.

        Returns:
            True if acquired, False if should drop.
        """
        if not self._config.enabled:
            return True

        async with self._lock:
            if self.should_pause():
                pause_duration = self.calculate_pause_duration()
                self._state.set_paused(True)
                self._metrics.record_pause(pause_duration * 1000)
                await asyncio.sleep(pause_duration)
                self._state.set_paused(False)

            # Update rate
            new_rate = self.calculate_rate()
            self._state.adjust_rate(new_rate)

            # Check if we should throttle
            if self._state.pressure_level == PressureLevel.CRITICAL:
                self._metrics.record_throttle()
                return True  # Still allow, but record

            return True

    def release(self) -> None:
        """Release after operation completes."""
        # Decay pressure over time
        if self._state.consecutive_low_pressure > 5:
            self._state.update_pressure(PressureLevel.NONE)

    def reset(self) -> None:
        """Reset to initial state."""
        self._metrics = BackpressureMetrics()
        self._state = BackpressureState(current_rate=self._config.base_rate)
