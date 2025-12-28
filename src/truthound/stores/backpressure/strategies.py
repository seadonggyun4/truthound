"""Backpressure strategy implementations.

This module provides various backpressure strategies:
- MemoryBasedBackpressure: Based on memory usage
- QueueDepthBackpressure: Based on queue depth
- LatencyBasedBackpressure: Based on operation latency
- TokenBucketBackpressure: Token bucket rate limiting
- AdaptiveBackpressure: Adaptive multi-factor strategy
- CompositeBackpressure: Combines multiple strategies
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from truthound.stores.backpressure.base import (
    BackpressureConfig,
    BackpressureMetrics,
    BackpressureState,
    BaseBackpressure,
    PressureLevel,
)

if TYPE_CHECKING:
    pass


class MemoryBasedBackpressure(BaseBackpressure):
    """Backpressure based on memory usage.

    Monitors memory usage and applies backpressure when memory
    exceeds configured thresholds.

    Example:
        >>> config = BackpressureConfig(memory_threshold_percent=75.0)
        >>> bp = MemoryBasedBackpressure(config)
        >>> bp.update_metrics(memory_percent=80.0)
        >>> bp.should_pause()
        True
    """

    def __init__(
        self,
        config: BackpressureConfig | None = None,
        memory_provider: callable | None = None,
    ) -> None:
        """Initialize memory-based backpressure.

        Args:
            config: Backpressure configuration.
            memory_provider: Optional callable that returns current memory percent.
        """
        super().__init__(config)
        self._memory_provider = memory_provider or self._default_memory_provider

    @staticmethod
    def _default_memory_provider() -> float:
        """Get current memory usage percentage."""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    def _update_pressure_level(self) -> None:
        """Update pressure level based on memory usage."""
        mem = self._metrics.current_memory_percent
        threshold = self._config.memory_threshold_percent

        if mem >= threshold + 15:
            level = PressureLevel.CRITICAL
        elif mem >= threshold + 10:
            level = PressureLevel.HIGH
        elif mem >= threshold + 5:
            level = PressureLevel.MEDIUM
        elif mem >= threshold:
            level = PressureLevel.LOW
        else:
            level = PressureLevel.NONE

        self._state.update_pressure(level)
        self._metrics.pressure_level = level

    def should_pause(self) -> bool:
        """Check if should pause based on memory."""
        # Auto-update from provider
        current = self._memory_provider()
        self.update_metrics(memory_percent=current)

        return self._state.pressure_level in (
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        )

    def calculate_pause_duration(self) -> float:
        """Calculate pause duration based on memory pressure."""
        mem = self._metrics.current_memory_percent
        threshold = self._config.memory_threshold_percent

        if mem <= threshold:
            return 0.0

        # Linear scaling based on how much over threshold
        excess = (mem - threshold) / (100 - threshold)
        duration_ms = self._config.min_pause_ms + (
            excess * (self._config.max_pause_ms - self._config.min_pause_ms)
        )

        return duration_ms / 1000.0

    def calculate_rate(self) -> float:
        """Calculate allowed rate based on memory pressure."""
        mem = self._metrics.current_memory_percent
        threshold = self._config.memory_threshold_percent

        if mem <= threshold:
            return self._config.base_rate

        # Reduce rate as memory increases
        excess = (mem - threshold) / (100 - threshold)
        reduction = 1.0 - (excess * 0.9)  # Max 90% reduction

        new_rate = self._config.base_rate * reduction
        return max(new_rate, self._config.min_rate)


class QueueDepthBackpressure(BaseBackpressure):
    """Backpressure based on queue depth.

    Monitors queue depth and applies backpressure when the queue
    exceeds configured thresholds.

    Example:
        >>> config = BackpressureConfig(queue_depth_threshold=5000)
        >>> bp = QueueDepthBackpressure(config)
        >>> bp.update_metrics(queue_depth=7500)
        >>> bp.calculate_rate()  # Returns reduced rate
    """

    def __init__(
        self,
        config: BackpressureConfig | None = None,
        queue_provider: callable | None = None,
    ) -> None:
        """Initialize queue-depth-based backpressure.

        Args:
            config: Backpressure configuration.
            queue_provider: Optional callable that returns current queue depth.
        """
        super().__init__(config)
        self._queue_provider = queue_provider

    def _update_pressure_level(self) -> None:
        """Update pressure level based on queue depth."""
        depth = self._metrics.current_queue_depth
        threshold = self._config.queue_depth_threshold

        if depth >= threshold * 2:
            level = PressureLevel.CRITICAL
        elif depth >= threshold * 1.5:
            level = PressureLevel.HIGH
        elif depth >= threshold * 1.2:
            level = PressureLevel.MEDIUM
        elif depth >= threshold:
            level = PressureLevel.LOW
        else:
            level = PressureLevel.NONE

        self._state.update_pressure(level)
        self._metrics.pressure_level = level

    def should_pause(self) -> bool:
        """Check if should pause based on queue depth."""
        if self._queue_provider:
            current = self._queue_provider()
            self.update_metrics(queue_depth=current)

        return self._state.pressure_level in (
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        )

    def calculate_pause_duration(self) -> float:
        """Calculate pause duration based on queue depth."""
        depth = self._metrics.current_queue_depth
        threshold = self._config.queue_depth_threshold

        if depth <= threshold:
            return 0.0

        # Exponential backoff based on queue depth
        excess_ratio = depth / threshold
        duration_ms = self._config.min_pause_ms * (2 ** min(excess_ratio - 1, 4))

        return min(duration_ms, self._config.max_pause_ms) / 1000.0

    def calculate_rate(self) -> float:
        """Calculate allowed rate based on queue depth."""
        depth = self._metrics.current_queue_depth
        threshold = self._config.queue_depth_threshold

        if depth <= threshold * 0.5:
            # Queue is low, can increase rate
            return min(
                self._config.base_rate * 1.2,
                self._config.base_rate,
            )
        elif depth <= threshold:
            return self._config.base_rate

        # Reduce rate inversely proportional to queue depth
        reduction = threshold / depth
        new_rate = self._config.base_rate * reduction

        return max(new_rate, self._config.min_rate)


class LatencyBasedBackpressure(BaseBackpressure):
    """Backpressure based on operation latency.

    Monitors operation latency and applies backpressure when
    latency exceeds configured thresholds.

    Example:
        >>> config = BackpressureConfig(latency_threshold_ms=50.0)
        >>> bp = LatencyBasedBackpressure(config)
        >>> bp.update_metrics(latency_ms=100.0)
        >>> bp.should_pause()
        True
    """

    def __init__(self, config: BackpressureConfig | None = None) -> None:
        """Initialize latency-based backpressure."""
        super().__init__(config)
        self._latency_window: list[float] = []
        self._window_size = config.adaptive_window_size if config else 100

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample."""
        self._latency_window.append(latency_ms)
        if len(self._latency_window) > self._window_size:
            self._latency_window.pop(0)

        # Use moving average for stability
        avg_latency = sum(self._latency_window) / len(self._latency_window)
        self.update_metrics(latency_ms=avg_latency)

    def _update_pressure_level(self) -> None:
        """Update pressure level based on latency."""
        latency = self._metrics.current_latency_ms
        threshold = self._config.latency_threshold_ms

        if latency >= threshold * 4:
            level = PressureLevel.CRITICAL
        elif latency >= threshold * 2:
            level = PressureLevel.HIGH
        elif latency >= threshold * 1.5:
            level = PressureLevel.MEDIUM
        elif latency >= threshold:
            level = PressureLevel.LOW
        else:
            level = PressureLevel.NONE

        self._state.update_pressure(level)
        self._metrics.pressure_level = level

    def should_pause(self) -> bool:
        """Check if should pause based on latency."""
        return self._state.pressure_level in (
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        )

    def calculate_pause_duration(self) -> float:
        """Calculate pause duration based on latency."""
        latency = self._metrics.current_latency_ms
        threshold = self._config.latency_threshold_ms

        if latency <= threshold:
            return 0.0

        # Pause proportional to excess latency
        excess_ratio = latency / threshold
        duration_ms = self._config.min_pause_ms * excess_ratio

        return min(duration_ms, self._config.max_pause_ms) / 1000.0

    def calculate_rate(self) -> float:
        """Calculate allowed rate based on latency."""
        latency = self._metrics.current_latency_ms
        threshold = self._config.latency_threshold_ms

        if latency <= threshold * 0.5:
            # Latency is good, can try higher rate
            return min(
                self._state.current_rate * (1 + self._config.recovery_rate),
                self._config.base_rate,
            )
        elif latency <= threshold:
            return self._state.current_rate

        # Reduce rate to bring latency down
        reduction = threshold / latency
        new_rate = self._state.current_rate * reduction

        return max(new_rate, self._config.min_rate)


@dataclass
class TokenBucketState:
    """State for token bucket rate limiter."""

    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class TokenBucketBackpressure(BaseBackpressure):
    """Token bucket rate limiting backpressure.

    Implements the token bucket algorithm for smooth rate limiting.
    Tokens are added at a steady rate and consumed per operation.

    Example:
        >>> config = BackpressureConfig(base_rate=1000.0)
        >>> bp = TokenBucketBackpressure(config)
        >>> await bp.acquire()  # Waits if no tokens available
    """

    def __init__(
        self,
        config: BackpressureConfig | None = None,
        bucket_size: float | None = None,
    ) -> None:
        """Initialize token bucket backpressure.

        Args:
            config: Backpressure configuration.
            bucket_size: Maximum tokens in bucket. Defaults to base_rate.
        """
        super().__init__(config)
        self._bucket_size = bucket_size or self._config.base_rate
        self._bucket = TokenBucketState(tokens=self._bucket_size)

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._bucket.last_refill
        tokens_to_add = elapsed * self._state.current_rate

        self._bucket.tokens = min(
            self._bucket.tokens + tokens_to_add,
            self._bucket_size,
        )
        self._bucket.last_refill = now

    def _update_pressure_level(self) -> None:
        """Update pressure level based on token availability."""
        self._refill_tokens()
        token_ratio = self._bucket.tokens / self._bucket_size

        if token_ratio <= 0.1:
            level = PressureLevel.CRITICAL
        elif token_ratio <= 0.25:
            level = PressureLevel.HIGH
        elif token_ratio <= 0.5:
            level = PressureLevel.MEDIUM
        elif token_ratio <= 0.75:
            level = PressureLevel.LOW
        else:
            level = PressureLevel.NONE

        self._state.update_pressure(level)
        self._metrics.pressure_level = level

    def should_pause(self) -> bool:
        """Check if should pause (no tokens available)."""
        self._refill_tokens()
        return self._bucket.tokens < 1.0

    def calculate_pause_duration(self) -> float:
        """Calculate time to wait for next token."""
        if self._bucket.tokens >= 1.0:
            return 0.0

        # Time to get 1 token
        tokens_needed = 1.0 - self._bucket.tokens
        wait_time = tokens_needed / self._state.current_rate

        return min(wait_time, self._config.max_pause_ms / 1000.0)

    def calculate_rate(self) -> float:
        """Calculate current rate (based on external factors if updated)."""
        return self._state.current_rate

    async def acquire(self) -> bool:
        """Acquire a token, waiting if necessary."""
        if not self._config.enabled:
            return True

        async with self._lock:
            self._refill_tokens()

            while self._bucket.tokens < 1.0:
                wait_time = self.calculate_pause_duration()
                self._metrics.record_pause(wait_time * 1000)
                await asyncio.sleep(wait_time)
                self._refill_tokens()

            self._bucket.tokens -= 1.0
            self._update_pressure_level()
            return True

    def try_acquire(self) -> bool:
        """Try to acquire a token without waiting."""
        self._refill_tokens()
        if self._bucket.tokens >= 1.0:
            self._bucket.tokens -= 1.0
            return True
        return False


class AdaptiveBackpressure(BaseBackpressure):
    """Adaptive multi-factor backpressure strategy.

    Combines memory, queue depth, and latency factors with
    adaptive rate adjustment based on system behavior.

    Example:
        >>> bp = AdaptiveBackpressure()
        >>> bp.update_metrics(
        ...     memory_percent=70.0,
        ...     queue_depth=5000,
        ...     latency_ms=80.0,
        ... )
        >>> rate = bp.calculate_rate()
    """

    def __init__(
        self,
        config: BackpressureConfig | None = None,
        memory_weight: float = 0.4,
        queue_weight: float = 0.3,
        latency_weight: float = 0.3,
    ) -> None:
        """Initialize adaptive backpressure.

        Args:
            config: Backpressure configuration.
            memory_weight: Weight for memory factor (0-1).
            queue_weight: Weight for queue depth factor (0-1).
            latency_weight: Weight for latency factor (0-1).
        """
        super().__init__(config)

        # Normalize weights
        total = memory_weight + queue_weight + latency_weight
        self._memory_weight = memory_weight / total
        self._queue_weight = queue_weight / total
        self._latency_weight = latency_weight / total

        # Track rate history for adaptation
        self._rate_history: list[tuple[float, float]] = []  # (time, rate)
        self._pressure_history: list[tuple[float, PressureLevel]] = []

    def _calculate_pressure_score(self) -> float:
        """Calculate combined pressure score (0-1)."""
        memory_score = 0.0
        queue_score = 0.0
        latency_score = 0.0

        # Memory pressure
        mem = self._metrics.current_memory_percent
        mem_threshold = self._config.memory_threshold_percent
        if mem > mem_threshold:
            memory_score = min((mem - mem_threshold) / (100 - mem_threshold), 1.0)

        # Queue pressure
        queue = self._metrics.current_queue_depth
        queue_threshold = self._config.queue_depth_threshold
        if queue > queue_threshold:
            queue_score = min((queue - queue_threshold) / queue_threshold, 1.0)

        # Latency pressure
        latency = self._metrics.current_latency_ms
        latency_threshold = self._config.latency_threshold_ms
        if latency > latency_threshold:
            latency_score = min(
                (latency - latency_threshold) / latency_threshold, 1.0
            )

        # Weighted combination
        return (
            self._memory_weight * memory_score
            + self._queue_weight * queue_score
            + self._latency_weight * latency_score
        )

    def _update_pressure_level(self) -> None:
        """Update pressure level based on combined score."""
        score = self._calculate_pressure_score()

        if score >= 0.8:
            level = PressureLevel.CRITICAL
        elif score >= 0.6:
            level = PressureLevel.HIGH
        elif score >= 0.4:
            level = PressureLevel.MEDIUM
        elif score >= 0.2:
            level = PressureLevel.LOW
        else:
            level = PressureLevel.NONE

        self._state.update_pressure(level)
        self._metrics.pressure_level = level

        # Track history
        now = time.monotonic()
        self._pressure_history.append((now, level))
        if len(self._pressure_history) > self._config.adaptive_window_size:
            self._pressure_history.pop(0)

    def should_pause(self) -> bool:
        """Check if should pause based on combined pressure."""
        self._update_pressure_level()
        return self._state.pressure_level in (
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        )

    def calculate_pause_duration(self) -> float:
        """Calculate adaptive pause duration."""
        score = self._calculate_pressure_score()
        if score <= 0.2:
            return 0.0

        # Base duration from score
        duration_ms = self._config.min_pause_ms + (
            score * (self._config.max_pause_ms - self._config.min_pause_ms)
        )

        # Adjust based on consecutive high pressure
        if self._state.consecutive_high_pressure > 3:
            duration_ms *= 1.5

        return min(duration_ms, self._config.max_pause_ms) / 1000.0

    def calculate_rate(self) -> float:
        """Calculate adaptive rate based on system behavior."""
        score = self._calculate_pressure_score()
        current_rate = self._state.current_rate

        if score <= 0.1 and self._state.consecutive_low_pressure > 3:
            # System is healthy, gradually increase rate
            new_rate = current_rate * (1 + self._config.recovery_rate)
        elif score >= 0.6:
            # High pressure, reduce rate significantly
            reduction = 1.0 - (score * 0.8)
            new_rate = current_rate * reduction
        elif score >= 0.3:
            # Moderate pressure, slight reduction
            reduction = 1.0 - (score * 0.5)
            new_rate = current_rate * reduction
        else:
            # Low pressure, maintain or slight increase
            new_rate = current_rate * (1 + self._config.recovery_rate * 0.5)

        # Clamp to bounds
        new_rate = max(min(new_rate, self._config.base_rate), self._config.min_rate)

        # Track history
        now = time.monotonic()
        self._rate_history.append((now, new_rate))
        if len(self._rate_history) > self._config.adaptive_window_size:
            self._rate_history.pop(0)

        self._state.adjust_rate(new_rate)
        return new_rate


class CompositeBackpressure(BaseBackpressure):
    """Composite backpressure combining multiple strategies.

    Combines multiple backpressure strategies and applies the
    most restrictive policy.

    Example:
        >>> memory_bp = MemoryBasedBackpressure()
        >>> queue_bp = QueueDepthBackpressure()
        >>> composite = CompositeBackpressure([memory_bp, queue_bp])
        >>> await composite.acquire()
    """

    def __init__(
        self,
        strategies: list[BaseBackpressure],
        config: BackpressureConfig | None = None,
    ) -> None:
        """Initialize composite backpressure.

        Args:
            strategies: List of backpressure strategies to combine.
            config: Optional override configuration.
        """
        super().__init__(config)
        self._strategies = strategies

    def _update_pressure_level(self) -> None:
        """Update pressure level to highest among strategies."""
        max_level = PressureLevel.NONE
        level_order = [
            PressureLevel.NONE,
            PressureLevel.LOW,
            PressureLevel.MEDIUM,
            PressureLevel.HIGH,
            PressureLevel.CRITICAL,
        ]

        for strategy in self._strategies:
            strategy._update_pressure_level()
            if level_order.index(strategy.state.pressure_level) > level_order.index(
                max_level
            ):
                max_level = strategy.state.pressure_level

        self._state.update_pressure(max_level)
        self._metrics.pressure_level = max_level

    def should_pause(self) -> bool:
        """Check if any strategy says to pause."""
        return any(s.should_pause() for s in self._strategies)

    def calculate_pause_duration(self) -> float:
        """Calculate maximum pause duration among strategies."""
        durations = [s.calculate_pause_duration() for s in self._strategies]
        return max(durations) if durations else 0.0

    def calculate_rate(self) -> float:
        """Calculate minimum rate among strategies."""
        rates = [s.calculate_rate() for s in self._strategies]
        return min(rates) if rates else self._config.base_rate

    def update_metrics(
        self,
        memory_percent: float | None = None,
        queue_depth: int | None = None,
        latency_ms: float | None = None,
    ) -> None:
        """Update metrics on all strategies."""
        super().update_metrics(memory_percent, queue_depth, latency_ms)
        for strategy in self._strategies:
            strategy.update_metrics(memory_percent, queue_depth, latency_ms)

    async def acquire(self) -> bool:
        """Acquire from all strategies."""
        results = await asyncio.gather(
            *[s.acquire() for s in self._strategies],
            return_exceptions=True,
        )
        return all(r is True for r in results if not isinstance(r, Exception))

    def release(self) -> None:
        """Release on all strategies."""
        super().release()
        for strategy in self._strategies:
            strategy.release()

    def reset(self) -> None:
        """Reset all strategies."""
        super().reset()
        for strategy in self._strategies:
            strategy.reset()
