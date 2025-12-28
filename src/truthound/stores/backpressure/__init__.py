"""Backpressure management for streaming stores.

This module provides backpressure strategies to prevent overwhelming
downstream systems during high-throughput streaming operations.

Features:
    - Adaptive backpressure with dynamic rate adjustment
    - Memory-based pressure detection
    - Queue depth monitoring
    - Latency-based throttling
    - Token bucket rate limiting
    - Circuit breaker pattern

Example:
    >>> from truthound.stores.backpressure import (
    ...     AdaptiveBackpressure,
    ...     BackpressureConfig,
    ...     BackpressureMonitor,
    ... )
    >>>
    >>> config = BackpressureConfig(
    ...     memory_threshold_percent=80.0,
    ...     queue_depth_threshold=10000,
    ...     latency_threshold_ms=100.0,
    ... )
    >>> backpressure = AdaptiveBackpressure(config)
    >>> monitor = BackpressureMonitor(backpressure)
    >>>
    >>> async with monitor:
    ...     while data_available:
    ...         await backpressure.acquire()
    ...         await process_batch(data)
    ...         backpressure.release()
"""

from truthound.stores.backpressure.base import (
    BackpressureConfig,
    BackpressureMetrics,
    BackpressureState,
    BackpressureStrategy,
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
from truthound.stores.backpressure.monitor import (
    BackpressureMonitor,
    BackpressureEvent,
    BackpressureEventType,
)
from truthound.stores.backpressure.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
)

__all__ = [
    # Base
    "BackpressureConfig",
    "BackpressureMetrics",
    "BackpressureState",
    "BackpressureStrategy",
    "PressureLevel",
    # Strategies
    "AdaptiveBackpressure",
    "CompositeBackpressure",
    "LatencyBasedBackpressure",
    "MemoryBasedBackpressure",
    "QueueDepthBackpressure",
    "TokenBucketBackpressure",
    # Monitor
    "BackpressureMonitor",
    "BackpressureEvent",
    "BackpressureEventType",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerState",
]
