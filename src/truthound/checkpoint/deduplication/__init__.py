"""Notification Deduplication System.

This module provides a comprehensive notification deduplication system
with support for multiple backends (InMemory, Redis Streams) and
time-window based duplicate detection.

Key Components:
    - DeduplicationStore: Protocol for deduplication state storage
    - TimeWindow: Time-based deduplication window configuration
    - NotificationFingerprint: Unique identifier for notifications
    - NotificationDeduplicator: Main deduplication service

Example:
    >>> from truthound.checkpoint.deduplication import (
    ...     NotificationDeduplicator,
    ...     InMemoryDeduplicationStore,
    ...     TimeWindow,
    ... )
    >>>
    >>> # Create deduplicator with 5-minute window
    >>> deduplicator = NotificationDeduplicator(
    ...     store=InMemoryDeduplicationStore(),
    ...     default_window=TimeWindow(seconds=300),
    ... )
    >>>
    >>> # Check for duplicates
    >>> fingerprint = deduplicator.generate_fingerprint(
    ...     checkpoint_name="data_quality",
    ...     action_type="slack",
    ...     severity="high",
    ... )
    >>> if not deduplicator.is_duplicate(fingerprint):
    ...     # Send notification
    ...     deduplicator.mark_sent(fingerprint)
"""

from truthound.checkpoint.deduplication.protocols import (
    DeduplicationRecord,
    DeduplicationResult,
    DeduplicationStats,
    DeduplicationStore,
    NotificationFingerprint,
    TimeWindow,
    WindowUnit,
)
from truthound.checkpoint.deduplication.stores import (
    InMemoryDeduplicationStore,
    RedisStreamsDeduplicationStore,
)
from truthound.checkpoint.deduplication.processor import (
    TimeWindowProcessor,
    WindowStrategy,
    SlidingWindowStrategy,
    TumblingWindowStrategy,
    SessionWindowStrategy,
)
from truthound.checkpoint.deduplication.service import (
    NotificationDeduplicator,
    DeduplicationConfig,
    DeduplicationPolicy,
)
from truthound.checkpoint.deduplication.middleware import (
    DeduplicationMiddleware,
    deduplicated,
)

__all__ = [
    # Protocols & Core Types
    "DeduplicationStore",
    "DeduplicationRecord",
    "DeduplicationResult",
    "DeduplicationStats",
    "NotificationFingerprint",
    "TimeWindow",
    "WindowUnit",
    # Stores
    "InMemoryDeduplicationStore",
    "RedisStreamsDeduplicationStore",
    # Window Processors
    "TimeWindowProcessor",
    "WindowStrategy",
    "SlidingWindowStrategy",
    "TumblingWindowStrategy",
    "SessionWindowStrategy",
    # Service
    "NotificationDeduplicator",
    "DeduplicationConfig",
    "DeduplicationPolicy",
    # Middleware
    "DeduplicationMiddleware",
    "deduplicated",
]
