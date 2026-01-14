"""Time Window Processors for Deduplication.

This module provides different windowing strategies for
notification deduplication:
- SlidingWindowStrategy: Continuous sliding window
- TumblingWindowStrategy: Fixed non-overlapping windows
- SessionWindowStrategy: Activity-based sessions
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from truthound.checkpoint.deduplication.protocols import (
    DeduplicationRecord,
    NotificationFingerprint,
    TimeWindow,
    WindowStrategy,
)


class WindowType(str, Enum):
    """Types of time windows."""

    SLIDING = "sliding"
    TUMBLING = "tumbling"
    SESSION = "session"


@dataclass
class SlidingWindowStrategy(WindowStrategy):
    """Sliding window strategy for continuous deduplication.

    Each notification creates its own window that slides forward
    with time. A duplicate is detected if a matching notification
    was sent within the window duration.

    This is the most common strategy for notification deduplication.

    Example:
        >>> strategy = SlidingWindowStrategy()
        >>> key = strategy.get_window_key(fingerprint, TimeWindow(minutes=5))
        >>> # Key doesn't change with time - window slides
    """

    @property
    def name(self) -> str:
        return "sliding"

    def get_window_key(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> str:
        """Get window key (same as base fingerprint for sliding windows)."""
        # Sliding windows don't bucket - they check actual time difference
        return fingerprint.key

    def is_in_window(
        self,
        record: DeduplicationRecord,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> bool:
        """Check if record is within the sliding window."""
        now = timestamp or datetime.now()
        window_start = now - window.to_timedelta()
        return record.sent_at >= window_start


@dataclass
class TumblingWindowStrategy(WindowStrategy):
    """Tumbling window strategy with fixed, non-overlapping windows.

    Time is divided into fixed-size windows. All notifications
    within the same window are considered duplicates.

    Example:
        Window size: 5 minutes
        00:00-00:05 -> Window 1
        00:05-00:10 -> Window 2
        Notification at 00:03 and 00:04 are in same window (duplicates)
        Notification at 00:03 and 00:06 are in different windows (not duplicates)

    Example:
        >>> strategy = TumblingWindowStrategy()
        >>> key = strategy.get_window_key(fingerprint, TimeWindow(minutes=5))
        >>> # Key changes every 5 minutes
    """

    @property
    def name(self) -> str:
        return "tumbling"

    def get_window_key(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> str:
        """Get window-bucketed key."""
        now = timestamp or datetime.now()
        window_seconds = window.total_seconds

        # Calculate window bucket
        bucket = int(now.timestamp() // window_seconds)

        # Create bucketed key
        bucket_key = f"{fingerprint.key}:tumbling:{bucket}"
        return hashlib.sha256(bucket_key.encode()).hexdigest()[:32]

    def is_in_window(
        self,
        record: DeduplicationRecord,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> bool:
        """Check if record is in the same tumbling window."""
        now = timestamp or datetime.now()
        window_seconds = window.total_seconds

        record_bucket = int(record.sent_at.timestamp() // window_seconds)
        current_bucket = int(now.timestamp() // window_seconds)

        return record_bucket == current_bucket


@dataclass
class SessionWindowStrategy(WindowStrategy):
    """Session window strategy based on activity.

    A session starts with the first notification and extends
    as long as notifications keep arriving within the gap duration.
    The session ends after a period of inactivity.

    Attributes:
        gap_duration: Maximum gap between notifications in a session.
        max_session_duration: Maximum total session duration.

    Example:
        Gap: 5 minutes, Max: 1 hour
        Notifications at 00:00, 00:03, 00:07, 00:11 -> Same session
        Next notification at 00:20 -> New session (gap > 5 min)

    Example:
        >>> strategy = SessionWindowStrategy(
        ...     gap_duration=TimeWindow(minutes=5),
        ...     max_session_duration=TimeWindow(hours=1),
        ... )
    """

    gap_duration: TimeWindow = field(default_factory=lambda: TimeWindow(minutes=5))
    max_session_duration: TimeWindow = field(
        default_factory=lambda: TimeWindow(hours=1)
    )

    @property
    def name(self) -> str:
        return "session"

    def get_window_key(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> str:
        """Get base key (session tracking is done via record state)."""
        # Session windows use the base key; session state is tracked separately
        return fingerprint.key

    def is_in_window(
        self,
        record: DeduplicationRecord,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> bool:
        """Check if current time is within the session window."""
        now = timestamp or datetime.now()

        # Check gap from last activity
        last_activity = record.last_duplicate_at or record.sent_at
        gap = now - last_activity

        if gap > self.gap_duration.to_timedelta():
            return False  # Session ended due to inactivity

        # Check max session duration
        session_duration = now - record.sent_at
        if session_duration > self.max_session_duration.to_timedelta():
            return False  # Session exceeded max duration

        return True


@dataclass
class TimeWindowProcessor:
    """Processor for time window-based deduplication.

    Coordinates window strategies and handles the logic for
    determining if a notification is a duplicate.

    Attributes:
        strategy: Window strategy to use.
        default_window: Default time window.

    Example:
        >>> processor = TimeWindowProcessor(
        ...     strategy=SlidingWindowStrategy(),
        ...     default_window=TimeWindow(minutes=5),
        ... )
        >>> key = processor.get_dedup_key(fingerprint)
        >>> is_dup = processor.is_duplicate(fingerprint, existing_record)
    """

    strategy: WindowStrategy = field(default_factory=SlidingWindowStrategy)
    default_window: TimeWindow = field(default_factory=lambda: TimeWindow(minutes=5))

    def get_dedup_key(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Get the deduplication key for a fingerprint.

        Args:
            fingerprint: Notification fingerprint.
            window: Time window (uses default if not specified).
            timestamp: Timestamp for window calculation.

        Returns:
            Deduplication key string.
        """
        effective_window = window or self.default_window
        return self.strategy.get_window_key(fingerprint, effective_window, timestamp)

    def is_duplicate(
        self,
        fingerprint: NotificationFingerprint,
        record: DeduplicationRecord,
        window: TimeWindow | None = None,
        timestamp: datetime | None = None,
    ) -> bool:
        """Check if a fingerprint is a duplicate of an existing record.

        Args:
            fingerprint: New notification fingerprint.
            record: Existing deduplication record.
            window: Time window for comparison.
            timestamp: Current timestamp.

        Returns:
            True if the notification is a duplicate.
        """
        effective_window = window or self.default_window

        # First check if keys match
        new_key = self.get_dedup_key(fingerprint, effective_window, timestamp)
        existing_key = self.get_dedup_key(
            record.fingerprint, effective_window, record.sent_at
        )

        if new_key != existing_key:
            return False

        # Check if still within window
        return self.strategy.is_in_window(record, effective_window, timestamp)

    def calculate_expiration(
        self,
        window: TimeWindow | None = None,
        timestamp: datetime | None = None,
    ) -> datetime:
        """Calculate when a deduplication record should expire.

        Args:
            window: Time window.
            timestamp: Start timestamp.

        Returns:
            Expiration datetime.
        """
        effective_window = window or self.default_window
        start = timestamp or datetime.now()
        return start + effective_window.to_timedelta()

    @classmethod
    def create_sliding(
        cls,
        default_window: TimeWindow | None = None,
    ) -> TimeWindowProcessor:
        """Create processor with sliding window strategy."""
        return cls(
            strategy=SlidingWindowStrategy(),
            default_window=default_window or TimeWindow(minutes=5),
        )

    @classmethod
    def create_tumbling(
        cls,
        default_window: TimeWindow | None = None,
    ) -> TimeWindowProcessor:
        """Create processor with tumbling window strategy."""
        return cls(
            strategy=TumblingWindowStrategy(),
            default_window=default_window or TimeWindow(minutes=5),
        )

    @classmethod
    def create_session(
        cls,
        gap_duration: TimeWindow | None = None,
        max_session_duration: TimeWindow | None = None,
        default_window: TimeWindow | None = None,
    ) -> TimeWindowProcessor:
        """Create processor with session window strategy."""
        strategy = SessionWindowStrategy(
            gap_duration=gap_duration or TimeWindow(minutes=5),
            max_session_duration=max_session_duration or TimeWindow(hours=1),
        )
        return cls(
            strategy=strategy,
            default_window=default_window or TimeWindow(minutes=5),
        )


@dataclass
class AdaptiveWindowStrategy(WindowStrategy):
    """Adaptive window strategy that adjusts based on load.

    Automatically adjusts window size based on notification frequency
    to prevent alert fatigue during high-volume periods.

    Attributes:
        base_window: Base window duration.
        min_window: Minimum window duration.
        max_window: Maximum window duration.
        rate_threshold: Notifications/minute to trigger scaling.
        scale_factor: How much to scale window on high load.

    Example:
        >>> strategy = AdaptiveWindowStrategy(
        ...     base_window=TimeWindow(minutes=5),
        ...     rate_threshold=10,  # 10 notifications/minute
        ...     scale_factor=2.0,   # Double window on high load
        ... )
    """

    base_window: TimeWindow = field(default_factory=lambda: TimeWindow(minutes=5))
    min_window: TimeWindow = field(default_factory=lambda: TimeWindow(minutes=1))
    max_window: TimeWindow = field(default_factory=lambda: TimeWindow(hours=1))
    rate_threshold: int = 10
    scale_factor: float = 2.0

    _recent_counts: dict[str, int] = field(default_factory=dict)
    _last_reset: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Initialize tracking."""
        self._recent_counts = {}
        self._last_reset = datetime.now()

    @property
    def name(self) -> str:
        return "adaptive"

    def get_window_key(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> str:
        """Get key (adaptive window tracking is internal)."""
        return fingerprint.key

    def is_in_window(
        self,
        record: DeduplicationRecord,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> bool:
        """Check if within adaptive window."""
        now = timestamp or datetime.now()

        # Calculate effective window based on rate
        effective_window = self._calculate_effective_window(
            record.fingerprint.checkpoint_name
        )

        window_start = now - effective_window.to_timedelta()
        return record.sent_at >= window_start

    def record_notification(self, checkpoint_name: str) -> None:
        """Record a notification for rate tracking."""
        now = datetime.now()

        # Reset counts every minute
        if (now - self._last_reset).total_seconds() >= 60:
            self._recent_counts = {}
            self._last_reset = now

        self._recent_counts[checkpoint_name] = (
            self._recent_counts.get(checkpoint_name, 0) + 1
        )

    def _calculate_effective_window(self, checkpoint_name: str) -> TimeWindow:
        """Calculate effective window based on current rate."""
        count = self._recent_counts.get(checkpoint_name, 0)

        if count <= self.rate_threshold:
            return self.base_window

        # Scale window based on how much we exceed threshold
        excess_ratio = count / self.rate_threshold
        scale = min(excess_ratio * self.scale_factor, 10.0)  # Cap at 10x

        new_seconds = int(self.base_window.total_seconds * scale)

        # Clamp to min/max
        new_seconds = max(new_seconds, self.min_window.total_seconds)
        new_seconds = min(new_seconds, self.max_window.total_seconds)

        return TimeWindow(seconds=new_seconds)

    def get_current_window(self, checkpoint_name: str) -> TimeWindow:
        """Get the current effective window for a checkpoint."""
        return self._calculate_effective_window(checkpoint_name)


@dataclass
class HierarchicalWindowStrategy(WindowStrategy):
    """Hierarchical window strategy with multiple levels.

    Uses different window sizes based on notification severity
    or other attributes.

    Attributes:
        windows: Mapping of attribute values to window sizes.
        default_window: Default window if no match found.
        key_extractor: Function to extract the key from fingerprint.

    Example:
        >>> strategy = HierarchicalWindowStrategy(
        ...     windows={
        ...         "critical": TimeWindow(minutes=1),
        ...         "high": TimeWindow(minutes=5),
        ...         "medium": TimeWindow(minutes=15),
        ...         "low": TimeWindow(hours=1),
        ...     },
        ...     key_extractor=lambda fp: fp.components.get("severity", "medium"),
        ... )
    """

    windows: dict[str, TimeWindow] = field(default_factory=dict)
    default_window: TimeWindow = field(default_factory=lambda: TimeWindow(minutes=5))
    key_extractor: Callable[[NotificationFingerprint], str] = field(
        default_factory=lambda: lambda fp: fp.components.get("severity", "medium")
    )

    @property
    def name(self) -> str:
        return "hierarchical"

    def get_window_for_fingerprint(
        self,
        fingerprint: NotificationFingerprint,
    ) -> TimeWindow:
        """Get the appropriate window for a fingerprint."""
        key = self.key_extractor(fingerprint)
        return self.windows.get(key, self.default_window)

    def get_window_key(
        self,
        fingerprint: NotificationFingerprint,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> str:
        """Get key (uses base fingerprint key)."""
        return fingerprint.key

    def is_in_window(
        self,
        record: DeduplicationRecord,
        window: TimeWindow,
        timestamp: datetime | None = None,
    ) -> bool:
        """Check if within the hierarchical window."""
        now = timestamp or datetime.now()

        # Use the window appropriate for the record's fingerprint
        effective_window = self.get_window_for_fingerprint(record.fingerprint)
        window_start = now - effective_window.to_timedelta()

        return record.sent_at >= window_start
