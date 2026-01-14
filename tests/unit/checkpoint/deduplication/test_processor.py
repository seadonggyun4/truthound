"""Tests for time window processors."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from truthound.checkpoint.deduplication.protocols import (
    DeduplicationRecord,
    NotificationFingerprint,
    TimeWindow,
)
from truthound.checkpoint.deduplication.processor import (
    AdaptiveWindowStrategy,
    HierarchicalWindowStrategy,
    SessionWindowStrategy,
    SlidingWindowStrategy,
    TimeWindowProcessor,
    TumblingWindowStrategy,
)


class TestSlidingWindowStrategy:
    """Tests for SlidingWindowStrategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        strategy = SlidingWindowStrategy()
        assert strategy.name == "sliding"

    def test_get_window_key(self) -> None:
        """Test window key is same as fingerprint key."""
        strategy = SlidingWindowStrategy()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)

        key = strategy.get_window_key(fp, window)
        assert key == fp.key

    def test_is_in_window_recent(self) -> None:
        """Test record within window."""
        strategy = SlidingWindowStrategy()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()
        window = TimeWindow(minutes=5)

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(minutes=2),
            expires_at=now + timedelta(minutes=3),
        )

        assert strategy.is_in_window(record, window, now)

    def test_is_in_window_expired(self) -> None:
        """Test record outside window."""
        strategy = SlidingWindowStrategy()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()
        window = TimeWindow(minutes=5)

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(minutes=10),
            expires_at=now - timedelta(minutes=5),
        )

        assert not strategy.is_in_window(record, window, now)


class TestTumblingWindowStrategy:
    """Tests for TumblingWindowStrategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        strategy = TumblingWindowStrategy()
        assert strategy.name == "tumbling"

    def test_get_window_key_same_bucket(self) -> None:
        """Test same bucket produces same key."""
        strategy = TumblingWindowStrategy()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)
        base_time = datetime(2024, 1, 1, 10, 2, 0)  # 10:02

        key1 = strategy.get_window_key(fp, window, base_time)
        key2 = strategy.get_window_key(fp, window, base_time + timedelta(minutes=2))

        assert key1 == key2  # Both in 10:00-10:05 window

    def test_get_window_key_different_buckets(self) -> None:
        """Test different buckets produce different keys."""
        strategy = TumblingWindowStrategy()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)
        time1 = datetime(2024, 1, 1, 10, 2, 0)  # 10:02 -> window 10:00-10:05
        time2 = datetime(2024, 1, 1, 10, 7, 0)  # 10:07 -> window 10:05-10:10

        key1 = strategy.get_window_key(fp, window, time1)
        key2 = strategy.get_window_key(fp, window, time2)

        assert key1 != key2

    def test_is_in_window_same_bucket(self) -> None:
        """Test record in same tumbling window."""
        strategy = TumblingWindowStrategy()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)
        base_time = datetime(2024, 1, 1, 10, 2, 0)

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=base_time,
            expires_at=base_time + timedelta(minutes=5),
        )

        check_time = base_time + timedelta(minutes=2)  # Still in same bucket
        assert strategy.is_in_window(record, window, check_time)

    def test_is_in_window_different_bucket(self) -> None:
        """Test record in different tumbling window."""
        strategy = TumblingWindowStrategy()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)
        base_time = datetime(2024, 1, 1, 10, 2, 0)

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=base_time,
            expires_at=base_time + timedelta(minutes=10),
        )

        check_time = base_time + timedelta(minutes=5)  # Next bucket
        assert not strategy.is_in_window(record, window, check_time)


class TestSessionWindowStrategy:
    """Tests for SessionWindowStrategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        strategy = SessionWindowStrategy()
        assert strategy.name == "session"

    def test_is_in_window_active_session(self) -> None:
        """Test active session is within window."""
        strategy = SessionWindowStrategy(
            gap_duration=TimeWindow(minutes=5),
            max_session_duration=TimeWindow(hours=1),
        )
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(minutes=2),
            expires_at=now + timedelta(hours=1),
            last_duplicate_at=now - timedelta(minutes=1),  # Recent activity
        )

        assert strategy.is_in_window(record, TimeWindow(minutes=5), now)

    def test_is_in_window_session_timeout(self) -> None:
        """Test session timed out due to inactivity."""
        strategy = SessionWindowStrategy(
            gap_duration=TimeWindow(minutes=5),
            max_session_duration=TimeWindow(hours=1),
        )
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(minutes=10),
            expires_at=now + timedelta(hours=1),
            last_duplicate_at=now - timedelta(minutes=8),  # No recent activity
        )

        assert not strategy.is_in_window(record, TimeWindow(minutes=5), now)

    def test_is_in_window_max_duration_exceeded(self) -> None:
        """Test session exceeded max duration."""
        strategy = SessionWindowStrategy(
            gap_duration=TimeWindow(minutes=5),
            max_session_duration=TimeWindow(hours=1),
        )
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(hours=2),  # Session too old
            expires_at=now + timedelta(hours=1),
            last_duplicate_at=now - timedelta(minutes=1),  # Recent activity
        )

        assert not strategy.is_in_window(record, TimeWindow(minutes=5), now)


class TestTimeWindowProcessor:
    """Tests for TimeWindowProcessor."""

    def test_create_sliding(self) -> None:
        """Test factory method for sliding window."""
        processor = TimeWindowProcessor.create_sliding(TimeWindow(minutes=5))
        assert processor.strategy.name == "sliding"
        assert processor.default_window.total_seconds == 300

    def test_create_tumbling(self) -> None:
        """Test factory method for tumbling window."""
        processor = TimeWindowProcessor.create_tumbling(TimeWindow(minutes=5))
        assert processor.strategy.name == "tumbling"

    def test_create_session(self) -> None:
        """Test factory method for session window."""
        processor = TimeWindowProcessor.create_session(
            gap_duration=TimeWindow(minutes=5),
            max_session_duration=TimeWindow(hours=1),
        )
        assert processor.strategy.name == "session"

    def test_get_dedup_key(self) -> None:
        """Test dedup key generation."""
        processor = TimeWindowProcessor.create_sliding()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )

        key = processor.get_dedup_key(fp)
        assert key == fp.key

    def test_is_duplicate_true(self) -> None:
        """Test duplicate detection."""
        processor = TimeWindowProcessor.create_sliding(TimeWindow(minutes=5))
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(minutes=2),
            expires_at=now + timedelta(minutes=3),
        )

        assert processor.is_duplicate(fp, record, timestamp=now)

    def test_is_duplicate_false_expired(self) -> None:
        """Test not duplicate when expired."""
        processor = TimeWindowProcessor.create_sliding(TimeWindow(minutes=5))
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(minutes=10),
            expires_at=now - timedelta(minutes=5),
        )

        assert not processor.is_duplicate(fp, record, timestamp=now)

    def test_calculate_expiration(self) -> None:
        """Test expiration calculation."""
        processor = TimeWindowProcessor(default_window=TimeWindow(minutes=5))
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        expiration = processor.calculate_expiration(timestamp=base_time)
        expected = base_time + timedelta(minutes=5)

        assert expiration == expected


class TestAdaptiveWindowStrategy:
    """Tests for AdaptiveWindowStrategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        strategy = AdaptiveWindowStrategy()
        assert strategy.name == "adaptive"

    def test_base_window_at_low_load(self) -> None:
        """Test base window used at low load."""
        strategy = AdaptiveWindowStrategy(
            base_window=TimeWindow(minutes=5),
            rate_threshold=10,
        )

        window = strategy.get_current_window("test_checkpoint")
        assert window.total_seconds == 300  # Base window

    def test_scaled_window_at_high_load(self) -> None:
        """Test window scales at high load."""
        strategy = AdaptiveWindowStrategy(
            base_window=TimeWindow(minutes=5),
            rate_threshold=10,
            scale_factor=2.0,
        )

        # Simulate high load
        for _ in range(20):
            strategy.record_notification("test_checkpoint")

        window = strategy.get_current_window("test_checkpoint")
        assert window.total_seconds > 300  # Should be scaled up

    def test_window_clamped_to_max(self) -> None:
        """Test window doesn't exceed max."""
        strategy = AdaptiveWindowStrategy(
            base_window=TimeWindow(minutes=5),
            max_window=TimeWindow(hours=1),
            rate_threshold=1,
            scale_factor=100.0,
        )

        # Simulate extreme load
        for _ in range(1000):
            strategy.record_notification("test_checkpoint")

        window = strategy.get_current_window("test_checkpoint")
        assert window.total_seconds <= 3600  # Max 1 hour


class TestHierarchicalWindowStrategy:
    """Tests for HierarchicalWindowStrategy."""

    def test_name(self) -> None:
        """Test strategy name."""
        strategy = HierarchicalWindowStrategy()
        assert strategy.name == "hierarchical"

    def test_severity_based_windows(self) -> None:
        """Test different windows for different severities."""
        strategy = HierarchicalWindowStrategy(
            windows={
                "critical": TimeWindow(minutes=1),
                "high": TimeWindow(minutes=5),
                "medium": TimeWindow(minutes=15),
                "low": TimeWindow(hours=1),
            },
            default_window=TimeWindow(minutes=10),
            key_extractor=lambda fp: fp.components.get("severity", "medium"),
        )

        # Critical severity
        fp_critical = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            severity="critical",
        )
        window = strategy.get_window_for_fingerprint(fp_critical)
        assert window.total_seconds == 60

        # Low severity
        fp_low = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            severity="low",
        )
        window = strategy.get_window_for_fingerprint(fp_low)
        assert window.total_seconds == 3600

        # Unknown severity uses default
        fp_unknown = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            severity="unknown",
        )
        window = strategy.get_window_for_fingerprint(fp_unknown)
        assert window.total_seconds == 600  # Default
