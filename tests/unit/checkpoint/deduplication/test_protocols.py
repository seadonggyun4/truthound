"""Tests for deduplication protocols and core types."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from truthound.checkpoint.deduplication.protocols import (
    DeduplicationRecord,
    DeduplicationResult,
    DeduplicationStats,
    NotificationFingerprint,
    TimeWindow,
    WindowUnit,
)


class TestWindowUnit:
    """Tests for WindowUnit enum."""

    def test_to_seconds_seconds(self) -> None:
        """Test seconds conversion."""
        assert WindowUnit.SECONDS.to_seconds(10) == 10

    def test_to_seconds_minutes(self) -> None:
        """Test minutes conversion."""
        assert WindowUnit.MINUTES.to_seconds(5) == 300

    def test_to_seconds_hours(self) -> None:
        """Test hours conversion."""
        assert WindowUnit.HOURS.to_seconds(2) == 7200

    def test_to_seconds_days(self) -> None:
        """Test days conversion."""
        assert WindowUnit.DAYS.to_seconds(1) == 86400


class TestTimeWindow:
    """Tests for TimeWindow class."""

    def test_default_initialization(self) -> None:
        """Test default values."""
        window = TimeWindow()
        assert window.value == 300
        assert window.unit == WindowUnit.SECONDS

    def test_value_unit_initialization(self) -> None:
        """Test value/unit initialization."""
        window = TimeWindow(10, WindowUnit.MINUTES)
        assert window.value == 10
        assert window.unit == WindowUnit.MINUTES

    def test_seconds_shorthand(self) -> None:
        """Test seconds shorthand."""
        window = TimeWindow(seconds=60)
        assert window.value == 60
        assert window.unit == WindowUnit.SECONDS

    def test_minutes_shorthand(self) -> None:
        """Test minutes shorthand."""
        window = TimeWindow(minutes=5)
        assert window.value == 5
        assert window.unit == WindowUnit.MINUTES

    def test_hours_shorthand(self) -> None:
        """Test hours shorthand."""
        window = TimeWindow(hours=1)
        assert window.value == 1
        assert window.unit == WindowUnit.HOURS

    def test_days_shorthand(self) -> None:
        """Test days shorthand."""
        window = TimeWindow(days=7)
        assert window.value == 7
        assert window.unit == WindowUnit.DAYS

    def test_total_seconds(self) -> None:
        """Test total_seconds property."""
        window = TimeWindow(minutes=5)
        assert window.total_seconds == 300

    def test_to_timedelta(self) -> None:
        """Test to_timedelta conversion."""
        window = TimeWindow(minutes=5)
        assert window.to_timedelta() == timedelta(minutes=5)

    def test_string_representation(self) -> None:
        """Test __str__."""
        window = TimeWindow(minutes=5)
        assert str(window) == "5 minutes"

    def test_frozen_immutable(self) -> None:
        """Test that TimeWindow is immutable."""
        window = TimeWindow(minutes=5)
        with pytest.raises(AttributeError):
            window.value = 10  # type: ignore


class TestNotificationFingerprint:
    """Tests for NotificationFingerprint class."""

    def test_generate_basic(self) -> None:
        """Test basic fingerprint generation."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test_checkpoint",
            action_type="slack",
        )
        assert fp.checkpoint_name == "test_checkpoint"
        assert fp.action_type == "slack"
        assert len(fp.key) == 32
        assert "checkpoint_name" in fp.components

    def test_generate_with_severity(self) -> None:
        """Test fingerprint with severity."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="email",
            severity="high",
        )
        assert fp.components.get("severity") == "high"

    def test_generate_with_data_asset(self) -> None:
        """Test fingerprint with data asset."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            data_asset="orders",
        )
        assert fp.components.get("data_asset") == "orders"

    def test_generate_with_issue_types(self) -> None:
        """Test fingerprint with issue types."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            issue_types=["NullValidator", "RangeValidator"],
        )
        assert fp.components.get("issue_types") == ["NullValidator", "RangeValidator"]

    def test_generate_with_custom_key(self) -> None:
        """Test fingerprint with custom key."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            custom_key="my-custom-key",
        )
        assert fp.key == "my-custom-key"

    def test_generate_deterministic(self) -> None:
        """Test that same inputs produce same fingerprint."""
        fp1 = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            severity="high",
        )
        fp2 = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            severity="high",
        )
        assert fp1.key == fp2.key

    def test_generate_different_for_different_inputs(self) -> None:
        """Test different inputs produce different fingerprints."""
        fp1 = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        fp2 = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="email",
        )
        assert fp1.key != fp2.key

    def test_with_window(self) -> None:
        """Test with_window creates bucketed fingerprint."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)
        bucketed = fp.with_window(window)

        assert bucketed.key != fp.key
        assert "_window_bucket" in bucketed.components

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
            severity="high",
        )
        data = fp.to_dict()
        restored = NotificationFingerprint.from_dict(data)

        assert restored.key == fp.key
        assert restored.checkpoint_name == fp.checkpoint_name
        assert restored.action_type == fp.action_type
        assert restored.components == fp.components

    def test_hash_and_eq(self) -> None:
        """Test hash and equality."""
        fp1 = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        fp2 = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        assert fp1 == fp2
        assert hash(fp1) == hash(fp2)

        # Can be used in sets
        s = {fp1, fp2}
        assert len(s) == 1


class TestDeduplicationRecord:
    """Tests for DeduplicationRecord class."""

    def test_create_record(self) -> None:
        """Test record creation."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()
        expires = now + timedelta(minutes=5)

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now,
            expires_at=expires,
        )

        assert record.fingerprint == fp
        assert record.sent_at == now
        assert record.expires_at == expires
        assert record.count == 1

    def test_is_expired(self) -> None:
        """Test expiration check."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()

        # Not expired
        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now,
            expires_at=now + timedelta(minutes=5),
        )
        assert not record.is_expired

        # Expired
        expired_record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now - timedelta(minutes=10),
            expires_at=now - timedelta(minutes=5),
        )
        assert expired_record.is_expired

    def test_remaining_ttl(self) -> None:
        """Test remaining TTL calculation."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()

        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now,
            expires_at=now + timedelta(minutes=5),
        )
        # TTL should be close to 5 minutes
        assert record.remaining_ttl > timedelta(minutes=4)
        assert record.remaining_ttl <= timedelta(minutes=5)

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization round-trip."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        now = datetime.now()
        record = DeduplicationRecord(
            fingerprint=fp,
            sent_at=now,
            expires_at=now + timedelta(minutes=5),
            count=3,
            last_duplicate_at=now + timedelta(seconds=30),
        )

        data = record.to_dict()
        restored = DeduplicationRecord.from_dict(data)

        assert restored.fingerprint.key == record.fingerprint.key
        assert restored.count == 3
        assert restored.last_duplicate_at is not None


class TestDeduplicationResult:
    """Tests for DeduplicationResult class."""

    def test_should_send_not_duplicate(self) -> None:
        """Test should_send when not duplicate."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        result = DeduplicationResult(
            is_duplicate=False,
            fingerprint=fp,
        )
        assert result.should_send

    def test_should_send_is_duplicate(self) -> None:
        """Test should_send when is duplicate."""
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        result = DeduplicationResult(
            is_duplicate=True,
            fingerprint=fp,
        )
        assert not result.should_send


class TestDeduplicationStats:
    """Tests for DeduplicationStats class."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = DeduplicationStats()
        assert stats.total_checked == 0
        assert stats.duplicates_found == 0
        assert stats.notifications_sent == 0

    def test_deduplication_rate(self) -> None:
        """Test deduplication rate calculation."""
        stats = DeduplicationStats(
            total_checked=100,
            duplicates_found=25,
        )
        assert stats.deduplication_rate == 25.0

    def test_deduplication_rate_zero_checked(self) -> None:
        """Test deduplication rate with zero checks."""
        stats = DeduplicationStats()
        assert stats.deduplication_rate == 0.0

    def test_to_dict(self) -> None:
        """Test to_dict."""
        stats = DeduplicationStats(
            total_checked=100,
            duplicates_found=25,
            notifications_sent=75,
            store_size=50,
        )
        data = stats.to_dict()

        assert data["total_checked"] == 100
        assert data["duplicates_found"] == 25
        assert data["deduplication_rate"] == 25.0
