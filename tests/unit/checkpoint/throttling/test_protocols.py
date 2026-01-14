"""Tests for throttling protocols and core types."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from truthound.checkpoint.throttling.protocols import (
    BaseThrottler,
    RateLimit,
    RateLimitScope,
    ThrottleResult,
    ThrottleStatus,
    ThrottlingConfig,
    ThrottlingKey,
    ThrottlingRecord,
    ThrottlingStats,
    TimeUnit,
)


class TestTimeUnit:
    """Tests for TimeUnit enum."""

    def test_to_seconds_second(self) -> None:
        """Test second to seconds conversion."""
        assert TimeUnit.SECOND.to_seconds() == 1

    def test_to_seconds_minute(self) -> None:
        """Test minute to seconds conversion."""
        assert TimeUnit.MINUTE.to_seconds() == 60

    def test_to_seconds_hour(self) -> None:
        """Test hour to seconds conversion."""
        assert TimeUnit.HOUR.to_seconds() == 3600

    def test_to_seconds_day(self) -> None:
        """Test day to seconds conversion."""
        assert TimeUnit.DAY.to_seconds() == 86400

    def test_to_timedelta(self) -> None:
        """Test timedelta conversion."""
        assert TimeUnit.MINUTE.to_timedelta() == timedelta(minutes=1)
        assert TimeUnit.HOUR.to_timedelta() == timedelta(hours=1)


class TestRateLimit:
    """Tests for RateLimit dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic rate limit creation."""
        limit = RateLimit(limit=100, time_unit=TimeUnit.HOUR)
        assert limit.limit == 100
        assert limit.time_unit == TimeUnit.HOUR
        assert limit.burst_multiplier == 1.0

    def test_window_seconds(self) -> None:
        """Test window_seconds property."""
        limit = RateLimit(limit=100, time_unit=TimeUnit.HOUR)
        assert limit.window_seconds == 3600

    def test_burst_limit(self) -> None:
        """Test burst_limit property."""
        limit = RateLimit(limit=100, time_unit=TimeUnit.HOUR, burst_multiplier=1.5)
        assert limit.burst_limit == 150

    def test_tokens_per_second(self) -> None:
        """Test tokens_per_second property."""
        limit = RateLimit(limit=60, time_unit=TimeUnit.MINUTE)
        assert limit.tokens_per_second == 1.0

    def test_per_minute_factory(self) -> None:
        """Test per_minute factory method."""
        limit = RateLimit.per_minute(10)
        assert limit.limit == 10
        assert limit.time_unit == TimeUnit.MINUTE

    def test_per_hour_factory(self) -> None:
        """Test per_hour factory method."""
        limit = RateLimit.per_hour(100, burst_multiplier=1.2)
        assert limit.limit == 100
        assert limit.time_unit == TimeUnit.HOUR
        assert limit.burst_multiplier == 1.2

    def test_per_day_factory(self) -> None:
        """Test per_day factory method."""
        limit = RateLimit.per_day(500)
        assert limit.limit == 500
        assert limit.time_unit == TimeUnit.DAY

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        limit = RateLimit.per_hour(100, burst_multiplier=1.5)
        data = limit.to_dict()

        assert data["limit"] == 100
        assert data["time_unit"] == "hour"
        assert data["burst_multiplier"] == 1.5
        assert data["window_seconds"] == 3600
        assert data["burst_limit"] == 150


class TestThrottlingConfig:
    """Tests for ThrottlingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ThrottlingConfig()
        assert config.per_minute_limit == 10
        assert config.per_hour_limit == 100
        assert config.per_day_limit == 500
        assert config.burst_multiplier == 1.0
        assert config.scope == RateLimitScope.GLOBAL
        assert config.algorithm == "token_bucket"
        assert config.enabled is True

    def test_get_rate_limits(self) -> None:
        """Test get_rate_limits method."""
        config = ThrottlingConfig(
            per_minute_limit=5,
            per_hour_limit=50,
            per_day_limit=200,
        )
        limits = config.get_rate_limits()

        assert len(limits) == 3
        assert limits[0].limit == 5
        assert limits[0].time_unit == TimeUnit.MINUTE
        assert limits[1].limit == 50
        assert limits[1].time_unit == TimeUnit.HOUR
        assert limits[2].limit == 200
        assert limits[2].time_unit == TimeUnit.DAY

    def test_get_rate_limits_partial(self) -> None:
        """Test get_rate_limits with partial config."""
        config = ThrottlingConfig(
            per_minute_limit=5,
            per_hour_limit=None,
            per_day_limit=None,
        )
        limits = config.get_rate_limits()

        assert len(limits) == 1
        assert limits[0].limit == 5

    def test_custom_limits_for_action(self) -> None:
        """Test custom limits for specific action."""
        custom_limits = {
            "slack": [RateLimit.per_minute(5)],
            "email": [RateLimit.per_minute(2)],
        }
        config = ThrottlingConfig(custom_limits=custom_limits)

        slack_limits = config.get_limits_for_action("slack")
        assert len(slack_limits) == 1
        assert slack_limits[0].limit == 5

        # Unknown action returns default
        default_limits = config.get_limits_for_action("unknown")
        assert len(default_limits) == 3

    def test_severity_limits(self) -> None:
        """Test severity-specific limits."""
        severity_limits = {
            "critical": [RateLimit.per_minute(100)],
            "low": [RateLimit.per_minute(2)],
        }
        config = ThrottlingConfig(severity_limits=severity_limits)

        critical_limits = config.get_limits_for_severity("critical")
        assert len(critical_limits) == 1
        assert critical_limits[0].limit == 100

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        config = ThrottlingConfig()
        data = config.to_dict()

        assert data["per_minute_limit"] == 10
        assert data["per_hour_limit"] == 100
        assert data["per_day_limit"] == 500
        assert data["scope"] == "global"
        assert data["algorithm"] == "token_bucket"


class TestThrottlingKey:
    """Tests for ThrottlingKey dataclass."""

    def test_global_key(self) -> None:
        """Test global scope key."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        assert key.scope == RateLimitScope.GLOBAL
        assert key.time_unit == TimeUnit.MINUTE

    def test_action_key(self) -> None:
        """Test per-action scope key."""
        key = ThrottlingKey.for_action("slack", TimeUnit.HOUR)
        assert key.scope == RateLimitScope.PER_ACTION
        assert key.action_type == "slack"
        assert key.time_unit == TimeUnit.HOUR

    def test_checkpoint_key(self) -> None:
        """Test per-checkpoint scope key."""
        key = ThrottlingKey.for_checkpoint("data_quality", TimeUnit.DAY)
        assert key.scope == RateLimitScope.PER_CHECKPOINT
        assert key.checkpoint_name == "data_quality"

    def test_action_checkpoint_key(self) -> None:
        """Test per-action-checkpoint scope key."""
        key = ThrottlingKey.for_action_checkpoint("slack", "data_quality", TimeUnit.MINUTE)
        assert key.scope == RateLimitScope.PER_ACTION_CHECKPOINT
        assert key.action_type == "slack"
        assert key.checkpoint_name == "data_quality"

    def test_key_uniqueness(self) -> None:
        """Test key string uniqueness."""
        key1 = ThrottlingKey.for_action("slack", TimeUnit.MINUTE)
        key2 = ThrottlingKey.for_action("email", TimeUnit.MINUTE)
        key3 = ThrottlingKey.for_action("slack", TimeUnit.HOUR)

        assert key1.key != key2.key
        assert key1.key != key3.key
        assert key2.key != key3.key

    def test_key_equality(self) -> None:
        """Test key equality."""
        key1 = ThrottlingKey.for_action("slack", TimeUnit.MINUTE)
        key2 = ThrottlingKey.for_action("slack", TimeUnit.MINUTE)

        assert key1 == key2
        assert hash(key1) == hash(key2)

    def test_key_hash_deterministic(self) -> None:
        """Test key hash is deterministic."""
        key1 = ThrottlingKey.for_global(TimeUnit.MINUTE)
        key2 = ThrottlingKey.for_global(TimeUnit.MINUTE)

        assert key1.key == key2.key


class TestThrottlingRecord:
    """Tests for ThrottlingRecord dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic record creation."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key, tokens=10.0)

        assert record.key == key
        assert record.tokens == 10.0
        assert record.count == 0
        assert record.total_allowed == 0
        assert record.total_throttled == 0

    def test_total_requests(self) -> None:
        """Test total_requests property."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(
            key=key,
            total_allowed=100,
            total_throttled=20,
        )

        assert record.total_requests == 120

    def test_throttle_rate(self) -> None:
        """Test throttle_rate property."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(
            key=key,
            total_allowed=80,
            total_throttled=20,
        )

        assert record.throttle_rate == 20.0

    def test_throttle_rate_zero(self) -> None:
        """Test throttle_rate with zero requests."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key)

        assert record.throttle_rate == 0.0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key, tokens=5.0, count=10)
        data = record.to_dict()

        assert data["tokens"] == 5.0
        assert data["count"] == 10
        assert "window_start" in data
        assert "last_updated" in data


class TestThrottleResult:
    """Tests for ThrottleResult dataclass."""

    def test_allowed_result(self) -> None:
        """Test allowed_result factory."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)
        result = ThrottleResult.allowed_result(
            key=key,
            remaining=5,
            limit=limit,
        )

        assert result.status == ThrottleStatus.ALLOWED
        assert result.allowed is True
        assert result.remaining == 5
        assert result.retry_after == 0.0

    def test_burst_allowed_result(self) -> None:
        """Test burst allowed result."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        result = ThrottleResult.allowed_result(
            key=key,
            remaining=15,
            is_burst=True,
        )

        assert result.status == ThrottleStatus.BURST_ALLOWED
        assert result.allowed is True
        assert result.metadata["is_burst"] is True

    def test_throttled_result(self) -> None:
        """Test throttled_result factory."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)
        result = ThrottleResult.throttled_result(
            key=key,
            retry_after=30.0,
            limit=limit,
        )

        assert result.status == ThrottleStatus.THROTTLED
        assert result.allowed is False
        assert result.retry_after == 30.0
        assert result.remaining == 0

    def test_error_result(self) -> None:
        """Test error_result factory."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        result = ThrottleResult.error_result(
            key=key,
            error="Connection failed",
        )

        assert result.status == ThrottleStatus.ERROR
        assert result.allowed is False
        assert "Connection failed" in result.message

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        limit = RateLimit.per_minute(10)
        result = ThrottleResult.allowed_result(key=key, remaining=5, limit=limit)
        data = result.to_dict()

        assert data["status"] == "allowed"
        assert data["allowed"] is True
        assert data["remaining"] == 5


class TestThrottlingStats:
    """Tests for ThrottlingStats dataclass."""

    def test_default_values(self) -> None:
        """Test default statistics values."""
        stats = ThrottlingStats()
        assert stats.total_checked == 0
        assert stats.total_allowed == 0
        assert stats.total_throttled == 0

    def test_throttle_rate(self) -> None:
        """Test throttle_rate calculation."""
        stats = ThrottlingStats(
            total_checked=100,
            total_allowed=70,
            total_throttled=30,
        )

        assert stats.throttle_rate == 30.0

    def test_throttle_rate_zero(self) -> None:
        """Test throttle_rate with zero checks."""
        stats = ThrottlingStats()
        assert stats.throttle_rate == 0.0

    def test_allow_rate(self) -> None:
        """Test allow_rate calculation."""
        stats = ThrottlingStats(
            total_checked=100,
            total_allowed=60,
            total_burst_allowed=20,
            total_throttled=20,
        )

        assert stats.allow_rate == 80.0

    def test_allow_rate_zero(self) -> None:
        """Test allow_rate with zero checks."""
        stats = ThrottlingStats()
        assert stats.allow_rate == 100.0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = ThrottlingStats(
            total_checked=100,
            total_allowed=80,
            total_throttled=20,
        )
        data = stats.to_dict()

        assert data["total_checked"] == 100
        assert data["throttle_rate"] == 20.0
        assert data["allow_rate"] == 80.0
