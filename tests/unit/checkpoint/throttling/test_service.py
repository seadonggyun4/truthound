"""Tests for the throttling service and builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from truthound.checkpoint.throttling.protocols import (
    RateLimit,
    RateLimitScope,
    ThrottleStatus,
    TimeUnit,
)
from truthound.checkpoint.throttling.service import (
    NotificationThrottler,
    ThrottlerBuilder,
    create_throttler,
)
from truthound.checkpoint.throttling.throttlers import CompositeThrottler, NoOpThrottler


@dataclass
class MockCheckpointResult:
    """Mock checkpoint result for testing."""

    checkpoint_name: str = "test_checkpoint"
    status: str = "failure"
    data_asset: str | None = "orders"
    validation_result: MagicMock | None = field(default_factory=lambda: MagicMock(issues=[]))


class TestNotificationThrottler:
    """Tests for NotificationThrottler."""

    def test_basic_check(self) -> None:
        """Test basic throttle check."""
        throttler = NotificationThrottler()

        result = throttler.check("slack", "test_checkpoint")

        assert result.allowed is True

    def test_basic_acquire(self) -> None:
        """Test basic throttle acquire."""
        throttler = NotificationThrottler()

        result = throttler.acquire("slack", "test_checkpoint")

        assert result.allowed is True

    def test_disabled_throttling(self) -> None:
        """Test disabled throttling always allows."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(enabled=False)
        throttler = NotificationThrottler(config=config)

        # Should always allow
        for _ in range(1000):
            result = throttler.acquire("slack", "test")
            assert result.allowed is True

    def test_rate_limit_enforcement(self) -> None:
        """Test rate limits are enforced."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(
            per_minute_limit=3,
            per_hour_limit=None,
            per_day_limit=None,
        )
        throttler = NotificationThrottler(config=config)

        # First 3 should pass
        for i in range(3):
            result = throttler.acquire("slack", "test")
            assert result.allowed is True, f"Request {i + 1} should be allowed"

        # 4th should be throttled
        result = throttler.acquire("slack", "test")
        assert result.allowed is False

    def test_is_throttled_helper(self) -> None:
        """Test is_throttled helper method."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(
            per_minute_limit=2,
            per_hour_limit=None,
            per_day_limit=None,
        )
        throttler = NotificationThrottler(config=config)

        # Not throttled initially
        assert throttler.is_throttled("slack") is False

        # Exhaust limits
        throttler.acquire("slack", "test")
        throttler.acquire("slack", "test")

        # Now throttled
        assert throttler.is_throttled("slack") is True

    def test_priority_bypass_critical(self) -> None:
        """Test priority bypass for critical severity."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(
            per_minute_limit=1,
            per_hour_limit=None,
            per_day_limit=None,
            priority_bypass=True,
            priority_threshold="critical",
        )
        throttler = NotificationThrottler(config=config)

        # Exhaust normal limit
        throttler.acquire("slack", "test", severity="low")

        # Low severity should be throttled
        result = throttler.acquire("slack", "test", severity="low")
        assert result.allowed is False

        # Critical should bypass
        result = throttler.acquire("slack", "test", severity="critical")
        assert result.allowed is True
        assert result.metadata.get("bypassed") is True

    def test_priority_bypass_high(self) -> None:
        """Test priority bypass for high severity."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(
            per_minute_limit=1,
            per_hour_limit=None,
            per_day_limit=None,
            priority_bypass=True,
            priority_threshold="high",
        )
        throttler = NotificationThrottler(config=config)

        # Exhaust normal limit
        throttler.acquire("slack", "test", severity="low")

        # High and critical should bypass
        result = throttler.acquire("slack", "test", severity="high")
        assert result.allowed is True

        result = throttler.acquire("slack", "test", severity="critical")
        assert result.allowed is True

    def test_check_result_with_checkpoint(self) -> None:
        """Test check_result with checkpoint result."""
        throttler = NotificationThrottler()
        checkpoint_result = MockCheckpointResult()

        result = throttler.check_result(checkpoint_result, "slack")

        assert result.allowed is True

    def test_acquire_result_with_checkpoint(self) -> None:
        """Test acquire_result with checkpoint result."""
        throttler = NotificationThrottler()
        checkpoint_result = MockCheckpointResult()

        result = throttler.acquire_result(checkpoint_result, "slack")

        assert result.allowed is True

    def test_per_action_scope(self) -> None:
        """Test per-action scope separation."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(
            per_minute_limit=2,
            per_hour_limit=None,
            per_day_limit=None,
            scope=RateLimitScope.PER_ACTION,
        )
        throttler = NotificationThrottler(config=config)

        # Exhaust slack
        throttler.acquire("slack", "test")
        throttler.acquire("slack", "test")
        result = throttler.acquire("slack", "test")
        assert result.allowed is False

        # Email should still be available
        result = throttler.acquire("email", "test")
        assert result.allowed is True

    def test_custom_action_limits(self) -> None:
        """Test custom limits per action with per-action scope."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(
            per_minute_limit=10,  # Default
            per_hour_limit=None,
            per_day_limit=None,
            scope=RateLimitScope.PER_ACTION,  # Need per-action scope for custom limits
            custom_limits={
                "pagerduty": [RateLimit.per_minute(2)],  # Stricter for PagerDuty
            },
        )
        throttler = NotificationThrottler(config=config)

        # PagerDuty should be throttled after 2
        throttler.acquire("pagerduty", "test")
        throttler.acquire("pagerduty", "test")
        result = throttler.acquire("pagerduty", "test")
        assert result.allowed is False

        # Slack should still have more capacity (uses default limits)
        for _ in range(5):
            result = throttler.acquire("slack", "test")
            assert result.allowed is True

    def test_reset_all(self) -> None:
        """Test reset all throttle state."""
        from truthound.checkpoint.throttling.protocols import ThrottlingConfig

        config = ThrottlingConfig(
            per_minute_limit=2,
            per_hour_limit=None,
            per_day_limit=None,
        )
        throttler = NotificationThrottler(config=config)

        # Exhaust limits
        throttler.acquire("slack", "test")
        throttler.acquire("slack", "test")
        assert throttler.is_throttled("slack") is True

        # Reset
        throttler.reset()

        # Should allow again
        assert throttler.is_throttled("slack") is False

    def test_get_stats(self) -> None:
        """Test statistics retrieval."""
        throttler = NotificationThrottler()

        for _ in range(5):
            throttler.acquire("slack", "test")

        stats = throttler.get_stats()
        assert stats.total_checked >= 5


class TestThrottlerBuilder:
    """Tests for ThrottlerBuilder."""

    def test_basic_build(self) -> None:
        """Test basic builder usage."""
        throttler = ThrottlerBuilder().build()

        assert throttler.config.per_minute_limit == 10
        assert throttler.config.per_hour_limit == 100
        assert throttler.config.per_day_limit == 500

    def test_custom_limits(self) -> None:
        """Test setting custom limits."""
        throttler = (
            ThrottlerBuilder()
            .with_per_minute_limit(5)
            .with_per_hour_limit(50)
            .with_per_day_limit(200)
            .build()
        )

        assert throttler.config.per_minute_limit == 5
        assert throttler.config.per_hour_limit == 50
        assert throttler.config.per_day_limit == 200

    def test_disable_limits(self) -> None:
        """Test disabling specific limits."""
        throttler = (
            ThrottlerBuilder()
            .with_per_minute_limit(10)
            .with_per_hour_limit(None)
            .with_per_day_limit(None)
            .build()
        )

        limits = throttler.config.get_rate_limits()
        assert len(limits) == 1
        assert limits[0].time_unit == TimeUnit.MINUTE

    def test_burst_allowance(self) -> None:
        """Test burst allowance setting."""
        throttler = (
            ThrottlerBuilder()
            .with_per_minute_limit(10)
            .with_burst_allowance(1.5)
            .build()
        )

        assert throttler.config.burst_multiplier == 1.5

    def test_scope_setting(self) -> None:
        """Test scope setting."""
        throttler = (
            ThrottlerBuilder()
            .with_scope(RateLimitScope.PER_ACTION)
            .build()
        )

        assert throttler.config.scope == RateLimitScope.PER_ACTION

    def test_algorithm_setting(self) -> None:
        """Test algorithm setting."""
        for algo in ["token_bucket", "sliding_window", "fixed_window"]:
            throttler = (
                ThrottlerBuilder()
                .with_algorithm(algo)
                .build()
            )

            assert throttler.config.algorithm == algo

    def test_invalid_algorithm(self) -> None:
        """Test invalid algorithm raises error."""
        with pytest.raises(ValueError):
            ThrottlerBuilder().with_algorithm("invalid")

    def test_priority_bypass(self) -> None:
        """Test priority bypass setting."""
        throttler = (
            ThrottlerBuilder()
            .with_priority_bypass("high")
            .build()
        )

        assert throttler.config.priority_bypass is True
        assert throttler.config.priority_threshold == "high"

    def test_priority_bypass_disable(self) -> None:
        """Test disabling priority bypass."""
        throttler = (
            ThrottlerBuilder()
            .with_priority_bypass()
            .without_priority_bypass()
            .build()
        )

        assert throttler.config.priority_bypass is False

    def test_action_limit(self) -> None:
        """Test action-specific limits."""
        throttler = (
            ThrottlerBuilder()
            .with_action_limit("pagerduty", per_minute=2, per_hour=10)
            .build()
        )

        limits = throttler.config.get_limits_for_action("pagerduty")
        assert len(limits) == 2

    def test_severity_limit(self) -> None:
        """Test severity-specific limits."""
        throttler = (
            ThrottlerBuilder()
            .with_severity_limit("low", per_minute=1, per_hour=5)
            .build()
        )

        limits = throttler.config.get_limits_for_severity("low")
        assert len(limits) == 2

    def test_queueing(self) -> None:
        """Test queueing configuration."""
        throttler = (
            ThrottlerBuilder()
            .with_queueing(max_size=500)
            .build()
        )

        assert throttler.config.queue_on_throttle is True
        assert throttler.config.max_queue_size == 500

    def test_enabled_disabled(self) -> None:
        """Test enable/disable methods."""
        throttler = ThrottlerBuilder().disabled().build()
        assert throttler.config.enabled is False

        throttler = ThrottlerBuilder().enabled(True).build()
        assert throttler.config.enabled is True

    def test_fluent_chaining(self) -> None:
        """Test fluent API chaining."""
        throttler = (
            ThrottlerBuilder()
            .with_per_minute_limit(10)
            .with_per_hour_limit(100)
            .with_per_day_limit(500)
            .with_burst_allowance(1.2)
            .with_scope(RateLimitScope.PER_ACTION)
            .with_algorithm("token_bucket")
            .with_priority_bypass("critical")
            .with_action_limit("slack", per_minute=20)
            .enabled()
            .build()
        )

        assert throttler.config.per_minute_limit == 10
        assert throttler.config.per_hour_limit == 100
        assert throttler.config.per_day_limit == 500
        assert throttler.config.burst_multiplier == 1.2
        assert throttler.config.scope == RateLimitScope.PER_ACTION
        assert throttler.config.priority_bypass is True


class TestCreateThrottler:
    """Tests for create_throttler convenience function."""

    def test_basic_creation(self) -> None:
        """Test basic throttler creation."""
        throttler = create_throttler()

        assert throttler.config.per_minute_limit == 10
        assert throttler.config.per_hour_limit == 100
        assert throttler.config.per_day_limit == 500

    def test_custom_parameters(self) -> None:
        """Test creation with custom parameters."""
        throttler = create_throttler(
            per_minute=5,
            per_hour=50,
            per_day=200,
            burst_multiplier=1.5,
            scope=RateLimitScope.PER_ACTION,
        )

        assert throttler.config.per_minute_limit == 5
        assert throttler.config.per_hour_limit == 50
        assert throttler.config.per_day_limit == 200
        assert throttler.config.burst_multiplier == 1.5
        assert throttler.config.scope == RateLimitScope.PER_ACTION

    def test_disabled_creation(self) -> None:
        """Test creating disabled throttler."""
        throttler = create_throttler(enabled=False)

        assert throttler.config.enabled is False

        # Should always allow
        for _ in range(100):
            result = throttler.acquire("slack", "test")
            assert result.allowed is True
