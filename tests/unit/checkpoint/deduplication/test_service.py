"""Tests for deduplication service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from truthound.checkpoint.deduplication.protocols import (
    NotificationFingerprint,
    TimeWindow,
)
from truthound.checkpoint.deduplication.service import (
    DeduplicationConfig,
    DeduplicationPolicy,
    DeduplicatorBuilder,
    NotificationDeduplicator,
    create_deduplicator,
)
from truthound.checkpoint.deduplication.stores import InMemoryDeduplicationStore


@dataclass
class MockValidationResult:
    """Mock validation result."""

    issues: list[Any]


@dataclass
class MockIssue:
    """Mock validation issue."""

    validator_name: str
    severity: str = "medium"


@dataclass
class MockCheckpointResult:
    """Mock checkpoint result for testing."""

    checkpoint_name: str
    status: str
    run_id: str = "test-run-id"
    data_asset: str = "test_asset"
    validation_result: MockValidationResult | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class TestDeduplicationConfig:
    """Tests for DeduplicationConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DeduplicationConfig()
        assert config.enabled
        assert config.policy == DeduplicationPolicy.SEVERITY
        assert config.default_window.total_seconds == 300
        assert config.max_suppressed_count == 100

    def test_get_window_for_action(self) -> None:
        """Test action-specific window override."""
        config = DeduplicationConfig(
            action_windows={
                "slack": TimeWindow(minutes=10),
                "email": TimeWindow(hours=1),
            }
        )

        assert config.get_window_for_action("slack").total_seconds == 600
        assert config.get_window_for_action("email").total_seconds == 3600
        assert config.get_window_for_action("pagerduty") == config.default_window

    def test_get_window_for_severity(self) -> None:
        """Test severity-based window."""
        config = DeduplicationConfig()

        assert config.get_window_for_severity("critical").total_seconds == 60
        assert config.get_window_for_severity("high").total_seconds == 300
        assert config.get_window_for_severity("low").total_seconds == 3600


class TestNotificationDeduplicator:
    """Tests for NotificationDeduplicator."""

    def test_check_no_duplicate(self) -> None:
        """Test check when no duplicate exists."""
        deduplicator = NotificationDeduplicator()
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        dedup_result = deduplicator.check(result, "slack")

        assert not dedup_result.is_duplicate
        assert dedup_result.should_send
        assert dedup_result.fingerprint.checkpoint_name == "test_checkpoint"

    def test_check_duplicate_found(self) -> None:
        """Test check when duplicate exists."""
        deduplicator = NotificationDeduplicator()
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        # First check - not duplicate
        first = deduplicator.check(result, "slack")
        assert not first.is_duplicate

        # Mark as sent
        deduplicator.mark_sent(first.fingerprint)

        # Second check - is duplicate
        second = deduplicator.check(result, "slack")
        assert second.is_duplicate
        # suppressed_count is 2 because: 1 (original) + 1 (this duplicate)
        assert second.suppressed_count >= 1

    def test_check_disabled(self) -> None:
        """Test check when deduplication is disabled."""
        config = DeduplicationConfig(enabled=False)
        deduplicator = NotificationDeduplicator(config=config)
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        # Mark as sent
        first = deduplicator.check(result, "slack")
        deduplicator.mark_sent(first.fingerprint)

        # Second check - still not duplicate (dedup disabled)
        second = deduplicator.check(result, "slack")
        assert not second.is_duplicate

    def test_mark_sent(self) -> None:
        """Test marking notification as sent."""
        deduplicator = NotificationDeduplicator()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )

        record = deduplicator.mark_sent(fp)

        assert record.fingerprint.key == fp.key
        assert record.count == 1

    def test_check_and_mark(self) -> None:
        """Test combined check and mark."""
        deduplicator = NotificationDeduplicator()
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        # First call - not duplicate, marks as sent
        first = deduplicator.check_and_mark(result, "slack")
        assert not first.is_duplicate

        # Second call - is duplicate
        second = deduplicator.check_and_mark(result, "slack")
        assert second.is_duplicate

    def test_is_duplicate_simple(self) -> None:
        """Test simple duplicate check."""
        deduplicator = NotificationDeduplicator()
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        assert not deduplicator.is_duplicate(result, "slack")

        # Use check_and_mark to properly mark with correct fingerprint
        deduplicator.check_and_mark(result, "slack")

        # Now is duplicate
        assert deduplicator.is_duplicate(result, "slack")

    def test_policy_basic(self) -> None:
        """Test BASIC policy fingerprinting."""
        config = DeduplicationConfig(policy=DeduplicationPolicy.BASIC)
        deduplicator = NotificationDeduplicator(config=config)

        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
            validation_result=MockValidationResult(
                issues=[MockIssue("NullValidator", "high")]
            ),
        )

        first = deduplicator.check(result, "slack")
        deduplicator.mark_sent(first.fingerprint)

        # Same checkpoint and action = duplicate
        second = deduplicator.check(result, "slack")
        assert second.is_duplicate

        # Different action = not duplicate
        third = deduplicator.check(result, "email")
        assert not third.is_duplicate

    def test_policy_severity(self) -> None:
        """Test SEVERITY policy fingerprinting."""
        config = DeduplicationConfig(policy=DeduplicationPolicy.SEVERITY)
        deduplicator = NotificationDeduplicator(config=config)

        result_high = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
            validation_result=MockValidationResult(
                issues=[MockIssue("NullValidator", "high")]
            ),
        )
        result_low = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
            validation_result=MockValidationResult(
                issues=[MockIssue("NullValidator", "low")]
            ),
        )

        first = deduplicator.check(result_high, "slack", severity="high")
        deduplicator.mark_sent(first.fingerprint)

        # Same severity = duplicate
        second = deduplicator.check(result_high, "slack", severity="high")
        assert second.is_duplicate

        # Different severity = not duplicate
        third = deduplicator.check(result_low, "slack", severity="low")
        assert not third.is_duplicate

    def test_policy_issue_based(self) -> None:
        """Test ISSUE_BASED policy fingerprinting."""
        config = DeduplicationConfig(policy=DeduplicationPolicy.ISSUE_BASED)
        deduplicator = NotificationDeduplicator(config=config)

        result_null = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
            validation_result=MockValidationResult(
                issues=[MockIssue("NullValidator", "high")]
            ),
        )
        result_range = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
            validation_result=MockValidationResult(
                issues=[MockIssue("RangeValidator", "high")]
            ),
        )

        first = deduplicator.check(result_null, "slack")
        deduplicator.mark_sent(first.fingerprint)

        # Same issues = duplicate
        second = deduplicator.check(result_null, "slack")
        assert second.is_duplicate

        # Different issues = not duplicate
        third = deduplicator.check(result_range, "slack")
        assert not third.is_duplicate

    def test_severity_based_windows(self) -> None:
        """Test different windows based on severity."""
        config = DeduplicationConfig(
            policy=DeduplicationPolicy.SEVERITY,
            severity_windows={
                "critical": TimeWindow(seconds=10),
                "high": TimeWindow(seconds=30),
                "medium": TimeWindow(minutes=1),
            },
        )
        deduplicator = NotificationDeduplicator(config=config)

        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        # Critical has shorter window
        first = deduplicator.check(result, "slack", severity="critical")
        deduplicator.mark_sent(first.fingerprint)

        # Check immediately - still duplicate
        second = deduplicator.check(result, "slack", severity="critical")
        assert second.is_duplicate

    def test_get_stats(self) -> None:
        """Test statistics retrieval."""
        deduplicator = NotificationDeduplicator()
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        # Perform some operations
        first = deduplicator.check(result, "slack")
        deduplicator.mark_sent(first.fingerprint)
        deduplicator.check(result, "slack")  # Duplicate

        stats = deduplicator.get_stats()
        assert stats.notifications_sent >= 1
        assert stats.store_size >= 1

    def test_cleanup(self) -> None:
        """Test cleanup of expired records."""
        store = InMemoryDeduplicationStore()
        deduplicator = NotificationDeduplicator(store=store)

        # Cleanup with no expired records
        removed = deduplicator.cleanup()
        assert removed == 0

    def test_clear(self) -> None:
        """Test clearing all records."""
        deduplicator = NotificationDeduplicator()
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        first = deduplicator.check(result, "slack")
        deduplicator.mark_sent(first.fingerprint)

        deduplicator.clear()

        # After clear, should not be duplicate
        second = deduplicator.check(result, "slack")
        assert not second.is_duplicate

    def test_max_suppression_callback(self) -> None:
        """Test callback when max suppression exceeded."""
        callback_called = []

        def on_exceeded(record: Any) -> None:
            callback_called.append(record)

        config = DeduplicationConfig(
            max_suppressed_count=3,
            on_suppression_exceeded=on_exceeded,
        )
        deduplicator = NotificationDeduplicator(config=config)
        result = MockCheckpointResult(
            checkpoint_name="test_checkpoint",
            status="failure",
        )

        # First - not duplicate
        first = deduplicator.check(result, "slack")
        deduplicator.mark_sent(first.fingerprint)

        # Next 4 - duplicates
        for _ in range(4):
            deduplicator.check(result, "slack")

        # Callback should have been called
        assert len(callback_called) >= 1


class TestDeduplicatorBuilder:
    """Tests for DeduplicatorBuilder."""

    def test_basic_build(self) -> None:
        """Test basic builder usage."""
        deduplicator = DeduplicatorBuilder().build()
        assert deduplicator is not None
        assert deduplicator.config.enabled

    def test_with_policy(self) -> None:
        """Test setting policy."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_policy(DeduplicationPolicy.STRICT)
            .build()
        )
        assert deduplicator.config.policy == DeduplicationPolicy.STRICT

    def test_with_default_window(self) -> None:
        """Test setting default window."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_default_window(TimeWindow(minutes=10))
            .build()
        )
        assert deduplicator.config.default_window.total_seconds == 600

    def test_with_action_window(self) -> None:
        """Test setting action-specific window."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_action_window("slack", TimeWindow(minutes=15))
            .with_action_window("email", TimeWindow(hours=1))
            .build()
        )
        assert deduplicator.config.action_windows["slack"].total_seconds == 900
        assert deduplicator.config.action_windows["email"].total_seconds == 3600

    def test_with_severity_window(self) -> None:
        """Test setting severity-specific window."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_severity_window("critical", TimeWindow(seconds=30))
            .build()
        )
        assert deduplicator.config.severity_windows["critical"].total_seconds == 30

    def test_with_max_suppression(self) -> None:
        """Test setting max suppression count."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_max_suppression(50)
            .build()
        )
        assert deduplicator.config.max_suppressed_count == 50

    def test_with_memory_store(self) -> None:
        """Test using memory store."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_memory_store(max_size=5000)
            .build()
        )
        assert isinstance(deduplicator.store, InMemoryDeduplicationStore)

    def test_with_sliding_window(self) -> None:
        """Test using sliding window strategy."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_sliding_window()
            .build()
        )
        assert deduplicator.processor.strategy.name == "sliding"

    def test_with_tumbling_window(self) -> None:
        """Test using tumbling window strategy."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_tumbling_window()
            .build()
        )
        assert deduplicator.processor.strategy.name == "tumbling"

    def test_with_session_window(self) -> None:
        """Test using session window strategy."""
        deduplicator = (
            DeduplicatorBuilder()
            .with_session_window(
                gap_duration=TimeWindow(minutes=10),
                max_duration=TimeWindow(hours=2),
            )
            .build()
        )
        assert deduplicator.processor.strategy.name == "session"

    def test_disabled(self) -> None:
        """Test disabling deduplication."""
        deduplicator = (
            DeduplicatorBuilder()
            .disabled()
            .build()
        )
        assert not deduplicator.config.enabled

    def test_enabled(self) -> None:
        """Test enabling deduplication."""
        deduplicator = (
            DeduplicatorBuilder()
            .disabled()
            .enabled()
            .build()
        )
        assert deduplicator.config.enabled

    def test_fluent_chain(self) -> None:
        """Test full fluent builder chain."""
        callback = MagicMock()
        deduplicator = (
            DeduplicatorBuilder()
            .with_policy(DeduplicationPolicy.SEVERITY)
            .with_default_window(TimeWindow(minutes=5))
            .with_action_window("pagerduty", TimeWindow(minutes=1))
            .with_severity_window("critical", TimeWindow(seconds=30))
            .with_max_suppression(10)
            .with_suppression_callback(callback)
            .with_memory_store(max_size=10000)
            .with_sliding_window()
            .enabled()
            .build()
        )

        assert deduplicator.config.policy == DeduplicationPolicy.SEVERITY
        assert deduplicator.config.default_window.total_seconds == 300
        assert deduplicator.config.action_windows["pagerduty"].total_seconds == 60
        assert deduplicator.config.max_suppressed_count == 10
        assert deduplicator.config.enabled


class TestCreateDeduplicator:
    """Tests for create_deduplicator factory function."""

    def test_default_creation(self) -> None:
        """Test default deduplicator creation."""
        deduplicator = create_deduplicator()
        assert deduplicator is not None
        assert deduplicator.config.policy == DeduplicationPolicy.SEVERITY

    def test_with_policy(self) -> None:
        """Test creation with specific policy."""
        deduplicator = create_deduplicator(policy=DeduplicationPolicy.BASIC)
        assert deduplicator.config.policy == DeduplicationPolicy.BASIC

    def test_with_window(self) -> None:
        """Test creation with specific window."""
        deduplicator = create_deduplicator(window=TimeWindow(minutes=10))
        assert deduplicator.config.default_window.total_seconds == 600
