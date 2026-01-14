"""Tests for deduplication middleware."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound.checkpoint.actions.base import ActionConfig, ActionResult, ActionStatus
from truthound.checkpoint.deduplication.middleware import (
    DeduplicatedAction,
    DeduplicationMiddleware,
    DeduplicationMixin,
    configure_global_deduplication,
    deduplicated,
    get_global_middleware,
    set_global_middleware,
)
from truthound.checkpoint.deduplication.protocols import TimeWindow
from truthound.checkpoint.deduplication.service import (
    DeduplicationConfig,
    DeduplicationPolicy,
    NotificationDeduplicator,
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


class MockAction:
    """Mock action for testing."""

    action_type = "mock"

    def __init__(self, name: str = "mock_action") -> None:
        self._name = name
        self._config = ActionConfig(name=name)
        self.execute_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> ActionConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def should_run(self, result_status: str) -> bool:
        return True

    def execute(self, checkpoint_result: Any) -> ActionResult:
        self.execute_count += 1
        return ActionResult(
            action_name=self.name,
            action_type=self.action_type,
            status=ActionStatus.SUCCESS,
            message="Mock executed",
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )


class TestDeduplicationMiddleware:
    """Tests for DeduplicationMiddleware."""

    def test_wrap_action(self) -> None:
        """Test wrapping an action."""
        middleware = DeduplicationMiddleware()
        action = MockAction()

        wrapped = middleware.wrap(action)

        assert isinstance(wrapped, DeduplicatedAction)
        assert wrapped.action_type == "mock"
        assert wrapped.name == "mock_action"

    def test_check_not_duplicate(self) -> None:
        """Test check returns not duplicate for new notification."""
        middleware = DeduplicationMiddleware()
        action = MockAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        dedup_result = middleware.check(result, action)

        assert not dedup_result.is_duplicate
        assert dedup_result.should_send

    def test_check_is_duplicate(self) -> None:
        """Test check returns duplicate for existing notification."""
        middleware = DeduplicationMiddleware()
        action = MockAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        # First check and mark
        first = middleware.check(result, action)
        middleware.mark_sent(result, action, first)

        # Second check
        second = middleware.check(result, action)
        assert second.is_duplicate

    def test_disabled_middleware(self) -> None:
        """Test disabled middleware doesn't deduplicate."""
        middleware = DeduplicationMiddleware(enabled=False)
        action = MockAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        # First check
        first = middleware.check(result, action)
        assert not first.is_duplicate

        # Even after marking, second check should not be duplicate
        # (deduplication is disabled)
        middleware.mark_sent(result, action, first)
        second = middleware.check(result, action)
        assert not second.is_duplicate


class TestDeduplicatedAction:
    """Tests for DeduplicatedAction."""

    def test_execute_not_duplicate(self) -> None:
        """Test execute proceeds when not duplicate."""
        middleware = DeduplicationMiddleware()
        action = MockAction()
        wrapped = middleware.wrap(action)
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        action_result = wrapped.execute(result)

        assert action_result.status == ActionStatus.SUCCESS
        assert action.execute_count == 1

    def test_execute_skipped_duplicate(self) -> None:
        """Test execute skipped when duplicate."""
        middleware = DeduplicationMiddleware()
        action = MockAction()
        wrapped = middleware.wrap(action)
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        # First execution
        first_result = wrapped.execute(result)
        assert first_result.status == ActionStatus.SUCCESS
        assert action.execute_count == 1

        # Second execution - should be skipped
        second_result = wrapped.execute(result)
        assert second_result.status == ActionStatus.SKIPPED
        assert "deduplication" in second_result.details.get("reason", "")
        assert action.execute_count == 1  # Not incremented

    def test_properties_delegated(self) -> None:
        """Test properties are delegated to wrapped action."""
        middleware = DeduplicationMiddleware()
        action = MockAction(name="test_action")
        wrapped = middleware.wrap(action)

        assert wrapped.action_type == "mock"
        assert wrapped.name == "test_action"
        assert wrapped.enabled
        assert wrapped.should_run("failure")


class TestDeduplicatedDecorator:
    """Tests for @deduplicated decorator."""

    def test_decorator_basic(self) -> None:
        """Test basic decorator usage."""

        class TestAction:
            action_type = "test"
            name = "test_action"

            @deduplicated()
            def _execute(self, checkpoint_result: Any) -> ActionResult:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Executed",
                )

        action = TestAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        # First call
        first = action._execute(result)
        assert first.status == ActionStatus.SUCCESS

        # Second call - should be deduplicated
        second = action._execute(result)
        assert second.status == ActionStatus.SKIPPED

    def test_decorator_with_policy(self) -> None:
        """Test decorator with custom policy."""

        class TestAction:
            action_type = "test"
            name = "test_action"

            @deduplicated(policy=DeduplicationPolicy.BASIC)
            def _execute(self, checkpoint_result: Any) -> ActionResult:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="Executed",
                )

        action = TestAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        first = action._execute(result)
        assert first.status == ActionStatus.SUCCESS


class TestDeduplicationMixin:
    """Tests for DeduplicationMixin."""

    def test_is_deduplicated(self) -> None:
        """Test is_deduplicated method."""

        class TestAction(DeduplicationMixin):
            action_type = "test"
            name = "test_action"
            dedup_enabled = True
            dedup_policy = DeduplicationPolicy.BASIC
            dedup_window = TimeWindow(minutes=5)
            _deduplicator = None

        action = TestAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        # First check - not deduplicated
        assert not action.is_deduplicated(result)

        # Mark as sent
        action.mark_notification_sent(result)

        # Second check - is deduplicated
        assert action.is_deduplicated(result)

    def test_check_deduplication(self) -> None:
        """Test check_deduplication method."""

        class TestAction(DeduplicationMixin):
            action_type = "test"
            name = "test_action"
            dedup_enabled = True
            dedup_policy = DeduplicationPolicy.BASIC
            dedup_window = TimeWindow(minutes=5)
            _deduplicator = None

        action = TestAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        dedup_result = action.check_deduplication(result)
        assert not dedup_result.is_duplicate
        assert dedup_result.should_send

    def test_disabled_mixin(self) -> None:
        """Test mixin with deduplication disabled."""

        class TestAction(DeduplicationMixin):
            action_type = "test"
            name = "test_action"
            dedup_enabled = False
            _deduplicator = None

        action = TestAction()
        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        # Should never be deduplicated when disabled
        assert not action.is_deduplicated(result)
        action.mark_notification_sent(result)
        assert not action.is_deduplicated(result)

    def test_skipped_result(self) -> None:
        """Test _skipped_result helper."""

        class TestAction(DeduplicationMixin):
            action_type = "test"
            name = "test_action"

        action = TestAction()
        result = action._skipped_result("Duplicate detected")

        assert result.status == ActionStatus.SKIPPED
        assert "Duplicate" in result.message


class TestGlobalMiddleware:
    """Tests for global middleware management."""

    def test_get_global_middleware(self) -> None:
        """Test getting global middleware."""
        middleware = get_global_middleware()
        assert middleware is not None
        assert isinstance(middleware, DeduplicationMiddleware)

    def test_set_global_middleware(self) -> None:
        """Test setting global middleware."""
        custom_middleware = DeduplicationMiddleware(enabled=False)
        set_global_middleware(custom_middleware)

        retrieved = get_global_middleware()
        assert not retrieved.enabled

        # Reset for other tests
        set_global_middleware(DeduplicationMiddleware())

    def test_configure_global_deduplication(self) -> None:
        """Test configuring global deduplication."""
        middleware = configure_global_deduplication(
            policy=DeduplicationPolicy.STRICT,
            window=TimeWindow(minutes=10),
            enabled=True,
        )

        assert middleware.enabled
        assert middleware.deduplicator.config.policy == DeduplicationPolicy.STRICT
        assert middleware.deduplicator.config.default_window.total_seconds == 600

        # Reset for other tests
        set_global_middleware(DeduplicationMiddleware())


class TestIntegration:
    """Integration tests for deduplication middleware."""

    def test_full_workflow(self) -> None:
        """Test full deduplication workflow."""
        # Create middleware with specific config
        deduplicator = NotificationDeduplicator(
            store=InMemoryDeduplicationStore(),
            config=DeduplicationConfig(
                policy=DeduplicationPolicy.SEVERITY,
                default_window=TimeWindow(minutes=5),
            ),
        )
        middleware = DeduplicationMiddleware(deduplicator=deduplicator)

        # Create and wrap action
        action = MockAction()
        wrapped = middleware.wrap(action)

        # Create checkpoint result
        result = MockCheckpointResult(
            checkpoint_name="production_data_check",
            status="failure",
            validation_result=MockValidationResult(
                issues=[MockIssue("NullValidator", "high")]
            ),
        )

        # First execution - should succeed
        first_result = wrapped.execute(result)
        assert first_result.status == ActionStatus.SUCCESS
        assert action.execute_count == 1

        # Second execution (same checkpoint) - should be skipped
        second_result = wrapped.execute(result)
        assert second_result.status == ActionStatus.SKIPPED
        assert action.execute_count == 1  # Not incremented

        # Different checkpoint - should succeed
        different_result = MockCheckpointResult(
            checkpoint_name="staging_data_check",
            status="failure",
            validation_result=MockValidationResult(
                issues=[MockIssue("NullValidator", "high")]
            ),
        )
        third_result = wrapped.execute(different_result)
        assert third_result.status == ActionStatus.SUCCESS
        assert action.execute_count == 2

    def test_multiple_action_types(self) -> None:
        """Test deduplication across different action types."""
        middleware = DeduplicationMiddleware()

        slack_action = MockAction(name="slack_notify")
        slack_action.action_type = "slack"
        email_action = MockAction(name="email_notify")
        email_action.action_type = "email"

        wrapped_slack = middleware.wrap(slack_action)
        wrapped_email = middleware.wrap(email_action)

        result = MockCheckpointResult(
            checkpoint_name="test",
            status="failure",
        )

        # Slack notification
        slack_result = wrapped_slack.execute(result)
        assert slack_result.status == ActionStatus.SUCCESS
        assert slack_action.execute_count == 1

        # Email notification - different action type, should not be duplicate
        email_result = wrapped_email.execute(result)
        assert email_result.status == ActionStatus.SUCCESS
        assert email_action.execute_count == 1

        # Second Slack - should be duplicate
        slack_result2 = wrapped_slack.execute(result)
        assert slack_result2.status == ActionStatus.SKIPPED
        assert slack_action.execute_count == 1
