"""Tests for throttling middleware."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound.checkpoint.actions.base import ActionResult, ActionStatus
from truthound.checkpoint.throttling.middleware import (
    ThrottledAction,
    ThrottlingMiddleware,
    ThrottlingMixin,
    configure_global_throttling,
    get_global_middleware,
    set_global_middleware,
    throttled,
)
from truthound.checkpoint.throttling.protocols import (
    RateLimitScope,
    ThrottleStatus,
    ThrottlingConfig,
)
from truthound.checkpoint.throttling.service import NotificationThrottler, ThrottlerBuilder


@dataclass
class MockCheckpointResult:
    """Mock checkpoint result for testing."""

    checkpoint_name: str = "test_checkpoint"
    status: str = "failure"
    data_asset: str | None = "orders"
    validation_result: MagicMock | None = field(default_factory=lambda: MagicMock(issues=[]))


@dataclass
class MockActionConfig:
    """Mock action config for testing."""

    name: str = "test_action"
    enabled: bool = True


class MockAction:
    """Mock action for testing."""

    action_type: str = "mock"

    def __init__(self, name: str = "test_action") -> None:
        self._name = name
        self._config = MockActionConfig(name=name)
        self._execute_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> MockActionConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def should_run(self, result_status: str) -> bool:
        return True

    def execute(self, checkpoint_result: Any) -> ActionResult:
        self._execute_count += 1
        return ActionResult(
            action_name=self._name,
            action_type=self.action_type,
            status=ActionStatus.SUCCESS,
            message="Success",
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )


class TestThrottlingMiddleware:
    """Tests for ThrottlingMiddleware."""

    def test_wrap_action(self) -> None:
        """Test wrapping an action."""
        middleware = ThrottlingMiddleware()
        action = MockAction()

        wrapped = middleware.wrap(action)

        assert isinstance(wrapped, ThrottledAction)
        assert wrapped.action_type == "mock"
        assert wrapped.name == "test_action"

    def test_check_not_throttled(self) -> None:
        """Test check when not throttled."""
        middleware = ThrottlingMiddleware()
        action = MockAction()
        checkpoint_result = MockCheckpointResult()

        result = middleware.check(checkpoint_result, action)

        assert result.allowed is True

    def test_check_when_disabled(self) -> None:
        """Test check when middleware is disabled."""
        middleware = ThrottlingMiddleware(enabled=False)
        action = MockAction()
        checkpoint_result = MockCheckpointResult()

        result = middleware.check(checkpoint_result, action)

        assert result.allowed is True

    def test_acquire_permits(self) -> None:
        """Test acquire consumes permits."""
        throttler = ThrottlerBuilder().with_per_minute_limit(2).build()
        middleware = ThrottlingMiddleware(throttler=throttler)
        action = MockAction()
        checkpoint_result = MockCheckpointResult()

        # First two should pass
        result = middleware.acquire(checkpoint_result, action)
        assert result.allowed is True

        result = middleware.acquire(checkpoint_result, action)
        assert result.allowed is True

        # Third should be throttled
        result = middleware.acquire(checkpoint_result, action)
        assert result.allowed is False


class TestThrottledAction:
    """Tests for ThrottledAction wrapper."""

    def test_delegates_properties(self) -> None:
        """Test property delegation to wrapped action."""
        middleware = ThrottlingMiddleware()
        action = MockAction("my_action")

        wrapped = middleware.wrap(action)

        assert wrapped.action_type == "mock"
        assert wrapped.name == "my_action"
        assert wrapped.enabled is True

    def test_execute_when_allowed(self) -> None:
        """Test execution when not throttled."""
        middleware = ThrottlingMiddleware()
        action = MockAction()
        wrapped = middleware.wrap(action)
        checkpoint_result = MockCheckpointResult()

        result = wrapped.execute(checkpoint_result)

        assert result.status == ActionStatus.SUCCESS
        assert action._execute_count == 1

    def test_execute_when_throttled(self) -> None:
        """Test execution when throttled."""
        throttler = ThrottlerBuilder().with_per_minute_limit(1).build()
        middleware = ThrottlingMiddleware(throttler=throttler)
        action = MockAction()
        wrapped = middleware.wrap(action)
        checkpoint_result = MockCheckpointResult()

        # First should execute
        result = wrapped.execute(checkpoint_result)
        assert result.status == ActionStatus.SUCCESS
        assert action._execute_count == 1

        # Second should be skipped
        result = wrapped.execute(checkpoint_result)
        assert result.status == ActionStatus.SKIPPED
        assert "Throttled" in result.message
        assert action._execute_count == 1  # Still 1

    def test_execute_includes_throttle_details(self) -> None:
        """Test throttled result includes details."""
        # Use per_minute_limit=1 and exhaust it first
        throttler = ThrottlerBuilder().with_per_minute_limit(1).with_per_hour_limit(None).with_per_day_limit(None).build()
        middleware = ThrottlingMiddleware(throttler=throttler)
        action = MockAction()
        wrapped = middleware.wrap(action)
        checkpoint_result = MockCheckpointResult()

        # Exhaust the limit
        wrapped.execute(checkpoint_result)

        # Second should be throttled
        result = wrapped.execute(checkpoint_result)

        assert result.status == ActionStatus.SKIPPED
        assert result.details.get("reason") == "throttled"
        assert "retry_after" in result.details

    def test_skip_on_error(self) -> None:
        """Test skip_on_error behavior."""
        middleware = ThrottlingMiddleware(skip_on_error=True)
        action = MockAction()
        wrapped = ThrottledAction(action=action, middleware=middleware)
        checkpoint_result = MockCheckpointResult()

        # Mock throttler to raise error
        with patch.object(
            middleware, "acquire", side_effect=RuntimeError("Test error")
        ):
            result = wrapped.execute(checkpoint_result)

        # Should still execute because skip_on_error=True
        assert result.status == ActionStatus.SUCCESS

    def test_raise_on_error(self) -> None:
        """Test raising error when skip_on_error=False."""
        middleware = ThrottlingMiddleware(skip_on_error=False)
        action = MockAction()
        wrapped = ThrottledAction(action=action, middleware=middleware)
        checkpoint_result = MockCheckpointResult()

        with patch.object(
            middleware, "acquire", side_effect=RuntimeError("Test error")
        ):
            with pytest.raises(RuntimeError):
                wrapped.execute(checkpoint_result)


class TestThrottledDecorator:
    """Tests for @throttled decorator."""

    def test_decorator_basic(self) -> None:
        """Test basic decorator usage."""

        class MyAction:
            action_type = "test"
            name = "test_action"

            @throttled(per_minute=10, per_hour=100)
            def _execute(self, checkpoint_result: Any) -> ActionResult:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="OK",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )

        action = MyAction()
        checkpoint_result = MockCheckpointResult()

        result = action._execute(checkpoint_result)
        assert result.status == ActionStatus.SUCCESS

    def test_decorator_throttling(self) -> None:
        """Test decorator enforces throttling."""

        class MyAction:
            action_type = "test"
            name = "test_action"

            @throttled(per_minute=2, per_hour=None, per_day=None)
            def _execute(self, checkpoint_result: Any) -> ActionResult:
                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SUCCESS,
                    message="OK",
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                )

        action = MyAction()
        checkpoint_result = MockCheckpointResult()

        # First two should pass
        result = action._execute(checkpoint_result)
        assert result.status == ActionStatus.SUCCESS

        result = action._execute(checkpoint_result)
        assert result.status == ActionStatus.SUCCESS

        # Third should be throttled
        result = action._execute(checkpoint_result)
        assert result.status == ActionStatus.SKIPPED


class TestThrottlingMixin:
    """Tests for ThrottlingMixin."""

    def test_mixin_basic(self) -> None:
        """Test basic mixin usage."""

        class MyAction(ThrottlingMixin):
            action_type = "test"
            name = "test_action"
            throttle_per_minute = 10

        action = MyAction()
        checkpoint_result = MockCheckpointResult()

        assert action.is_throttled(checkpoint_result) is False

    def test_mixin_throttling(self) -> None:
        """Test mixin enforces throttling."""

        class MyAction(ThrottlingMixin):
            action_type = "test"
            name = "test_action"
            throttle_per_minute = 2
            throttle_per_hour = None
            throttle_per_day = None

        action = MyAction()
        checkpoint_result = MockCheckpointResult()

        # Acquire permits
        action.acquire_throttle(checkpoint_result)
        action.acquire_throttle(checkpoint_result)

        # Now should be throttled
        assert action.is_throttled(checkpoint_result) is True

    def test_mixin_disabled(self) -> None:
        """Test mixin when disabled."""

        class MyAction(ThrottlingMixin):
            action_type = "test"
            name = "test_action"
            throttle_enabled = False
            throttle_per_minute = 1

        action = MyAction()
        checkpoint_result = MockCheckpointResult()

        # Should never be throttled when disabled
        for _ in range(100):
            assert action.is_throttled(checkpoint_result) is False

    def test_throttled_result(self) -> None:
        """Test _throttled_result helper."""

        class MyAction(ThrottlingMixin):
            action_type = "test"
            name = "test_action"

        action = MyAction()
        result = action._throttled_result(retry_after=30.0)

        assert result.status == ActionStatus.SKIPPED
        assert result.details["retry_after"] == 30.0
        assert result.details["reason"] == "throttled"


class TestGlobalMiddleware:
    """Tests for global middleware management."""

    def test_get_global_creates_default(self) -> None:
        """Test get_global_middleware creates default."""
        # Reset global state
        set_global_middleware(None)  # type: ignore

        middleware = get_global_middleware()

        assert middleware is not None
        assert isinstance(middleware, ThrottlingMiddleware)

    def test_set_global_middleware(self) -> None:
        """Test setting global middleware."""
        custom = ThrottlingMiddleware(enabled=False)

        set_global_middleware(custom)
        retrieved = get_global_middleware()

        assert retrieved is custom
        assert retrieved.enabled is False

    def test_configure_global_throttling(self) -> None:
        """Test configure_global_throttling function."""
        middleware = configure_global_throttling(
            per_minute=5,
            per_hour=50,
            per_day=200,
            burst_multiplier=1.5,
            scope=RateLimitScope.PER_ACTION,
            priority_bypass=True,
        )

        assert middleware.throttler.config.per_minute_limit == 5
        assert middleware.throttler.config.per_hour_limit == 50
        assert middleware.throttler.config.per_day_limit == 200
        assert middleware.throttler.config.burst_multiplier == 1.5
        assert middleware.throttler.config.scope == RateLimitScope.PER_ACTION
        assert middleware.throttler.config.priority_bypass is True

        # Should be set as global
        retrieved = get_global_middleware()
        assert retrieved is middleware

    def test_configure_global_disabled(self) -> None:
        """Test configuring global throttling as disabled."""
        middleware = configure_global_throttling(enabled=False)

        assert middleware.enabled is False
