"""Throttling Middleware for Actions.

This module provides middleware and decorators for integrating
throttling with checkpoint actions.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from truthound.checkpoint.actions.base import ActionResult, ActionStatus
from truthound.checkpoint.throttling.protocols import (
    RateLimitScope,
    ThrottleResult,
    ThrottleStatus,
    ThrottlingConfig,
)
from truthound.checkpoint.throttling.service import (
    NotificationThrottler,
    ThrottlerBuilder,
)

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ThrottlingMiddleware:
    """Middleware for transparent action throttling.

    Wraps action execution to automatically check rate limits
    before sending notifications.

    Attributes:
        throttler: The throttling service.
        enabled: Whether middleware is active.
        skip_on_error: Skip throttling on errors (allow notification).
        log_throttled: Log when notifications are throttled.

    Example:
        >>> middleware = ThrottlingMiddleware(
        ...     throttler=NotificationThrottler(),
        ... )
        >>>
        >>> # Wrap an action
        >>> wrapped_action = middleware.wrap(slack_action)
        >>> result = wrapped_action.execute(checkpoint_result)
    """

    throttler: NotificationThrottler = field(
        default_factory=lambda: NotificationThrottler(
            config=ThrottlingConfig()
        )
    )
    enabled: bool = True
    skip_on_error: bool = True
    log_throttled: bool = True

    def wrap(self, action: BaseAction[Any]) -> ThrottledAction:
        """Wrap an action with throttling.

        Args:
            action: The action to wrap.

        Returns:
            Wrapped action with throttling.
        """
        return ThrottledAction(
            action=action,
            middleware=self,
        )

    def check(
        self,
        checkpoint_result: CheckpointResult,
        action: BaseAction[Any],
    ) -> ThrottleResult:
        """Check if an action should be throttled.

        Args:
            checkpoint_result: The checkpoint result.
            action: The action to check.

        Returns:
            Throttle result.
        """
        if not self.enabled:
            from truthound.checkpoint.throttling.protocols import ThrottlingKey, TimeUnit

            key = ThrottlingKey.for_action(action.action_type, TimeUnit.MINUTE)
            return ThrottleResult.allowed_result(
                key=key,
                remaining=999,
                limit=None,
            )

        return self.throttler.check_result(checkpoint_result, action.action_type)

    def acquire(
        self,
        checkpoint_result: CheckpointResult,
        action: BaseAction[Any],
    ) -> ThrottleResult:
        """Acquire throttle permit for an action.

        Args:
            checkpoint_result: The checkpoint result.
            action: The action to throttle.

        Returns:
            Throttle result.
        """
        if not self.enabled:
            from truthound.checkpoint.throttling.protocols import ThrottlingKey, TimeUnit

            key = ThrottlingKey.for_action(action.action_type, TimeUnit.MINUTE)
            return ThrottleResult.allowed_result(
                key=key,
                remaining=999,
                limit=None,
            )

        return self.throttler.acquire_result(checkpoint_result, action.action_type)


@dataclass
class ThrottledAction:
    """Action wrapper that adds throttling.

    Delegates to the wrapped action but checks rate limits first.

    Attributes:
        action: The wrapped action.
        middleware: The throttling middleware.
    """

    action: BaseAction[Any]
    middleware: ThrottlingMiddleware

    @property
    def action_type(self) -> str:
        """Get the action type."""
        return self.action.action_type

    @property
    def name(self) -> str:
        """Get the action name."""
        return self.action.name

    @property
    def config(self) -> Any:
        """Get the action config."""
        return self.action.config

    @property
    def enabled(self) -> bool:
        """Check if action is enabled."""
        return self.action.enabled

    def should_run(self, result_status: str) -> bool:
        """Check if action should run."""
        return self.action.should_run(result_status)

    def execute(self, checkpoint_result: CheckpointResult) -> ActionResult:
        """Execute with throttle check.

        Args:
            checkpoint_result: The checkpoint result.

        Returns:
            Action result (may be throttled).
        """
        started_at = datetime.now()

        # Check throttle first
        try:
            throttle_result = self.middleware.acquire(checkpoint_result, self.action)

            if not throttle_result.allowed:
                # Throttled
                if self.middleware.log_throttled:
                    logger.info(
                        f"Notification throttled: "
                        f"{self.action.action_type} for "
                        f"{checkpoint_result.checkpoint_name} "
                        f"(retry after {throttle_result.retry_after:.1f}s)"
                    )

                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SKIPPED,
                    message=f"Throttled: {throttle_result.message}",
                    started_at=started_at,
                    completed_at=datetime.now(),
                    details={
                        "reason": "throttled",
                        "retry_after": throttle_result.retry_after,
                        "remaining": throttle_result.remaining,
                        "key": throttle_result.key.key,
                    },
                )

        except Exception as e:
            if not self.middleware.skip_on_error:
                raise

            logger.warning(f"Throttle check failed, proceeding: {e}")

        # Execute the actual action
        return self.action.execute(checkpoint_result)


def throttled(
    throttler: NotificationThrottler | None = None,
    per_minute: int | None = 10,
    per_hour: int | None = 100,
    per_day: int | None = 500,
    skip_on_error: bool = True,
) -> Callable[[F], F]:
    """Decorator for adding throttling to action methods.

    Can be applied to the _execute method of BaseAction subclasses.

    Args:
        throttler: Throttling service (creates default if None).
        per_minute: Per-minute limit (used if throttler is None).
        per_hour: Per-hour limit (used if throttler is None).
        per_day: Per-day limit (used if throttler is None).
        skip_on_error: Skip throttling on errors.

    Returns:
        Decorator function.

    Example:
        >>> class MyNotificationAction(BaseAction):
        ...     @throttled(per_minute=5, per_hour=50)
        ...     def _execute(self, checkpoint_result):
        ...         # Send notification
        ...         pass
    """

    def decorator(func: F) -> F:
        # Create or use provided throttler
        nonlocal throttler
        if throttler is None:
            throttler = (
                ThrottlerBuilder()
                .with_per_minute_limit(per_minute)
                .with_per_hour_limit(per_hour)
                .with_per_day_limit(per_day)
                .build()
            )

        @functools.wraps(func)
        def wrapper(self: Any, checkpoint_result: CheckpointResult) -> ActionResult:
            started_at = datetime.now()

            # Check throttle
            try:
                result = throttler.acquire_result(
                    checkpoint_result,
                    self.action_type,
                )

                if not result.allowed:
                    logger.debug(
                        f"Notification throttled: {self.action_type} "
                        f"(retry after {result.retry_after:.1f}s)"
                    )
                    return ActionResult(
                        action_name=self.name,
                        action_type=self.action_type,
                        status=ActionStatus.SKIPPED,
                        message=f"Throttled: {result.message}",
                        started_at=started_at,
                        completed_at=datetime.now(),
                        details={"retry_after": result.retry_after},
                    )

            except Exception as e:
                if not skip_on_error:
                    raise
                logger.warning(f"Throttle check failed: {e}")

            # Execute the actual method
            return func(self, checkpoint_result)

        return wrapper  # type: ignore

    return decorator


class ThrottlingMixin:
    """Mixin class for adding throttling support to actions.

    Add this mixin to BaseAction subclasses to enable throttling.

    Example:
        >>> class MyAction(ThrottlingMixin, BaseAction[MyConfig]):
        ...     throttle_per_minute = 10
        ...     throttle_per_hour = 100
        ...
        ...     def _execute(self, checkpoint_result):
        ...         if self.is_throttled(checkpoint_result):
        ...             return self._throttled_result()
        ...         # ... actual implementation
    """

    throttle_enabled: bool = True
    throttle_per_minute: int | None = 10
    throttle_per_hour: int | None = 100
    throttle_per_day: int | None = 500
    _throttler: NotificationThrottler | None = None

    def _get_throttler(self) -> NotificationThrottler:
        """Get or create throttler."""
        if self._throttler is None:
            self._throttler = (
                ThrottlerBuilder()
                .with_per_minute_limit(self.throttle_per_minute)
                .with_per_hour_limit(self.throttle_per_hour)
                .with_per_day_limit(self.throttle_per_day)
                .build()
            )
        return self._throttler

    def is_throttled(
        self,
        checkpoint_result: CheckpointResult,
    ) -> bool:
        """Check if this notification would be throttled.

        Args:
            checkpoint_result: The checkpoint result.

        Returns:
            True if this is throttled.
        """
        if not self.throttle_enabled:
            return False

        throttler = self._get_throttler()
        return throttler.is_throttled(
            getattr(self, "action_type", "unknown"),
            checkpoint_result.checkpoint_name,
        )

    def check_throttle(
        self,
        checkpoint_result: CheckpointResult,
    ) -> ThrottleResult:
        """Check throttle and get full result.

        Args:
            checkpoint_result: The checkpoint result.

        Returns:
            Full throttle result.
        """
        throttler = self._get_throttler()
        return throttler.check_result(
            checkpoint_result,
            getattr(self, "action_type", "unknown"),
        )

    def acquire_throttle(
        self,
        checkpoint_result: CheckpointResult,
    ) -> ThrottleResult:
        """Acquire throttle permit.

        Args:
            checkpoint_result: The checkpoint result.

        Returns:
            Throttle result.
        """
        if not self.throttle_enabled:
            from truthound.checkpoint.throttling.protocols import ThrottlingKey, TimeUnit

            key = ThrottlingKey.for_global(TimeUnit.MINUTE)
            return ThrottleResult.allowed_result(key=key, remaining=999, limit=None)

        throttler = self._get_throttler()
        return throttler.acquire_result(
            checkpoint_result,
            getattr(self, "action_type", "unknown"),
        )

    def _throttled_result(
        self,
        retry_after: float = 60.0,
    ) -> ActionResult:
        """Create a throttled action result.

        Args:
            retry_after: Seconds until retry is allowed.

        Returns:
            ActionResult with SKIPPED status.
        """
        return ActionResult(
            action_name=getattr(self, "name", "unknown"),
            action_type=getattr(self, "action_type", "unknown"),
            status=ActionStatus.SKIPPED,
            message=f"Throttled. Retry after {retry_after:.1f}s",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            details={"retry_after": retry_after, "reason": "throttled"},
        )


# Global middleware instance for convenience
_global_middleware: ThrottlingMiddleware | None = None


def get_global_middleware() -> ThrottlingMiddleware:
    """Get the global throttling middleware."""
    global _global_middleware
    if _global_middleware is None:
        _global_middleware = ThrottlingMiddleware()
    return _global_middleware


def set_global_middleware(middleware: ThrottlingMiddleware) -> None:
    """Set the global throttling middleware."""
    global _global_middleware
    _global_middleware = middleware


def configure_global_throttling(
    per_minute: int | None = 10,
    per_hour: int | None = 100,
    per_day: int | None = 500,
    burst_multiplier: float = 1.0,
    scope: RateLimitScope = RateLimitScope.GLOBAL,
    priority_bypass: bool = False,
    enabled: bool = True,
) -> ThrottlingMiddleware:
    """Configure global throttling settings.

    Args:
        per_minute: Maximum notifications per minute.
        per_hour: Maximum notifications per hour.
        per_day: Maximum notifications per day.
        burst_multiplier: Burst capacity multiplier.
        scope: Rate limit scope.
        priority_bypass: Enable priority bypass.
        enabled: Whether throttling is enabled.

    Returns:
        Configured middleware.
    """
    builder = (
        ThrottlerBuilder()
        .with_per_minute_limit(per_minute)
        .with_per_hour_limit(per_hour)
        .with_per_day_limit(per_day)
        .with_burst_allowance(burst_multiplier)
        .with_scope(scope)
        .enabled(enabled)
    )

    if priority_bypass:
        builder.with_priority_bypass()

    throttler = builder.build()

    middleware = ThrottlingMiddleware(
        throttler=throttler,
        enabled=enabled,
    )

    set_global_middleware(middleware)
    return middleware
