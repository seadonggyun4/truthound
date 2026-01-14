"""Deduplication Middleware for Actions.

This module provides middleware and decorators for integrating
deduplication with checkpoint actions.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from truthound.checkpoint.actions.base import ActionResult, ActionStatus
from truthound.checkpoint.deduplication.protocols import (
    DeduplicationResult,
    TimeWindow,
)
from truthound.checkpoint.deduplication.service import (
    DeduplicationConfig,
    DeduplicationPolicy,
    NotificationDeduplicator,
)
from truthound.checkpoint.deduplication.stores import InMemoryDeduplicationStore

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class DeduplicationMiddleware:
    """Middleware for transparent action deduplication.

    Wraps action execution to automatically check for duplicates
    before sending notifications.

    Attributes:
        deduplicator: The deduplication service.
        enabled: Whether middleware is active.
        skip_on_error: Skip deduplication on errors.
        log_suppressed: Log when notifications are suppressed.

    Example:
        >>> middleware = DeduplicationMiddleware(
        ...     deduplicator=NotificationDeduplicator(),
        ... )
        >>>
        >>> # Wrap an action
        >>> wrapped_action = middleware.wrap(slack_action)
        >>> result = wrapped_action.execute(checkpoint_result)
    """

    deduplicator: NotificationDeduplicator = field(
        default_factory=lambda: NotificationDeduplicator(
            store=InMemoryDeduplicationStore(),
            config=DeduplicationConfig(),
        )
    )
    enabled: bool = True
    skip_on_error: bool = True
    log_suppressed: bool = True

    def wrap(self, action: BaseAction[Any]) -> DeduplicatedAction:
        """Wrap an action with deduplication.

        Args:
            action: The action to wrap.

        Returns:
            Wrapped action with deduplication.
        """
        return DeduplicatedAction(
            action=action,
            middleware=self,
        )

    def check(
        self,
        checkpoint_result: CheckpointResult,
        action: BaseAction[Any],
    ) -> DeduplicationResult:
        """Check if an action should be deduplicated.

        Args:
            checkpoint_result: The checkpoint result.
            action: The action to check.

        Returns:
            Deduplication result.
        """
        if not self.enabled:
            from truthound.checkpoint.deduplication.protocols import (
                NotificationFingerprint,
            )

            return DeduplicationResult(
                is_duplicate=False,
                fingerprint=NotificationFingerprint.generate(
                    checkpoint_name=checkpoint_result.checkpoint_name,
                    action_type=action.action_type,
                ),
                message="Deduplication middleware disabled",
            )

        # Extract severity from result
        severity = self._extract_severity(checkpoint_result)

        return self.deduplicator.check(
            checkpoint_result=checkpoint_result,
            action_type=action.action_type,
            severity=severity,
        )

    def mark_sent(
        self,
        checkpoint_result: CheckpointResult,
        action: BaseAction[Any],
        result: DeduplicationResult,
    ) -> None:
        """Mark an action as sent after successful execution.

        Args:
            checkpoint_result: The checkpoint result.
            action: The executed action.
            result: The deduplication result.
        """
        if not self.enabled:
            return

        self.deduplicator.mark_sent(
            fingerprint=result.fingerprint,
            metadata={
                "action_name": action.name,
                "checkpoint_status": str(checkpoint_result.status),
            },
        )

    def _extract_severity(self, checkpoint_result: CheckpointResult) -> str:
        """Extract severity from checkpoint result."""
        if not checkpoint_result.validation_result:
            return "medium"

        severity_order = ["critical", "high", "medium", "low", "info"]
        highest = "info"

        for issue in checkpoint_result.validation_result.issues:
            issue_severity = getattr(issue, "severity", "medium")
            if isinstance(issue_severity, str):
                sev = issue_severity.lower()
            else:
                sev = str(issue_severity).lower()

            try:
                if severity_order.index(sev) < severity_order.index(highest):
                    highest = sev
            except ValueError:
                continue

        return highest


@dataclass
class DeduplicatedAction:
    """Action wrapper that adds deduplication.

    Delegates to the wrapped action but checks for duplicates first.

    Attributes:
        action: The wrapped action.
        middleware: The deduplication middleware.
    """

    action: BaseAction[Any]
    middleware: DeduplicationMiddleware

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
        """Execute with deduplication check.

        Args:
            checkpoint_result: The checkpoint result.

        Returns:
            Action result (may be skipped due to deduplication).
        """
        started_at = datetime.now()

        # Check for duplicates first
        try:
            dedup_result = self.middleware.check(checkpoint_result, self.action)

            if dedup_result.is_duplicate:
                # Suppressed due to deduplication
                if self.middleware.log_suppressed:
                    logger.info(
                        f"Notification suppressed (duplicate): "
                        f"{self.action.action_type} for "
                        f"{checkpoint_result.checkpoint_name} "
                        f"(suppressed {dedup_result.suppressed_count} times)"
                    )

                return ActionResult(
                    action_name=self.name,
                    action_type=self.action_type,
                    status=ActionStatus.SKIPPED,
                    message=f"Suppressed: {dedup_result.message}",
                    started_at=started_at,
                    completed_at=datetime.now(),
                    details={
                        "reason": "deduplication",
                        "suppressed_count": dedup_result.suppressed_count,
                        "fingerprint": dedup_result.fingerprint.key,
                    },
                )

        except Exception as e:
            if not self.middleware.skip_on_error:
                raise

            logger.warning(f"Deduplication check failed, proceeding: {e}")
            dedup_result = None

        # Execute the actual action
        result = self.action.execute(checkpoint_result)

        # Mark as sent if successful
        if result.status == ActionStatus.SUCCESS and dedup_result:
            try:
                self.middleware.mark_sent(
                    checkpoint_result, self.action, dedup_result
                )
            except Exception as e:
                logger.warning(f"Failed to mark notification as sent: {e}")

        return result


def deduplicated(
    deduplicator: NotificationDeduplicator | None = None,
    policy: DeduplicationPolicy = DeduplicationPolicy.SEVERITY,
    window: TimeWindow | None = None,
    skip_on_error: bool = True,
) -> Callable[[F], F]:
    """Decorator for adding deduplication to action methods.

    Can be applied to the _execute method of BaseAction subclasses.

    Args:
        deduplicator: Deduplication service (creates default if None).
        policy: Deduplication policy.
        window: Default time window.
        skip_on_error: Skip deduplication on errors.

    Returns:
        Decorator function.

    Example:
        >>> class MyNotificationAction(BaseAction):
        ...     @deduplicated(policy=DeduplicationPolicy.SEVERITY)
        ...     def _execute(self, checkpoint_result):
        ...         # Send notification
        ...         pass
    """

    def decorator(func: F) -> F:
        # Create or use provided deduplicator
        nonlocal deduplicator
        if deduplicator is None:
            from truthound.checkpoint.deduplication.service import (
                DeduplicatorBuilder,
            )

            deduplicator = (
                DeduplicatorBuilder()
                .with_policy(policy)
                .with_default_window(window or TimeWindow(minutes=5))
                .build()
            )

        @functools.wraps(func)
        def wrapper(self: Any, checkpoint_result: CheckpointResult) -> ActionResult:
            started_at = datetime.now()

            # Check for duplicates
            try:
                result = deduplicator.check(
                    checkpoint_result=checkpoint_result,
                    action_type=self.action_type,
                )

                if result.is_duplicate:
                    logger.debug(
                        f"Notification deduplicated: {self.action_type} "
                        f"(count: {result.suppressed_count})"
                    )
                    return ActionResult(
                        action_name=self.name,
                        action_type=self.action_type,
                        status=ActionStatus.SKIPPED,
                        message=f"Deduplicated: {result.message}",
                        started_at=started_at,
                        completed_at=datetime.now(),
                        details={"suppressed_count": result.suppressed_count},
                    )

            except Exception as e:
                if not skip_on_error:
                    raise
                logger.warning(f"Deduplication check failed: {e}")
                result = None

            # Execute the actual method
            action_result = func(self, checkpoint_result)

            # Mark as sent if successful
            if action_result.status == ActionStatus.SUCCESS and result:
                try:
                    deduplicator.mark_sent(result.fingerprint)
                except Exception as e:
                    logger.warning(f"Failed to mark sent: {e}")

            return action_result

        return wrapper  # type: ignore

    return decorator


class DeduplicationMixin:
    """Mixin class for adding deduplication support to actions.

    Add this mixin to BaseAction subclasses to enable deduplication.

    Example:
        >>> class MyAction(DeduplicationMixin, BaseAction[MyConfig]):
        ...     dedup_policy = DeduplicationPolicy.SEVERITY
        ...     dedup_window = TimeWindow(minutes=5)
        ...
        ...     def _execute(self, checkpoint_result):
        ...         if self.is_deduplicated(checkpoint_result):
        ...             return self._skipped_result("Deduplicated")
        ...         # ... actual implementation
    """

    dedup_enabled: bool = True
    dedup_policy: DeduplicationPolicy = DeduplicationPolicy.SEVERITY
    dedup_window: TimeWindow = field(default_factory=lambda: TimeWindow(minutes=5))
    _deduplicator: NotificationDeduplicator | None = None

    def _get_deduplicator(self) -> NotificationDeduplicator:
        """Get or create deduplicator."""
        if self._deduplicator is None:
            from truthound.checkpoint.deduplication.service import (
                DeduplicatorBuilder,
            )

            self._deduplicator = (
                DeduplicatorBuilder()
                .with_policy(self.dedup_policy)
                .with_default_window(self.dedup_window)
                .build()
            )
        return self._deduplicator

    def is_deduplicated(
        self,
        checkpoint_result: CheckpointResult,
        severity: str | None = None,
    ) -> bool:
        """Check if this notification would be a duplicate.

        Args:
            checkpoint_result: The checkpoint result.
            severity: Optional severity override.

        Returns:
            True if this is a duplicate.
        """
        if not self.dedup_enabled:
            return False

        deduplicator = self._get_deduplicator()
        return deduplicator.is_duplicate(
            checkpoint_result,
            getattr(self, "action_type", "unknown"),
            severity=severity,
        )

    def check_deduplication(
        self,
        checkpoint_result: CheckpointResult,
        severity: str | None = None,
    ) -> DeduplicationResult:
        """Check deduplication and get full result.

        Args:
            checkpoint_result: The checkpoint result.
            severity: Optional severity override.

        Returns:
            Full deduplication result.
        """
        deduplicator = self._get_deduplicator()
        return deduplicator.check(
            checkpoint_result,
            getattr(self, "action_type", "unknown"),
            severity=severity,
        )

    def mark_notification_sent(
        self,
        checkpoint_result: CheckpointResult,
        severity: str | None = None,
    ) -> None:
        """Mark notification as sent for deduplication.

        Args:
            checkpoint_result: The checkpoint result.
            severity: Optional severity override.
        """
        if not self.dedup_enabled:
            return

        deduplicator = self._get_deduplicator()
        result = deduplicator.check(
            checkpoint_result,
            getattr(self, "action_type", "unknown"),
            severity=severity,
        )
        deduplicator.mark_sent(result.fingerprint)

    def _skipped_result(self, message: str) -> ActionResult:
        """Create a skipped action result.

        Args:
            message: Skip reason message.

        Returns:
            ActionResult with SKIPPED status.
        """
        return ActionResult(
            action_name=getattr(self, "name", "unknown"),
            action_type=getattr(self, "action_type", "unknown"),
            status=ActionStatus.SKIPPED,
            message=message,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )


# Global middleware instance for convenience
_global_middleware: DeduplicationMiddleware | None = None


def get_global_middleware() -> DeduplicationMiddleware:
    """Get the global deduplication middleware."""
    global _global_middleware
    if _global_middleware is None:
        _global_middleware = DeduplicationMiddleware()
    return _global_middleware


def set_global_middleware(middleware: DeduplicationMiddleware) -> None:
    """Set the global deduplication middleware."""
    global _global_middleware
    _global_middleware = middleware


def configure_global_deduplication(
    policy: DeduplicationPolicy = DeduplicationPolicy.SEVERITY,
    window: TimeWindow | None = None,
    redis_url: str | None = None,
    enabled: bool = True,
) -> DeduplicationMiddleware:
    """Configure global deduplication settings.

    Args:
        policy: Deduplication policy.
        window: Default time window.
        redis_url: Redis URL for distributed deduplication.
        enabled: Whether deduplication is enabled.

    Returns:
        Configured middleware.
    """
    from truthound.checkpoint.deduplication.service import create_deduplicator

    deduplicator = create_deduplicator(
        policy=policy,
        window=window,
        redis_url=redis_url,
    )

    middleware = DeduplicationMiddleware(
        deduplicator=deduplicator,
        enabled=enabled,
    )

    set_global_middleware(middleware)
    return middleware
