"""Notification Throttling Service.

This module provides the main NotificationThrottler service and
ThrottlerBuilder for flexible configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from truthound.checkpoint.throttling.protocols import (
    RateLimit,
    RateLimitScope,
    ThrottleResult,
    ThrottleStatus,
    ThrottlerProtocol,
    ThrottlingConfig,
    ThrottlingKey,
    ThrottlingStats,
    TimeUnit,
)
from truthound.checkpoint.throttling.stores import InMemoryThrottlingStore
from truthound.checkpoint.throttling.throttlers import (
    CompositeThrottler,
    FixedWindowThrottler,
    NoOpThrottler,
    SlidingWindowThrottler,
    TokenBucketThrottler,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult

logger = logging.getLogger(__name__)


# Severity priority for bypass checks
SEVERITY_PRIORITY = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}


@dataclass
class NotificationThrottler:
    """Main throttling service for notifications.

    Provides high-level throttling functionality with support for:
    - Multiple rate limits (per-minute, per-hour, per-day)
    - Different scopes (global, per-action, per-checkpoint)
    - Severity-based overrides
    - Priority bypass for critical notifications

    Attributes:
        config: Throttling configuration.
        throttler: The underlying throttler implementation.

    Example:
        >>> throttler = NotificationThrottler(
        ...     config=ThrottlingConfig(
        ...         per_minute_limit=10,
        ...         per_hour_limit=100,
        ...         per_day_limit=500,
        ...     )
        ... )
        >>>
        >>> result = throttler.check("slack", "data_quality_check")
        >>> if result.allowed:
        ...     send_notification()
    """

    config: ThrottlingConfig = field(default_factory=ThrottlingConfig)
    throttler: ThrottlerProtocol | None = None
    _store: InMemoryThrottlingStore = field(
        default_factory=InMemoryThrottlingStore,
        init=False,
    )

    def __post_init__(self) -> None:
        """Initialize throttler after dataclass init."""
        if self.throttler is None:
            self.throttler = self._create_throttler()

    def _create_throttler(self) -> ThrottlerProtocol:
        """Create throttler based on config."""
        if not self.config.enabled:
            return NoOpThrottler()

        limits = self.config.get_rate_limits()
        if not limits:
            return NoOpThrottler()

        # Use composite throttler for multiple limits
        if len(limits) > 1:
            composite = CompositeThrottler(
                name="notification_throttler",
                algorithm=self.config.algorithm,
            )
            for limit in limits:
                composite.add_limit(limit)
            return composite

        # Single limit - use the configured algorithm
        if self.config.algorithm == "sliding_window":
            return SlidingWindowThrottler("notification_throttler")
        elif self.config.algorithm == "fixed_window":
            return FixedWindowThrottler("notification_throttler")
        else:
            return TokenBucketThrottler("notification_throttler")

    def check(
        self,
        action_type: str,
        checkpoint_name: str | None = None,
        *,
        severity: str | None = None,
        data_asset: str | None = None,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check if a notification is allowed.

        Does not consume permits - use for pre-flight checks.

        Args:
            action_type: Type of notification action.
            checkpoint_name: Name of the checkpoint.
            severity: Severity level of the notification.
            data_asset: Data asset being validated.
            permits: Number of permits to check.

        Returns:
            ThrottleResult indicating if notification is allowed.
        """
        if not self.config.enabled:
            key = self._create_key(action_type, checkpoint_name, severity, data_asset)
            return ThrottleResult.allowed_result(
                key=key,
                remaining=self.config.per_minute_limit or 999,
                limit=None,
            )

        # Check priority bypass
        if self._should_bypass(severity):
            key = self._create_key(action_type, checkpoint_name, severity, data_asset)
            result = ThrottleResult.allowed_result(
                key=key,
                remaining=self.config.per_minute_limit or 999,
                limit=None,
            )
            result.message = "Bypassed due to priority"
            result.metadata["bypassed"] = True
            return result

        return self._check_limits(
            action_type=action_type,
            checkpoint_name=checkpoint_name,
            severity=severity,
            data_asset=data_asset,
            permits=permits,
            consume=False,
        )

    def acquire(
        self,
        action_type: str,
        checkpoint_name: str | None = None,
        *,
        severity: str | None = None,
        data_asset: str | None = None,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire permits for a notification.

        Checks and consumes permits if allowed.

        Args:
            action_type: Type of notification action.
            checkpoint_name: Name of the checkpoint.
            severity: Severity level of the notification.
            data_asset: Data asset being validated.
            permits: Number of permits to acquire.

        Returns:
            ThrottleResult indicating if notification is allowed.
        """
        if not self.config.enabled:
            key = self._create_key(action_type, checkpoint_name, severity, data_asset)
            return ThrottleResult.allowed_result(
                key=key,
                remaining=self.config.per_minute_limit or 999,
                limit=None,
            )

        # Check priority bypass
        if self._should_bypass(severity):
            key = self._create_key(action_type, checkpoint_name, severity, data_asset)
            result = ThrottleResult.allowed_result(
                key=key,
                remaining=self.config.per_minute_limit or 999,
                limit=None,
            )
            result.message = "Bypassed due to priority"
            result.metadata["bypassed"] = True
            logger.debug(
                f"Throttle bypassed for {action_type}: priority={severity}"
            )
            return result

        return self._check_limits(
            action_type=action_type,
            checkpoint_name=checkpoint_name,
            severity=severity,
            data_asset=data_asset,
            permits=permits,
            consume=True,
        )

    def check_result(
        self,
        checkpoint_result: CheckpointResult,
        action_type: str,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check throttle for a checkpoint result.

        Args:
            checkpoint_result: The checkpoint result.
            action_type: Type of notification action.
            permits: Number of permits to check.

        Returns:
            ThrottleResult indicating if notification is allowed.
        """
        severity = self._extract_severity(checkpoint_result)
        data_asset = getattr(checkpoint_result, "data_asset", None)

        return self.check(
            action_type=action_type,
            checkpoint_name=checkpoint_result.checkpoint_name,
            severity=severity,
            data_asset=data_asset,
            permits=permits,
        )

    def acquire_result(
        self,
        checkpoint_result: CheckpointResult,
        action_type: str,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire throttle permits for a checkpoint result.

        Args:
            checkpoint_result: The checkpoint result.
            action_type: Type of notification action.
            permits: Number of permits to acquire.

        Returns:
            ThrottleResult indicating if notification is allowed.
        """
        severity = self._extract_severity(checkpoint_result)
        data_asset = getattr(checkpoint_result, "data_asset", None)

        return self.acquire(
            action_type=action_type,
            checkpoint_name=checkpoint_result.checkpoint_name,
            severity=severity,
            data_asset=data_asset,
            permits=permits,
        )

    def is_throttled(
        self,
        action_type: str,
        checkpoint_name: str | None = None,
        severity: str | None = None,
    ) -> bool:
        """Check if a notification would be throttled.

        Convenience method that returns bool instead of ThrottleResult.

        Args:
            action_type: Type of notification action.
            checkpoint_name: Name of the checkpoint.
            severity: Severity level.

        Returns:
            True if throttled, False if allowed.
        """
        result = self.check(
            action_type=action_type,
            checkpoint_name=checkpoint_name,
            severity=severity,
        )
        return not result.allowed

    def _check_limits(
        self,
        action_type: str,
        checkpoint_name: str | None,
        severity: str | None,
        data_asset: str | None,
        permits: int,
        consume: bool,
    ) -> ThrottleResult:
        """Internal method to check all applicable limits."""
        key = self._create_key(action_type, checkpoint_name, severity, data_asset)

        # Get limits for this action/severity
        limits = self._get_applicable_limits(action_type, severity)

        if not limits:
            return ThrottleResult.allowed_result(
                key=key,
                remaining=999,
                limit=None,
            )

        # If using composite throttler with multiple limits
        if isinstance(self.throttler, CompositeThrottler):
            if consume:
                return self.throttler.acquire(key, permits=permits)
            else:
                return self.throttler.check(key, permits=permits)

        # Single limit
        limit = limits[0]
        if consume:
            return self.throttler.acquire(key, limit, permits)
        else:
            return self.throttler.check(key, limit, permits)

    def _create_key(
        self,
        action_type: str,
        checkpoint_name: str | None,
        severity: str | None,
        data_asset: str | None,
    ) -> ThrottlingKey:
        """Create throttling key based on scope."""
        scope = self.config.scope

        if scope == RateLimitScope.GLOBAL:
            return ThrottlingKey.for_global(TimeUnit.MINUTE)
        elif scope == RateLimitScope.PER_ACTION:
            return ThrottlingKey.for_action(action_type, TimeUnit.MINUTE)
        elif scope == RateLimitScope.PER_CHECKPOINT:
            return ThrottlingKey.for_checkpoint(
                checkpoint_name or "unknown",
                TimeUnit.MINUTE,
            )
        elif scope == RateLimitScope.PER_ACTION_CHECKPOINT:
            return ThrottlingKey.for_action_checkpoint(
                action_type,
                checkpoint_name or "unknown",
                TimeUnit.MINUTE,
            )
        elif scope == RateLimitScope.PER_SEVERITY:
            return ThrottlingKey(
                scope=RateLimitScope.PER_SEVERITY,
                severity=severity,
                time_unit=TimeUnit.MINUTE,
            )
        elif scope == RateLimitScope.PER_DATA_ASSET:
            return ThrottlingKey(
                scope=RateLimitScope.PER_DATA_ASSET,
                data_asset=data_asset,
                time_unit=TimeUnit.MINUTE,
            )
        else:
            return ThrottlingKey.for_global(TimeUnit.MINUTE)

    def _get_applicable_limits(
        self,
        action_type: str,
        severity: str | None,
    ) -> list[RateLimit]:
        """Get rate limits applicable for this action/severity."""
        # Check for action-specific limits
        if action_type in self.config.custom_limits:
            return self.config.custom_limits[action_type]

        # Check for severity-specific limits
        if severity and severity in self.config.severity_limits:
            return self.config.severity_limits[severity]

        # Use default limits
        return self.config.get_rate_limits()

    def _should_bypass(self, severity: str | None) -> bool:
        """Check if this severity should bypass throttling."""
        if not self.config.priority_bypass:
            return False

        if not severity:
            return False

        threshold_priority = SEVERITY_PRIORITY.get(
            self.config.priority_threshold, 0
        )
        current_priority = SEVERITY_PRIORITY.get(severity.lower(), 999)

        return current_priority <= threshold_priority

    def _extract_severity(self, checkpoint_result: CheckpointResult) -> str:
        """Extract highest severity from checkpoint result."""
        if not checkpoint_result.validation_result:
            return "medium"

        highest = "info"
        for issue in checkpoint_result.validation_result.issues:
            issue_severity = getattr(issue, "severity", "medium")
            if isinstance(issue_severity, str):
                sev = issue_severity.lower()
            else:
                sev = str(issue_severity).lower()

            current_priority = SEVERITY_PRIORITY.get(sev, 999)
            highest_priority = SEVERITY_PRIORITY.get(highest, 999)

            if current_priority < highest_priority:
                highest = sev

        return highest

    def get_stats(self) -> ThrottlingStats:
        """Get throttler statistics."""
        return self.throttler.get_stats()

    def reset(self, action_type: str | None = None) -> None:
        """Reset throttler state.

        Args:
            action_type: Reset for specific action, or all if None.
        """
        if action_type:
            key = ThrottlingKey.for_action(action_type, TimeUnit.MINUTE)
            self.throttler.reset(key)
        else:
            self.throttler.reset()


class ThrottlerBuilder:
    """Builder for creating NotificationThrottler instances.

    Provides fluent interface for configuring throttling.

    Example:
        >>> throttler = (
        ...     ThrottlerBuilder()
        ...     .with_per_minute_limit(10)
        ...     .with_per_hour_limit(100)
        ...     .with_per_day_limit(500)
        ...     .with_burst_allowance(1.5)
        ...     .with_algorithm("token_bucket")
        ...     .with_scope(RateLimitScope.PER_ACTION)
        ...     .with_priority_bypass("critical")
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._per_minute: int | None = 10
        self._per_hour: int | None = 100
        self._per_day: int | None = 500
        self._burst_multiplier: float = 1.0
        self._scope: RateLimitScope = RateLimitScope.GLOBAL
        self._algorithm: str = "token_bucket"
        self._enabled: bool = True
        self._custom_limits: dict[str, list[RateLimit]] = {}
        self._severity_limits: dict[str, list[RateLimit]] = {}
        self._priority_bypass: bool = False
        self._priority_threshold: str = "critical"
        self._queue_on_throttle: bool = False
        self._max_queue_size: int = 1000

    def with_per_minute_limit(self, limit: int | None) -> ThrottlerBuilder:
        """Set per-minute rate limit.

        Args:
            limit: Maximum notifications per minute, or None to disable.

        Returns:
            Self for chaining.
        """
        self._per_minute = limit
        return self

    def with_per_hour_limit(self, limit: int | None) -> ThrottlerBuilder:
        """Set per-hour rate limit.

        Args:
            limit: Maximum notifications per hour, or None to disable.

        Returns:
            Self for chaining.
        """
        self._per_hour = limit
        return self

    def with_per_day_limit(self, limit: int | None) -> ThrottlerBuilder:
        """Set per-day rate limit.

        Args:
            limit: Maximum notifications per day, or None to disable.

        Returns:
            Self for chaining.
        """
        self._per_day = limit
        return self

    def with_burst_allowance(self, multiplier: float) -> ThrottlerBuilder:
        """Set burst capacity multiplier.

        Args:
            multiplier: Multiplier for burst capacity (e.g., 1.5 = 50% extra).

        Returns:
            Self for chaining.
        """
        self._burst_multiplier = multiplier
        return self

    def with_scope(self, scope: RateLimitScope) -> ThrottlerBuilder:
        """Set rate limit scope.

        Args:
            scope: Scope for rate limiting.

        Returns:
            Self for chaining.
        """
        self._scope = scope
        return self

    def with_algorithm(self, algorithm: str) -> ThrottlerBuilder:
        """Set throttling algorithm.

        Args:
            algorithm: Algorithm name (token_bucket, sliding_window, fixed_window).

        Returns:
            Self for chaining.
        """
        if algorithm not in ("token_bucket", "sliding_window", "fixed_window"):
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self._algorithm = algorithm
        return self

    def with_priority_bypass(
        self,
        threshold: str = "critical",
    ) -> ThrottlerBuilder:
        """Enable priority bypass for high-severity notifications.

        Args:
            threshold: Severity threshold for bypass (critical, high, etc.).

        Returns:
            Self for chaining.
        """
        self._priority_bypass = True
        self._priority_threshold = threshold
        return self

    def without_priority_bypass(self) -> ThrottlerBuilder:
        """Disable priority bypass.

        Returns:
            Self for chaining.
        """
        self._priority_bypass = False
        return self

    def with_action_limit(
        self,
        action_type: str,
        per_minute: int | None = None,
        per_hour: int | None = None,
        per_day: int | None = None,
    ) -> ThrottlerBuilder:
        """Set custom limits for a specific action type.

        Args:
            action_type: Action type (slack, email, etc.).
            per_minute: Per-minute limit.
            per_hour: Per-hour limit.
            per_day: Per-day limit.

        Returns:
            Self for chaining.
        """
        limits: list[RateLimit] = []
        if per_minute is not None:
            limits.append(RateLimit.per_minute(per_minute, self._burst_multiplier))
        if per_hour is not None:
            limits.append(RateLimit.per_hour(per_hour, self._burst_multiplier))
        if per_day is not None:
            limits.append(RateLimit.per_day(per_day, self._burst_multiplier))

        if limits:
            self._custom_limits[action_type] = limits
        return self

    def with_severity_limit(
        self,
        severity: str,
        per_minute: int | None = None,
        per_hour: int | None = None,
        per_day: int | None = None,
    ) -> ThrottlerBuilder:
        """Set custom limits for a specific severity level.

        Args:
            severity: Severity level (critical, high, medium, low, info).
            per_minute: Per-minute limit.
            per_hour: Per-hour limit.
            per_day: Per-day limit.

        Returns:
            Self for chaining.
        """
        limits: list[RateLimit] = []
        if per_minute is not None:
            limits.append(RateLimit.per_minute(per_minute, self._burst_multiplier))
        if per_hour is not None:
            limits.append(RateLimit.per_hour(per_hour, self._burst_multiplier))
        if per_day is not None:
            limits.append(RateLimit.per_day(per_day, self._burst_multiplier))

        if limits:
            self._severity_limits[severity] = limits
        return self

    def with_queueing(self, max_size: int = 1000) -> ThrottlerBuilder:
        """Enable queueing for throttled notifications.

        Args:
            max_size: Maximum queue size.

        Returns:
            Self for chaining.
        """
        self._queue_on_throttle = True
        self._max_queue_size = max_size
        return self

    def enabled(self, value: bool = True) -> ThrottlerBuilder:
        """Enable or disable throttling.

        Args:
            value: Whether throttling is enabled.

        Returns:
            Self for chaining.
        """
        self._enabled = value
        return self

    def disabled(self) -> ThrottlerBuilder:
        """Disable throttling.

        Returns:
            Self for chaining.
        """
        self._enabled = False
        return self

    def build(self) -> NotificationThrottler:
        """Build the NotificationThrottler instance.

        Returns:
            Configured NotificationThrottler.
        """
        config = ThrottlingConfig(
            per_minute_limit=self._per_minute,
            per_hour_limit=self._per_hour,
            per_day_limit=self._per_day,
            burst_multiplier=self._burst_multiplier,
            scope=self._scope,
            algorithm=self._algorithm,
            enabled=self._enabled,
            custom_limits=self._custom_limits,
            severity_limits=self._severity_limits,
            priority_bypass=self._priority_bypass,
            priority_threshold=self._priority_threshold,
            queue_on_throttle=self._queue_on_throttle,
            max_queue_size=self._max_queue_size,
        )

        return NotificationThrottler(config=config)


def create_throttler(
    per_minute: int | None = 10,
    per_hour: int | None = 100,
    per_day: int | None = 500,
    burst_multiplier: float = 1.0,
    scope: RateLimitScope = RateLimitScope.GLOBAL,
    algorithm: str = "token_bucket",
    enabled: bool = True,
) -> NotificationThrottler:
    """Create a throttler with common configuration.

    Convenience function for quick throttler creation.

    Args:
        per_minute: Maximum notifications per minute.
        per_hour: Maximum notifications per hour.
        per_day: Maximum notifications per day.
        burst_multiplier: Burst capacity multiplier.
        scope: Rate limit scope.
        algorithm: Throttling algorithm.
        enabled: Whether throttling is enabled.

    Returns:
        Configured NotificationThrottler.

    Example:
        >>> throttler = create_throttler(
        ...     per_minute=10,
        ...     per_hour=100,
        ...     priority_bypass=True,
        ... )
    """
    return (
        ThrottlerBuilder()
        .with_per_minute_limit(per_minute)
        .with_per_hour_limit(per_hour)
        .with_per_day_limit(per_day)
        .with_burst_allowance(burst_multiplier)
        .with_scope(scope)
        .with_algorithm(algorithm)
        .enabled(enabled)
        .build()
    )
