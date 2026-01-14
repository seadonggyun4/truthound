"""Throttling Protocols and Core Types.

This module defines the core protocols and data types for the
notification throttling system.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class TimeUnit(str, Enum):
    """Time unit for rate limits."""

    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"

    def to_seconds(self) -> int:
        """Convert to seconds."""
        multipliers = {
            TimeUnit.SECOND: 1,
            TimeUnit.MINUTE: 60,
            TimeUnit.HOUR: 3600,
            TimeUnit.DAY: 86400,
        }
        return multipliers[self]

    def to_timedelta(self) -> timedelta:
        """Convert to timedelta."""
        return timedelta(seconds=self.to_seconds())


class RateLimitScope(str, Enum):
    """Scope of rate limiting."""

    GLOBAL = "global"  # All notifications combined
    PER_ACTION = "per_action"  # Per action type (slack, email, etc.)
    PER_CHECKPOINT = "per_checkpoint"  # Per checkpoint
    PER_ACTION_CHECKPOINT = "per_action_checkpoint"  # Per action + checkpoint
    PER_SEVERITY = "per_severity"  # Per severity level
    PER_DATA_ASSET = "per_data_asset"  # Per data asset
    CUSTOM = "custom"  # Custom key function


class ThrottleStatus(str, Enum):
    """Status of a throttle check."""

    ALLOWED = "allowed"  # Request allowed
    THROTTLED = "throttled"  # Request throttled (rate limit exceeded)
    QUEUED = "queued"  # Request queued for later
    BURST_ALLOWED = "burst_allowed"  # Allowed via burst capacity
    ERROR = "error"  # Error during check


@dataclass(frozen=True)
class RateLimit:
    """Rate limit configuration.

    Defines the maximum number of requests allowed within a time window.

    Attributes:
        limit: Maximum number of requests.
        time_unit: Time unit for the window.
        burst_multiplier: Multiplier for burst capacity (1.0 = no burst).

    Example:
        >>> limit = RateLimit(limit=100, time_unit=TimeUnit.HOUR)
        >>> limit.window_seconds
        3600
    """

    limit: int
    time_unit: TimeUnit
    burst_multiplier: float = 1.0

    @property
    def window_seconds(self) -> int:
        """Get window duration in seconds."""
        return self.time_unit.to_seconds()

    @property
    def burst_limit(self) -> int:
        """Get burst limit (limit * multiplier)."""
        return int(self.limit * self.burst_multiplier)

    @property
    def tokens_per_second(self) -> float:
        """Get token refill rate per second."""
        return self.limit / self.window_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "limit": self.limit,
            "time_unit": self.time_unit.value,
            "burst_multiplier": self.burst_multiplier,
            "window_seconds": self.window_seconds,
            "burst_limit": self.burst_limit,
        }

    @classmethod
    def per_minute(cls, limit: int, burst_multiplier: float = 1.0) -> RateLimit:
        """Create a per-minute rate limit."""
        return cls(limit=limit, time_unit=TimeUnit.MINUTE, burst_multiplier=burst_multiplier)

    @classmethod
    def per_hour(cls, limit: int, burst_multiplier: float = 1.0) -> RateLimit:
        """Create a per-hour rate limit."""
        return cls(limit=limit, time_unit=TimeUnit.HOUR, burst_multiplier=burst_multiplier)

    @classmethod
    def per_day(cls, limit: int, burst_multiplier: float = 1.0) -> RateLimit:
        """Create a per-day rate limit."""
        return cls(limit=limit, time_unit=TimeUnit.DAY, burst_multiplier=burst_multiplier)


@dataclass
class ThrottlingConfig:
    """Configuration for the throttling system.

    Attributes:
        per_minute_limit: Maximum notifications per minute.
        per_hour_limit: Maximum notifications per hour.
        per_day_limit: Maximum notifications per day.
        burst_multiplier: Burst capacity multiplier.
        scope: Rate limit scope.
        algorithm: Throttling algorithm (token_bucket, sliding_window, fixed_window).
        enabled: Whether throttling is enabled.
        custom_limits: Custom rate limits per action/checkpoint.
        severity_limits: Override limits per severity level.
        priority_bypass: Bypass throttling for high priority notifications.
        queue_on_throttle: Queue throttled notifications instead of dropping.
        max_queue_size: Maximum queue size when queueing enabled.

    Example:
        >>> config = ThrottlingConfig(
        ...     per_minute_limit=10,
        ...     per_hour_limit=100,
        ...     per_day_limit=500,
        ...     burst_multiplier=1.5,
        ... )
    """

    per_minute_limit: int | None = 10
    per_hour_limit: int | None = 100
    per_day_limit: int | None = 500
    burst_multiplier: float = 1.0
    scope: RateLimitScope = RateLimitScope.GLOBAL
    algorithm: str = "token_bucket"
    enabled: bool = True
    custom_limits: dict[str, list[RateLimit]] = field(default_factory=dict)
    severity_limits: dict[str, list[RateLimit]] = field(default_factory=dict)
    priority_bypass: bool = False
    priority_threshold: str = "critical"
    queue_on_throttle: bool = False
    max_queue_size: int = 1000

    def get_rate_limits(self) -> list[RateLimit]:
        """Get all configured rate limits."""
        limits: list[RateLimit] = []

        if self.per_minute_limit is not None:
            limits.append(
                RateLimit(
                    limit=self.per_minute_limit,
                    time_unit=TimeUnit.MINUTE,
                    burst_multiplier=self.burst_multiplier,
                )
            )

        if self.per_hour_limit is not None:
            limits.append(
                RateLimit(
                    limit=self.per_hour_limit,
                    time_unit=TimeUnit.HOUR,
                    burst_multiplier=self.burst_multiplier,
                )
            )

        if self.per_day_limit is not None:
            limits.append(
                RateLimit(
                    limit=self.per_day_limit,
                    time_unit=TimeUnit.DAY,
                    burst_multiplier=self.burst_multiplier,
                )
            )

        return limits

    def get_limits_for_action(self, action_type: str) -> list[RateLimit]:
        """Get rate limits for a specific action type."""
        if action_type in self.custom_limits:
            return self.custom_limits[action_type]
        return self.get_rate_limits()

    def get_limits_for_severity(self, severity: str) -> list[RateLimit]:
        """Get rate limits for a specific severity level."""
        if severity in self.severity_limits:
            return self.severity_limits[severity]
        return self.get_rate_limits()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "per_minute_limit": self.per_minute_limit,
            "per_hour_limit": self.per_hour_limit,
            "per_day_limit": self.per_day_limit,
            "burst_multiplier": self.burst_multiplier,
            "scope": self.scope.value,
            "algorithm": self.algorithm,
            "enabled": self.enabled,
            "priority_bypass": self.priority_bypass,
            "queue_on_throttle": self.queue_on_throttle,
        }


@dataclass
class ThrottlingKey:
    """Key identifying a throttling bucket.

    Attributes:
        scope: The throttling scope.
        action_type: Action type (for per-action scopes).
        checkpoint_name: Checkpoint name (for per-checkpoint scopes).
        severity: Severity level (for per-severity scopes).
        data_asset: Data asset name (for per-asset scopes).
        custom_key: Custom key value.
        time_unit: Time unit for this bucket.
    """

    scope: RateLimitScope
    action_type: str | None = None
    checkpoint_name: str | None = None
    severity: str | None = None
    data_asset: str | None = None
    custom_key: str | None = None
    time_unit: TimeUnit = TimeUnit.MINUTE

    @property
    def key(self) -> str:
        """Generate unique key string."""
        components: dict[str, Any] = {
            "scope": self.scope.value,
            "time_unit": self.time_unit.value,
        }

        if self.action_type:
            components["action_type"] = self.action_type
        if self.checkpoint_name:
            components["checkpoint_name"] = self.checkpoint_name
        if self.severity:
            components["severity"] = self.severity
        if self.data_asset:
            components["data_asset"] = self.data_asset
        if self.custom_key:
            components["custom_key"] = self.custom_key

        canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    @classmethod
    def for_global(cls, time_unit: TimeUnit) -> ThrottlingKey:
        """Create a global scope key."""
        return cls(scope=RateLimitScope.GLOBAL, time_unit=time_unit)

    @classmethod
    def for_action(cls, action_type: str, time_unit: TimeUnit) -> ThrottlingKey:
        """Create a per-action scope key."""
        return cls(
            scope=RateLimitScope.PER_ACTION,
            action_type=action_type,
            time_unit=time_unit,
        )

    @classmethod
    def for_checkpoint(cls, checkpoint_name: str, time_unit: TimeUnit) -> ThrottlingKey:
        """Create a per-checkpoint scope key."""
        return cls(
            scope=RateLimitScope.PER_CHECKPOINT,
            checkpoint_name=checkpoint_name,
            time_unit=time_unit,
        )

    @classmethod
    def for_action_checkpoint(
        cls, action_type: str, checkpoint_name: str, time_unit: TimeUnit
    ) -> ThrottlingKey:
        """Create a per-action-checkpoint scope key."""
        return cls(
            scope=RateLimitScope.PER_ACTION_CHECKPOINT,
            action_type=action_type,
            checkpoint_name=checkpoint_name,
            time_unit=time_unit,
        )

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ThrottlingKey):
            return self.key == other.key
        return False


@dataclass
class ThrottlingRecord:
    """Record tracking throttling state.

    Attributes:
        key: The throttling key.
        tokens: Current token count (for token bucket).
        count: Request count in current window (for window-based).
        window_start: Start of current window.
        last_updated: Last update timestamp.
        last_request_at: Last request timestamp.
        total_allowed: Total allowed requests.
        total_throttled: Total throttled requests.
    """

    key: ThrottlingKey
    tokens: float = 0.0
    count: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_request_at: datetime | None = None
    total_allowed: int = 0
    total_throttled: int = 0

    @property
    def total_requests(self) -> int:
        """Get total requests (allowed + throttled)."""
        return self.total_allowed + self.total_throttled

    @property
    def throttle_rate(self) -> float:
        """Get throttle rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.total_throttled / self.total_requests) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key.key,
            "tokens": self.tokens,
            "count": self.count,
            "window_start": self.window_start.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "last_request_at": self.last_request_at.isoformat() if self.last_request_at else None,
            "total_allowed": self.total_allowed,
            "total_throttled": self.total_throttled,
            "throttle_rate": round(self.throttle_rate, 2),
        }


@dataclass
class ThrottleResult:
    """Result of a throttle check.

    Attributes:
        status: Throttle status.
        key: Throttling key that was checked.
        allowed: Whether the request is allowed.
        retry_after: Seconds to wait before retrying.
        remaining: Remaining tokens/requests.
        limit: The rate limit that was checked.
        message: Human-readable message.
        metadata: Additional result metadata.
    """

    status: ThrottleStatus
    key: ThrottlingKey
    allowed: bool
    retry_after: float = 0.0
    remaining: int = 0
    limit: RateLimit | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allowed_result(
        cls,
        key: ThrottlingKey,
        remaining: int,
        limit: RateLimit | None = None,
        is_burst: bool = False,
    ) -> ThrottleResult:
        """Create an allowed result."""
        status = ThrottleStatus.BURST_ALLOWED if is_burst else ThrottleStatus.ALLOWED
        return cls(
            status=status,
            key=key,
            allowed=True,
            remaining=remaining,
            limit=limit,
            message="Request allowed",
            metadata={"is_burst": is_burst},
        )

    @classmethod
    def throttled_result(
        cls,
        key: ThrottlingKey,
        retry_after: float,
        limit: RateLimit | None = None,
    ) -> ThrottleResult:
        """Create a throttled result."""
        return cls(
            status=ThrottleStatus.THROTTLED,
            key=key,
            allowed=False,
            retry_after=retry_after,
            remaining=0,
            limit=limit,
            message=f"Rate limit exceeded. Retry after {retry_after:.1f}s",
        )

    @classmethod
    def error_result(
        cls,
        key: ThrottlingKey,
        error: str,
    ) -> ThrottleResult:
        """Create an error result."""
        return cls(
            status=ThrottleStatus.ERROR,
            key=key,
            allowed=False,
            message=f"Throttle check error: {error}",
            metadata={"error": error},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "key": self.key.key,
            "allowed": self.allowed,
            "retry_after": self.retry_after,
            "remaining": self.remaining,
            "limit": self.limit.to_dict() if self.limit else None,
            "message": self.message,
        }


@dataclass
class ThrottlingStats:
    """Statistics for the throttling system.

    Attributes:
        total_checked: Total requests checked.
        total_allowed: Total requests allowed.
        total_throttled: Total requests throttled.
        total_burst_allowed: Total requests allowed via burst.
        total_queued: Total requests queued.
        total_errors: Total errors during checks.
        buckets_active: Number of active throttling buckets.
        oldest_bucket: Timestamp of oldest bucket.
        newest_bucket: Timestamp of newest bucket.
    """

    total_checked: int = 0
    total_allowed: int = 0
    total_throttled: int = 0
    total_burst_allowed: int = 0
    total_queued: int = 0
    total_errors: int = 0
    buckets_active: int = 0
    oldest_bucket: datetime | None = None
    newest_bucket: datetime | None = None

    @property
    def throttle_rate(self) -> float:
        """Get throttle rate as percentage."""
        if self.total_checked == 0:
            return 0.0
        return (self.total_throttled / self.total_checked) * 100

    @property
    def allow_rate(self) -> float:
        """Get allow rate as percentage."""
        if self.total_checked == 0:
            return 100.0
        return ((self.total_allowed + self.total_burst_allowed) / self.total_checked) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_checked": self.total_checked,
            "total_allowed": self.total_allowed,
            "total_throttled": self.total_throttled,
            "total_burst_allowed": self.total_burst_allowed,
            "total_queued": self.total_queued,
            "total_errors": self.total_errors,
            "throttle_rate": round(self.throttle_rate, 2),
            "allow_rate": round(self.allow_rate, 2),
            "buckets_active": self.buckets_active,
            "oldest_bucket": self.oldest_bucket.isoformat() if self.oldest_bucket else None,
            "newest_bucket": self.newest_bucket.isoformat() if self.newest_bucket else None,
        }


@runtime_checkable
class ThrottlerProtocol(Protocol):
    """Protocol for throttler implementations.

    All throttlers must implement this protocol.
    """

    @property
    def name(self) -> str:
        """Get throttler name."""
        ...

    def check(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check if a request is allowed.

        Args:
            key: Throttling key.
            limit: Rate limit to check against.
            permits: Number of permits to acquire.

        Returns:
            ThrottleResult indicating if request is allowed.
        """
        ...

    def acquire(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire permits (check and consume).

        Args:
            key: Throttling key.
            limit: Rate limit to check against.
            permits: Number of permits to acquire.

        Returns:
            ThrottleResult indicating if request was allowed.
        """
        ...

    def get_stats(self) -> ThrottlingStats:
        """Get throttler statistics."""
        ...

    def reset(self, key: ThrottlingKey | None = None) -> None:
        """Reset throttler state.

        Args:
            key: Specific key to reset, or None for all.
        """
        ...


class BaseThrottler(ABC):
    """Abstract base class for throttler implementations."""

    def __init__(self, name: str):
        """Initialize base throttler.

        Args:
            name: Unique name for this throttler.
        """
        self._name = name
        self._stats = ThrottlingStats()

    @property
    def name(self) -> str:
        """Get throttler name."""
        return self._name

    @abstractmethod
    def check(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Check if a request is allowed without consuming."""
        pass

    @abstractmethod
    def acquire(
        self,
        key: ThrottlingKey,
        limit: RateLimit,
        permits: int = 1,
    ) -> ThrottleResult:
        """Acquire permits (check and consume)."""
        pass

    def get_stats(self) -> ThrottlingStats:
        """Get throttler statistics."""
        return self._stats

    @abstractmethod
    def reset(self, key: ThrottlingKey | None = None) -> None:
        """Reset throttler state."""
        pass

    def _update_stats(self, result: ThrottleResult) -> None:
        """Update statistics based on result."""
        self._stats.total_checked += 1

        if result.status == ThrottleStatus.ALLOWED:
            self._stats.total_allowed += 1
        elif result.status == ThrottleStatus.BURST_ALLOWED:
            self._stats.total_burst_allowed += 1
        elif result.status == ThrottleStatus.THROTTLED:
            self._stats.total_throttled += 1
        elif result.status == ThrottleStatus.QUEUED:
            self._stats.total_queued += 1
        elif result.status == ThrottleStatus.ERROR:
            self._stats.total_errors += 1
