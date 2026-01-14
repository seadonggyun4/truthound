"""Notification Throttling System.

This module provides rate limiting and throttling for notification actions.
It supports Token Bucket algorithm with minute/hour/day rate limits.

Key Features:
- Multi-level rate limits (per minute, hour, day)
- Per-action and per-provider throttling
- Composable throttlers for complex policies
- Pluggable storage backends (InMemory, Redis)
- Builder pattern for flexible configuration
- Middleware integration with actions

Example:
    >>> from truthound.checkpoint.throttling import (
    ...     ThrottlerBuilder,
    ...     ThrottlingMiddleware,
    ... )
    >>>
    >>> # Create a multi-level throttler
    >>> throttler = (
    ...     ThrottlerBuilder()
    ...     .with_per_minute_limit(10)
    ...     .with_per_hour_limit(100)
    ...     .with_per_day_limit(500)
    ...     .with_burst_allowance(1.5)
    ...     .build()
    ... )
    >>>
    >>> # Wrap actions with throttling
    >>> middleware = ThrottlingMiddleware(throttler=throttler)
    >>> throttled_action = middleware.wrap(slack_action)
"""

from truthound.checkpoint.throttling.protocols import (
    RateLimit,
    RateLimitScope,
    ThrottleResult,
    ThrottleStatus,
    ThrottlerProtocol,
    ThrottlingConfig,
    ThrottlingKey,
    ThrottlingRecord,
    ThrottlingStats,
    TimeUnit,
)
from truthound.checkpoint.throttling.service import (
    NotificationThrottler,
    ThrottlerBuilder,
    create_throttler,
)
from truthound.checkpoint.throttling.throttlers import (
    CompositeThrottler,
    FixedWindowThrottler,
    NoOpThrottler,
    SlidingWindowThrottler,
    TokenBucketThrottler,
)
from truthound.checkpoint.throttling.middleware import (
    ThrottledAction,
    ThrottlingMiddleware,
    ThrottlingMixin,
    configure_global_throttling,
    get_global_middleware,
    set_global_middleware,
    throttled,
)
from truthound.checkpoint.throttling.stores import (
    InMemoryThrottlingStore,
    ThrottlingStore,
)

__all__ = [
    # Protocols & Types
    "RateLimit",
    "RateLimitScope",
    "ThrottleResult",
    "ThrottleStatus",
    "ThrottlerProtocol",
    "ThrottlingConfig",
    "ThrottlingKey",
    "ThrottlingRecord",
    "ThrottlingStats",
    "TimeUnit",
    # Service
    "NotificationThrottler",
    "ThrottlerBuilder",
    "create_throttler",
    # Throttlers
    "CompositeThrottler",
    "FixedWindowThrottler",
    "NoOpThrottler",
    "SlidingWindowThrottler",
    "TokenBucketThrottler",
    # Middleware
    "ThrottledAction",
    "ThrottlingMiddleware",
    "ThrottlingMixin",
    "configure_global_throttling",
    "get_global_middleware",
    "set_global_middleware",
    "throttled",
    # Stores
    "InMemoryThrottlingStore",
    "ThrottlingStore",
]
