"""Rate limiting module for Truthound.

This module provides a comprehensive rate limiting system with support for
multiple algorithms, storage backends, and policy-based configuration.

Features:
    - Multiple algorithms: Token Bucket, Sliding Window, Fixed Window, Leaky Bucket
    - Pluggable storage: Memory, Redis, Memcached
    - Policy-based configuration for tiered limits
    - Quota management for long-term limits
    - Decorators and middleware for easy integration
    - Metrics and observability

Architecture:
    The rate limiting system follows a layered design:

    RateLimiter
        │
        ├── Algorithm (Token Bucket, Sliding Window, etc.)
        │       │
        │       └── Storage (Memory, Redis, etc.)
        │
        ├── Policy Registry
        │       │
        │       └── Policies (Tier, Endpoint, IP, Custom)
        │
        └── Quota Manager

Usage:
    >>> from truthound.ratelimit import (
    ...     RateLimiter, RateLimitConfig,
    ...     rate_limit, RateLimitStrategy,
    ... )
    >>>
    >>> # Simple rate limiting
    >>> limiter = RateLimiter(
    ...     config=RateLimitConfig(
    ...         requests_per_second=10,
    ...         burst_size=20,
    ...     )
    ... )
    >>> result = limiter.acquire("user:123")
    >>> if result.allowed:
    ...     process_request()
    >>>
    >>> # Using decorator
    >>> @rate_limit(key=lambda user_id: f"user:{user_id}")
    ... def api_call(user_id: str):
    ...     return process(user_id)
    >>>
    >>> # Policy-based limits
    >>> from truthound.ratelimit import TierBasedPolicy, PolicyRegistry
    >>> registry = PolicyRegistry()
    >>> registry.register(TierBasedPolicy(
    ...     tier_configs={
    ...         "free": RateLimitConfig(requests_per_second=1),
    ...         "pro": RateLimitConfig(requests_per_second=10),
    ...     },
    ...     tier_extractor=lambda ctx: ctx.get("tier"),
    ... ))
"""

# Core types and configuration
from truthound.ratelimit.core import (
    # Enums
    RateLimitStrategy,
    RateLimitScope,
    RateLimitAction,
    # Result types
    RateLimitResult,
    TokenBucketState,
    WindowState,
    # Configuration
    RateLimitConfig,
    QuotaConfig,
    # Exceptions
    RateLimitError,
    RateLimitExceeded,
    QuotaExceeded,
    RateLimitConfigError,
    # Interfaces
    RateLimitAlgorithm,
    RateLimitStorage,
    KeyExtractor,
    # Key extractors
    GlobalKeyExtractor,
    AttributeKeyExtractor,
    CompositeKeyExtractor,
    CallableKeyExtractor,
    # Utilities
    current_time,
    calculate_retry_after,
)

# Algorithms
from truthound.ratelimit.algorithms import (
    TokenBucketAlgorithm,
    SlidingWindowAlgorithm,
    FixedWindowAlgorithm,
    LeakyBucketAlgorithm,
    create_algorithm,
)

# Storage backends
from truthound.ratelimit.storage import (
    MemoryStorage,
    RedisStorage,
    MemcachedStorage,
    DistributedStorage,
    create_storage,
)

# Policies
from truthound.ratelimit.policy import (
    # Types
    PolicyPriority,
    PolicyMatch,
    # Interface
    RateLimitPolicy,
    # Built-in policies
    DefaultPolicy,
    TierBasedPolicy,
    EndpointPolicy,
    IPBasedPolicy,
    CompositePolicy,
    ConditionalPolicy,
    # Quota management
    QuotaUsage,
    QuotaManager,
    # Registry
    PolicyRegistry,
    DynamicPolicyConfig,
)

# Main limiter
from truthound.ratelimit.limiter import (
    RateLimiter,
    RateLimiterRegistry,
    get_limiter,
    configure_rate_limit,
    rate_limit,
    rate_limit_async,
    RateLimitContext,
    AsyncRateLimitContext,
)

# Middleware
from truthound.ratelimit.middleware import (
    RateLimitMiddleware,
    ASGIRateLimitMiddleware,
    WSGIRateLimitMiddleware,
    create_flask_limiter,
    RetryHandler,
    RateLimitMetrics,
    MetricsMiddleware,
)

# Integration with other modules
from truthound.ratelimit.integration import (
    RateLimitedCheckpointConfig,
    RateLimitedCheckpointMixin,
    rate_limited_checkpoint,
    RateLimitObserver,
    InstrumentedRateLimiter,
    rate_limit_action,
)


__all__ = [
    # Enums
    "RateLimitStrategy",
    "RateLimitScope",
    "RateLimitAction",
    # Result types
    "RateLimitResult",
    "TokenBucketState",
    "WindowState",
    # Configuration
    "RateLimitConfig",
    "QuotaConfig",
    # Exceptions
    "RateLimitError",
    "RateLimitExceeded",
    "QuotaExceeded",
    "RateLimitConfigError",
    # Interfaces
    "RateLimitAlgorithm",
    "RateLimitStorage",
    "KeyExtractor",
    # Key extractors
    "GlobalKeyExtractor",
    "AttributeKeyExtractor",
    "CompositeKeyExtractor",
    "CallableKeyExtractor",
    # Utilities
    "current_time",
    "calculate_retry_after",
    # Algorithms
    "TokenBucketAlgorithm",
    "SlidingWindowAlgorithm",
    "FixedWindowAlgorithm",
    "LeakyBucketAlgorithm",
    "create_algorithm",
    # Storage
    "MemoryStorage",
    "RedisStorage",
    "MemcachedStorage",
    "DistributedStorage",
    "create_storage",
    # Policies
    "PolicyPriority",
    "PolicyMatch",
    "RateLimitPolicy",
    "DefaultPolicy",
    "TierBasedPolicy",
    "EndpointPolicy",
    "IPBasedPolicy",
    "CompositePolicy",
    "ConditionalPolicy",
    "QuotaUsage",
    "QuotaManager",
    "PolicyRegistry",
    "DynamicPolicyConfig",
    # Main limiter
    "RateLimiter",
    "RateLimiterRegistry",
    "get_limiter",
    "configure_rate_limit",
    "rate_limit",
    "rate_limit_async",
    "RateLimitContext",
    "AsyncRateLimitContext",
    # Middleware
    "RateLimitMiddleware",
    "ASGIRateLimitMiddleware",
    "WSGIRateLimitMiddleware",
    "create_flask_limiter",
    "RetryHandler",
    "RateLimitMetrics",
    "MetricsMiddleware",
    # Integration
    "RateLimitedCheckpointConfig",
    "RateLimitedCheckpointMixin",
    "rate_limited_checkpoint",
    "RateLimitObserver",
    "InstrumentedRateLimiter",
    "rate_limit_action",
]
