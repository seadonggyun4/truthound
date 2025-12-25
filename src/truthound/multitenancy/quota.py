"""Quota tracking and enforcement for multi-tenancy.

This module provides resource usage tracking, quota enforcement,
and usage analytics for multi-tenant deployments.
"""

from __future__ import annotations

import threading
import time
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from truthound.multitenancy.core import (
    QuotaTracker,
    ResourceType,
    Tenant,
    TenantQuota,
    TenantQuotaExceededError,
)


# =============================================================================
# Usage Record
# =============================================================================


@dataclass
class UsageRecord:
    """Record of resource usage."""

    tenant_id: str
    resource_type: ResourceType
    amount: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageSummary:
    """Summary of resource usage for a time window."""

    tenant_id: str
    resource_type: ResourceType
    window: str  # "hour", "day", "month"
    window_start: datetime
    window_end: datetime
    total_usage: int
    limit: int
    remaining: int
    percentage_used: float

    @property
    def is_exceeded(self) -> bool:
        """Check if usage exceeds limit."""
        return self.total_usage >= self.limit

    @property
    def is_warning(self) -> bool:
        """Check if usage is at warning level (>80%)."""
        return self.percentage_used >= 0.8


# =============================================================================
# In-Memory Quota Tracker
# =============================================================================


class MemoryQuotaTracker(QuotaTracker):
    """In-memory quota tracking for testing and development.

    Thread-safe implementation using sliding windows.

    Example:
        >>> tracker = MemoryQuotaTracker()
        >>> quota = TenantQuota.for_tier(TenantTier.PROFESSIONAL)
        >>> if tracker.check_quota("tenant_1", ResourceType.VALIDATIONS, quota):
        ...     tracker.increment("tenant_1", ResourceType.VALIDATIONS)
        ...     run_validation()
    """

    def __init__(self) -> None:
        # Structure: {tenant_id: {resource_type: [(timestamp, amount)]}}
        self._usage: dict[str, dict[ResourceType, list[tuple[float, int]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self._lock = threading.RLock()

    def _get_window_seconds(self, window: str) -> int:
        """Get window duration in seconds."""
        windows = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
            "month": 86400 * 30,
        }
        return windows.get(window, 86400)

    def _clean_old_entries(
        self,
        entries: list[tuple[float, int]],
        window_seconds: int,
    ) -> list[tuple[float, int]]:
        """Remove entries older than the window."""
        cutoff = time.time() - window_seconds
        return [(ts, amt) for ts, amt in entries if ts > cutoff]

    def get_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        window: str = "day",
    ) -> int:
        """Get current usage for a resource."""
        with self._lock:
            window_seconds = self._get_window_seconds(window)
            entries = self._usage[tenant_id][resource_type]
            entries = self._clean_old_entries(entries, window_seconds)
            self._usage[tenant_id][resource_type] = entries
            return sum(amt for _, amt in entries)

    def increment(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: int = 1,
    ) -> int:
        """Increment usage counter."""
        with self._lock:
            self._usage[tenant_id][resource_type].append((time.time(), amount))
            return self.get_usage(tenant_id, resource_type)

    def check_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        quota: TenantQuota,
        required: int = 1,
    ) -> bool:
        """Check if quota allows the requested amount."""
        limit = quota.get_limit(resource_type)
        if limit == 0:
            return True  # No limit

        current = self.get_usage(tenant_id, resource_type)
        return current + required <= limit

    def reset(
        self,
        tenant_id: str,
        resource_type: ResourceType | None = None,
    ) -> None:
        """Reset usage counters."""
        with self._lock:
            if resource_type is not None:
                if tenant_id in self._usage:
                    self._usage[tenant_id][resource_type] = []
            else:
                if tenant_id in self._usage:
                    del self._usage[tenant_id]

    def get_summary(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        quota: TenantQuota,
        window: str = "day",
    ) -> UsageSummary:
        """Get usage summary for a resource."""
        usage = self.get_usage(tenant_id, resource_type, window)
        limit = quota.get_limit(resource_type)
        remaining = max(0, limit - usage)
        percentage = usage / limit if limit > 0 else 0.0

        window_seconds = self._get_window_seconds(window)
        now = datetime.now(timezone.utc)
        window_start = datetime.fromtimestamp(
            time.time() - window_seconds, tz=timezone.utc
        )

        return UsageSummary(
            tenant_id=tenant_id,
            resource_type=resource_type,
            window=window,
            window_start=window_start,
            window_end=now,
            total_usage=usage,
            limit=limit,
            remaining=remaining,
            percentage_used=percentage,
        )


# =============================================================================
# Redis-Based Quota Tracker
# =============================================================================


@dataclass
class RedisQuotaConfig:
    """Configuration for Redis-based quota tracking."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    key_prefix: str = "truthound:quota:"
    use_sliding_window: bool = True


class RedisQuotaTracker(QuotaTracker):
    """Redis-based quota tracking for production deployments.

    Uses Redis sorted sets for efficient sliding window calculations.

    Example:
        >>> tracker = RedisQuotaTracker(config=RedisQuotaConfig(
        ...     host="redis.example.com",
        ... ))
    """

    def __init__(self, config: RedisQuotaConfig | None = None) -> None:
        self._config = config or RedisQuotaConfig()
        self._client: Any = None

    def _get_client(self) -> Any:
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis

                self._client = redis.Redis(
                    host=self._config.host,
                    port=self._config.port,
                    db=self._config.db,
                    password=self._config.password or None,
                    decode_responses=True,
                )
            except ImportError:
                raise ImportError(
                    "redis package is required for RedisQuotaTracker. "
                    "Install with: pip install redis"
                )
        return self._client

    def _get_key(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        window: str = "day",
    ) -> str:
        """Generate Redis key for quota tracking."""
        return f"{self._config.key_prefix}{tenant_id}:{resource_type.value}:{window}"

    def _get_window_seconds(self, window: str) -> int:
        """Get window duration in seconds."""
        windows = {
            "minute": 60,
            "hour": 3600,
            "day": 86400,
            "month": 86400 * 30,
        }
        return windows.get(window, 86400)

    def get_usage(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        window: str = "day",
    ) -> int:
        """Get current usage for a resource."""
        client = self._get_client()
        key = self._get_key(tenant_id, resource_type, window)

        if self._config.use_sliding_window:
            # Clean old entries and count
            window_seconds = self._get_window_seconds(window)
            cutoff = time.time() - window_seconds
            client.zremrangebyscore(key, "-inf", cutoff)
            return client.zcard(key)
        else:
            # Simple counter
            result = client.get(key)
            return int(result) if result else 0

    def increment(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        amount: int = 1,
    ) -> int:
        """Increment usage counter."""
        client = self._get_client()
        now = time.time()

        # Add to all relevant windows
        for window in ["minute", "hour", "day"]:
            key = self._get_key(tenant_id, resource_type, window)
            window_seconds = self._get_window_seconds(window)

            if self._config.use_sliding_window:
                # Use sorted set for sliding window
                pipeline = client.pipeline()
                for i in range(amount):
                    # Unique member for each increment
                    member = f"{now}:{i}"
                    pipeline.zadd(key, {member: now})
                pipeline.expire(key, window_seconds)
                pipeline.execute()
            else:
                # Simple increment
                pipeline = client.pipeline()
                pipeline.incrby(key, amount)
                pipeline.expire(key, window_seconds)
                pipeline.execute()

        return self.get_usage(tenant_id, resource_type)

    def check_quota(
        self,
        tenant_id: str,
        resource_type: ResourceType,
        quota: TenantQuota,
        required: int = 1,
    ) -> bool:
        """Check if quota allows the requested amount."""
        limit = quota.get_limit(resource_type)
        if limit == 0:
            return True

        current = self.get_usage(tenant_id, resource_type)
        return current + required <= limit

    def reset(
        self,
        tenant_id: str,
        resource_type: ResourceType | None = None,
    ) -> None:
        """Reset usage counters."""
        client = self._get_client()

        if resource_type is not None:
            for window in ["minute", "hour", "day"]:
                key = self._get_key(tenant_id, resource_type, window)
                client.delete(key)
        else:
            # Delete all keys for tenant
            pattern = f"{self._config.key_prefix}{tenant_id}:*"
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)


# =============================================================================
# Quota Enforcement
# =============================================================================


class QuotaEnforcer:
    """Enforces quota limits for tenant operations.

    Provides a convenient interface for checking and tracking quota
    usage with automatic enforcement.

    Example:
        >>> enforcer = QuotaEnforcer(tracker=MemoryQuotaTracker())
        >>> with enforcer.acquire(tenant, ResourceType.VALIDATIONS):
        ...     run_validation()  # Automatically tracked
    """

    def __init__(
        self,
        tracker: QuotaTracker,
        raise_on_exceeded: bool = True,
    ) -> None:
        self._tracker = tracker
        self._raise_on_exceeded = raise_on_exceeded

    def check(
        self,
        tenant: Tenant,
        resource_type: ResourceType,
        required: int = 1,
    ) -> bool:
        """Check if quota allows the operation."""
        return self._tracker.check_quota(
            tenant.id,
            resource_type,
            tenant.quota,
            required,
        )

    def require(
        self,
        tenant: Tenant,
        resource_type: ResourceType,
        required: int = 1,
    ) -> None:
        """Require quota availability, raising if exceeded."""
        if not self.check(tenant, resource_type, required):
            current = self._tracker.get_usage(tenant.id, resource_type)
            limit = tenant.quota.get_limit(resource_type)
            raise TenantQuotaExceededError(
                f"Quota exceeded for {resource_type.value}: {current}/{limit}",
                tenant_id=tenant.id,
                resource_type=resource_type,
                limit=limit,
                current=current,
            )

    def track(
        self,
        tenant: Tenant,
        resource_type: ResourceType,
        amount: int = 1,
    ) -> int:
        """Track resource usage."""
        return self._tracker.increment(tenant.id, resource_type, amount)

    def acquire(
        self,
        tenant: Tenant,
        resource_type: ResourceType,
        amount: int = 1,
    ) -> "QuotaContext":
        """Context manager for quota-tracked operations.

        Example:
            >>> with enforcer.acquire(tenant, ResourceType.VALIDATIONS):
            ...     run_validation()
        """
        return QuotaContext(self, tenant, resource_type, amount)

    def get_summary(
        self,
        tenant: Tenant,
        resource_type: ResourceType,
        window: str = "day",
    ) -> UsageSummary:
        """Get usage summary."""
        if isinstance(self._tracker, MemoryQuotaTracker):
            return self._tracker.get_summary(
                tenant.id, resource_type, tenant.quota, window
            )

        # Generic fallback
        usage = self._tracker.get_usage(tenant.id, resource_type, window)
        limit = tenant.quota.get_limit(resource_type)
        return UsageSummary(
            tenant_id=tenant.id,
            resource_type=resource_type,
            window=window,
            window_start=datetime.now(timezone.utc),
            window_end=datetime.now(timezone.utc),
            total_usage=usage,
            limit=limit,
            remaining=max(0, limit - usage),
            percentage_used=usage / limit if limit > 0 else 0.0,
        )

    def get_all_summaries(
        self,
        tenant: Tenant,
        window: str = "day",
    ) -> dict[ResourceType, UsageSummary]:
        """Get usage summaries for all resource types."""
        return {
            resource_type: self.get_summary(tenant, resource_type, window)
            for resource_type in ResourceType
        }


class QuotaContext:
    """Context manager for quota-tracked operations."""

    def __init__(
        self,
        enforcer: QuotaEnforcer,
        tenant: Tenant,
        resource_type: ResourceType,
        amount: int = 1,
    ) -> None:
        self._enforcer = enforcer
        self._tenant = tenant
        self._resource_type = resource_type
        self._amount = amount
        self._tracked = False

    def __enter__(self) -> "QuotaContext":
        self._enforcer.require(self._tenant, self._resource_type, self._amount)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Track usage on successful completion
        if exc_type is None and not self._tracked:
            self._enforcer.track(self._tenant, self._resource_type, self._amount)
            self._tracked = True

    def track_now(self) -> None:
        """Manually track usage before context exit."""
        if not self._tracked:
            self._enforcer.track(self._tenant, self._resource_type, self._amount)
            self._tracked = True


# =============================================================================
# Factory Function
# =============================================================================


def create_quota_tracker(
    backend: str = "memory",
    **kwargs: Any,
) -> QuotaTracker:
    """Create a quota tracker.

    Args:
        backend: Tracker backend ("memory", "redis")
        **kwargs: Backend-specific configuration

    Returns:
        Configured QuotaTracker instance.

    Example:
        >>> tracker = create_quota_tracker("memory")
        >>> tracker = create_quota_tracker("redis", host="redis.example.com")
    """
    if backend == "memory":
        return MemoryQuotaTracker()
    elif backend == "redis":
        config = RedisQuotaConfig(**kwargs)
        return RedisQuotaTracker(config=config)
    else:
        raise ValueError(f"Unknown quota tracker backend: {backend}")
