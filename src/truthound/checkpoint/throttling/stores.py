"""Throttling Store Backends.

This module provides storage backends for throttling state:
- InMemoryThrottlingStore: Thread-safe in-memory storage
- RedisThrottlingStore: Distributed Redis-based storage (planned)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Protocol, runtime_checkable

from truthound.checkpoint.throttling.protocols import (
    RateLimit,
    ThrottlingKey,
    ThrottlingRecord,
    ThrottlingStats,
    TimeUnit,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class ThrottlingStore(Protocol):
    """Protocol for throttling state storage backends.

    Implementations must provide thread-safe storage for
    throttling records with automatic cleanup.
    """

    def get(self, key: str) -> ThrottlingRecord | None:
        """Get a throttling record by key.

        Args:
            key: The throttling key string.

        Returns:
            The record if found, None otherwise.
        """
        ...

    def put(
        self,
        key: str,
        record: ThrottlingRecord,
    ) -> ThrottlingRecord:
        """Store a throttling record.

        Args:
            key: The throttling key string.
            record: The record to store.

        Returns:
            The stored record.
        """
        ...

    def update_tokens(
        self,
        key: str,
        tokens: float,
    ) -> ThrottlingRecord | None:
        """Update token count for a record.

        Args:
            key: The throttling key string.
            tokens: New token count.

        Returns:
            The updated record if found, None otherwise.
        """
        ...

    def increment_count(
        self,
        key: str,
        amount: int = 1,
    ) -> ThrottlingRecord | None:
        """Increment request count for a record.

        Args:
            key: The throttling key string.
            amount: Amount to increment by.

        Returns:
            The updated record if found, None otherwise.
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete a record.

        Args:
            key: The throttling key string.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Remove records older than max age.

        Args:
            max_age_seconds: Maximum age in seconds.

        Returns:
            Number of records removed.
        """
        ...

    def get_stats(self) -> ThrottlingStats:
        """Get store statistics.

        Returns:
            Current statistics.
        """
        ...

    def clear(self) -> None:
        """Clear all records."""
        ...

    def keys(self) -> Iterator[str]:
        """Iterate over all keys.

        Returns:
            Iterator of key strings.
        """
        ...


@dataclass
class InMemoryThrottlingStore:
    """In-memory throttling store.

    Thread-safe storage for throttling state using OrderedDict
    for LRU-style eviction.

    Attributes:
        max_size: Maximum number of records to store.
        auto_cleanup_interval: Interval for automatic cleanup (seconds).

    Example:
        >>> store = InMemoryThrottlingStore(max_size=10000)
        >>> record = ThrottlingRecord(
        ...     key=ThrottlingKey.for_global(TimeUnit.MINUTE),
        ...     tokens=10.0,
        ... )
        >>> store.put(record.key.key, record)
    """

    max_size: int = 10000
    auto_cleanup_interval: float = 60.0
    _records: OrderedDict[str, ThrottlingRecord] = field(
        default_factory=OrderedDict,
        init=False,
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _last_cleanup: float = field(default_factory=time.time, init=False)
    _stats: ThrottlingStats = field(default_factory=ThrottlingStats, init=False)

    def get(self, key: str) -> ThrottlingRecord | None:
        """Get a throttling record by key."""
        with self._lock:
            self._maybe_cleanup()
            record = self._records.get(key)
            if record:
                # Move to end for LRU
                self._records.move_to_end(key)
            return record

    def put(
        self,
        key: str,
        record: ThrottlingRecord,
    ) -> ThrottlingRecord:
        """Store a throttling record."""
        with self._lock:
            self._maybe_cleanup()

            # Check size limit
            if len(self._records) >= self.max_size and key not in self._records:
                # Remove oldest
                self._records.popitem(last=False)

            self._records[key] = record
            self._records.move_to_end(key)

            # Update stats
            self._stats.buckets_active = len(self._records)
            if self._stats.oldest_bucket is None:
                self._stats.oldest_bucket = record.window_start
            self._stats.newest_bucket = record.last_updated

            return record

    def update_tokens(
        self,
        key: str,
        tokens: float,
    ) -> ThrottlingRecord | None:
        """Update token count for a record."""
        with self._lock:
            record = self._records.get(key)
            if record:
                record.tokens = tokens
                record.last_updated = datetime.now()
                self._records.move_to_end(key)
            return record

    def increment_count(
        self,
        key: str,
        amount: int = 1,
    ) -> ThrottlingRecord | None:
        """Increment request count for a record."""
        with self._lock:
            record = self._records.get(key)
            if record:
                record.count += amount
                record.last_updated = datetime.now()
                record.last_request_at = datetime.now()
                self._records.move_to_end(key)
            return record

    def record_allowed(self, key: str) -> ThrottlingRecord | None:
        """Record an allowed request."""
        with self._lock:
            record = self._records.get(key)
            if record:
                record.total_allowed += 1
                record.last_updated = datetime.now()
                record.last_request_at = datetime.now()
                self._stats.total_allowed += 1
            return record

    def record_throttled(self, key: str) -> ThrottlingRecord | None:
        """Record a throttled request."""
        with self._lock:
            record = self._records.get(key)
            if record:
                record.total_throttled += 1
                record.last_updated = datetime.now()
                self._stats.total_throttled += 1
            return record

    def delete(self, key: str) -> bool:
        """Delete a record."""
        with self._lock:
            if key in self._records:
                del self._records[key]
                self._stats.buckets_active = len(self._records)
                return True
            return False

    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Remove records older than max age."""
        with self._lock:
            now = datetime.now()
            keys_to_remove: list[str] = []

            for key, record in self._records.items():
                age = (now - record.last_updated).total_seconds()
                if age > max_age_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._records[key]

            self._stats.buckets_active = len(self._records)
            self._last_cleanup = time.time()

            logger.debug(f"Cleaned up {len(keys_to_remove)} expired throttling records")
            return len(keys_to_remove)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup >= self.auto_cleanup_interval:
            # Clean up records older than 1 day by default
            self.cleanup_expired(86400.0)

    def get_stats(self) -> ThrottlingStats:
        """Get store statistics."""
        with self._lock:
            self._stats.buckets_active = len(self._records)

            if self._records:
                records = list(self._records.values())
                self._stats.oldest_bucket = min(r.window_start for r in records)
                self._stats.newest_bucket = max(r.last_updated for r in records)

            return self._stats

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()
            self._stats = ThrottlingStats()

    def keys(self) -> Iterator[str]:
        """Iterate over all keys."""
        with self._lock:
            return iter(list(self._records.keys()))

    def __len__(self) -> int:
        """Get number of records."""
        with self._lock:
            return len(self._records)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            return key in self._records

    def to_dict(self) -> dict[str, Any]:
        """Export store state as dictionary."""
        with self._lock:
            return {
                "max_size": self.max_size,
                "current_size": len(self._records),
                "stats": self._stats.to_dict(),
                "records": {k: v.to_dict() for k, v in self._records.items()},
            }


class RedisThrottlingStore:
    """Redis-based throttling store (placeholder).

    Provides distributed throttling state storage using Redis.
    Supports atomic operations and automatic TTL-based expiration.

    Note:
        This is a placeholder for future implementation.
        The full implementation will use Redis sorted sets and
        Lua scripts for atomic rate limiting operations.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "truthound:throttle:",
        default_ttl_seconds: int = 86400,
    ):
        """Initialize Redis throttling store.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for Redis keys.
            default_ttl_seconds: Default TTL for records.
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl_seconds = default_ttl_seconds
        self._stats = ThrottlingStats()

        logger.info(
            f"RedisThrottlingStore initialized (placeholder) - URL: {redis_url}"
        )

    def get(self, key: str) -> ThrottlingRecord | None:
        """Get a throttling record by key (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )

    def put(
        self,
        key: str,
        record: ThrottlingRecord,
    ) -> ThrottlingRecord:
        """Store a throttling record (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )

    def update_tokens(
        self,
        key: str,
        tokens: float,
    ) -> ThrottlingRecord | None:
        """Update token count (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )

    def increment_count(
        self,
        key: str,
        amount: int = 1,
    ) -> ThrottlingRecord | None:
        """Increment request count (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )

    def delete(self, key: str) -> bool:
        """Delete a record (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )

    def cleanup_expired(self, max_age_seconds: float) -> int:
        """Remove expired records (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )

    def get_stats(self) -> ThrottlingStats:
        """Get store statistics (placeholder)."""
        return self._stats

    def clear(self) -> None:
        """Clear all records (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )

    def keys(self) -> Iterator[str]:
        """Iterate over all keys (placeholder)."""
        raise NotImplementedError(
            "RedisThrottlingStore is not yet implemented. "
            "Use InMemoryThrottlingStore for now."
        )
