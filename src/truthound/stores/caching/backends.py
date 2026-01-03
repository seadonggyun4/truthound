"""Cache backend implementations.

This module provides various cache backend implementations
including LRU, LFU, and TTL-based caches.
"""

from __future__ import annotations

import heapq
import random
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Generic, TypeVar

from truthound.stores.caching.base import (
    BaseCache,
    CacheConfig,
    CacheEntry,
    CacheMetrics,
    EvictionPolicy,
)


T = TypeVar("T")


class InMemoryCache(BaseCache[T]):
    """Simple in-memory cache with configurable eviction.

    A basic cache implementation that stores entries in memory
    with optional TTL and size limits.

    Example:
        >>> cache = InMemoryCache[dict](CacheConfig(max_size=1000))
        >>> cache.set("key1", {"data": "value"})
        >>> result = cache.get("key1")
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize in-memory cache."""
        super().__init__(config)
        self._cache: dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._total_memory_bytes = 0

    @property
    def size(self) -> int:
        """Get current number of entries."""
        with self._lock:
            return len(self._cache)

    def get(self, key: str) -> T | None:
        """Get value by key."""
        start_time = time.monotonic()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            if entry.is_expired:
                self._remove_entry(key)
                self._metrics.record_expiration()
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            entry.touch()
            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_hit(elapsed)
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set value with optional TTL."""
        start_time = time.monotonic()
        ttl = ttl_seconds if ttl_seconds is not None else self._config.ttl_seconds

        with self._lock:
            # Calculate size
            size_bytes = sys.getsizeof(value)

            # Create entry
            expires_at = None
            if ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=ttl)

            entry = CacheEntry[T](
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )

            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self._config.max_size:
                # Evict at least 1, but don't evict more than needed to make room
                evict_count = min(self._config.eviction_batch_size, max(1, len(self._cache) - self._config.max_size + 1))
                self._evict(evict_count)

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_memory_bytes -= old_entry.size_bytes

            # Add new entry
            self._cache[key] = entry
            self._total_memory_bytes += size_bytes

            # Update metrics
            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_set(elapsed)
            self._metrics.update_size(len(self._cache), self._total_memory_bytes)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._total_memory_bytes = 0
            self._metrics.update_size(0, 0)
            return count

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        entry = self._cache.pop(key, None)
        if entry:
            self._total_memory_bytes -= entry.size_bytes

    def _evict(self, count: int = 1) -> int:
        """Evict entries according to eviction policy."""
        if not self._cache:
            return 0

        evicted = 0
        policy = self._config.eviction_policy

        if policy == EvictionPolicy.LRU:
            # Sort by last accessed time
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].last_accessed,
            )
            for key, _ in entries[:count]:
                self._remove_entry(key)
                evicted += 1

        elif policy == EvictionPolicy.LFU:
            # Sort by access count
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count,
            )
            for key, _ in entries[:count]:
                self._remove_entry(key)
                evicted += 1

        elif policy == EvictionPolicy.TTL:
            # Remove expired first, then oldest
            now = datetime.now()
            expired = [
                k for k, v in self._cache.items()
                if v.expires_at and v.expires_at <= now
            ]
            for key in expired[:count]:
                self._remove_entry(key)
                evicted += 1

            if evicted < count:
                entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].created_at,
                )
                for key, _ in entries[: count - evicted]:
                    self._remove_entry(key)
                    evicted += 1

        elif policy == EvictionPolicy.FIFO:
            # Remove oldest created
            entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at,
            )
            for key, _ in entries[:count]:
                self._remove_entry(key)
                evicted += 1

        elif policy == EvictionPolicy.RANDOM:
            # Random eviction
            keys = list(self._cache.keys())
            for key in random.sample(keys, min(count, len(keys))):
                self._remove_entry(key)
                evicted += 1

        self._metrics.record_eviction(evicted)
        self._metrics.update_size(len(self._cache), self._total_memory_bytes)
        return evicted


class LRUCache(BaseCache[T]):
    """Least Recently Used cache implementation.

    Uses OrderedDict for O(1) access and eviction of least
    recently used entries.

    Example:
        >>> cache = LRUCache[dict](max_size=1000, ttl_seconds=3600)
        >>> cache.set("key1", {"data": "value"})
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600.0,
        config: CacheConfig | None = None,
    ) -> None:
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries.
            ttl_seconds: Default TTL in seconds.
            config: Optional full configuration.
        """
        if config is None:
            config = CacheConfig(
                max_size=max_size,
                ttl_seconds=ttl_seconds,
                eviction_policy=EvictionPolicy.LRU,
            )
        super().__init__(config)
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._total_memory_bytes = 0

    @property
    def size(self) -> int:
        """Get current number of entries."""
        with self._lock:
            return len(self._cache)

    def get(self, key: str) -> T | None:
        """Get value by key, moving to end (most recent)."""
        start_time = time.monotonic()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            if entry.is_expired:
                self._remove_entry(key)
                self._metrics.record_expiration()
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_hit(elapsed)
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set value with optional TTL."""
        start_time = time.monotonic()
        ttl = ttl_seconds if ttl_seconds is not None else self._config.ttl_seconds

        with self._lock:
            size_bytes = sys.getsizeof(value)

            expires_at = None
            if ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=ttl)

            entry = CacheEntry[T](
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_memory_bytes -= old_entry.size_bytes
                del self._cache[key]
            elif len(self._cache) >= self._config.max_size:
                self._evict(1)

            # Add to end (most recent)
            self._cache[key] = entry
            self._total_memory_bytes += size_bytes

            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_set(elapsed)
            self._metrics.update_size(len(self._cache), self._total_memory_bytes)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._total_memory_bytes = 0
            self._metrics.update_size(0, 0)
            return count

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        entry = self._cache.pop(key, None)
        if entry:
            self._total_memory_bytes -= entry.size_bytes

    def _evict(self, count: int = 1) -> int:
        """Evict least recently used entries."""
        evicted = 0
        while evicted < count and self._cache:
            # Pop from beginning (least recently used)
            key, entry = self._cache.popitem(last=False)
            self._total_memory_bytes -= entry.size_bytes
            evicted += 1

        self._metrics.record_eviction(evicted)
        self._metrics.update_size(len(self._cache), self._total_memory_bytes)
        return evicted


@dataclass(order=True)
class LFUEntry(Generic[T]):
    """Entry for LFU cache with frequency tracking."""

    frequency: int
    last_accessed: float = field(compare=False)
    key: str = field(compare=False)
    entry: CacheEntry[T] = field(compare=False)


class LFUCache(BaseCache[T]):
    """Least Frequently Used cache implementation.

    Uses a heap to efficiently track and evict least frequently
    accessed entries.

    Example:
        >>> cache = LFUCache[dict](max_size=1000)
        >>> cache.set("key1", {"data": "value"})
    """

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: float = 3600.0,
        config: CacheConfig | None = None,
    ) -> None:
        """Initialize LFU cache."""
        if config is None:
            config = CacheConfig(
                max_size=max_size,
                ttl_seconds=ttl_seconds,
                eviction_policy=EvictionPolicy.LFU,
            )
        super().__init__(config)
        self._cache: dict[str, LFUEntry[T]] = {}
        self._lock = threading.RLock()
        self._total_memory_bytes = 0

    @property
    def size(self) -> int:
        """Get current number of entries."""
        with self._lock:
            return len(self._cache)

    def get(self, key: str) -> T | None:
        """Get value by key, incrementing frequency."""
        start_time = time.monotonic()

        with self._lock:
            lfu_entry = self._cache.get(key)

            if lfu_entry is None:
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            entry = lfu_entry.entry
            if entry.is_expired:
                self._remove_entry(key)
                self._metrics.record_expiration()
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            # Increment frequency
            lfu_entry.frequency += 1
            lfu_entry.last_accessed = time.monotonic()
            entry.touch()

            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_hit(elapsed)
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set value with optional TTL."""
        start_time = time.monotonic()
        ttl = ttl_seconds if ttl_seconds is not None else self._config.ttl_seconds

        with self._lock:
            size_bytes = sys.getsizeof(value)

            expires_at = None
            if ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=ttl)

            entry = CacheEntry[T](
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_memory_bytes -= old_entry.entry.size_bytes
            elif len(self._cache) >= self._config.max_size:
                self._evict(1)

            # Add entry with frequency 1
            self._cache[key] = LFUEntry[T](
                frequency=1,
                last_accessed=time.monotonic(),
                key=key,
                entry=entry,
            )
            self._total_memory_bytes += size_bytes

            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_set(elapsed)
            self._metrics.update_size(len(self._cache), self._total_memory_bytes)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._total_memory_bytes = 0
            self._metrics.update_size(0, 0)
            return count

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        lfu_entry = self._cache.pop(key, None)
        if lfu_entry:
            self._total_memory_bytes -= lfu_entry.entry.size_bytes

    def _evict(self, count: int = 1) -> int:
        """Evict least frequently used entries."""
        if not self._cache:
            return 0

        # Build min-heap of entries by frequency
        heap = list(self._cache.values())
        heapq.heapify(heap)

        evicted = 0
        while evicted < count and heap:
            lfu_entry = heapq.heappop(heap)
            if lfu_entry.key in self._cache:
                self._remove_entry(lfu_entry.key)
                evicted += 1

        self._metrics.record_eviction(evicted)
        self._metrics.update_size(len(self._cache), self._total_memory_bytes)
        return evicted


class TTLCache(BaseCache[T]):
    """TTL-based cache with automatic expiration.

    Optimized for time-based expiration with background cleanup.

    Example:
        >>> cache = TTLCache[dict](ttl_seconds=300)  # 5 minutes
        >>> cache.set("key1", {"data": "value"})
    """

    def __init__(
        self,
        ttl_seconds: float = 3600.0,
        max_size: int = 10000,
        config: CacheConfig | None = None,
    ) -> None:
        """Initialize TTL cache."""
        if config is None:
            config = CacheConfig(
                max_size=max_size,
                ttl_seconds=ttl_seconds,
                eviction_policy=EvictionPolicy.TTL,
            )
        super().__init__(config)
        self._cache: dict[str, CacheEntry[T]] = {}
        self._expiry_times: dict[str, float] = {}  # key -> expiry timestamp
        self._lock = threading.RLock()
        self._total_memory_bytes = 0

    @property
    def size(self) -> int:
        """Get current number of entries."""
        with self._lock:
            return len(self._cache)

    def get(self, key: str) -> T | None:
        """Get value by key."""
        start_time = time.monotonic()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            if entry.is_expired:
                self._remove_entry(key)
                self._metrics.record_expiration()
                elapsed = (time.monotonic() - start_time) * 1000
                self._metrics.record_miss(elapsed)
                return None

            entry.touch()
            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_hit(elapsed)
            return entry.value

    def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set value with TTL."""
        start_time = time.monotonic()
        ttl = ttl_seconds if ttl_seconds is not None else self._config.ttl_seconds

        with self._lock:
            size_bytes = sys.getsizeof(value)

            expires_at = None
            expiry_timestamp = float("inf")
            if ttl > 0:
                expires_at = datetime.now() + timedelta(seconds=ttl)
                expiry_timestamp = time.monotonic() + ttl

            entry = CacheEntry[T](
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size_bytes,
            )

            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._total_memory_bytes -= old_entry.size_bytes
            elif len(self._cache) >= self._config.max_size:
                self._evict(1)

            self._cache[key] = entry
            self._expiry_times[key] = expiry_timestamp
            self._total_memory_bytes += size_bytes

            elapsed = (time.monotonic() - start_time) * 1000
            self._metrics.record_set(elapsed)
            self._metrics.update_size(len(self._cache), self._total_memory_bytes)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._expiry_times.clear()
            self._total_memory_bytes = 0
            self._metrics.update_size(0, 0)
            return count

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache."""
        entry = self._cache.pop(key, None)
        self._expiry_times.pop(key, None)
        if entry:
            self._total_memory_bytes -= entry.size_bytes

    def _evict(self, count: int = 1) -> int:
        """Evict expired entries first, then oldest."""
        if not self._cache:
            return 0

        evicted = 0
        now = time.monotonic()

        # First evict expired entries
        expired_keys = [
            k for k, exp in self._expiry_times.items()
            if exp <= now
        ]
        for key in expired_keys[:count]:
            self._remove_entry(key)
            evicted += 1

        # If not enough, evict soonest to expire
        if evicted < count:
            remaining = count - evicted
            sorted_keys = sorted(
                self._expiry_times.keys(),
                key=lambda k: self._expiry_times[k],
            )
            for key in sorted_keys[:remaining]:
                if key in self._cache:
                    self._remove_entry(key)
                    evicted += 1

        self._metrics.record_eviction(evicted)
        self._metrics.update_size(len(self._cache), self._total_memory_bytes)
        return evicted

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            now = time.monotonic()
            expired_keys = [
                k for k, exp in self._expiry_times.items()
                if exp <= now
            ]

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                self._metrics.record_expiration(len(expired_keys))
                self._metrics.update_size(len(self._cache), self._total_memory_bytes)

            return len(expired_keys)
