"""Base classes and protocols for caching layer.

This module defines the abstract interfaces and data structures
for cache implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable


T = TypeVar("T")


class EvictionPolicy(str, Enum):
    """Cache eviction policies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    RANDOM = "random"  # Random eviction


@dataclass
class CacheConfig:
    """Configuration for cache.

    Attributes:
        max_size: Maximum number of items in cache.
        max_memory_mb: Maximum memory usage in MB.
        ttl_seconds: Default TTL for cache entries.
        eviction_policy: Eviction policy to use.
        eviction_batch_size: Items to evict at once.
        enable_statistics: Enable cache statistics.
        warm_on_startup: Warm cache on startup.
        background_refresh: Refresh entries in background.
        refresh_threshold_percent: Refresh when TTL is this % remaining.
    """

    max_size: int = 10000
    max_memory_mb: float = 100.0
    ttl_seconds: float = 3600.0  # 1 hour
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    eviction_batch_size: int = 100
    enable_statistics: bool = True
    warm_on_startup: bool = False
    background_refresh: bool = False
    refresh_threshold_percent: float = 20.0

    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_size <= 0:
            raise ValueError("max_size must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        if self.eviction_batch_size <= 0:
            raise ValueError("eviction_batch_size must be positive")
        if not 0 <= self.refresh_threshold_percent <= 100:
            raise ValueError("refresh_threshold_percent must be between 0 and 100")


@dataclass
class CacheEntry(Generic[T]):
    """A cache entry with metadata.

    Attributes:
        key: Cache key.
        value: Cached value.
        created_at: When the entry was created.
        expires_at: When the entry expires.
        access_count: Number of times accessed.
        last_accessed: Last access time.
        size_bytes: Estimated size in bytes.
    """

    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    @property
    def ttl_remaining_seconds(self) -> float | None:
        """Get remaining TTL in seconds."""
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, remaining)

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class CacheMetrics:
    """Metrics for cache operations.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        sets: Number of cache sets.
        evictions: Number of evictions.
        expirations: Number of expirations.
        size: Current cache size.
        memory_bytes: Current memory usage.
        hit_rate: Cache hit rate percentage.
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    memory_bytes: int = 0
    _total_get_time_ms: float = 0.0
    _total_set_time_ms: float = 0.0

    def record_hit(self, time_ms: float = 0.0) -> None:
        """Record a cache hit."""
        self.hits += 1
        self._total_get_time_ms += time_ms

    def record_miss(self, time_ms: float = 0.0) -> None:
        """Record a cache miss."""
        self.misses += 1
        self._total_get_time_ms += time_ms

    def record_set(self, time_ms: float = 0.0) -> None:
        """Record a cache set."""
        self.sets += 1
        self._total_set_time_ms += time_ms

    def record_eviction(self, count: int = 1) -> None:
        """Record evictions."""
        self.evictions += count

    def record_expiration(self, count: int = 1) -> None:
        """Record expirations."""
        self.expirations += count

    def update_size(self, size: int, memory_bytes: int) -> None:
        """Update size metrics."""
        self.size = size
        self.memory_bytes = memory_bytes

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def average_get_time_ms(self) -> float:
        """Get average get operation time."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self._total_get_time_ms / total

    @property
    def average_set_time_ms(self) -> float:
        """Get average set operation time."""
        if self.sets == 0:
            return 0.0
        return self._total_set_time_ms / self.sets

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "memory_bytes": self.memory_bytes,
            "hit_rate_percent": self.hit_rate,
            "average_get_time_ms": self.average_get_time_ms,
            "average_set_time_ms": self.average_set_time_ms,
        }


@runtime_checkable
class CacheStrategy(Protocol[T]):
    """Protocol for cache implementations."""

    def get(self, key: str) -> T | None:
        """Get value by key, returns None if not found."""
        ...

    def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set value with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a key, returns True if deleted."""
        ...

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...

    def clear(self) -> int:
        """Clear all entries, returns count cleared."""
        ...

    @property
    def size(self) -> int:
        """Get current number of entries."""
        ...

    @property
    def metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        ...


class BaseCache(ABC, Generic[T]):
    """Abstract base class for cache implementations."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize cache.

        Args:
            config: Cache configuration.
        """
        self._config = config or CacheConfig()
        self._config.validate()
        self._metrics = CacheMetrics()

    @property
    def config(self) -> CacheConfig:
        """Get cache configuration."""
        return self._config

    @property
    def metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        return self._metrics

    @property
    @abstractmethod
    def size(self) -> int:
        """Get current number of entries."""
        pass

    @abstractmethod
    def get(self, key: str) -> T | None:
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: T, ttl_seconds: float | None = None) -> None:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key."""
        pass

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    @abstractmethod
    def clear(self) -> int:
        """Clear all entries."""
        pass

    def get_many(self, keys: list[str]) -> dict[str, T]:
        """Get multiple values by keys.

        Args:
            keys: List of keys to get.

        Returns:
            Dictionary of found key-value pairs.
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def set_many(
        self, items: dict[str, T], ttl_seconds: float | None = None
    ) -> None:
        """Set multiple values.

        Args:
            items: Dictionary of key-value pairs.
            ttl_seconds: Optional TTL for all entries.
        """
        for key, value in items.items():
            self.set(key, value, ttl_seconds)

    def delete_many(self, keys: list[str]) -> int:
        """Delete multiple keys.

        Args:
            keys: List of keys to delete.

        Returns:
            Number of keys deleted.
        """
        deleted = 0
        for key in keys:
            if self.delete(key):
                deleted += 1
        return deleted

    @abstractmethod
    def _evict(self, count: int = 1) -> int:
        """Evict entries according to eviction policy.

        Args:
            count: Number of entries to evict.

        Returns:
            Number of entries actually evicted.
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "config": {
                "max_size": self._config.max_size,
                "ttl_seconds": self._config.ttl_seconds,
                "eviction_policy": self._config.eviction_policy.value,
            },
            "metrics": self._metrics.to_dict(),
        }
