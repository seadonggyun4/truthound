"""Cached store wrapper implementation.

This module provides a wrapper around storage backends that
adds caching capabilities.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.stores.caching.base import (
    BaseCache,
    CacheConfig,
    CacheMetrics,
)
from truthound.stores.caching.backends import LRUCache

if TYPE_CHECKING:
    from truthound.stores.base import ValidationStore


T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


class CacheMode(str, Enum):
    """Cache operation modes."""

    READ_THROUGH = "read_through"  # Read from cache, fallback to store
    WRITE_THROUGH = "write_through"  # Write to both cache and store
    WRITE_BEHIND = "write_behind"  # Write to cache, async write to store
    CACHE_ASIDE = "cache_aside"  # Application manages cache


class CachedStore(Generic[T, ConfigT]):
    """Store wrapper with caching support.

    Wraps an underlying store to provide caching capabilities,
    reducing latency and load on the storage backend.

    Example:
        >>> from truthound.stores import FileSystemStore
        >>> from truthound.stores.caching import CachedStore, LRUCache
        >>>
        >>> inner_store = FileSystemStore()
        >>> cache = LRUCache[dict](max_size=10000, ttl_seconds=3600)
        >>> cached = CachedStore(inner_store, cache=cache)
        >>>
        >>> result = cached.get("run-123")  # First call hits store
        >>> result = cached.get("run-123")  # Second call hits cache
    """

    def __init__(
        self,
        store: "ValidationStore[ConfigT]",
        cache: BaseCache[T] | None = None,
        mode: CacheMode = CacheMode.WRITE_THROUGH,
        cache_reads: bool = True,
        cache_writes: bool = True,
        cache_config: CacheConfig | None = None,
    ) -> None:
        """Initialize cached store.

        Args:
            store: Underlying store to wrap.
            cache: Cache backend to use.
            mode: Cache operation mode.
            cache_reads: Cache read results.
            cache_writes: Cache write results.
            cache_config: Configuration for default cache.
        """
        self._store = store
        self._cache = cache or LRUCache[T](
            config=cache_config or CacheConfig(),
        )
        self._mode = mode
        self._cache_reads = cache_reads
        self._cache_writes = cache_writes

    @property
    def store(self) -> "ValidationStore[ConfigT]":
        """Get underlying store."""
        return self._store

    @property
    def cache(self) -> BaseCache[T]:
        """Get cache backend."""
        return self._cache

    @property
    def mode(self) -> CacheMode:
        """Get cache mode."""
        return self._mode

    @property
    def metrics(self) -> CacheMetrics:
        """Get cache metrics."""
        return self._cache.metrics

    def _cache_key(self, item_id: str) -> str:
        """Generate cache key from item ID."""
        return f"store:{item_id}"

    def get(self, item_id: str) -> T | None:
        """Get an item, using cache.

        Args:
            item_id: Item identifier.

        Returns:
            The item or None if not found.
        """
        if self._cache_reads:
            cache_key = self._cache_key(item_id)
            cached = self._cache.get(cache_key)

            if cached is not None:
                return cached

        # Cache miss - get from store
        try:
            result = self._store.get(item_id)
        except Exception:
            return None

        # Cache the result
        if self._cache_reads and result is not None:
            self._cache.set(self._cache_key(item_id), result)

        return result

    def save(self, item: T) -> str:
        """Save an item, updating cache.

        Args:
            item: Item to save.

        Returns:
            Item ID.
        """
        # Save to store
        item_id = self._store.save(item)

        # Update cache
        if self._cache_writes:
            self._cache.set(self._cache_key(item_id), item)

        return item_id

    def delete(self, item_id: str) -> bool:
        """Delete an item, invalidating cache.

        Args:
            item_id: Item identifier.

        Returns:
            True if deleted.
        """
        # Invalidate cache first
        self._cache.delete(self._cache_key(item_id))

        # Delete from store
        return self._store.delete(item_id)

    def exists(self, item_id: str) -> bool:
        """Check if an item exists.

        Args:
            item_id: Item identifier.

        Returns:
            True if exists.
        """
        # Check cache first
        if self._cache.exists(self._cache_key(item_id)):
            return True

        # Check store
        return self._store.exists(item_id)

    def list_ids(self, **kwargs: Any) -> list[str]:
        """List item IDs.

        Note: This always goes to the store as listing is
        typically not cached.

        Returns:
            List of item IDs.
        """
        return self._store.list_ids(**kwargs)

    def get_many(self, item_ids: list[str]) -> dict[str, T]:
        """Get multiple items.

        Args:
            item_ids: List of item IDs.

        Returns:
            Dictionary of found items.
        """
        results = {}
        missing_ids = []

        # Check cache first
        if self._cache_reads:
            for item_id in item_ids:
                cache_key = self._cache_key(item_id)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    results[item_id] = cached
                else:
                    missing_ids.append(item_id)
        else:
            missing_ids = item_ids

        # Get missing from store
        for item_id in missing_ids:
            try:
                result = self._store.get(item_id)
                if result is not None:
                    results[item_id] = result
                    if self._cache_reads:
                        self._cache.set(self._cache_key(item_id), result)
            except Exception:
                pass

        return results

    def invalidate(self, item_id: str) -> bool:
        """Invalidate a cache entry without deleting from store.

        Args:
            item_id: Item identifier.

        Returns:
            True if invalidated.
        """
        return self._cache.delete(self._cache_key(item_id))

    def invalidate_all(self) -> int:
        """Invalidate all cache entries.

        Returns:
            Number of entries invalidated.
        """
        return self._cache.clear()

    def refresh(self, item_id: str) -> T | None:
        """Refresh cache entry from store.

        Args:
            item_id: Item identifier.

        Returns:
            The refreshed item or None.
        """
        # Get fresh from store
        try:
            result = self._store.get(item_id)
        except Exception:
            return None

        # Update cache
        if result is not None:
            self._cache.set(self._cache_key(item_id), result)

        return result

    def warm(self, item_ids: list[str]) -> int:
        """Warm cache with specific items.

        Args:
            item_ids: List of item IDs to cache.

        Returns:
            Number of items cached.
        """
        cached = 0
        for item_id in item_ids:
            try:
                result = self._store.get(item_id)
                if result is not None:
                    self._cache.set(self._cache_key(item_id), result)
                    cached += 1
            except Exception:
                pass
        return cached

    def get_stats(self) -> dict[str, Any]:
        """Get cached store statistics."""
        return {
            "mode": self._mode.value,
            "cache_reads": self._cache_reads,
            "cache_writes": self._cache_writes,
            "cache": self._cache.get_stats(),
        }

    def __enter__(self) -> "CachedStore[T, ConfigT]":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        pass
