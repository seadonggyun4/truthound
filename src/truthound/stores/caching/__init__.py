"""Result caching layer for storage backends.

This module provides caching capabilities to reduce latency and load
on underlying storage backends by caching frequently accessed results.

Features:
    - Multiple cache backends (in-memory, Redis, Memcached)
    - Configurable TTL and eviction policies
    - LRU, LFU, and TTL-based eviction
    - Write-through and write-behind caching
    - Cache statistics and monitoring

Example:
    >>> from truthound.stores.caching import (
    ...     CachedStore,
    ...     CacheConfig,
    ...     LRUCache,
    ... )
    >>>
    >>> cache = LRUCache(max_size=10000, ttl_seconds=3600)
    >>> cached_store = CachedStore(underlying_store, cache=cache)
    >>>
    >>> result = cached_store.get("run-123")  # First call hits storage
    >>> result = cached_store.get("run-123")  # Second call hits cache
"""

from truthound.stores.caching.base import (
    CacheConfig,
    CacheEntry,
    CacheMetrics,
    CacheStrategy,
    EvictionPolicy,
)
from truthound.stores.caching.backends import (
    InMemoryCache,
    LRUCache,
    LFUCache,
    TTLCache,
)
from truthound.stores.caching.store import (
    CachedStore,
    CacheMode,
)

__all__ = [
    # Base
    "CacheConfig",
    "CacheEntry",
    "CacheMetrics",
    "CacheStrategy",
    "EvictionPolicy",
    # Backends
    "InMemoryCache",
    "LRUCache",
    "LFUCache",
    "TTLCache",
    # Store
    "CachedStore",
    "CacheMode",
]
