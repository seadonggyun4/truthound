"""Tests for caching backends."""

from __future__ import annotations

import time
import pytest

from truthound.stores.caching.base import CacheConfig, EvictionPolicy
from truthound.stores.caching.backends import (
    InMemoryCache,
    LRUCache,
    LFUCache,
    TTLCache,
)


class TestInMemoryCache:
    """Tests for InMemoryCache."""

    def test_creation(self) -> None:
        """Test cache creation."""
        cache = InMemoryCache[dict]()
        assert cache.size == 0

    def test_set_get(self) -> None:
        """Test basic set and get."""
        cache = InMemoryCache[dict]()
        cache.set("key1", {"value": 1})
        result = cache.get("key1")
        assert result == {"value": 1}

    def test_get_miss(self) -> None:
        """Test cache miss."""
        cache = InMemoryCache[dict]()
        result = cache.get("nonexistent")
        assert result is None
        assert cache.metrics.misses == 1

    def test_delete(self) -> None:
        """Test delete."""
        cache = InMemoryCache[dict]()
        cache.set("key1", {"value": 1})
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("key1") is False

    def test_exists(self) -> None:
        """Test exists."""
        cache = InMemoryCache[dict]()
        cache.set("key1", {"value": 1})
        assert cache.exists("key1") is True
        assert cache.exists("key2") is False

    def test_clear(self) -> None:
        """Test clear."""
        cache = InMemoryCache[dict]()
        cache.set("key1", {"value": 1})
        cache.set("key2", {"value": 2})
        count = cache.clear()
        assert count == 2
        assert cache.size == 0

    def test_max_size_eviction(self) -> None:
        """Test eviction when max size reached."""
        config = CacheConfig(max_size=3, eviction_policy=EvictionPolicy.LRU)
        cache = InMemoryCache[int](config)

        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.set("key3", 3)
        cache.set("key4", 4)

        assert cache.size == 3
        assert cache.metrics.evictions >= 1

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        config = CacheConfig(ttl_seconds=0.1)
        cache = InMemoryCache[int](config)

        cache.set("key1", 1)
        assert cache.get("key1") == 1

        time.sleep(0.15)
        assert cache.get("key1") is None
        assert cache.metrics.expirations == 1

    def test_get_set_many(self) -> None:
        """Test get_many and set_many."""
        cache = InMemoryCache[int]()
        cache.set_many({"a": 1, "b": 2, "c": 3})

        result = cache.get_many(["a", "b", "d"])
        assert result == {"a": 1, "b": 2}

    def test_delete_many(self) -> None:
        """Test delete_many."""
        cache = InMemoryCache[int]()
        cache.set_many({"a": 1, "b": 2, "c": 3})

        deleted = cache.delete_many(["a", "b", "d"])
        assert deleted == 2
        assert cache.size == 1

    def test_metrics(self) -> None:
        """Test metrics collection."""
        cache = InMemoryCache[int]()

        cache.set("key1", 1)
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        metrics = cache.metrics
        assert metrics.hits == 1
        assert metrics.misses == 1
        assert metrics.sets == 1
        assert metrics.hit_rate == 50.0


class TestLRUCache:
    """Tests for LRUCache."""

    def test_creation(self) -> None:
        """Test cache creation."""
        cache = LRUCache[dict](max_size=100)
        assert cache.size == 0

    def test_set_get(self) -> None:
        """Test basic set and get."""
        cache = LRUCache[dict]()
        cache.set("key1", {"value": 1})
        result = cache.get("key1")
        assert result == {"value": 1}

    def test_lru_eviction(self) -> None:
        """Test LRU eviction order."""
        cache = LRUCache[int](max_size=3)

        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.set("key3", 3)

        # Access key1 to make it recently used
        cache.get("key1")

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", 4)

        assert cache.get("key1") == 1  # Still present
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == 3
        assert cache.get("key4") == 4

    def test_update_existing(self) -> None:
        """Test updating existing key."""
        cache = LRUCache[int](max_size=3)

        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.set("key3", 3)

        # Update key1
        cache.set("key1", 10)

        # Add key4, should evict key2
        cache.set("key4", 4)

        assert cache.get("key1") == 10
        assert cache.get("key2") is None

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration in LRU cache."""
        cache = LRUCache[int](ttl_seconds=0.1)

        cache.set("key1", 1)
        time.sleep(0.15)
        assert cache.get("key1") is None


class TestLFUCache:
    """Tests for LFUCache."""

    def test_creation(self) -> None:
        """Test cache creation."""
        cache = LFUCache[dict](max_size=100)
        assert cache.size == 0

    def test_set_get(self) -> None:
        """Test basic set and get."""
        cache = LFUCache[dict]()
        cache.set("key1", {"value": 1})
        result = cache.get("key1")
        assert result == {"value": 1}

    def test_lfu_eviction(self) -> None:
        """Test LFU eviction order."""
        cache = LFUCache[int](max_size=3)

        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.set("key3", 3)

        # Access key1 multiple times
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")

        # Access key2 once
        cache.get("key2")

        # key3 has least accesses (only set, no gets)

        # Add key4, should evict key3 (least frequently used)
        cache.set("key4", 4)

        assert cache.get("key1") == 1  # Most frequent
        assert cache.get("key2") == 2  # Second most
        assert cache.get("key3") is None  # Evicted
        assert cache.get("key4") == 4

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration in LFU cache."""
        cache = LFUCache[int](ttl_seconds=0.1)

        cache.set("key1", 1)
        time.sleep(0.15)
        assert cache.get("key1") is None


class TestTTLCache:
    """Tests for TTLCache."""

    def test_creation(self) -> None:
        """Test cache creation."""
        cache = TTLCache[dict](ttl_seconds=3600)
        assert cache.size == 0

    def test_set_get(self) -> None:
        """Test basic set and get."""
        cache = TTLCache[dict]()
        cache.set("key1", {"value": 1})
        result = cache.get("key1")
        assert result == {"value": 1}

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        cache = TTLCache[int](ttl_seconds=0.1)

        cache.set("key1", 1)
        assert cache.get("key1") == 1

        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_custom_ttl(self) -> None:
        """Test custom TTL per entry."""
        cache = TTLCache[int](ttl_seconds=10.0)

        cache.set("long", 1)  # Default 10 second TTL
        cache.set("short", 2, ttl_seconds=0.1)  # 0.1 second TTL

        time.sleep(0.15)

        assert cache.get("long") == 1  # Still valid
        assert cache.get("short") is None  # Expired

    def test_cleanup_expired(self) -> None:
        """Test manual cleanup of expired entries."""
        cache = TTLCache[int](ttl_seconds=0.1)

        cache.set("key1", 1)
        cache.set("key2", 2)
        cache.set("key3", 3)

        time.sleep(0.15)

        removed = cache.cleanup_expired()
        assert removed == 3
        assert cache.size == 0

    def test_eviction_prefers_expired(self) -> None:
        """Test that eviction prefers expired entries."""
        cache = TTLCache[int](ttl_seconds=0.05, max_size=3)

        cache.set("key1", 1)
        time.sleep(0.06)  # key1 expires

        cache.set("key2", 2)
        cache.set("key3", 3)
        cache.set("key4", 4)  # Triggers eviction

        # key1 should be evicted (expired)
        assert cache.get("key2") == 2
        assert cache.get("key3") == 3
        assert cache.get("key4") == 4
