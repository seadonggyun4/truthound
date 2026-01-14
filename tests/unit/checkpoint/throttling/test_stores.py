"""Tests for throttling store backends."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from truthound.checkpoint.throttling.protocols import (
    ThrottlingKey,
    ThrottlingRecord,
    TimeUnit,
)
from truthound.checkpoint.throttling.stores import (
    InMemoryThrottlingStore,
    RedisThrottlingStore,
)


class TestInMemoryThrottlingStore:
    """Tests for InMemoryThrottlingStore."""

    def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        store = InMemoryThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key, tokens=10.0)

        store.put(key.key, record)
        retrieved = store.get(key.key)

        assert retrieved is not None
        assert retrieved.tokens == 10.0

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent key."""
        store = InMemoryThrottlingStore()

        result = store.get("nonexistent")

        assert result is None

    def test_update_tokens(self) -> None:
        """Test token update."""
        store = InMemoryThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key, tokens=10.0)

        store.put(key.key, record)
        store.update_tokens(key.key, 5.0)
        retrieved = store.get(key.key)

        assert retrieved is not None
        assert retrieved.tokens == 5.0

    def test_increment_count(self) -> None:
        """Test count increment."""
        store = InMemoryThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key, count=0)

        store.put(key.key, record)
        store.increment_count(key.key, 5)
        retrieved = store.get(key.key)

        assert retrieved is not None
        assert retrieved.count == 5

    def test_record_allowed(self) -> None:
        """Test recording allowed request."""
        store = InMemoryThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key)

        store.put(key.key, record)
        store.record_allowed(key.key)
        retrieved = store.get(key.key)

        assert retrieved is not None
        assert retrieved.total_allowed == 1

    def test_record_throttled(self) -> None:
        """Test recording throttled request."""
        store = InMemoryThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key)

        store.put(key.key, record)
        store.record_throttled(key.key)
        retrieved = store.get(key.key)

        assert retrieved is not None
        assert retrieved.total_throttled == 1

    def test_delete(self) -> None:
        """Test deletion."""
        store = InMemoryThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key)

        store.put(key.key, record)
        assert store.get(key.key) is not None

        result = store.delete(key.key)
        assert result is True
        assert store.get(key.key) is None

    def test_delete_nonexistent(self) -> None:
        """Test deleting nonexistent key."""
        store = InMemoryThrottlingStore()

        result = store.delete("nonexistent")

        assert result is False

    def test_max_size_eviction(self) -> None:
        """Test LRU eviction when max size reached."""
        store = InMemoryThrottlingStore(max_size=3)

        # Add 3 items
        for i in range(3):
            key = ThrottlingKey.for_action(f"action_{i}", TimeUnit.MINUTE)
            store.put(key.key, ThrottlingRecord(key=key))

        # Access first to make it recently used
        first_key = ThrottlingKey.for_action("action_0", TimeUnit.MINUTE)
        store.get(first_key.key)

        # Add 4th item (should evict action_1, the oldest)
        new_key = ThrottlingKey.for_action("action_3", TimeUnit.MINUTE)
        store.put(new_key.key, ThrottlingRecord(key=new_key))

        # action_0 should still exist (was accessed)
        assert store.get(first_key.key) is not None

        # action_1 should be evicted (oldest)
        evicted_key = ThrottlingKey.for_action("action_1", TimeUnit.MINUTE)
        assert store.get(evicted_key.key) is None

        # action_3 should exist
        assert store.get(new_key.key) is not None

    def test_cleanup_expired(self) -> None:
        """Test cleanup of expired records."""
        store = InMemoryThrottlingStore(auto_cleanup_interval=9999)  # Disable auto
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key)

        store.put(key.key, record)

        # Immediate cleanup should not remove anything
        removed = store.cleanup_expired(max_age_seconds=1.0)
        assert removed == 0

        # Wait and cleanup
        time.sleep(0.1)
        removed = store.cleanup_expired(max_age_seconds=0.05)
        assert removed == 1

    def test_clear(self) -> None:
        """Test clearing all records."""
        store = InMemoryThrottlingStore()

        for i in range(5):
            key = ThrottlingKey.for_action(f"action_{i}", TimeUnit.MINUTE)
            store.put(key.key, ThrottlingRecord(key=key))

        assert len(store) == 5

        store.clear()

        assert len(store) == 0

    def test_keys_iterator(self) -> None:
        """Test keys iterator."""
        store = InMemoryThrottlingStore()

        for i in range(3):
            key = ThrottlingKey.for_action(f"action_{i}", TimeUnit.MINUTE)
            store.put(key.key, ThrottlingRecord(key=key))

        keys = list(store.keys())

        assert len(keys) == 3

    def test_contains(self) -> None:
        """Test contains check."""
        store = InMemoryThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key)

        assert key.key not in store

        store.put(key.key, record)

        assert key.key in store

    def test_get_stats(self) -> None:
        """Test statistics retrieval."""
        store = InMemoryThrottlingStore()

        for i in range(5):
            key = ThrottlingKey.for_action(f"action_{i}", TimeUnit.MINUTE)
            record = ThrottlingRecord(key=key)
            store.put(key.key, record)
            store.record_allowed(key.key)

        stats = store.get_stats()

        assert stats.buckets_active == 5
        assert stats.total_allowed == 5

    def test_to_dict(self) -> None:
        """Test dictionary export."""
        store = InMemoryThrottlingStore(max_size=100)
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        store.put(key.key, ThrottlingRecord(key=key, tokens=5.0))

        data = store.to_dict()

        assert data["max_size"] == 100
        assert data["current_size"] == 1
        assert "stats" in data
        assert "records" in data

    def test_thread_safety(self) -> None:
        """Test thread safety of operations."""
        store = InMemoryThrottlingStore(max_size=1000)
        errors: list[Exception] = []

        def worker(action_id: int) -> None:
            try:
                key = ThrottlingKey.for_action(f"action_{action_id}", TimeUnit.MINUTE)
                record = ThrottlingRecord(key=key, tokens=10.0)
                store.put(key.key, record)
                store.update_tokens(key.key, 5.0)
                store.increment_count(key.key)
                store.get(key.key)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(worker, range(100)))

        assert len(errors) == 0
        assert len(store) == 100

    def test_lru_order(self) -> None:
        """Test LRU ordering on get."""
        store = InMemoryThrottlingStore(max_size=3)

        # Add items in order
        keys = []
        for i in range(3):
            key = ThrottlingKey.for_action(f"action_{i}", TimeUnit.MINUTE)
            keys.append(key)
            store.put(key.key, ThrottlingRecord(key=key))

        # Access first item (moves to end)
        store.get(keys[0].key)

        # Add new item (should evict action_1, not action_0)
        new_key = ThrottlingKey.for_action("action_3", TimeUnit.MINUTE)
        store.put(new_key.key, ThrottlingRecord(key=new_key))

        # action_0 should exist (recently accessed)
        assert keys[0].key in store

        # action_1 should be evicted (oldest)
        assert keys[1].key not in store

        # action_2 should exist
        assert keys[2].key in store


class TestRedisThrottlingStore:
    """Tests for RedisThrottlingStore placeholder."""

    def test_initialization(self) -> None:
        """Test store initialization."""
        store = RedisThrottlingStore(
            redis_url="redis://localhost:6379",
            key_prefix="test:",
        )

        assert store.redis_url == "redis://localhost:6379"
        assert store.key_prefix == "test:"

    def test_get_raises_not_implemented(self) -> None:
        """Test get raises NotImplementedError."""
        store = RedisThrottlingStore()

        with pytest.raises(NotImplementedError):
            store.get("key")

    def test_put_raises_not_implemented(self) -> None:
        """Test put raises NotImplementedError."""
        store = RedisThrottlingStore()
        key = ThrottlingKey.for_global(TimeUnit.MINUTE)
        record = ThrottlingRecord(key=key)

        with pytest.raises(NotImplementedError):
            store.put(key.key, record)

    def test_get_stats_returns_empty(self) -> None:
        """Test get_stats returns empty stats."""
        store = RedisThrottlingStore()

        stats = store.get_stats()

        assert stats.total_checked == 0
        assert stats.buckets_active == 0
