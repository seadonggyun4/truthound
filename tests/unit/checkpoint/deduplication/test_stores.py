"""Tests for deduplication stores."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from truthound.checkpoint.deduplication.protocols import (
    NotificationFingerprint,
    TimeWindow,
)
from truthound.checkpoint.deduplication.stores import (
    InMemoryDeduplicationStore,
)


class TestInMemoryDeduplicationStore:
    """Tests for InMemoryDeduplicationStore."""

    def test_put_and_get(self) -> None:
        """Test basic put and get operations."""
        store = InMemoryDeduplicationStore()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)

        record = store.put(fp, window)
        assert record.fingerprint.key == fp.key
        assert record.count == 1

        retrieved = store.get(fp.key)
        assert retrieved is not None
        assert retrieved.fingerprint.key == fp.key

    def test_get_nonexistent(self) -> None:
        """Test get with nonexistent key."""
        store = InMemoryDeduplicationStore()
        assert store.get("nonexistent") is None

    def test_get_expired(self) -> None:
        """Test that expired records return None."""
        store = InMemoryDeduplicationStore()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        # Very short window
        window = TimeWindow(seconds=1)

        store.put(fp, window)
        time.sleep(1.1)

        assert store.get(fp.key) is None

    def test_increment(self) -> None:
        """Test increment duplicate count."""
        store = InMemoryDeduplicationStore()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)

        store.put(fp, window)

        updated = store.increment(fp.key)
        assert updated is not None
        assert updated.count == 2
        assert updated.last_duplicate_at is not None

        updated = store.increment(fp.key)
        assert updated is not None
        assert updated.count == 3

    def test_increment_nonexistent(self) -> None:
        """Test increment with nonexistent key."""
        store = InMemoryDeduplicationStore()
        assert store.increment("nonexistent") is None

    def test_delete(self) -> None:
        """Test delete operation."""
        store = InMemoryDeduplicationStore()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        window = TimeWindow(minutes=5)

        store.put(fp, window)
        assert store.delete(fp.key)
        assert store.get(fp.key) is None

    def test_delete_nonexistent(self) -> None:
        """Test delete with nonexistent key."""
        store = InMemoryDeduplicationStore()
        assert not store.delete("nonexistent")

    def test_cleanup_expired(self) -> None:
        """Test cleanup of expired records."""
        store = InMemoryDeduplicationStore()

        # Add records with short TTL
        for i in range(5):
            fp = NotificationFingerprint.generate(
                checkpoint_name=f"test_{i}",
                action_type="slack",
            )
            store.put(fp, TimeWindow(seconds=1))

        # Add record with long TTL
        fp_long = NotificationFingerprint.generate(
            checkpoint_name="test_long",
            action_type="slack",
        )
        store.put(fp_long, TimeWindow(minutes=5))

        time.sleep(1.1)

        removed = store.cleanup_expired()
        assert removed == 5

        # Long TTL record should still exist
        assert store.get(fp_long.key) is not None

    def test_clear(self) -> None:
        """Test clear all records."""
        store = InMemoryDeduplicationStore()

        for i in range(10):
            fp = NotificationFingerprint.generate(
                checkpoint_name=f"test_{i}",
                action_type="slack",
            )
            store.put(fp, TimeWindow(minutes=5))

        store.clear()
        stats = store.get_stats()
        assert stats.store_size == 0

    def test_max_size_enforcement(self) -> None:
        """Test max size is enforced."""
        store = InMemoryDeduplicationStore(max_size=5)

        for i in range(10):
            fp = NotificationFingerprint.generate(
                checkpoint_name=f"test_{i}",
                action_type="slack",
            )
            store.put(fp, TimeWindow(minutes=5))

        stats = store.get_stats()
        assert stats.store_size <= 5

    def test_get_stats(self) -> None:
        """Test statistics tracking."""
        store = InMemoryDeduplicationStore()

        # Add some records
        for i in range(5):
            fp = NotificationFingerprint.generate(
                checkpoint_name=f"test_{i}",
                action_type="slack",
            )
            store.put(fp, TimeWindow(minutes=5))

        # Record some checks
        store.record_check()
        store.record_check()
        store.record_check()

        # Increment some duplicates
        fp = NotificationFingerprint.generate(
            checkpoint_name="test_0",
            action_type="slack",
        )
        store.put(fp, TimeWindow(minutes=5))
        store.increment(fp.key)
        store.increment(fp.key)

        stats = store.get_stats()
        assert stats.notifications_sent >= 5
        assert stats.total_checked == 3
        assert stats.duplicates_found == 2
        assert stats.store_size > 0

    def test_get_all_records(self) -> None:
        """Test getting all non-expired records."""
        store = InMemoryDeduplicationStore()

        for i in range(5):
            fp = NotificationFingerprint.generate(
                checkpoint_name=f"test_{i}",
                action_type="slack",
            )
            store.put(fp, TimeWindow(minutes=5))

        records = store.get_all_records()
        assert len(records) == 5

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        import threading

        store = InMemoryDeduplicationStore()
        errors = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(100):
                    fp = NotificationFingerprint.generate(
                        checkpoint_name=f"test_{worker_id}_{i}",
                        action_type="slack",
                    )
                    store.put(fp, TimeWindow(minutes=5))
                    store.get(fp.key)
                    store.record_check()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_lru_behavior(self) -> None:
        """Test LRU-like behavior (accessed records move to end)."""
        store = InMemoryDeduplicationStore(max_size=5)

        # Add records
        keys = []
        for i in range(5):
            fp = NotificationFingerprint.generate(
                checkpoint_name=f"test_{i}",
                action_type="slack",
            )
            store.put(fp, TimeWindow(minutes=5))
            keys.append(fp.key)

        # Access first record (should move to end)
        store.get(keys[0])

        # Add new record (should evict second record, not first)
        fp_new = NotificationFingerprint.generate(
            checkpoint_name="test_new",
            action_type="slack",
        )
        store.put(fp_new, TimeWindow(minutes=5))

        # First record should still exist (was accessed)
        assert store.get(keys[0]) is not None

    def test_metadata_storage(self) -> None:
        """Test metadata is stored with record."""
        store = InMemoryDeduplicationStore()
        fp = NotificationFingerprint.generate(
            checkpoint_name="test",
            action_type="slack",
        )
        metadata = {"source": "test", "priority": 1}

        record = store.put(fp, TimeWindow(minutes=5), metadata)
        assert record.metadata == metadata

        retrieved = store.get(fp.key)
        assert retrieved is not None
        assert retrieved.metadata == metadata


class TestRedisStreamsDeduplicationStore:
    """Tests for RedisStreamsDeduplicationStore.

    These tests are skipped if Redis is not available.
    """

    @pytest.fixture
    def redis_available(self) -> bool:
        """Check if Redis is available."""
        try:
            import redis

            r = redis.Redis(host="localhost", port=6379)
            r.ping()
            return True
        except Exception:
            return False

    @pytest.mark.skipif(
        True,  # Skip by default, enable when Redis is available
        reason="Redis not available for testing",
    )
    def test_redis_store_basic(self, redis_available: bool) -> None:
        """Test basic Redis store operations."""
        if not redis_available:
            pytest.skip("Redis not available")

        import asyncio

        from truthound.checkpoint.deduplication.stores import (
            RedisStreamsDeduplicationStore,
        )

        async def run_test() -> None:
            store = RedisStreamsDeduplicationStore(
                redis_url="redis://localhost:6379",
                stream_key="test:dedup:stream",
                hash_key="test:dedup:records",
            )
            await store.initialize()

            try:
                fp = NotificationFingerprint.generate(
                    checkpoint_name="test",
                    action_type="slack",
                )
                window = TimeWindow(minutes=5)

                # Test put
                record = await store.put_async(fp, window)
                assert record.fingerprint.key == fp.key

                # Test get
                retrieved = await store.get_async(fp.key)
                assert retrieved is not None
                assert retrieved.fingerprint.key == fp.key

                # Test increment
                updated = await store.increment_async(fp.key)
                assert updated is not None
                assert updated.count == 2

                # Test delete
                assert await store.delete_async(fp.key)
                assert await store.get_async(fp.key) is None

            finally:
                await store.clear_async()
                await store.close()

        asyncio.run(run_test())
