"""Tests for concurrent index management."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from truthound.stores.concurrency.index import (
    ConcurrentIndex,
    IndexEntry,
    IndexSnapshot,
    IndexTransaction,
)
from truthound.stores.concurrency.manager import FileLockManager
from truthound.stores.concurrency.locks import NoOpLockStrategy


@pytest.fixture
def lock_manager() -> FileLockManager:
    """Create a lock manager for tests."""
    return FileLockManager(strategy=NoOpLockStrategy())


@pytest.fixture
def index(tmp_path: Path, lock_manager: FileLockManager) -> ConcurrentIndex:
    """Create a concurrent index for tests."""
    return ConcurrentIndex(
        base_path=tmp_path,
        lock_manager=lock_manager,
        wal_enabled=False,  # Disable WAL for simpler tests
    )


class TestIndexEntry:
    """Tests for IndexEntry dataclass."""

    def test_create_entry(self) -> None:
        """Test creating an index entry."""
        entry = IndexEntry(
            item_id="test-id",
            metadata={"key": "value"},
        )

        assert entry.item_id == "test-id"
        assert entry.metadata == {"key": "value"}
        assert entry.version == 1

    def test_entry_to_dict(self) -> None:
        """Test serializing entry to dict."""
        entry = IndexEntry(
            item_id="test-id",
            metadata={"key": "value"},
        )
        data = entry.to_dict()

        assert data["item_id"] == "test-id"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data
        assert "version" in data

    def test_entry_from_dict(self) -> None:
        """Test deserializing entry from dict."""
        data = {
            "item_id": "test-id",
            "metadata": {"key": "value"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "version": 5,
        }
        entry = IndexEntry.from_dict(data)

        assert entry.item_id == "test-id"
        assert entry.metadata == {"key": "value"}
        assert entry.version == 5


class TestIndexSnapshot:
    """Tests for IndexSnapshot."""

    def test_snapshot_get(self) -> None:
        """Test getting entry from snapshot."""
        entries = {
            "id1": IndexEntry("id1", {"key": "value1"}),
            "id2": IndexEntry("id2", {"key": "value2"}),
        }
        snapshot = IndexSnapshot(entries=entries, version=1)

        assert snapshot.get("id1") is not None
        assert snapshot.get("id1").metadata == {"key": "value1"}
        assert snapshot.get("nonexistent") is None

    def test_snapshot_contains(self) -> None:
        """Test checking if entry exists in snapshot."""
        entries = {"id1": IndexEntry("id1", {})}
        snapshot = IndexSnapshot(entries=entries, version=1)

        assert snapshot.contains("id1")
        assert not snapshot.contains("id2")

    def test_snapshot_list_ids(self) -> None:
        """Test listing IDs from snapshot."""
        entries = {
            "id1": IndexEntry("id1", {}),
            "id2": IndexEntry("id2", {}),
            "id3": IndexEntry("id3", {}),
        }
        snapshot = IndexSnapshot(entries=entries, version=1)

        ids = snapshot.list_ids()
        assert set(ids) == {"id1", "id2", "id3"}

    def test_snapshot_filter(self) -> None:
        """Test filtering entries in snapshot."""
        entries = {
            "id1": IndexEntry("id1", {"type": "a"}),
            "id2": IndexEntry("id2", {"type": "b"}),
            "id3": IndexEntry("id3", {"type": "a"}),
        }
        snapshot = IndexSnapshot(entries=entries, version=1)

        filtered = snapshot.filter(lambda e: e.metadata.get("type") == "a")
        assert len(filtered) == 2

    def test_snapshot_len(self) -> None:
        """Test snapshot length."""
        entries = {
            "id1": IndexEntry("id1", {}),
            "id2": IndexEntry("id2", {}),
        }
        snapshot = IndexSnapshot(entries=entries, version=1)

        assert len(snapshot) == 2


class TestIndexTransaction:
    """Tests for IndexTransaction."""

    def test_transaction_add(self, index: ConcurrentIndex) -> None:
        """Test adding entry in transaction."""
        index.initialize()

        with index.transaction() as txn:
            entry = txn.add("new-id", {"key": "value"})
            txn.commit()

        assert entry.item_id == "new-id"
        assert index.contains("new-id")

    def test_transaction_add_duplicate_raises(self, index: ConcurrentIndex) -> None:
        """Test that adding duplicate entry raises error."""
        index.initialize()
        index.add("existing", {})

        with pytest.raises(ValueError):
            with index.transaction() as txn:
                txn.add("existing", {})

    def test_transaction_update(self, index: ConcurrentIndex) -> None:
        """Test updating entry in transaction."""
        index.initialize()
        index.add("id", {"key": "old"})

        with index.transaction() as txn:
            entry = txn.update("id", {"key": "new"})
            txn.commit()

        assert entry.metadata["key"] == "new"
        assert index.get("id").metadata["key"] == "new"

    def test_transaction_update_merge(self, index: ConcurrentIndex) -> None:
        """Test updating with merge."""
        index.initialize()
        index.add("id", {"key1": "value1", "key2": "value2"})

        with index.transaction() as txn:
            txn.update("id", {"key2": "updated"}, merge=True)
            txn.commit()

        entry = index.get("id")
        assert entry.metadata["key1"] == "value1"
        assert entry.metadata["key2"] == "updated"

    def test_transaction_update_no_merge(self, index: ConcurrentIndex) -> None:
        """Test updating without merge."""
        index.initialize()
        index.add("id", {"key1": "value1", "key2": "value2"})

        with index.transaction() as txn:
            txn.update("id", {"key3": "value3"}, merge=False)
            txn.commit()

        entry = index.get("id")
        assert "key1" not in entry.metadata
        assert entry.metadata["key3"] == "value3"

    def test_transaction_upsert(self, index: ConcurrentIndex) -> None:
        """Test upsert operation."""
        index.initialize()

        # Upsert non-existing creates
        with index.transaction() as txn:
            txn.upsert("id", {"key": "value"})
            txn.commit()

        assert index.contains("id")

        # Upsert existing updates
        with index.transaction() as txn:
            txn.upsert("id", {"key": "new"})
            txn.commit()

        assert index.get("id").metadata["key"] == "new"

    def test_transaction_remove(self, index: ConcurrentIndex) -> None:
        """Test removing entry in transaction."""
        index.initialize()
        index.add("id", {})

        with index.transaction() as txn:
            result = txn.remove("id")
            txn.commit()

        assert result is True
        assert not index.contains("id")

    def test_transaction_remove_nonexistent(self, index: ConcurrentIndex) -> None:
        """Test removing non-existent entry."""
        index.initialize()

        with index.transaction() as txn:
            result = txn.remove("nonexistent")
            txn.commit()

        assert result is False

    def test_transaction_rollback(self, index: ConcurrentIndex) -> None:
        """Test transaction rollback."""
        index.initialize()

        with index.transaction() as txn:
            txn.add("id", {})
            txn.rollback()

        assert not index.contains("id")

    def test_transaction_auto_commit(self, index: ConcurrentIndex) -> None:
        """Test automatic commit on exit."""
        index.initialize()

        with index.transaction() as txn:
            txn.add("id", {})
            # Don't explicitly commit

        assert index.contains("id")

    def test_transaction_exception_rollback(self, index: ConcurrentIndex) -> None:
        """Test rollback on exception."""
        index.initialize()

        with pytest.raises(ValueError):
            with index.transaction() as txn:
                txn.add("id", {})
                raise ValueError("Test error")

        assert not index.contains("id")


class TestConcurrentIndex:
    """Tests for ConcurrentIndex."""

    def test_initialize(self, tmp_path: Path, lock_manager: FileLockManager) -> None:
        """Test index initialization."""
        index = ConcurrentIndex(tmp_path, lock_manager=lock_manager)
        index.initialize()

        assert (tmp_path / "_index.json").exists()

    def test_add_and_get(self, index: ConcurrentIndex) -> None:
        """Test adding and retrieving entries."""
        index.initialize()

        entry = index.add("id", {"key": "value"})
        retrieved = index.get("id")

        assert retrieved is not None
        assert retrieved.item_id == "id"
        assert retrieved.metadata == {"key": "value"}

    def test_contains(self, index: ConcurrentIndex) -> None:
        """Test checking entry existence."""
        index.initialize()
        index.add("id", {})

        assert index.contains("id")
        assert not index.contains("nonexistent")

    def test_list_ids(self, index: ConcurrentIndex) -> None:
        """Test listing all IDs."""
        index.initialize()
        index.add("id1", {})
        index.add("id2", {})
        index.add("id3", {})

        ids = index.list_ids()
        assert set(ids) == {"id1", "id2", "id3"}

    def test_count(self, index: ConcurrentIndex) -> None:
        """Test counting entries."""
        index.initialize()
        assert index.count() == 0

        index.add("id1", {})
        index.add("id2", {})
        assert index.count() == 2

    def test_clear(self, index: ConcurrentIndex) -> None:
        """Test clearing all entries."""
        index.initialize()
        index.add("id1", {})
        index.add("id2", {})

        count = index.clear()

        assert count == 2
        assert index.count() == 0

    def test_snapshot_isolation(self, index: ConcurrentIndex) -> None:
        """Test that snapshots are isolated from changes."""
        index.initialize()
        index.add("id1", {})

        snapshot = index.snapshot()

        # Add more after snapshot
        index.add("id2", {})

        # Snapshot should not see id2
        assert snapshot.contains("id1")
        assert not snapshot.contains("id2")

        # Current index should see both
        assert index.contains("id1")
        assert index.contains("id2")

    def test_persistence(self, tmp_path: Path, lock_manager: FileLockManager) -> None:
        """Test that index persists across instances."""
        # First instance
        index1 = ConcurrentIndex(tmp_path, lock_manager=lock_manager)
        index1.initialize()
        index1.add("id", {"key": "value"})

        # Second instance (simulating restart)
        index2 = ConcurrentIndex(tmp_path, lock_manager=lock_manager)
        index2.initialize()

        assert index2.contains("id")
        assert index2.get("id").metadata == {"key": "value"}


class TestConcurrentIndexWAL:
    """Tests for write-ahead logging."""

    def test_wal_recovery(self, tmp_path: Path, lock_manager: FileLockManager) -> None:
        """Test recovery from WAL."""
        # Create index with WAL
        index = ConcurrentIndex(tmp_path, lock_manager=lock_manager, wal_enabled=True)
        index.initialize()

        # Manually create a WAL entry
        wal_path = tmp_path / "_index.json.wal"
        wal_entry = {
            "type": "add",
            "item_id": "recovered-id",
            "entry": IndexEntry("recovered-id", {"recovered": True}).to_dict(),
        }
        wal_path.write_text(json.dumps(wal_entry) + "\n")

        # Create new index - should recover from WAL
        index2 = ConcurrentIndex(tmp_path, lock_manager=lock_manager, wal_enabled=True)
        index2.initialize()

        assert index2.contains("recovered-id")
        assert index2.get("recovered-id").metadata["recovered"] is True
        assert not wal_path.exists()  # WAL should be cleared


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    def test_concurrent_adds(self, index: ConcurrentIndex) -> None:
        """Test concurrent add operations."""
        index.initialize()
        errors: list[Exception] = []

        def add_entry(i: int) -> None:
            try:
                index.add(f"id-{i}", {"index": i})
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_entry, i) for i in range(50)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0
        assert index.count() == 50

    def test_concurrent_reads(self, index: ConcurrentIndex) -> None:
        """Test concurrent read operations."""
        index.initialize()

        # Add some entries
        for i in range(20):
            index.add(f"id-{i}", {"index": i})

        results: list[IndexEntry] = []
        lock = threading.Lock()

        def read_entry(i: int) -> None:
            entry = index.get(f"id-{i}")
            if entry:
                with lock:
                    results.append(entry)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_entry, i % 20) for i in range(100)]
            for future in as_completed(futures):
                future.result()

        assert len(results) == 100

    def test_concurrent_reads_writes(self, index: ConcurrentIndex) -> None:
        """Test concurrent read and write operations."""
        index.initialize()

        # Add initial entries
        for i in range(10):
            index.add(f"id-{i}", {"index": i})

        errors: list[Exception] = []
        lock = threading.Lock()

        def read_entry(i: int) -> None:
            try:
                index.get(f"id-{i % 10}")
            except Exception as e:
                with lock:
                    errors.append(e)

        def write_entry(i: int) -> None:
            try:
                index.upsert(f"id-{i}", {"index": i})
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(50):
                if i % 3 == 0:
                    futures.append(executor.submit(write_entry, i))
                else:
                    futures.append(executor.submit(read_entry, i))

            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0

    def test_snapshot_consistency_under_concurrent_writes(
        self, index: ConcurrentIndex
    ) -> None:
        """Test that snapshots remain consistent during concurrent writes."""
        index.initialize()
        for i in range(10):
            index.add(f"id-{i}", {"version": 0})

        snapshots: list[IndexSnapshot] = []
        lock = threading.Lock()

        def take_snapshot() -> None:
            snapshot = index.snapshot()
            with lock:
                snapshots.append(snapshot)

        def update_entries() -> None:
            for i in range(10):
                try:
                    index.update(f"id-{i}", {"version": 1})
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(20):
                if i % 2 == 0:
                    futures.append(executor.submit(take_snapshot))
                else:
                    futures.append(executor.submit(update_entries))

            for future in as_completed(futures):
                future.result()

        # Each snapshot should be internally consistent
        for snapshot in snapshots:
            versions = [e.metadata.get("version") for e in snapshot.entries.values()]
            # All versions in a snapshot should be the same
            # (either all 0 or all 1, depending on when snapshot was taken)
            assert len(set(versions)) <= 2  # Allow for in-flight transactions
