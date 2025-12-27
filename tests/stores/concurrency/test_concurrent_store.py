"""Tests for ConcurrentFileSystemStore."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest

from truthound.stores.backends.concurrent_filesystem import (
    ConcurrentFileSystemStore,
    ConcurrentFileSystemConfig,
    ConcurrencyConfig,
    LockStrategyType,
    BatchContext,
)
from truthound.stores.base import StoreQuery, StoreNotFoundError
from truthound.stores.results import ValidationResult, ResultStatus, ResultStatistics


def create_validation_result(
    run_id: str | None = None,
    data_asset: str = "test.csv",
    status: ResultStatus = ResultStatus.SUCCESS,
) -> ValidationResult:
    """Create a test validation result."""
    return ValidationResult(
        run_id=run_id or str(uuid4()),
        data_asset=data_asset,
        run_time=datetime.now(),
        status=status,
        statistics=ResultStatistics(
            total_validators=10,
            passed_validators=8,
            failed_validators=2,
            error_validators=0,
        ),
        results=[],
        metadata={},
        tags={},
    )


@pytest.fixture
def store(tmp_path: Path) -> ConcurrentFileSystemStore:
    """Create a concurrent store for tests."""
    return ConcurrentFileSystemStore(
        base_path=str(tmp_path / "store"),
        concurrency=ConcurrencyConfig(
            lock_strategy=LockStrategyType.NONE,  # Use NoOp for faster tests
            enable_wal=False,
        ),
    )


@pytest.fixture
def store_with_locking(tmp_path: Path) -> ConcurrentFileSystemStore:
    """Create a store with actual locking for concurrency tests."""
    return ConcurrentFileSystemStore(
        base_path=str(tmp_path / "store"),
        concurrency=ConcurrencyConfig(
            lock_strategy=LockStrategyType.AUTO,
            enable_wal=True,
            enable_statistics=True,
        ),
    )


class TestConcurrentFileSystemStore:
    """Tests for ConcurrentFileSystemStore."""

    def test_save_and_get(self, store: ConcurrentFileSystemStore) -> None:
        """Test saving and retrieving a result."""
        result = create_validation_result()

        run_id = store.save(result)
        retrieved = store.get(run_id)

        assert retrieved.run_id == result.run_id
        assert retrieved.data_asset == result.data_asset

    def test_exists(self, store: ConcurrentFileSystemStore) -> None:
        """Test checking existence."""
        result = create_validation_result()
        run_id = store.save(result)

        assert store.exists(run_id)
        assert not store.exists("nonexistent")

    def test_delete(self, store: ConcurrentFileSystemStore) -> None:
        """Test deleting a result."""
        result = create_validation_result()
        run_id = store.save(result)

        assert store.delete(run_id)
        assert not store.exists(run_id)

    def test_delete_nonexistent(self, store: ConcurrentFileSystemStore) -> None:
        """Test deleting non-existent result."""
        assert not store.delete("nonexistent")

    def test_get_nonexistent_raises(self, store: ConcurrentFileSystemStore) -> None:
        """Test that getting non-existent result raises error."""
        with pytest.raises(StoreNotFoundError):
            store.get("nonexistent")

    def test_list_ids(self, store: ConcurrentFileSystemStore) -> None:
        """Test listing all IDs."""
        results = [create_validation_result() for _ in range(5)]
        ids = [store.save(r) for r in results]

        listed_ids = store.list_ids()
        assert set(listed_ids) == set(ids)

    def test_list_ids_with_query(self, store: ConcurrentFileSystemStore) -> None:
        """Test listing IDs with query filter."""
        # Create results with different data assets
        r1 = create_validation_result(data_asset="customers.csv")
        r2 = create_validation_result(data_asset="orders.csv")
        r3 = create_validation_result(data_asset="customers.csv")

        store.save(r1)
        store.save(r2)
        store.save(r3)

        query = StoreQuery(data_asset="customers.csv")
        ids = store.list_ids(query)

        assert len(ids) == 2

    def test_query(self, store: ConcurrentFileSystemStore) -> None:
        """Test querying results."""
        for i in range(5):
            store.save(create_validation_result())

        query = StoreQuery(limit=3)
        results = store.query(query)

        assert len(results) == 3

    def test_compression(self, tmp_path: Path) -> None:
        """Test storing with compression."""
        store = ConcurrentFileSystemStore(
            base_path=str(tmp_path / "compressed"),
            compression=True,
            concurrency=ConcurrencyConfig(lock_strategy=LockStrategyType.NONE),
        )

        result = create_validation_result()
        run_id = store.save(result)
        retrieved = store.get(run_id)

        assert retrieved.run_id == result.run_id

    def test_rebuild_index(self, store: ConcurrentFileSystemStore) -> None:
        """Test rebuilding index from files."""
        # Save some results
        results = [create_validation_result() for _ in range(5)]
        ids = [store.save(r) for r in results]

        # Clear and rebuild index
        count = store.rebuild_index()

        assert count == 5
        assert set(store.list_ids()) == set(ids)


class TestBatchOperations:
    """Tests for batch operations."""

    def test_batch_save(self, store: ConcurrentFileSystemStore) -> None:
        """Test batch save operation."""
        # Initialize store first
        store.initialize()
        results = [create_validation_result() for _ in range(5)]

        with store._index.transaction() as txn:
            batch = BatchContext(store, txn)
            for r in results:
                batch.save(r)
            batch.commit()

        assert store.list_ids().__len__() == 5

    def test_batch_delete(self, store: ConcurrentFileSystemStore) -> None:
        """Test batch delete operation."""
        # Save some results first
        results = [create_validation_result() for _ in range(5)]
        ids = [store.save(r) for r in results]

        # Delete in batch
        with store._index.transaction() as txn:
            batch = BatchContext(store, txn)
            for id_ in ids[:3]:
                batch.delete(id_)
            batch.commit()

        assert len(store.list_ids()) == 2

    def test_batch_rollback(self, store: ConcurrentFileSystemStore) -> None:
        """Test batch rollback."""
        # Initialize store first
        store.initialize()
        initial_count = len(store.list_ids())

        with store._index.transaction() as txn:
            batch = BatchContext(store, txn)
            batch.save(create_validation_result())
            batch.save(create_validation_result())
            batch.rollback()

        assert len(store.list_ids()) == initial_count


class TestConcurrency:
    """Tests for concurrent access.

    These tests use NoOp locking for thread safety within the test process.
    For full multi-process concurrency testing, see integration tests.
    """

    def test_concurrent_saves(self, store: ConcurrentFileSystemStore) -> None:
        """Test concurrent save operations."""
        errors: list[Exception] = []
        lock = threading.Lock()

        def save_result(i: int) -> str:
            try:
                result = create_validation_result()
                return store.save(result)
            except Exception as e:
                with lock:
                    errors.append(e)
                return ""

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(save_result, i) for i in range(10)]
            ids = [f.result() for f in as_completed(futures)]

        assert len(errors) == 0
        assert len([id_ for id_ in ids if id_]) == 10
        assert len(store.list_ids()) == 10

    def test_concurrent_reads(self, store: ConcurrentFileSystemStore) -> None:
        """Test concurrent read operations."""
        # Save some results first
        results = [create_validation_result() for _ in range(5)]
        ids = [store.save(r) for r in results]

        read_results: list[ValidationResult] = []
        lock = threading.Lock()

        def read_result(run_id: str) -> None:
            result = store.get(run_id)
            with lock:
                read_results.append(result)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_result, id_) for id_ in ids * 3]
            for f in as_completed(futures):
                f.result()

        assert len(read_results) == 15

    def test_concurrent_reads_writes(self, store: ConcurrentFileSystemStore) -> None:
        """Test concurrent read and write operations."""
        # Save initial results
        initial_results = [create_validation_result() for _ in range(5)]
        initial_ids = [store.save(r) for r in initial_results]

        errors: list[Exception] = []
        lock = threading.Lock()

        def read_result(run_id: str) -> None:
            try:
                store.get(run_id)
            except StoreNotFoundError:
                pass  # Expected if result was deleted
            except Exception as e:
                with lock:
                    errors.append(e)

        def write_result() -> None:
            try:
                result = create_validation_result()
                store.save(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(20):
                if i % 3 == 0:
                    futures.append(executor.submit(write_result))
                else:
                    futures.append(executor.submit(read_result, initial_ids[i % 5]))

            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0

    def test_concurrent_deletes(self, store: ConcurrentFileSystemStore) -> None:
        """Test concurrent delete operations."""
        # Save results
        results = [create_validation_result() for _ in range(10)]
        ids = [store.save(r) for r in results]

        delete_results: list[bool] = []
        lock = threading.Lock()

        def delete_result(run_id: str) -> None:
            result = store.delete(run_id)
            with lock:
                delete_results.append(result)

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Each ID deleted twice - only one should succeed
            futures = [executor.submit(delete_result, id_) for id_ in ids * 2]
            for f in as_completed(futures):
                f.result()

        # Exactly 10 should succeed, 10 should fail
        assert sum(delete_results) == 10
        assert len(store.list_ids()) == 0


class TestLockStatistics:
    """Tests for lock statistics."""

    def test_statistics_tracking(
        self, store_with_locking: ConcurrentFileSystemStore
    ) -> None:
        """Test that lock statistics are tracked."""
        # Perform some operations
        for _ in range(10):
            result = create_validation_result()
            store_with_locking.save(result)

        stats = store_with_locking.lock_statistics
        assert stats is not None
        assert stats.total_acquisitions > 0


class TestConfiguration:
    """Tests for store configuration."""

    def test_custom_config(self, tmp_path: Path) -> None:
        """Test creating store with custom config."""
        store = ConcurrentFileSystemStore(
            base_path=str(tmp_path / "custom"),
            namespace="production",
            prefix="results",
            compression=True,
            concurrency=ConcurrencyConfig(
                lock_strategy=LockStrategyType.NONE,
                enable_wal=False,
                lock_timeout=60.0,
            ),
        )

        result = create_validation_result()
        run_id = store.save(result)

        assert store.exists(run_id)

    def test_config_get_full_path(self) -> None:
        """Test configuration path generation."""
        config = ConcurrentFileSystemConfig(
            base_path="/base",
            namespace="prod",
            prefix="results",
        )

        path = config.get_full_path()
        assert path == Path("/base/prod/results")

    def test_lock_strategy_auto(self, tmp_path: Path) -> None:
        """Test auto lock strategy selection."""
        store = ConcurrentFileSystemStore(
            base_path=str(tmp_path / "auto"),
            concurrency=ConcurrencyConfig(lock_strategy=LockStrategyType.AUTO),
        )

        # Should initialize without error
        result = create_validation_result()
        store.save(result)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_store(self, store: ConcurrentFileSystemStore) -> None:
        """Test operations on empty store."""
        assert store.list_ids() == []
        assert store.query(StoreQuery()) == []

    def test_special_characters_in_id(self, store: ConcurrentFileSystemStore) -> None:
        """Test handling special characters in IDs."""
        result = create_validation_result(run_id="test/special:chars")
        run_id = store.save(result)
        retrieved = store.get(run_id)

        assert retrieved.run_id == "test/special:chars"

    def test_large_result(self, store: ConcurrentFileSystemStore) -> None:
        """Test storing large results."""
        # Create result with lots of data
        result = create_validation_result()
        result.metadata = {"data": "x" * 100000}  # 100KB of data

        run_id = store.save(result)
        retrieved = store.get(run_id)

        assert len(retrieved.metadata["data"]) == 100000

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test store as context manager."""
        with ConcurrentFileSystemStore(
            base_path=str(tmp_path / "ctx"),
            concurrency=ConcurrencyConfig(lock_strategy=LockStrategyType.NONE),
        ) as store:
            result = create_validation_result()
            store.save(result)
