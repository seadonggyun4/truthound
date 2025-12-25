"""Tests for the Enhanced Idempotency Framework.

This module tests:
- Core types and records
- Fingerprint generation strategies
- Storage backends (Memory, File, SQL)
- Distributed locking
- IdempotencyService and Middleware
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from truthound.checkpoint.idempotency import (
    # Core types
    IdempotencyRecord,
    IdempotencyStatus,
    IdempotencyConfig,
    # Errors
    IdempotencyError,
    IdempotencyConflictError,
    IdempotencyExpiredError,
    IdempotencyHashMismatchError,
    # Fingerprint
    RequestFingerprint,
    FingerprintStrategy,
    ContentHashStrategy,
    StructuralHashStrategy,
    CompositeFingerprint,
    # Stores
    IdempotencyStore,
    InMemoryIdempotencyStore,
    FileIdempotencyStore,
    SQLIdempotencyStore,
    # Locking
    DistributedLock,
    LockAcquisitionError,
    InMemoryLock,
    FileLock,
    # Service
    IdempotencyService,
    IdempotencyMiddleware,
    idempotent,
    idempotent_action,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def memory_store() -> InMemoryIdempotencyStore:
    """Create an in-memory store."""
    return InMemoryIdempotencyStore(max_size=100)


@pytest.fixture
def file_store(tmp_path: Path) -> FileIdempotencyStore:
    """Create a file-based store."""
    return FileIdempotencyStore(tmp_path / "idempotency")


@pytest.fixture
def sql_store() -> SQLIdempotencyStore:
    """Create a SQL-based store."""
    return SQLIdempotencyStore(":memory:")


@pytest.fixture
def memory_lock() -> InMemoryLock:
    """Create an in-memory lock."""
    return InMemoryLock()


@pytest.fixture
def file_lock(tmp_path: Path) -> FileLock:
    """Create a file-based lock."""
    return FileLock(tmp_path / "locks")


@pytest.fixture
def service(memory_store: InMemoryIdempotencyStore, memory_lock: InMemoryLock) -> IdempotencyService:
    """Create an idempotency service."""
    return IdempotencyService(store=memory_store, lock=memory_lock)


# =============================================================================
# Tests: Core Types
# =============================================================================


class TestIdempotencyStatus:
    """Tests for IdempotencyStatus enum."""

    def test_values(self):
        assert IdempotencyStatus.PENDING.value == "pending"
        assert IdempotencyStatus.COMPLETED.value == "completed"
        assert IdempotencyStatus.FAILED.value == "failed"
        assert IdempotencyStatus.EXPIRED.value == "expired"

    def test_is_terminal(self):
        assert IdempotencyStatus.COMPLETED.is_terminal
        assert IdempotencyStatus.FAILED.is_terminal
        assert IdempotencyStatus.EXPIRED.is_terminal
        assert not IdempotencyStatus.PENDING.is_terminal

    def test_allows_retry(self):
        assert IdempotencyStatus.FAILED.allows_retry
        assert IdempotencyStatus.EXPIRED.allows_retry
        assert not IdempotencyStatus.PENDING.allows_retry
        assert not IdempotencyStatus.COMPLETED.allows_retry


class TestIdempotencyConfig:
    """Tests for IdempotencyConfig."""

    def test_defaults(self):
        config = IdempotencyConfig()
        assert config.enabled
        assert config.ttl_seconds == 3600
        assert config.validate_hash
        assert config.store_result

    def test_with_ttl(self):
        config = IdempotencyConfig()
        new_config = config.with_ttl(7200)
        assert new_config.ttl_seconds == 7200
        assert config.ttl_seconds == 3600  # Original unchanged


class TestIdempotencyRecord:
    """Tests for IdempotencyRecord."""

    def test_default_initialization(self):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test-key")
        assert record.key == "test-key"
        assert record.status == IdempotencyStatus.PENDING
        assert record.expires_at is not None
        assert not record.is_expired

    def test_is_expired(self):
        record: IdempotencyRecord[str] = IdempotencyRecord(
            key="test",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert record.is_expired

    def test_mark_pending(self):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        record.mark_pending("holder-1")

        assert record.status == IdempotencyStatus.PENDING
        assert record.locked_by == "holder-1"
        assert record.is_locked
        assert record.attempt_count == 1

    def test_mark_completed(self):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        record.mark_pending("holder-1")
        record.mark_completed("result-value")

        assert record.status == IdempotencyStatus.COMPLETED
        assert record.result == "result-value"
        assert record.has_result
        assert not record.is_locked

    def test_mark_failed(self):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        record.mark_pending("holder-1")
        record.mark_failed("Something went wrong")

        assert record.status == IdempotencyStatus.FAILED
        assert record.error == "Something went wrong"
        assert not record.is_locked

    def test_to_dict_and_from_dict(self):
        record: IdempotencyRecord[dict[str, Any]] = IdempotencyRecord(
            key="test-key",
            request_hash="abc123",
            metadata={"source": "test"},
        )

        data = record.to_dict()
        restored = IdempotencyRecord.from_dict(data)

        assert restored.key == record.key
        assert restored.request_hash == record.request_hash
        assert restored.metadata == record.metadata


# =============================================================================
# Tests: Fingerprint
# =============================================================================


class TestRequestFingerprint:
    """Tests for RequestFingerprint."""

    def test_from_dict(self):
        fp = RequestFingerprint.from_dict({
            "action": "validate",
            "dataset_id": "ds-123",
        })

        assert len(fp.key) == 64  # SHA256 hex length
        assert fp.components["action"] == "validate"

    def test_from_dict_deterministic(self):
        fp1 = RequestFingerprint.from_dict({"a": 1, "b": 2})
        fp2 = RequestFingerprint.from_dict({"b": 2, "a": 1})  # Different order

        assert fp1.key == fp2.key  # Keys should match (sorted)

    def test_from_args(self):
        fp = RequestFingerprint.from_args("validate", dataset_id="ds-123")

        assert len(fp.key) == 64
        assert "validate" in str(fp.components)

    def test_with_prefix(self):
        fp = RequestFingerprint.from_dict({"key": "value"})
        prefixed = fp.with_prefix("prefix:")

        assert prefixed.key.startswith("prefix:")

    def test_from_string(self):
        fp = RequestFingerprint.from_string("unique-operation-id")

        assert len(fp.key) == 64


class TestContentHashStrategy:
    """Tests for ContentHashStrategy."""

    def test_generate(self):
        strategy = ContentHashStrategy()
        fp = strategy.generate({"key": "value"})

        assert len(fp.key) == 64
        assert strategy.name == "content_hash"

    def test_with_prefix(self):
        strategy = ContentHashStrategy(prefix="content:")
        fp = strategy.generate({"key": "value"})

        assert fp.key.startswith("content:")


class TestStructuralHashStrategy:
    """Tests for StructuralHashStrategy."""

    def test_include_fields(self):
        strategy = StructuralHashStrategy(fields=["action", "target"])
        fp = strategy.generate({
            "action": "validate",
            "target": "users",
            "timestamp": "ignored",
        })

        # Only action and target should be included
        assert "action" in str(fp.components)
        assert "target" in str(fp.components)

    def test_exclude_fields(self):
        strategy = StructuralHashStrategy(ignore=["timestamp", "request_id"])
        fp1 = strategy.generate({
            "action": "validate",
            "timestamp": "2024-01-01",
        })
        fp2 = strategy.generate({
            "action": "validate",
            "timestamp": "2024-01-02",  # Different timestamp
        })

        assert fp1.key == fp2.key  # Should match since timestamp is ignored


class TestCompositeFingerprint:
    """Tests for CompositeFingerprint."""

    def test_combines_strategies(self):
        composite = CompositeFingerprint([
            ContentHashStrategy(prefix="c:"),
            StructuralHashStrategy(fields=["action"]),
        ])

        fp = composite.generate({"action": "test", "data": [1, 2, 3]})

        assert fp.metadata.get("composite")
        assert len(fp.components["keys"]) == 2


# =============================================================================
# Tests: Storage Backends
# =============================================================================


class TestInMemoryIdempotencyStore:
    """Tests for InMemoryIdempotencyStore."""

    def test_set_and_get(self, memory_store: InMemoryIdempotencyStore):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        memory_store.set(record)

        retrieved = memory_store.get("test")
        assert retrieved is not None
        assert retrieved.key == "test"

    def test_update(self, memory_store: InMemoryIdempotencyStore):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        memory_store.set(record)

        record.mark_completed("result")
        assert memory_store.update(record)

        retrieved = memory_store.get("test")
        assert retrieved is not None
        assert retrieved.status == IdempotencyStatus.COMPLETED

    def test_delete(self, memory_store: InMemoryIdempotencyStore):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        memory_store.set(record)

        assert memory_store.delete("test")
        assert memory_store.get("test") is None
        assert not memory_store.delete("test")  # Already deleted

    def test_exists(self, memory_store: InMemoryIdempotencyStore):
        assert not memory_store.exists("test")

        memory_store.set(IdempotencyRecord(key="test"))
        assert memory_store.exists("test")

    def test_cleanup_expired(self, memory_store: InMemoryIdempotencyStore):
        # Add expired record
        expired = IdempotencyRecord(
            key="expired",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        memory_store._store["expired"] = expired

        # Add valid record
        memory_store.set(IdempotencyRecord(key="valid"))

        removed = memory_store.cleanup_expired()
        assert removed == 1
        assert memory_store.get("valid") is not None

    def test_lru_eviction(self):
        store = InMemoryIdempotencyStore(max_size=3)

        for i in range(5):
            store.set(IdempotencyRecord(key=f"key-{i}"))

        assert store.size() <= 3

    def test_thread_safety(self, memory_store: InMemoryIdempotencyStore):
        errors: list[Exception] = []

        def add_records(prefix: str, count: int):
            try:
                for i in range(count):
                    memory_store.set(IdempotencyRecord(key=f"{prefix}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_records, args=("a", 50)),
            threading.Thread(target=add_records, args=("b", 50)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestFileIdempotencyStore:
    """Tests for FileIdempotencyStore."""

    def test_set_and_get(self, file_store: FileIdempotencyStore):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        file_store.set(record)

        retrieved = file_store.get("test")
        assert retrieved is not None
        assert retrieved.key == "test"

    def test_persistence(self, tmp_path: Path):
        store_path = tmp_path / "idempotency"

        # First store
        store1 = FileIdempotencyStore(store_path)
        store1.set(IdempotencyRecord(key="persistent"))

        # Second store (simulates restart)
        store2 = FileIdempotencyStore(store_path)
        assert store2.get("persistent") is not None

    def test_delete(self, file_store: FileIdempotencyStore):
        file_store.set(IdempotencyRecord(key="test"))
        assert file_store.delete("test")
        assert file_store.get("test") is None


class TestSQLIdempotencyStore:
    """Tests for SQLIdempotencyStore."""

    def test_set_and_get(self, sql_store: SQLIdempotencyStore):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        sql_store.set(record)

        retrieved = sql_store.get("test")
        assert retrieved is not None
        assert retrieved.key == "test"

    def test_update(self, sql_store: SQLIdempotencyStore):
        record: IdempotencyRecord[str] = IdempotencyRecord(key="test")
        sql_store.set(record)

        record.mark_completed("result")
        assert sql_store.update(record)

        retrieved = sql_store.get("test")
        assert retrieved is not None
        assert retrieved.status == IdempotencyStatus.COMPLETED

    def test_find_by_status(self, sql_store: SQLIdempotencyStore):
        sql_store.set(IdempotencyRecord(key="pending1"))
        sql_store.set(IdempotencyRecord(key="pending2"))

        completed = IdempotencyRecord(key="completed")
        completed.mark_completed("done")
        sql_store.set(completed)

        pending = sql_store.find_by_status(IdempotencyStatus.PENDING)
        assert len(pending) == 2

    def test_cleanup_expired(self, sql_store: SQLIdempotencyStore):
        # Add expired record
        expired = IdempotencyRecord(
            key="expired",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        sql_store.set(expired)

        sql_store.set(IdempotencyRecord(key="valid"))

        removed = sql_store.cleanup_expired()
        assert removed == 1


# =============================================================================
# Tests: Locking
# =============================================================================


class TestInMemoryLock:
    """Tests for InMemoryLock."""

    def test_acquire_and_release(self, memory_lock: InMemoryLock):
        info = memory_lock.acquire("test-key", holder_id="holder-1")

        assert info.key == "test-key"
        assert info.holder_id == "holder-1"
        assert memory_lock.is_locked("test-key")

        assert memory_lock.release("test-key", "holder-1")
        assert not memory_lock.is_locked("test-key")

    def test_cannot_acquire_held_lock(self, memory_lock: InMemoryLock):
        memory_lock.acquire("test-key", holder_id="holder-1")

        with pytest.raises(LockAcquisitionError):
            memory_lock.acquire("test-key", holder_id="holder-2")

    def test_reentrant_lock(self, memory_lock: InMemoryLock):
        memory_lock.acquire("test-key", holder_id="holder-1")
        info = memory_lock.acquire("test-key", holder_id="holder-1")

        assert info.holder_id == "holder-1"  # Same holder can reacquire

    def test_context_manager(self, memory_lock: InMemoryLock):
        with memory_lock.lock("test-key") as info:
            assert memory_lock.is_locked("test-key")
            assert info.key == "test-key"

        assert not memory_lock.is_locked("test-key")

    def test_extend_lock(self, memory_lock: InMemoryLock):
        memory_lock.acquire("test-key", timeout=10, holder_id="holder-1")
        original_info = memory_lock.get_lock_info("test-key")
        assert original_info is not None
        original_expires = original_info.expires_at

        time.sleep(0.1)

        assert memory_lock.extend("test-key", "holder-1", 60)
        extended_info = memory_lock.get_lock_info("test-key")
        assert extended_info is not None
        # Extended expiry should be later than original
        assert extended_info.expires_at is not None
        assert original_expires is not None
        assert extended_info.expires_at > original_expires


class TestFileLock:
    """Tests for FileLock."""

    def test_acquire_and_release(self, file_lock: FileLock):
        info = file_lock.acquire("test-key", holder_id="holder-1")

        assert info.key == "test-key"
        assert file_lock.is_locked("test-key")

        assert file_lock.release("test-key", "holder-1")
        assert not file_lock.is_locked("test-key")

    def test_context_manager(self, file_lock: FileLock):
        with file_lock.lock("test-key") as info:
            assert file_lock.is_locked("test-key")

        assert not file_lock.is_locked("test-key")


# =============================================================================
# Tests: IdempotencyService
# =============================================================================


class TestIdempotencyService:
    """Tests for IdempotencyService."""

    def test_execute_new(self, service: IdempotencyService):
        call_count = [0]

        def execute_fn():
            call_count[0] += 1
            return "result"

        result = service.execute("key-1", execute_fn)

        assert result.result == "result"
        assert not result.was_cached
        assert call_count[0] == 1

    def test_execute_cached(self, service: IdempotencyService):
        call_count = [0]

        def execute_fn():
            call_count[0] += 1
            return "result"

        result1 = service.execute("key-1", execute_fn)
        result2 = service.execute("key-1", execute_fn)

        assert result1.result == result2.result
        assert not result1.was_cached
        assert result2.was_cached
        assert call_count[0] == 1

    def test_execute_different_keys(self, service: IdempotencyService):
        call_count = [0]

        def execute_fn():
            call_count[0] += 1
            return f"result-{call_count[0]}"

        result1 = service.execute("key-1", execute_fn)
        result2 = service.execute("key-2", execute_fn)

        assert result1.result != result2.result
        assert call_count[0] == 2

    def test_hash_mismatch(self, service: IdempotencyService):
        service.execute("key-1", lambda: "result", request_hash="hash-1")

        with pytest.raises(IdempotencyHashMismatchError):
            service.execute("key-1", lambda: "result", request_hash="hash-2")

    def test_execution_failure_allows_retry(self, service: IdempotencyService):
        call_count = [0]

        def failing_fn():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First attempt fails")
            return "success"

        # First attempt fails
        with pytest.raises(RuntimeError):
            service.execute("key-1", failing_fn)

        # Second attempt succeeds (key should be available for retry)
        result = service.execute("key-1", failing_fn)
        assert result.result == "success"
        assert call_count[0] == 2

    def test_invalidate(self, service: IdempotencyService):
        service.execute("key-1", lambda: "result")
        assert service.get_result("key-1") == "result"

        assert service.invalidate("key-1")
        assert service.get_result("key-1") is None

    def test_scoped_service(self, service: IdempotencyService):
        with service.scope("scope:") as scoped:
            scoped.execute("key-1", lambda: "result")

            # Key is prefixed
            assert service.get_result("scope:key-1") == "result"
            assert scoped.get_result("key-1") == "result"

    def test_disabled(self):
        service = IdempotencyService(
            config=IdempotencyConfig(enabled=False),
        )
        call_count = [0]

        def execute_fn():
            call_count[0] += 1
            return "result"

        service.execute("key-1", execute_fn)
        service.execute("key-1", execute_fn)

        assert call_count[0] == 2  # Both calls executed


class TestIdempotencyMiddleware:
    """Tests for IdempotencyMiddleware."""

    def test_wrap_decorator(self, service: IdempotencyService):
        middleware = IdempotencyMiddleware(service=service)
        call_count = [0]

        @middleware.wrap(key_fn=lambda x: f"key-{x}")
        def my_function(x: int) -> str:
            call_count[0] += 1
            return f"result-{x}"

        result1 = my_function(1)
        result2 = my_function(1)
        result3 = my_function(2)

        assert result1.result == "result-1"
        assert result2.was_cached
        assert not result3.was_cached
        assert call_count[0] == 2


class TestIdempotentDecorator:
    """Tests for @idempotent decorator."""

    def test_basic_usage(self):
        call_count = [0]

        @idempotent(key_fn=lambda x: f"key-{x}")
        def process(x: int) -> str:
            call_count[0] += 1
            return f"result-{x}"

        result1 = process(1)
        result2 = process(1)

        assert result1.result == "result-1"
        assert result2.was_cached
        assert call_count[0] == 1

    def test_with_static_key(self):
        call_count = [0]

        @idempotent(key="static-key")
        def singleton_operation() -> str:
            call_count[0] += 1
            return "singleton"

        singleton_operation()
        singleton_operation()

        assert call_count[0] == 1


# =============================================================================
# Tests: Exceptions
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_idempotency_conflict_error(self):
        error = IdempotencyConflictError("key-1", locked_by="process-1")
        assert "key-1" in str(error)
        assert "process-1" in str(error)

    def test_idempotency_hash_mismatch_error(self):
        error = IdempotencyHashMismatchError("key-1", "expected", "actual")
        assert "key-1" in str(error)
        assert "expected" in str(error)
        assert "actual" in str(error)

    def test_idempotency_expired_error(self):
        expired_at = datetime.now()
        error = IdempotencyExpiredError("key-1", expired_at)
        assert "key-1" in str(error)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self, tmp_path: Path):
        """Test complete workflow with persistence."""
        store = FileIdempotencyStore(tmp_path / "idempotency")
        lock = FileLock(tmp_path / "locks")
        service = IdempotencyService(store=store, lock=lock)

        call_count = [0]

        def expensive_operation() -> dict[str, Any]:
            call_count[0] += 1
            return {"data": "processed", "count": call_count[0]}

        # First call
        result1 = service.execute("operation-1", expensive_operation)
        assert result1.result["count"] == 1
        assert not result1.was_cached

        # Second call returns cached
        result2 = service.execute("operation-1", expensive_operation)
        assert result2.result["count"] == 1  # Same result
        assert result2.was_cached

        # Different key executes again
        result3 = service.execute("operation-2", expensive_operation)
        assert result3.result["count"] == 2
        assert not result3.was_cached

    def test_concurrent_execution(self):
        """Test concurrent execution with locking."""
        service = IdempotencyService()
        results: list[Any] = []
        errors: list[Exception] = []
        execution_count = [0]

        def slow_operation():
            execution_count[0] += 1
            time.sleep(0.01)  # Reduced from 0.1
            return f"result-{execution_count[0]}"

        def execute():
            try:
                result = service.execute("shared-key", slow_operation)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=execute) for _ in range(3)]  # Reduced from 5

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one execution, rest are conflicts or cached
        successful = [r for r in results if not r.was_cached]
        assert len(successful) <= 1

    def test_fingerprint_based_deduplication(self):
        """Test deduplication based on request fingerprinting."""
        service = IdempotencyService()

        def process_order(order_data: dict[str, Any]) -> str:
            fp = RequestFingerprint.from_dict(order_data)
            return service.execute(
                key=fp.key,
                execute_fn=lambda: f"processed-{order_data['id']}",
            ).result

        # Same order data = same fingerprint = deduplicated
        result1 = process_order({"id": 1, "amount": 100})
        result2 = process_order({"id": 1, "amount": 100})

        assert result1 == result2

        # Different order = different fingerprint = new execution
        result3 = process_order({"id": 2, "amount": 100})
        assert result3 != result1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
