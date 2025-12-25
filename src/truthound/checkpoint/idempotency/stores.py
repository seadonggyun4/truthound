"""Idempotency Storage Backends.

This module provides various storage implementations for idempotency records,
from simple in-memory stores to database-backed solutions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Protocol, runtime_checkable

from truthound.checkpoint.idempotency.core import (
    IdempotencyRecord,
    IdempotencyStatus,
)


logger = logging.getLogger(__name__)


@runtime_checkable
class IdempotencyStore(Protocol):
    """Protocol for idempotency record storage.

    All implementations must provide thread-safe operations for
    storing and retrieving idempotency records.
    """

    def get(self, key: str) -> IdempotencyRecord[Any] | None:
        """Get a record by key.

        Args:
            key: The idempotency key.

        Returns:
            IdempotencyRecord if found, None otherwise.
        """
        ...

    def set(self, record: IdempotencyRecord[Any]) -> None:
        """Store a record.

        Args:
            record: The record to store.
        """
        ...

    def update(self, record: IdempotencyRecord[Any]) -> bool:
        """Update an existing record.

        Args:
            record: The record to update.

        Returns:
            True if updated, False if not found.
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete a record.

        Args:
            key: The key to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if exists.
        """
        ...

    def cleanup_expired(self) -> int:
        """Remove expired records.

        Returns:
            Number of records removed.
        """
        ...


class InMemoryIdempotencyStore:
    """In-memory implementation of IdempotencyStore.

    Thread-safe, suitable for single-process scenarios.
    Supports LRU eviction and automatic expiry cleanup.

    Example:
        >>> store = InMemoryIdempotencyStore(max_size=10000)
        >>> record = IdempotencyRecord(key="req-123")
        >>> store.set(record)
        >>> retrieved = store.get("req-123")
    """

    def __init__(
        self,
        max_size: int = 10000,
        auto_cleanup: bool = True,
        cleanup_threshold: float = 0.9,
    ) -> None:
        """Initialize the store.

        Args:
            max_size: Maximum number of records.
            auto_cleanup: Automatically cleanup when near capacity.
            cleanup_threshold: Trigger cleanup at this capacity ratio.
        """
        self._store: dict[str, IdempotencyRecord[Any]] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._auto_cleanup = auto_cleanup
        self._cleanup_threshold = cleanup_threshold
        self._access_order: list[str] = []  # For LRU

    def get(self, key: str) -> IdempotencyRecord[Any] | None:
        with self._lock:
            record = self._store.get(key)
            if record is None:
                return None

            if record.is_expired:
                del self._store[key]
                self._access_order.remove(key)
                return None

            # Update access order for LRU
            self._update_access(key)
            return record

    def set(self, record: IdempotencyRecord[Any]) -> None:
        with self._lock:
            if self._auto_cleanup and len(self._store) >= self._max_size * self._cleanup_threshold:
                self._evict()

            if record.key in self._store:
                self._access_order.remove(record.key)

            self._store[record.key] = record
            self._access_order.append(record.key)

    def update(self, record: IdempotencyRecord[Any]) -> bool:
        with self._lock:
            if record.key not in self._store:
                return False

            self._store[record.key] = record
            self._update_access(record.key)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._access_order.remove(key)
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            record = self._store.get(key)
            if record and record.is_expired:
                del self._store[key]
                self._access_order.remove(key)
                return False
            return record is not None

    def cleanup_expired(self) -> int:
        with self._lock:
            now = datetime.now()
            expired = [
                k for k, v in self._store.items()
                if v.expires_at and v.expires_at < now
            ]
            for key in expired:
                del self._store[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            return len(expired)

    def _update_access(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict(self) -> None:
        """Evict entries to make room."""
        # First cleanup expired
        expired_count = self.cleanup_expired()
        if expired_count > 0 and len(self._store) < self._max_size * self._cleanup_threshold:
            return

        # Then evict LRU entries
        while len(self._store) >= self._max_size:
            if not self._access_order:
                break
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._store:
                del self._store[oldest_key]

    def size(self) -> int:
        """Get current size."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._store.clear()
            self._access_order.clear()

    def keys(self) -> list[str]:
        """Get all keys."""
        with self._lock:
            return list(self._store.keys())


class FileIdempotencyStore:
    """File-based implementation of IdempotencyStore.

    Each record is stored as a separate JSON file.
    Provides persistence across process restarts.

    Example:
        >>> store = FileIdempotencyStore("/var/lib/truthound/idempotency")
        >>> record = IdempotencyRecord(key="req-123")
        >>> store.set(record)
    """

    def __init__(
        self,
        storage_path: str | Path,
        use_subdirs: bool = True,
        subdir_depth: int = 2,
    ) -> None:
        """Initialize the store.

        Args:
            storage_path: Base directory for storage.
            use_subdirs: Use subdirectories to avoid too many files in one dir.
            subdir_depth: Number of subdirectory levels.
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._use_subdirs = use_subdirs
        self._subdir_depth = subdir_depth

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        hashed = hashlib.sha256(key.encode()).hexdigest()

        if self._use_subdirs:
            # Create subdirectory structure
            parts = [hashed[i:i+2] for i in range(0, self._subdir_depth * 2, 2)]
            subdir = self._storage_path.joinpath(*parts)
            subdir.mkdir(parents=True, exist_ok=True)
            return subdir / f"{hashed}.json"
        else:
            return self._storage_path / f"{hashed}.json"

    def get(self, key: str) -> IdempotencyRecord[Any] | None:
        with self._lock:
            path = self._key_to_path(key)
            if not path.exists():
                return None

            try:
                data = json.loads(path.read_text())
                record = IdempotencyRecord.from_dict(data)

                if record.is_expired:
                    path.unlink()
                    return None

                return record
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to read record {key}: {e}")
                return None

    def set(self, record: IdempotencyRecord[Any]) -> None:
        with self._lock:
            path = self._key_to_path(record.key)
            data = record.to_dict()
            path.write_text(json.dumps(data, indent=2, default=str))

    def update(self, record: IdempotencyRecord[Any]) -> bool:
        with self._lock:
            path = self._key_to_path(record.key)
            if not path.exists():
                return False
            self.set(record)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            path = self._key_to_path(key)
            if path.exists():
                path.unlink()
                return True
            return False

    def exists(self, key: str) -> bool:
        with self._lock:
            path = self._key_to_path(key)
            if not path.exists():
                return False

            # Check expiry
            try:
                data = json.loads(path.read_text())
                if data.get("expires_at"):
                    expires = datetime.fromisoformat(data["expires_at"])
                    if expires < datetime.now():
                        path.unlink()
                        return False
                return True
            except Exception:
                return False

    def cleanup_expired(self) -> int:
        with self._lock:
            removed = 0
            now = datetime.now()

            for path in self._storage_path.rglob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    if data.get("expires_at"):
                        expires = datetime.fromisoformat(data["expires_at"])
                        if expires < now:
                            path.unlink()
                            removed += 1
                except Exception:
                    # Remove corrupted files
                    path.unlink()
                    removed += 1

            return removed

    def size(self) -> int:
        """Get number of stored records."""
        return len(list(self._storage_path.rglob("*.json")))

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            for path in self._storage_path.rglob("*.json"):
                path.unlink()


class SQLIdempotencyStore:
    """SQLite-based implementation of IdempotencyStore.

    Provides efficient storage with SQL query capabilities.
    Supports atomic operations and index-based lookups.

    Example:
        >>> store = SQLIdempotencyStore("idempotency.db")
        >>> record = IdempotencyRecord(key="req-123")
        >>> store.set(record)
    """

    CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS idempotency_records (
            key TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            request_hash TEXT,
            result TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT,
            attempt_count INTEGER DEFAULT 0,
            locked_by TEXT,
            locked_at TEXT,
            metadata TEXT
        )
    """

    CREATE_INDEX_SQL = """
        CREATE INDEX IF NOT EXISTS idx_expires_at ON idempotency_records(expires_at);
        CREATE INDEX IF NOT EXISTS idx_status ON idempotency_records(status);
    """

    def __init__(
        self,
        database_path: str | Path = ":memory:",
        pool_size: int = 5,
    ) -> None:
        """Initialize the store.

        Args:
            database_path: Path to SQLite database or ":memory:".
            pool_size: Connection pool size.
        """
        self._database_path = str(database_path)
        self._pool_size = pool_size
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection for the current thread."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                self._database_path,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row

        if not self._initialized:
            with self._init_lock:
                if not self._initialized:
                    self._init_database(self._local.connection)
                    self._initialized = True

        return self._local.connection

    def _init_database(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema."""
        conn.executescript(self.CREATE_TABLE_SQL)
        conn.executescript(self.CREATE_INDEX_SQL)
        conn.commit()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get(self, key: str) -> IdempotencyRecord[Any] | None:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM idempotency_records WHERE key = ?",
            (key,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        record = self._row_to_record(row)

        if record.is_expired:
            self.delete(key)
            return None

        return record

    def set(self, record: IdempotencyRecord[Any]) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO idempotency_records
                (key, status, request_hash, result, error, created_at, updated_at,
                 expires_at, attempt_count, locked_by, locked_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._record_to_tuple(record),
            )

    def update(self, record: IdempotencyRecord[Any]) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE idempotency_records SET
                    status = ?, request_hash = ?, result = ?, error = ?,
                    updated_at = ?, expires_at = ?, attempt_count = ?,
                    locked_by = ?, locked_at = ?, metadata = ?
                WHERE key = ?
                """,
                (
                    record.status.value,
                    record.request_hash,
                    json.dumps(record._serialize_result()) if record.result else None,
                    record.error,
                    record.updated_at.isoformat(),
                    record.expires_at.isoformat() if record.expires_at else None,
                    record.attempt_count,
                    record.locked_by,
                    record.locked_at.isoformat() if record.locked_at else None,
                    json.dumps(record.metadata),
                    record.key,
                ),
            )
            return cursor.rowcount > 0

    def delete(self, key: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM idempotency_records WHERE key = ?",
                (key,),
            )
            return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM idempotency_records WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
            (key, datetime.now().isoformat()),
        )
        return cursor.fetchone() is not None

    def cleanup_expired(self) -> int:
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM idempotency_records WHERE expires_at < ?",
                (datetime.now().isoformat(),),
            )
            return cursor.rowcount

    def find_by_status(
        self,
        status: IdempotencyStatus,
        limit: int = 100,
    ) -> list[IdempotencyRecord[Any]]:
        """Find records by status.

        Args:
            status: Status to filter by.
            limit: Maximum records to return.

        Returns:
            List of matching records.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM idempotency_records WHERE status = ? LIMIT ?",
            (status.value, limit),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def find_stale_locks(
        self,
        max_lock_age_seconds: int = 300,
    ) -> list[IdempotencyRecord[Any]]:
        """Find records with stale locks.

        Args:
            max_lock_age_seconds: Maximum age of lock before considered stale.

        Returns:
            List of records with stale locks.
        """
        conn = self._get_connection()
        cutoff = datetime.now().isoformat()
        cursor = conn.execute(
            """
            SELECT * FROM idempotency_records
            WHERE locked_by IS NOT NULL
            AND locked_at < datetime(?, '-' || ? || ' seconds')
            """,
            (cutoff, max_lock_age_seconds),
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def release_stale_locks(self, max_lock_age_seconds: int = 300) -> int:
        """Release stale locks.

        Args:
            max_lock_age_seconds: Maximum age of lock before release.

        Returns:
            Number of locks released.
        """
        with self._transaction() as conn:
            cutoff = datetime.now().isoformat()
            cursor = conn.execute(
                """
                UPDATE idempotency_records
                SET locked_by = NULL, locked_at = NULL, status = 'failed',
                    error = 'Lock expired', updated_at = ?
                WHERE locked_by IS NOT NULL
                AND locked_at < datetime(?, '-' || ? || ' seconds')
                """,
                (cutoff, cutoff, max_lock_age_seconds),
            )
            return cursor.rowcount

    def size(self) -> int:
        """Get total record count."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM idempotency_records")
        return cursor.fetchone()[0]

    def clear(self) -> None:
        """Clear all records."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM idempotency_records")

    def _record_to_tuple(self, record: IdempotencyRecord[Any]) -> tuple[Any, ...]:
        """Convert record to tuple for SQL."""
        return (
            record.key,
            record.status.value,
            record.request_hash,
            json.dumps(record._serialize_result()) if record.result else None,
            record.error,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            record.expires_at.isoformat() if record.expires_at else None,
            record.attempt_count,
            record.locked_by,
            record.locked_at.isoformat() if record.locked_at else None,
            json.dumps(record.metadata),
        )

    def _row_to_record(self, row: sqlite3.Row) -> IdempotencyRecord[Any]:
        """Convert SQL row to record."""
        return IdempotencyRecord(
            key=row["key"],
            status=IdempotencyStatus(row["status"]),
            request_hash=row["request_hash"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=(
                datetime.fromisoformat(row["expires_at"])
                if row["expires_at"]
                else None
            ),
            attempt_count=row["attempt_count"],
            locked_by=row["locked_by"],
            locked_at=(
                datetime.fromisoformat(row["locked_at"])
                if row["locked_at"]
                else None
            ),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
