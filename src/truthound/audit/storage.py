"""Storage backends for audit logging.

This module provides various storage implementations for persisting
audit events:
- Memory: Fast, in-process storage for development/testing
- File: JSON/JSONL file-based storage
- SQLite: Local database storage
- SQL: Generic SQL database storage (PostgreSQL, MySQL, etc.)
- Elasticsearch: Full-text search capable storage
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import threading
from abc import ABC
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from truthound.audit.core import (
    AuditEvent,
    AuditEventType,
    AuditOutcome,
    AuditStorage,
    AuditStorageError,
    current_timestamp,
)


# =============================================================================
# Memory Storage
# =============================================================================


class MemoryAuditStorage(AuditStorage):
    """In-memory storage for audit events.

    Suitable for development, testing, and short-lived processes.

    Example:
        >>> storage = MemoryAuditStorage(max_events=10000)
        >>> storage.write(event)
        >>> events = storage.query(limit=10)
    """

    def __init__(
        self,
        max_events: int = 10000,
        auto_expire: bool = True,
    ) -> None:
        """Initialize memory storage.

        Args:
            max_events: Maximum events to keep.
            auto_expire: Automatically remove oldest events when full.
        """
        self._events: deque[AuditEvent] = deque(maxlen=max_events if auto_expire else None)
        self._index: dict[str, AuditEvent] = {}
        self._lock = threading.Lock()
        self._max_events = max_events
        self._auto_expire = auto_expire

    def write(self, event: AuditEvent) -> None:
        """Write a single audit event."""
        with self._lock:
            # Remove from index if we're at capacity and auto-expiring
            if self._auto_expire and len(self._events) >= self._max_events:
                old_event = self._events[0]
                self._index.pop(old_event.id, None)

            self._events.append(event)
            self._index[event.id] = event

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write multiple audit events."""
        with self._lock:
            for event in events:
                if self._auto_expire and len(self._events) >= self._max_events:
                    old_event = self._events[0]
                    self._index.pop(old_event.id, None)

                self._events.append(event)
                self._index[event.id] = event

    def read(self, event_id: str) -> AuditEvent | None:
        """Read a single audit event by ID."""
        with self._lock:
            return self._index.get(event_id)

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query audit events."""
        with self._lock:
            results = []

            for event in reversed(self._events):  # Most recent first
                if self._matches_filters(
                    event,
                    start_time,
                    end_time,
                    event_types,
                    actor_id,
                    resource_id,
                    outcome,
                ):
                    results.append(event)

            # Apply offset and limit
            return results[offset : offset + limit]

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count matching audit events."""
        with self._lock:
            count = 0
            for event in self._events:
                if self._matches_filters(
                    event,
                    start_time,
                    end_time,
                    event_types,
                    None,
                    None,
                    None,
                ):
                    count += 1
            return count

    def delete_before(self, before: datetime) -> int:
        """Delete events before a given time."""
        with self._lock:
            deleted = 0
            events_to_keep = []

            for event in self._events:
                if event.timestamp >= before:
                    events_to_keep.append(event)
                else:
                    self._index.pop(event.id, None)
                    deleted += 1

            self._events.clear()
            for event in events_to_keep:
                self._events.append(event)

            return deleted

    def clear(self) -> None:
        """Clear all events."""
        with self._lock:
            self._events.clear()
            self._index.clear()

    def _matches_filters(
        self,
        event: AuditEvent,
        start_time: datetime | None,
        end_time: datetime | None,
        event_types: list[AuditEventType] | None,
        actor_id: str | None,
        resource_id: str | None,
        outcome: AuditOutcome | None,
    ) -> bool:
        """Check if event matches all filters."""
        if start_time and event.timestamp < start_time:
            return False
        if end_time and event.timestamp > end_time:
            return False
        if event_types and event.event_type not in event_types:
            return False
        if actor_id and (not event.actor or event.actor.id != actor_id):
            return False
        if resource_id and (not event.resource or event.resource.id != resource_id):
            return False
        if outcome and event.outcome != outcome:
            return False
        return True


# =============================================================================
# File Storage
# =============================================================================


@dataclass
class FileStorageConfig:
    """Configuration for file-based audit storage."""

    path: str = "./audit_logs"
    filename_pattern: str = "audit_{date}.jsonl"
    compress: bool = False
    max_file_size_mb: int = 100
    rotate_daily: bool = True
    encoding: str = "utf-8"


class FileAuditStorage(AuditStorage):
    """File-based storage for audit events.

    Supports JSONL format with optional compression and rotation.

    Example:
        >>> storage = FileAuditStorage(
        ...     config=FileStorageConfig(
        ...         path="./audit_logs",
        ...         compress=True,
        ...     )
        ... )
    """

    def __init__(self, config: FileStorageConfig | None = None) -> None:
        """Initialize file storage.

        Args:
            config: Storage configuration.
        """
        self._config = config or FileStorageConfig()
        self._lock = threading.Lock()
        self._current_file: Any = None
        self._current_filename: str = ""

        # Ensure directory exists
        Path(self._config.path).mkdir(parents=True, exist_ok=True)

    def write(self, event: AuditEvent) -> None:
        """Write a single audit event."""
        with self._lock:
            self._ensure_file_open()
            line = json.dumps(event.to_dict()) + "\n"
            self._current_file.write(line)
            self._current_file.flush()

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write multiple audit events."""
        with self._lock:
            self._ensure_file_open()
            for event in events:
                line = json.dumps(event.to_dict()) + "\n"
                self._current_file.write(line)
            self._current_file.flush()

    def read(self, event_id: str) -> AuditEvent | None:
        """Read a single audit event by ID."""
        # Search through all files
        for filepath in self._get_all_files():
            try:
                for event in self._read_file(filepath):
                    if event.id == event_id:
                        return event
            except Exception:
                continue
        return None

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query audit events."""
        results = []
        skipped = 0

        for filepath in reversed(self._get_all_files()):  # Most recent first
            if len(results) >= limit:
                break

            try:
                for event in self._read_file(filepath):
                    if self._matches_filters(
                        event,
                        start_time,
                        end_time,
                        event_types,
                        actor_id,
                        resource_id,
                        outcome,
                    ):
                        if skipped < offset:
                            skipped += 1
                        else:
                            results.append(event)
                            if len(results) >= limit:
                                break
            except Exception:
                continue

        return results

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count matching audit events."""
        count = 0

        for filepath in self._get_all_files():
            try:
                for event in self._read_file(filepath):
                    if self._matches_filters(
                        event,
                        start_time,
                        end_time,
                        event_types,
                        None,
                        None,
                        None,
                    ):
                        count += 1
            except Exception:
                continue

        return count

    def delete_before(self, before: datetime) -> int:
        """Delete events before a given time."""
        deleted = 0

        for filepath in self._get_all_files():
            # Check if entire file is before cutoff
            try:
                file_date = self._extract_date_from_filename(filepath)
                if file_date and file_date.date() < before.date():
                    os.remove(filepath)
                    deleted += 1  # Count as 1 per file
            except Exception:
                continue

        return deleted

    def close(self) -> None:
        """Close storage."""
        with self._lock:
            if self._current_file:
                self._current_file.close()
                self._current_file = None

    def flush(self) -> None:
        """Flush buffered data."""
        with self._lock:
            if self._current_file:
                self._current_file.flush()

    def _ensure_file_open(self) -> None:
        """Ensure current file is open."""
        filename = self._get_current_filename()

        if filename != self._current_filename:
            if self._current_file:
                self._current_file.close()

            filepath = os.path.join(self._config.path, filename)

            if self._config.compress:
                self._current_file = gzip.open(
                    filepath + ".gz",
                    "at",
                    encoding=self._config.encoding,
                )
            else:
                self._current_file = open(
                    filepath,
                    "a",
                    encoding=self._config.encoding,
                )

            self._current_filename = filename

    def _get_current_filename(self) -> str:
        """Get current filename based on pattern."""
        now = current_timestamp()
        return self._config.filename_pattern.format(
            date=now.strftime("%Y-%m-%d"),
            datetime=now.strftime("%Y-%m-%d_%H"),
        )

    def _get_all_files(self) -> list[str]:
        """Get all audit log files sorted by date."""
        path = Path(self._config.path)
        files = []

        for f in path.glob("audit_*.jsonl*"):
            files.append(str(f))

        return sorted(files)

    def _read_file(self, filepath: str) -> Iterator[AuditEvent]:
        """Read events from a file."""
        opener = gzip.open if filepath.endswith(".gz") else open

        with opener(filepath, "rt", encoding=self._config.encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        yield AuditEvent.from_dict(data)
                    except (json.JSONDecodeError, TypeError):
                        continue

    def _extract_date_from_filename(self, filepath: str) -> datetime | None:
        """Extract date from filename."""
        import re

        match = re.search(r"audit_(\d{4}-\d{2}-\d{2})", filepath)
        if match:
            return datetime.strptime(match.group(1), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        return None

    def _matches_filters(
        self,
        event: AuditEvent,
        start_time: datetime | None,
        end_time: datetime | None,
        event_types: list[AuditEventType] | None,
        actor_id: str | None,
        resource_id: str | None,
        outcome: AuditOutcome | None,
    ) -> bool:
        """Check if event matches all filters."""
        if start_time and event.timestamp < start_time:
            return False
        if end_time and event.timestamp > end_time:
            return False
        if event_types and event.event_type not in event_types:
            return False
        if actor_id and (not event.actor or event.actor.id != actor_id):
            return False
        if resource_id and (not event.resource or event.resource.id != resource_id):
            return False
        if outcome and event.outcome != outcome:
            return False
        return True


# =============================================================================
# SQLite Storage
# =============================================================================


class SQLiteAuditStorage(AuditStorage):
    """SQLite-based storage for audit events.

    Provides persistent local storage with SQL query capabilities.

    Example:
        >>> storage = SQLiteAuditStorage("./audit.db")
        >>> storage.write(event)
    """

    def __init__(
        self,
        db_path: str = "./audit.db",
        *,
        create_indexes: bool = True,
    ) -> None:
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file.
            create_indexes: Create indexes for common queries.
        """
        self._db_path = db_path
        self._lock = threading.Lock()
        self._local = threading.local()

        # Initialize schema
        self._init_schema(create_indexes)

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self, create_indexes: bool) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                timestamp_unix REAL NOT NULL,
                event_type TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                action TEXT,
                outcome TEXT NOT NULL,
                message TEXT,
                reason TEXT,
                actor_id TEXT,
                actor_type TEXT,
                actor_name TEXT,
                actor_ip TEXT,
                resource_id TEXT,
                resource_type TEXT,
                resource_name TEXT,
                target_id TEXT,
                context_request_id TEXT,
                context_trace_id TEXT,
                context_environment TEXT,
                data_json TEXT,
                tags_json TEXT,
                duration_ms REAL,
                checksum TEXT
            )
        """)

        if create_indexes:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp_unix)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_actor_id ON audit_events(actor_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_resource_id ON audit_events(resource_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_outcome ON audit_events(outcome)"
            )

        conn.commit()

    def write(self, event: AuditEvent) -> None:
        """Write a single audit event."""
        with self._lock:
            conn = self._get_connection()
            self._insert_event(conn, event)
            conn.commit()

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write multiple audit events."""
        with self._lock:
            conn = self._get_connection()
            for event in events:
                self._insert_event(conn, event)
            conn.commit()

    def _insert_event(self, conn: sqlite3.Connection, event: AuditEvent) -> None:
        """Insert a single event."""
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO audit_events (
                id, timestamp, timestamp_unix, event_type, category, severity,
                action, outcome, message, reason,
                actor_id, actor_type, actor_name, actor_ip,
                resource_id, resource_type, resource_name, target_id,
                context_request_id, context_trace_id, context_environment,
                data_json, tags_json, duration_ms, checksum
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.id,
                event.timestamp_iso,
                event.timestamp_unix,
                event.event_type.value,
                event.category.value,
                event.severity.value,
                event.action,
                event.outcome.value,
                event.message,
                event.reason,
                event.actor.id if event.actor else None,
                event.actor.type if event.actor else None,
                event.actor.name if event.actor else None,
                event.actor.ip_address if event.actor else None,
                event.resource.id if event.resource else None,
                event.resource.type if event.resource else None,
                event.resource.name if event.resource else None,
                event.target.id if event.target else None,
                event.context.request_id,
                event.context.trace_id,
                event.context.environment,
                json.dumps(event.data),
                json.dumps(event.tags),
                event.duration_ms,
                event.compute_checksum(),
            ),
        )

    def read(self, event_id: str) -> AuditEvent | None:
        """Read a single audit event by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_event(row)
        return None

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query audit events."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM audit_events WHERE 1=1"
        params: list[Any] = []

        if start_time:
            query += " AND timestamp_unix >= ?"
            params.append(start_time.timestamp())
        if end_time:
            query += " AND timestamp_unix <= ?"
            params.append(end_time.timestamp())
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(et.value for et in event_types)
        if actor_id:
            query += " AND actor_id = ?"
            params.append(actor_id)
        if resource_id:
            query += " AND resource_id = ?"
            params.append(resource_id)
        if outcome:
            query += " AND outcome = ?"
            params.append(outcome.value)

        query += " ORDER BY timestamp_unix DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count matching audit events."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT COUNT(*) FROM audit_events WHERE 1=1"
        params: list[Any] = []

        if start_time:
            query += " AND timestamp_unix >= ?"
            params.append(start_time.timestamp())
        if end_time:
            query += " AND timestamp_unix <= ?"
            params.append(end_time.timestamp())
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(et.value for et in event_types)

        cursor.execute(query, params)
        return cursor.fetchone()[0]

    def delete_before(self, before: datetime) -> int:
        """Delete events before a given time."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM audit_events WHERE timestamp_unix < ?",
                (before.timestamp(),),
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def close(self) -> None:
        """Close storage."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn

    def _row_to_event(self, row: sqlite3.Row) -> AuditEvent:
        """Convert database row to AuditEvent."""
        from truthound.audit.core import (
            AuditActor,
            AuditResource,
            AuditContext,
        )

        actor = None
        if row["actor_id"]:
            actor = AuditActor(
                id=row["actor_id"],
                type=row["actor_type"] or "user",
                name=row["actor_name"] or "",
                ip_address=row["actor_ip"] or "",
            )

        resource = None
        if row["resource_id"]:
            resource = AuditResource(
                id=row["resource_id"],
                type=row["resource_type"] or "",
                name=row["resource_name"] or "",
            )

        target = None
        if row["target_id"]:
            target = AuditResource(
                id=row["target_id"],
                type="",
                name="",
            )

        context = AuditContext(
            request_id=row["context_request_id"] or "",
            trace_id=row["context_trace_id"] or "",
            environment=row["context_environment"] or "",
        )

        timestamp = datetime.fromisoformat(
            row["timestamp"].replace("Z", "+00:00")
        )

        return AuditEvent(
            id=row["id"],
            timestamp=timestamp,
            event_type=AuditEventType(row["event_type"]),
            category=AuditCategory(row["category"]),
            severity=AuditSeverity(row["severity"]),
            action=row["action"] or "",
            outcome=AuditOutcome(row["outcome"]),
            message=row["message"] or "",
            reason=row["reason"] or "",
            actor=actor,
            resource=resource,
            target=target,
            context=context,
            data=json.loads(row["data_json"]) if row["data_json"] else {},
            tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
            duration_ms=row["duration_ms"],
            checksum=row["checksum"] or "",
        )


# Need to import these for the _row_to_event method
from truthound.audit.core import AuditCategory, AuditSeverity


# =============================================================================
# Async File Storage
# =============================================================================


class AsyncBufferedStorage(AuditStorage):
    """Buffered storage that writes asynchronously.

    Wraps any storage backend with buffering for improved performance.

    Example:
        >>> base_storage = FileAuditStorage(config)
        >>> storage = AsyncBufferedStorage(
        ...     base_storage,
        ...     buffer_size=100,
        ...     flush_interval=5.0,
        ... )
    """

    def __init__(
        self,
        storage: AuditStorage,
        buffer_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        """Initialize async buffered storage.

        Args:
            storage: Underlying storage backend.
            buffer_size: Maximum buffer size before auto-flush.
            flush_interval: Seconds between auto-flushes.
        """
        self._storage = storage
        self._buffer: list[AuditEvent] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None

        # Start flush thread
        self._start_flush_thread()

    def _start_flush_thread(self) -> None:
        """Start background flush thread."""
        def flush_loop() -> None:
            while not self._stop_event.wait(self._flush_interval):
                self.flush()

        self._flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self._flush_thread.start()

    def write(self, event: AuditEvent) -> None:
        """Write event to buffer."""
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self._buffer_size:
                self._flush_internal()

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write events to buffer."""
        with self._lock:
            self._buffer.extend(events)
            if len(self._buffer) >= self._buffer_size:
                self._flush_internal()

    def read(self, event_id: str) -> AuditEvent | None:
        """Read from underlying storage."""
        # Check buffer first
        with self._lock:
            for event in self._buffer:
                if event.id == event_id:
                    return event

        return self._storage.read(event_id)

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query from underlying storage."""
        # Flush buffer first to ensure complete results
        self.flush()
        return self._storage.query(
            start_time,
            end_time,
            event_types,
            actor_id,
            resource_id,
            outcome,
            limit,
            offset,
        )

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count from underlying storage."""
        self.flush()
        return self._storage.count(start_time, end_time, event_types)

    def delete_before(self, before: datetime) -> int:
        """Delete from underlying storage."""
        self.flush()
        return self._storage.delete_before(before)

    def flush(self) -> None:
        """Flush buffer to storage."""
        with self._lock:
            self._flush_internal()

    def _flush_internal(self) -> None:
        """Internal flush without lock."""
        if self._buffer:
            try:
                self._storage.write_batch(self._buffer)
                self._buffer = []
            except Exception as e:
                # Keep events in buffer on failure
                raise AuditStorageError(f"Failed to flush buffer: {e}") from e

    def close(self) -> None:
        """Close storage."""
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=self._flush_interval * 2)
        self.flush()
        self._storage.close()


# =============================================================================
# Composite Storage
# =============================================================================


class CompositeAuditStorage(AuditStorage):
    """Storage that writes to multiple backends.

    Useful for writing to both fast (memory) and persistent (file/db) storage.

    Example:
        >>> storage = CompositeAuditStorage([
        ...     MemoryAuditStorage(max_events=1000),
        ...     SQLiteAuditStorage("./audit.db"),
        ... ])
    """

    def __init__(
        self,
        storages: list[AuditStorage],
        *,
        primary_index: int = 0,
        fail_fast: bool = False,
    ) -> None:
        """Initialize composite storage.

        Args:
            storages: List of storage backends.
            primary_index: Index of primary storage for reads.
            fail_fast: Fail if any storage fails.
        """
        self._storages = storages
        self._primary_index = primary_index
        self._fail_fast = fail_fast

    @property
    def primary(self) -> AuditStorage:
        """Get primary storage."""
        return self._storages[self._primary_index]

    def write(self, event: AuditEvent) -> None:
        """Write to all storages."""
        errors = []
        for storage in self._storages:
            try:
                storage.write(event)
            except Exception as e:
                if self._fail_fast:
                    raise
                errors.append(e)

        if errors and len(errors) == len(self._storages):
            raise AuditStorageError(f"All storages failed: {errors}")

    def write_batch(self, events: list[AuditEvent]) -> None:
        """Write batch to all storages."""
        errors = []
        for storage in self._storages:
            try:
                storage.write_batch(events)
            except Exception as e:
                if self._fail_fast:
                    raise
                errors.append(e)

        if errors and len(errors) == len(self._storages):
            raise AuditStorageError(f"All storages failed: {errors}")

    def read(self, event_id: str) -> AuditEvent | None:
        """Read from primary storage."""
        return self.primary.read(event_id)

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
        actor_id: str | None = None,
        resource_id: str | None = None,
        outcome: AuditOutcome | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEvent]:
        """Query from primary storage."""
        return self.primary.query(
            start_time,
            end_time,
            event_types,
            actor_id,
            resource_id,
            outcome,
            limit,
            offset,
        )

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[AuditEventType] | None = None,
    ) -> int:
        """Count from primary storage."""
        return self.primary.count(start_time, end_time, event_types)

    def delete_before(self, before: datetime) -> int:
        """Delete from all storages."""
        max_deleted = 0
        for storage in self._storages:
            try:
                deleted = storage.delete_before(before)
                max_deleted = max(max_deleted, deleted)
            except Exception:
                if self._fail_fast:
                    raise
        return max_deleted

    def flush(self) -> None:
        """Flush all storages."""
        for storage in self._storages:
            storage.flush()

    def close(self) -> None:
        """Close all storages."""
        for storage in self._storages:
            storage.close()


# =============================================================================
# Storage Factory
# =============================================================================


def create_storage(
    backend: str = "memory",
    **kwargs: Any,
) -> AuditStorage:
    """Create storage backend from configuration.

    Args:
        backend: Storage backend type.
        **kwargs: Backend-specific configuration.

    Returns:
        Storage instance.

    Example:
        >>> storage = create_storage("memory", max_events=10000)
        >>> storage = create_storage("file", path="./audit_logs")
        >>> storage = create_storage("sqlite", db_path="./audit.db")
    """
    if backend == "memory":
        return MemoryAuditStorage(**kwargs)
    elif backend == "file":
        config = FileStorageConfig(**kwargs)
        return FileAuditStorage(config)
    elif backend == "sqlite":
        return SQLiteAuditStorage(**kwargs)
    else:
        raise ValueError(f"Unknown storage backend: {backend}")
