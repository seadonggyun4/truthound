"""Escalation Policy Store Implementations.

This module provides storage backends for escalation records
with support for in-memory, Redis, and SQLite.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

from truthound.checkpoint.escalation.protocols import (
    BaseEscalationStore,
    EscalationRecord,
    EscalationStats,
)

logger = logging.getLogger(__name__)


class InMemoryEscalationStore(BaseEscalationStore):
    """Thread-safe in-memory store for escalation records.

    Suitable for single-process deployments and testing.

    Example:
        >>> store = InMemoryEscalationStore()
        >>> record = EscalationRecord.create("incident-1", "policy-1")
        >>> store.save(record)
        >>> store.get(record.id)
    """

    def __init__(self, name: str = "memory") -> None:
        """Initialize in-memory store."""
        super().__init__(name)
        self._records: dict[str, EscalationRecord] = {}
        self._by_incident: dict[str, set[str]] = {}
        self._by_policy: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def get(self, record_id: str) -> EscalationRecord | None:
        """Get a record by ID."""
        with self._lock:
            return self._records.get(record_id)

    def get_by_incident(self, incident_id: str) -> list[EscalationRecord]:
        """Get all records for an incident."""
        with self._lock:
            record_ids = self._by_incident.get(incident_id, set())
            return [self._records[rid] for rid in record_ids if rid in self._records]

    def get_active(self, policy_name: str | None = None) -> list[EscalationRecord]:
        """Get all active escalations."""
        with self._lock:
            if policy_name:
                record_ids = self._by_policy.get(policy_name, set())
                records = [self._records[rid] for rid in record_ids if rid in self._records]
            else:
                records = list(self._records.values())

            return [r for r in records if r.is_active]

    def get_pending_escalations(self, before: datetime) -> list[EscalationRecord]:
        """Get records with pending escalations before a timestamp."""
        with self._lock:
            result = []
            for record in self._records.values():
                if (
                    record.is_active
                    and record.next_escalation_at
                    and record.next_escalation_at <= before
                ):
                    result.append(record)
            return result

    def save(self, record: EscalationRecord) -> EscalationRecord:
        """Save a record."""
        with self._lock:
            is_new = record.id not in self._records

            self._records[record.id] = record

            # Update indices
            if record.incident_id not in self._by_incident:
                self._by_incident[record.incident_id] = set()
            self._by_incident[record.incident_id].add(record.id)

            if record.policy_name not in self._by_policy:
                self._by_policy[record.policy_name] = set()
            self._by_policy[record.policy_name].add(record.id)

            self._update_stats_on_save(record, is_new)

            return record

    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        with self._lock:
            record = self._records.pop(record_id, None)
            if not record:
                return False

            # Clean up indices
            if record.incident_id in self._by_incident:
                self._by_incident[record.incident_id].discard(record_id)
            if record.policy_name in self._by_policy:
                self._by_policy[record.policy_name].discard(record_id)

            return True

    def cleanup_resolved(self, older_than: timedelta) -> int:
        """Remove resolved records older than threshold."""
        with self._lock:
            threshold = datetime.now() - older_than
            to_delete = []

            for record_id, record in self._records.items():
                if record.is_resolved and record.resolved_at and record.resolved_at < threshold:
                    to_delete.append(record_id)

            for record_id in to_delete:
                self.delete(record_id)

            return len(to_delete)

    def get_stats(self) -> EscalationStats:
        """Get store statistics."""
        with self._lock:
            stats = EscalationStats(
                total_escalations=len(self._records),
                active_escalations=len(self.get_active()),
            )

            for record in self._records.values():
                if record.is_acknowledged:
                    stats.acknowledged_count += 1
                if record.is_resolved:
                    stats.resolved_count += 1
                if record.state == "timed_out":
                    stats.timed_out_count += 1
                if record.state == "cancelled":
                    stats.cancelled_count += 1

                # Count by level
                level_count = stats.escalations_by_level.get(record.current_level, 0)
                stats.escalations_by_level[record.current_level] = level_count + 1

                # Count by policy
                policy_count = stats.escalations_by_policy.get(record.policy_name, 0)
                stats.escalations_by_policy[record.policy_name] = policy_count + 1

            return stats

    def clear(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()
            self._by_incident.clear()
            self._by_policy.clear()


class SQLiteEscalationStore(BaseEscalationStore):
    """SQLite-based persistent store for escalation records.

    Provides durable storage with SQL query capabilities.

    Example:
        >>> store = SQLiteEscalationStore("escalations.db")
        >>> record = EscalationRecord.create("incident-1", "policy-1")
        >>> store.save(record)
    """

    def __init__(
        self,
        database: str = ":memory:",
        name: str = "sqlite",
    ) -> None:
        """Initialize SQLite store.

        Args:
            database: Database file path or ":memory:".
            name: Store name.
        """
        super().__init__(name)
        self._database = database
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                self._database,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        """Get a cursor with automatic commit/rollback."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalation_records (
                    id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    current_level INTEGER DEFAULT 1,
                    state TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    next_escalation_at TEXT,
                    escalation_count INTEGER DEFAULT 0,
                    notification_count INTEGER DEFAULT 0,
                    history TEXT DEFAULT '[]',
                    context TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}'
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_incident_id
                ON escalation_records(incident_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_policy_name
                ON escalation_records(policy_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_state
                ON escalation_records(state)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_next_escalation
                ON escalation_records(next_escalation_at)
            """)

    def _row_to_record(self, row: sqlite3.Row) -> EscalationRecord:
        """Convert a database row to an EscalationRecord."""
        return EscalationRecord(
            id=row["id"],
            incident_id=row["incident_id"],
            policy_name=row["policy_name"],
            current_level=row["current_level"],
            state=row["state"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            acknowledged_at=datetime.fromisoformat(row["acknowledged_at"])
            if row["acknowledged_at"]
            else None,
            acknowledged_by=row["acknowledged_by"],
            resolved_at=datetime.fromisoformat(row["resolved_at"])
            if row["resolved_at"]
            else None,
            resolved_by=row["resolved_by"],
            next_escalation_at=datetime.fromisoformat(row["next_escalation_at"])
            if row["next_escalation_at"]
            else None,
            escalation_count=row["escalation_count"],
            notification_count=row["notification_count"],
            history=json.loads(row["history"]),
            context=json.loads(row["context"]),
            metadata=json.loads(row["metadata"]),
        )

    def get(self, record_id: str) -> EscalationRecord | None:
        """Get a record by ID."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM escalation_records WHERE id = ?",
                (record_id,),
            )
            row = cursor.fetchone()
            return self._row_to_record(row) if row else None

    def get_by_incident(self, incident_id: str) -> list[EscalationRecord]:
        """Get all records for an incident."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM escalation_records WHERE incident_id = ?",
                (incident_id,),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_active(self, policy_name: str | None = None) -> list[EscalationRecord]:
        """Get all active escalations."""
        with self._cursor() as cursor:
            if policy_name:
                cursor.execute(
                    """SELECT * FROM escalation_records
                       WHERE state IN ('pending', 'active', 'escalating')
                       AND policy_name = ?""",
                    (policy_name,),
                )
            else:
                cursor.execute(
                    """SELECT * FROM escalation_records
                       WHERE state IN ('pending', 'active', 'escalating')"""
                )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_pending_escalations(self, before: datetime) -> list[EscalationRecord]:
        """Get records with pending escalations before a timestamp."""
        with self._cursor() as cursor:
            cursor.execute(
                """SELECT * FROM escalation_records
                   WHERE state IN ('pending', 'active', 'escalating')
                   AND next_escalation_at IS NOT NULL
                   AND next_escalation_at <= ?""",
                (before.isoformat(),),
            )
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def save(self, record: EscalationRecord) -> EscalationRecord:
        """Save a record."""
        with self._cursor() as cursor:
            cursor.execute(
                """INSERT OR REPLACE INTO escalation_records (
                    id, incident_id, policy_name, current_level, state,
                    created_at, updated_at, acknowledged_at, acknowledged_by,
                    resolved_at, resolved_by, next_escalation_at,
                    escalation_count, notification_count, history, context, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.incident_id,
                    record.policy_name,
                    record.current_level,
                    record.state,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.acknowledged_at.isoformat() if record.acknowledged_at else None,
                    record.acknowledged_by,
                    record.resolved_at.isoformat() if record.resolved_at else None,
                    record.resolved_by,
                    record.next_escalation_at.isoformat()
                    if record.next_escalation_at
                    else None,
                    record.escalation_count,
                    record.notification_count,
                    json.dumps(record.history),
                    json.dumps(record.context),
                    json.dumps(record.metadata),
                ),
            )
        return record

    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        with self._cursor() as cursor:
            cursor.execute(
                "DELETE FROM escalation_records WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    def cleanup_resolved(self, older_than: timedelta) -> int:
        """Remove resolved records older than threshold."""
        threshold = datetime.now() - older_than
        with self._cursor() as cursor:
            cursor.execute(
                """DELETE FROM escalation_records
                   WHERE state = 'resolved'
                   AND resolved_at IS NOT NULL
                   AND resolved_at < ?""",
                (threshold.isoformat(),),
            )
            return cursor.rowcount

    def get_stats(self) -> EscalationStats:
        """Get store statistics."""
        with self._cursor() as cursor:
            stats = EscalationStats()

            # Total count
            cursor.execute("SELECT COUNT(*) FROM escalation_records")
            stats.total_escalations = cursor.fetchone()[0]

            # Active count
            cursor.execute(
                """SELECT COUNT(*) FROM escalation_records
                   WHERE state IN ('pending', 'active', 'escalating')"""
            )
            stats.active_escalations = cursor.fetchone()[0]

            # Acknowledged count
            cursor.execute(
                "SELECT COUNT(*) FROM escalation_records WHERE acknowledged_at IS NOT NULL"
            )
            stats.acknowledged_count = cursor.fetchone()[0]

            # Resolved count
            cursor.execute("SELECT COUNT(*) FROM escalation_records WHERE state = 'resolved'")
            stats.resolved_count = cursor.fetchone()[0]

            # Timed out count
            cursor.execute("SELECT COUNT(*) FROM escalation_records WHERE state = 'timed_out'")
            stats.timed_out_count = cursor.fetchone()[0]

            # Cancelled count
            cursor.execute("SELECT COUNT(*) FROM escalation_records WHERE state = 'cancelled'")
            stats.cancelled_count = cursor.fetchone()[0]

            # By level
            cursor.execute(
                """SELECT current_level, COUNT(*) as count
                   FROM escalation_records
                   GROUP BY current_level"""
            )
            for row in cursor.fetchall():
                stats.escalations_by_level[row[0]] = row[1]

            # By policy
            cursor.execute(
                """SELECT policy_name, COUNT(*) as count
                   FROM escalation_records
                   GROUP BY policy_name"""
            )
            for row in cursor.fetchall():
                stats.escalations_by_policy[row[0]] = row[1]

            return stats

    def clear(self) -> None:
        """Clear all records."""
        with self._cursor() as cursor:
            cursor.execute("DELETE FROM escalation_records")


class RedisEscalationStore(BaseEscalationStore):
    """Redis-based distributed store for escalation records.

    Suitable for multi-process and distributed deployments.
    Requires: pip install redis

    Example:
        >>> store = RedisEscalationStore(redis_url="redis://localhost:6379/0")
        >>> record = EscalationRecord.create("incident-1", "policy-1")
        >>> store.save(record)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "escalation:",
        name: str = "redis",
    ) -> None:
        """Initialize Redis store.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for all keys.
            name: Store name.
        """
        super().__init__(name)
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Ensure Redis client is connected."""
        if self._client is None:
            try:
                import redis

                self._client = redis.from_url(self._redis_url)
            except ImportError:
                raise RuntimeError("Redis not available. Install with: pip install redis")
        return self._client

    def _record_key(self, record_id: str) -> str:
        """Get Redis key for a record."""
        return f"{self._key_prefix}record:{record_id}"

    def _incident_key(self, incident_id: str) -> str:
        """Get Redis key for incident index."""
        return f"{self._key_prefix}incident:{incident_id}"

    def _policy_key(self, policy_name: str) -> str:
        """Get Redis key for policy index."""
        return f"{self._key_prefix}policy:{policy_name}"

    def _active_key(self) -> str:
        """Get Redis key for active records set."""
        return f"{self._key_prefix}active"

    def _pending_key(self) -> str:
        """Get Redis key for pending escalations sorted set."""
        return f"{self._key_prefix}pending"

    def get(self, record_id: str) -> EscalationRecord | None:
        """Get a record by ID."""
        client = self._ensure_client()
        data = client.get(self._record_key(record_id))
        if data:
            return EscalationRecord.from_dict(json.loads(data))
        return None

    def get_by_incident(self, incident_id: str) -> list[EscalationRecord]:
        """Get all records for an incident."""
        client = self._ensure_client()
        record_ids = client.smembers(self._incident_key(incident_id))
        records = []
        for record_id in record_ids:
            record = self.get(record_id.decode() if isinstance(record_id, bytes) else record_id)
            if record:
                records.append(record)
        return records

    def get_active(self, policy_name: str | None = None) -> list[EscalationRecord]:
        """Get all active escalations."""
        client = self._ensure_client()

        if policy_name:
            # Intersection of active and policy sets
            record_ids = client.sinter(
                self._active_key(),
                self._policy_key(policy_name),
            )
        else:
            record_ids = client.smembers(self._active_key())

        records = []
        for record_id in record_ids:
            record = self.get(record_id.decode() if isinstance(record_id, bytes) else record_id)
            if record and record.is_active:
                records.append(record)

        return records

    def get_pending_escalations(self, before: datetime) -> list[EscalationRecord]:
        """Get records with pending escalations before a timestamp."""
        client = self._ensure_client()
        score = before.timestamp()

        record_ids = client.zrangebyscore(
            self._pending_key(),
            "-inf",
            score,
        )

        records = []
        for record_id in record_ids:
            record = self.get(record_id.decode() if isinstance(record_id, bytes) else record_id)
            if record and record.is_active:
                records.append(record)

        return records

    def save(self, record: EscalationRecord) -> EscalationRecord:
        """Save a record."""
        client = self._ensure_client()
        pipe = client.pipeline()

        # Save record
        pipe.set(
            self._record_key(record.id),
            json.dumps(record.to_dict()),
        )

        # Update indices
        pipe.sadd(self._incident_key(record.incident_id), record.id)
        pipe.sadd(self._policy_key(record.policy_name), record.id)

        # Update active set
        if record.is_active:
            pipe.sadd(self._active_key(), record.id)
        else:
            pipe.srem(self._active_key(), record.id)

        # Update pending sorted set
        if record.next_escalation_at and record.is_active:
            pipe.zadd(
                self._pending_key(),
                {record.id: record.next_escalation_at.timestamp()},
            )
        else:
            pipe.zrem(self._pending_key(), record.id)

        pipe.execute()

        return record

    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        client = self._ensure_client()
        record = self.get(record_id)
        if not record:
            return False

        pipe = client.pipeline()

        pipe.delete(self._record_key(record_id))
        pipe.srem(self._incident_key(record.incident_id), record_id)
        pipe.srem(self._policy_key(record.policy_name), record_id)
        pipe.srem(self._active_key(), record_id)
        pipe.zrem(self._pending_key(), record_id)

        pipe.execute()

        return True

    def cleanup_resolved(self, older_than: timedelta) -> int:
        """Remove resolved records older than threshold."""
        client = self._ensure_client()
        threshold = datetime.now() - older_than
        removed = 0

        # Scan all records
        for key in client.scan_iter(f"{self._key_prefix}record:*"):
            data = client.get(key)
            if data:
                record = EscalationRecord.from_dict(json.loads(data))
                if (
                    record.is_resolved
                    and record.resolved_at
                    and record.resolved_at < threshold
                ):
                    if self.delete(record.id):
                        removed += 1

        return removed

    def get_stats(self) -> EscalationStats:
        """Get store statistics."""
        client = self._ensure_client()
        stats = EscalationStats()

        # Count all records
        stats.total_escalations = 0
        stats.active_escalations = client.scard(self._active_key())

        # Scan for detailed stats
        for key in client.scan_iter(f"{self._key_prefix}record:*"):
            stats.total_escalations += 1
            data = client.get(key)
            if data:
                record = EscalationRecord.from_dict(json.loads(data))

                if record.is_acknowledged:
                    stats.acknowledged_count += 1
                if record.is_resolved:
                    stats.resolved_count += 1
                if record.state == "timed_out":
                    stats.timed_out_count += 1
                if record.state == "cancelled":
                    stats.cancelled_count += 1

                level_count = stats.escalations_by_level.get(record.current_level, 0)
                stats.escalations_by_level[record.current_level] = level_count + 1

                policy_count = stats.escalations_by_policy.get(record.policy_name, 0)
                stats.escalations_by_policy[record.policy_name] = policy_count + 1

        return stats

    def clear(self) -> None:
        """Clear all records."""
        client = self._ensure_client()

        # Delete all keys with prefix
        for key in client.scan_iter(f"{self._key_prefix}*"):
            client.delete(key)


def create_store(
    store_type: str = "memory",
    **kwargs: Any,
) -> BaseEscalationStore:
    """Factory function to create a store.

    Args:
        store_type: Type of store (memory, sqlite, redis).
        **kwargs: Store-specific configuration.

    Returns:
        Appropriate store implementation.
    """
    if store_type == "memory":
        return InMemoryEscalationStore(**kwargs)
    elif store_type == "sqlite":
        return SQLiteEscalationStore(**kwargs)
    elif store_type == "redis":
        return RedisEscalationStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
