"""Tests for escalation policy store implementations."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from truthound.checkpoint.escalation.protocols import EscalationRecord
from truthound.checkpoint.escalation.stores import (
    InMemoryEscalationStore,
    SQLiteEscalationStore,
    create_store,
)


class TestInMemoryEscalationStore:
    """Tests for InMemoryEscalationStore."""

    def test_save_and_get(self) -> None:
        """Test saving and retrieving a record."""
        store = InMemoryEscalationStore()
        record = EscalationRecord.create("inc-1", "policy-1")

        saved = store.save(record)
        retrieved = store.get(record.id)

        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.incident_id == "inc-1"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent record."""
        store = InMemoryEscalationStore()
        result = store.get("nonexistent")
        assert result is None

    def test_get_by_incident(self) -> None:
        """Test getting records by incident ID."""
        store = InMemoryEscalationStore()

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record2 = EscalationRecord.create("inc-1", "policy-2")
        record3 = EscalationRecord.create("inc-2", "policy-1")

        store.save(record1)
        store.save(record2)
        store.save(record3)

        records = store.get_by_incident("inc-1")
        assert len(records) == 2
        assert all(r.incident_id == "inc-1" for r in records)

    def test_get_active(self) -> None:
        """Test getting active escalations."""
        store = InMemoryEscalationStore()

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "active"

        record2 = EscalationRecord.create("inc-2", "policy-1")
        record2.state = "resolved"

        record3 = EscalationRecord.create("inc-3", "policy-2")
        record3.state = "pending"

        store.save(record1)
        store.save(record2)
        store.save(record3)

        active = store.get_active()
        assert len(active) == 2

        active_policy1 = store.get_active("policy-1")
        assert len(active_policy1) == 1
        assert active_policy1[0].incident_id == "inc-1"

    def test_get_pending_escalations(self) -> None:
        """Test getting pending escalations."""
        store = InMemoryEscalationStore()
        now = datetime.now()

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "active"
        record1.next_escalation_at = now - timedelta(minutes=5)

        record2 = EscalationRecord.create("inc-2", "policy-1")
        record2.state = "active"
        record2.next_escalation_at = now + timedelta(minutes=5)

        store.save(record1)
        store.save(record2)

        pending = store.get_pending_escalations(now)
        assert len(pending) == 1
        assert pending[0].incident_id == "inc-1"

    def test_delete(self) -> None:
        """Test deleting a record."""
        store = InMemoryEscalationStore()
        record = EscalationRecord.create("inc-1", "policy-1")
        store.save(record)

        assert store.delete(record.id)
        assert store.get(record.id) is None
        assert not store.delete(record.id)  # Second delete returns False

    def test_cleanup_resolved(self) -> None:
        """Test cleaning up old resolved records."""
        store = InMemoryEscalationStore()

        # Old resolved record
        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "resolved"
        record1.resolved_at = datetime.now() - timedelta(days=2)
        store.save(record1)

        # Recent resolved record
        record2 = EscalationRecord.create("inc-2", "policy-1")
        record2.state = "resolved"
        record2.resolved_at = datetime.now()
        store.save(record2)

        # Active record
        record3 = EscalationRecord.create("inc-3", "policy-1")
        record3.state = "active"
        store.save(record3)

        removed = store.cleanup_resolved(timedelta(days=1))
        assert removed == 1
        assert store.get(record1.id) is None
        assert store.get(record2.id) is not None
        assert store.get(record3.id) is not None

    def test_get_stats(self) -> None:
        """Test getting store statistics."""
        store = InMemoryEscalationStore()

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "active"
        record1.current_level = 1
        store.save(record1)

        record2 = EscalationRecord.create("inc-2", "policy-1")
        record2.state = "acknowledged"
        record2.acknowledged_at = datetime.now()
        record2.current_level = 2
        store.save(record2)

        record3 = EscalationRecord.create("inc-3", "policy-2")
        record3.state = "resolved"
        record3.resolved_at = datetime.now()
        record3.current_level = 1
        store.save(record3)

        stats = store.get_stats()
        assert stats.total_escalations == 3
        assert stats.active_escalations == 1
        assert stats.acknowledged_count == 1
        assert stats.resolved_count == 1
        assert stats.escalations_by_level[1] == 2
        assert stats.escalations_by_level[2] == 1
        assert stats.escalations_by_policy["policy-1"] == 2
        assert stats.escalations_by_policy["policy-2"] == 1

    def test_clear(self) -> None:
        """Test clearing all records."""
        store = InMemoryEscalationStore()

        for i in range(5):
            record = EscalationRecord.create(f"inc-{i}", "policy-1")
            store.save(record)

        store.clear()
        assert len(store.get_active()) == 0


class TestSQLiteEscalationStore:
    """Tests for SQLiteEscalationStore."""

    def test_save_and_get(self) -> None:
        """Test saving and retrieving a record."""
        store = SQLiteEscalationStore(":memory:")
        record = EscalationRecord.create("inc-1", "policy-1")
        record.context = {"severity": "critical"}

        saved = store.save(record)
        retrieved = store.get(record.id)

        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.incident_id == "inc-1"
        assert retrieved.context["severity"] == "critical"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent record."""
        store = SQLiteEscalationStore(":memory:")
        result = store.get("nonexistent")
        assert result is None

    def test_get_by_incident(self) -> None:
        """Test getting records by incident ID."""
        store = SQLiteEscalationStore(":memory:")

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record2 = EscalationRecord.create("inc-1", "policy-2")
        record3 = EscalationRecord.create("inc-2", "policy-1")

        store.save(record1)
        store.save(record2)
        store.save(record3)

        records = store.get_by_incident("inc-1")
        assert len(records) == 2

    def test_get_active(self) -> None:
        """Test getting active escalations."""
        store = SQLiteEscalationStore(":memory:")

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "active"
        store.save(record1)

        record2 = EscalationRecord.create("inc-2", "policy-1")
        record2.state = "resolved"
        store.save(record2)

        active = store.get_active()
        assert len(active) == 1
        assert active[0].incident_id == "inc-1"

    def test_get_pending_escalations(self) -> None:
        """Test getting pending escalations."""
        store = SQLiteEscalationStore(":memory:")
        now = datetime.now()

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "active"
        record1.next_escalation_at = now - timedelta(minutes=5)
        store.save(record1)

        record2 = EscalationRecord.create("inc-2", "policy-1")
        record2.state = "active"
        record2.next_escalation_at = now + timedelta(minutes=5)
        store.save(record2)

        pending = store.get_pending_escalations(now)
        assert len(pending) == 1
        assert pending[0].incident_id == "inc-1"

    def test_update_record(self) -> None:
        """Test updating an existing record."""
        store = SQLiteEscalationStore(":memory:")

        record = EscalationRecord.create("inc-1", "policy-1")
        record.state = "active"
        store.save(record)

        record.state = "acknowledged"
        record.acknowledged_at = datetime.now()
        record.acknowledged_by = "user-123"
        store.save(record)

        retrieved = store.get(record.id)
        assert retrieved is not None
        assert retrieved.state == "acknowledged"
        assert retrieved.acknowledged_by == "user-123"

    def test_delete(self) -> None:
        """Test deleting a record."""
        store = SQLiteEscalationStore(":memory:")
        record = EscalationRecord.create("inc-1", "policy-1")
        store.save(record)

        assert store.delete(record.id)
        assert store.get(record.id) is None

    def test_cleanup_resolved(self) -> None:
        """Test cleaning up old resolved records."""
        store = SQLiteEscalationStore(":memory:")

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "resolved"
        record1.resolved_at = datetime.now() - timedelta(days=2)
        store.save(record1)

        record2 = EscalationRecord.create("inc-2", "policy-1")
        record2.state = "active"
        store.save(record2)

        removed = store.cleanup_resolved(timedelta(days=1))
        assert removed == 1

    def test_get_stats(self) -> None:
        """Test getting store statistics."""
        store = SQLiteEscalationStore(":memory:")

        record1 = EscalationRecord.create("inc-1", "policy-1")
        record1.state = "active"
        store.save(record1)

        record2 = EscalationRecord.create("inc-2", "policy-2")
        record2.state = "resolved"
        record2.resolved_at = datetime.now()
        store.save(record2)

        stats = store.get_stats()
        assert stats.total_escalations == 2
        assert stats.active_escalations == 1
        assert stats.resolved_count == 1

    def test_clear(self) -> None:
        """Test clearing all records."""
        store = SQLiteEscalationStore(":memory:")

        for i in range(3):
            record = EscalationRecord.create(f"inc-{i}", "policy-1")
            store.save(record)

        store.clear()
        stats = store.get_stats()
        assert stats.total_escalations == 0

    def test_record_with_history(self) -> None:
        """Test saving record with history."""
        store = SQLiteEscalationStore(":memory:")

        record = EscalationRecord.create("inc-1", "policy-1")
        record.add_history_event("created", {"by": "system"})
        record.add_history_event("notified", {"target": "user-1"})
        store.save(record)

        retrieved = store.get(record.id)
        assert retrieved is not None
        assert len(retrieved.history) == 2
        assert retrieved.history[0]["event_type"] == "created"


class TestCreateStore:
    """Tests for store factory function."""

    def test_create_memory_store(self) -> None:
        """Test creating memory store."""
        store = create_store("memory")
        assert isinstance(store, InMemoryEscalationStore)

    def test_create_sqlite_store(self) -> None:
        """Test creating SQLite store."""
        store = create_store("sqlite", database=":memory:")
        assert isinstance(store, SQLiteEscalationStore)

    def test_create_unknown_store(self) -> None:
        """Test creating unknown store type."""
        with pytest.raises(ValueError, match="Unknown store type"):
            create_store("unknown")
