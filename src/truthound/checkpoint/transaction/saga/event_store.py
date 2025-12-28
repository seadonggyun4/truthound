"""Saga Event Store Module.

This module provides event sourcing capabilities for saga execution,
enabling replay, recovery, and auditing of saga transactions.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from truthound.checkpoint.transaction.saga.state_machine import (
    SagaEvent,
    SagaEventType,
    SagaState,
    SagaStateMachine,
)


logger = logging.getLogger(__name__)


@dataclass
class SagaSnapshot:
    """Snapshot of saga state at a point in time.

    Attributes:
        saga_id: Saga identifier.
        state: Current saga state.
        version: Snapshot version (event count).
        timestamp: When snapshot was taken.
        step_states: State of each step.
        current_step: Current step being executed.
        completed_steps: List of completed step IDs.
        compensated_steps: List of compensated step IDs.
        metadata: Additional snapshot metadata.
    """

    saga_id: str
    state: SagaState
    version: int
    timestamp: datetime = field(default_factory=datetime.now)
    step_states: dict[str, str] = field(default_factory=dict)
    current_step: str | None = None
    completed_steps: list[str] = field(default_factory=list)
    compensated_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "saga_id": self.saga_id,
            "state": self.state.value,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "step_states": self.step_states,
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "compensated_steps": self.compensated_steps,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SagaSnapshot":
        """Create from dictionary."""
        return cls(
            saga_id=data["saga_id"],
            state=SagaState(data["state"]),
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            step_states=data.get("step_states", {}),
            current_step=data.get("current_step"),
            completed_steps=data.get("completed_steps", []),
            compensated_steps=data.get("compensated_steps", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_state_machine(cls, machine: SagaStateMachine) -> "SagaSnapshot":
        """Create snapshot from state machine.

        Args:
            machine: State machine to snapshot.

        Returns:
            Snapshot of current state.
        """
        return cls(
            saga_id=machine.saga_id,
            state=machine.state,
            version=len(machine.events),
            current_step=machine.current_step,
            step_states={k: v.value for k, v in machine._step_states.items()},
            completed_steps=machine.get_completed_steps(),
            compensated_steps=machine.get_compensated_steps(),
            metadata=machine._metadata.copy(),
        )


class SagaEventStore(ABC):
    """Abstract base class for saga event stores.

    Event stores provide persistence and querying of saga events,
    enabling:
    - Event sourcing and replay
    - Saga recovery after failures
    - Audit trails
    - Analytics and monitoring
    """

    @abstractmethod
    def append(self, event: SagaEvent) -> None:
        """Append an event to the store.

        Args:
            event: Event to append.
        """
        pass

    @abstractmethod
    def get_events(
        self,
        saga_id: str,
        from_version: int = 0,
    ) -> list[SagaEvent]:
        """Get all events for a saga.

        Args:
            saga_id: Saga identifier.
            from_version: Start from this event version (0-indexed).

        Returns:
            List of events.
        """
        pass

    @abstractmethod
    def get_latest_snapshot(self, saga_id: str) -> SagaSnapshot | None:
        """Get the latest snapshot for a saga.

        Args:
            saga_id: Saga identifier.

        Returns:
            Latest snapshot or None if not found.
        """
        pass

    @abstractmethod
    def save_snapshot(self, snapshot: SagaSnapshot) -> None:
        """Save a snapshot.

        Args:
            snapshot: Snapshot to save.
        """
        pass

    def replay(
        self,
        saga_id: str,
        to_version: int | None = None,
    ) -> SagaStateMachine:
        """Replay events to reconstruct saga state.

        Args:
            saga_id: Saga identifier.
            to_version: Replay up to this version (None = all events).

        Returns:
            Reconstructed state machine.
        """
        events = self.get_events(saga_id)

        if to_version is not None:
            events = events[:to_version]

        return SagaStateMachine.from_events(saga_id, events)

    def replay_from_snapshot(self, saga_id: str) -> SagaStateMachine:
        """Replay from latest snapshot plus subsequent events.

        More efficient than replaying all events for long-running sagas.

        Args:
            saga_id: Saga identifier.

        Returns:
            Reconstructed state machine.
        """
        snapshot = self.get_latest_snapshot(saga_id)

        if snapshot is None:
            return self.replay(saga_id)

        # Create machine from snapshot
        machine = SagaStateMachine(saga_id, initial_state=snapshot.state)
        machine._step_states = {
            k: SagaState(v) for k, v in snapshot.step_states.items()
        }
        machine._current_step = snapshot.current_step
        machine._metadata = snapshot.metadata.copy()

        # Replay events after snapshot
        events = self.get_events(saga_id, from_version=snapshot.version)
        for event in events:
            machine._events.append(event)
            if event.target_state:
                machine._state = event.target_state
            if event.step_id:
                machine._current_step = event.step_id
                machine._step_states[event.step_id] = (
                    event.target_state or machine._state
                )

        return machine

    @abstractmethod
    def list_sagas(
        self,
        state: SagaState | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """List saga IDs, optionally filtered by state.

        Args:
            state: Filter by saga state.
            limit: Maximum number of results.
            offset: Skip this many results.

        Returns:
            List of saga IDs.
        """
        pass

    @abstractmethod
    def delete_saga(self, saga_id: str) -> bool:
        """Delete all data for a saga.

        Args:
            saga_id: Saga identifier.

        Returns:
            True if saga was deleted.
        """
        pass


class InMemorySagaEventStore(SagaEventStore):
    """In-memory implementation of saga event store.

    Suitable for testing and single-process applications.
    Data is lost on restart.
    """

    def __init__(self, max_events_per_saga: int = 10000) -> None:
        """Initialize in-memory store.

        Args:
            max_events_per_saga: Maximum events to store per saga.
        """
        self._max_events = max_events_per_saga
        self._events: dict[str, list[SagaEvent]] = {}
        self._snapshots: dict[str, list[SagaSnapshot]] = {}
        self._lock = threading.RLock()

    def append(self, event: SagaEvent) -> None:
        """Append an event to the store."""
        with self._lock:
            if event.saga_id not in self._events:
                self._events[event.saga_id] = []

            events = self._events[event.saga_id]
            events.append(event)

            # Trim if exceeds max
            if len(events) > self._max_events:
                # Take a snapshot before trimming
                machine = SagaStateMachine.from_events(event.saga_id, events)
                snapshot = SagaSnapshot.from_state_machine(machine)
                self.save_snapshot(snapshot)

                # Keep only recent events
                trim_point = len(events) - (self._max_events // 2)
                self._events[event.saga_id] = events[trim_point:]

    def get_events(
        self,
        saga_id: str,
        from_version: int = 0,
    ) -> list[SagaEvent]:
        """Get events for a saga."""
        with self._lock:
            events = self._events.get(saga_id, [])
            return list(events[from_version:])

    def get_latest_snapshot(self, saga_id: str) -> SagaSnapshot | None:
        """Get latest snapshot for a saga."""
        with self._lock:
            snapshots = self._snapshots.get(saga_id, [])
            return snapshots[-1] if snapshots else None

    def save_snapshot(self, snapshot: SagaSnapshot) -> None:
        """Save a snapshot."""
        with self._lock:
            if snapshot.saga_id not in self._snapshots:
                self._snapshots[snapshot.saga_id] = []
            self._snapshots[snapshot.saga_id].append(snapshot)

            # Keep only last 5 snapshots per saga
            if len(self._snapshots[snapshot.saga_id]) > 5:
                self._snapshots[snapshot.saga_id] = self._snapshots[snapshot.saga_id][-5:]

    def list_sagas(
        self,
        state: SagaState | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """List saga IDs."""
        with self._lock:
            saga_ids = list(self._events.keys())

            if state is not None:
                # Filter by state
                filtered = []
                for saga_id in saga_ids:
                    events = self._events[saga_id]
                    if events:
                        last_state = events[-1].target_state
                        if last_state == state:
                            filtered.append(saga_id)
                saga_ids = filtered

            return saga_ids[offset : offset + limit]

    def delete_saga(self, saga_id: str) -> bool:
        """Delete all data for a saga."""
        with self._lock:
            deleted = saga_id in self._events
            self._events.pop(saga_id, None)
            self._snapshots.pop(saga_id, None)
            return deleted

    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._events.clear()
            self._snapshots.clear()


class FileSagaEventStore(SagaEventStore):
    """File-based implementation of saga event store.

    Stores events as JSONL files and snapshots as JSON files.
    Suitable for single-node production deployments.
    """

    def __init__(
        self,
        base_path: str | Path,
        snapshot_interval: int = 100,
    ) -> None:
        """Initialize file-based store.

        Args:
            base_path: Base directory for storage.
            snapshot_interval: Take snapshot every N events.
        """
        self._base_path = Path(base_path)
        self._snapshot_interval = snapshot_interval
        self._events_dir = self._base_path / "events"
        self._snapshots_dir = self._base_path / "snapshots"
        self._lock = threading.RLock()

        # Ensure directories exist
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _events_file(self, saga_id: str) -> Path:
        """Get path to events file for a saga."""
        return self._events_dir / f"{saga_id}.jsonl"

    def _snapshot_file(self, saga_id: str) -> Path:
        """Get path to snapshot file for a saga."""
        return self._snapshots_dir / f"{saga_id}.json"

    def append(self, event: SagaEvent) -> None:
        """Append an event to the store."""
        with self._lock:
            events_file = self._events_file(event.saga_id)

            # Append event as JSONL
            with open(events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")

            # Check if snapshot needed
            event_count = sum(1 for _ in open(events_file, encoding="utf-8"))
            if event_count % self._snapshot_interval == 0:
                machine = self.replay(event.saga_id)
                snapshot = SagaSnapshot.from_state_machine(machine)
                self.save_snapshot(snapshot)

    def get_events(
        self,
        saga_id: str,
        from_version: int = 0,
    ) -> list[SagaEvent]:
        """Get events for a saga."""
        with self._lock:
            events_file = self._events_file(saga_id)

            if not events_file.exists():
                return []

            events = []
            with open(events_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= from_version:
                        event_data = json.loads(line.strip())
                        events.append(SagaEvent.from_dict(event_data))

            return events

    def get_latest_snapshot(self, saga_id: str) -> SagaSnapshot | None:
        """Get latest snapshot for a saga."""
        with self._lock:
            snapshot_file = self._snapshot_file(saga_id)

            if not snapshot_file.exists():
                return None

            with open(snapshot_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return SagaSnapshot.from_dict(data)

    def save_snapshot(self, snapshot: SagaSnapshot) -> None:
        """Save a snapshot."""
        with self._lock:
            snapshot_file = self._snapshot_file(snapshot.saga_id)

            with open(snapshot_file, "w", encoding="utf-8") as f:
                json.dump(snapshot.to_dict(), f, indent=2)

    def list_sagas(
        self,
        state: SagaState | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[str]:
        """List saga IDs."""
        with self._lock:
            saga_ids = [
                f.stem for f in self._events_dir.glob("*.jsonl")
            ]

            if state is not None:
                filtered = []
                for saga_id in saga_ids:
                    snapshot = self.get_latest_snapshot(saga_id)
                    if snapshot and snapshot.state == state:
                        filtered.append(saga_id)
                    elif not snapshot:
                        # Check last event
                        events = self.get_events(saga_id)
                        if events and events[-1].target_state == state:
                            filtered.append(saga_id)
                saga_ids = filtered

            return saga_ids[offset : offset + limit]

    def delete_saga(self, saga_id: str) -> bool:
        """Delete all data for a saga."""
        with self._lock:
            events_file = self._events_file(saga_id)
            snapshot_file = self._snapshot_file(saga_id)

            deleted = events_file.exists()

            if events_file.exists():
                events_file.unlink()
            if snapshot_file.exists():
                snapshot_file.unlink()

            return deleted

    def compact(self, saga_id: str) -> None:
        """Compact event log by taking snapshot and removing old events.

        Args:
            saga_id: Saga identifier.
        """
        with self._lock:
            events = self.get_events(saga_id)

            if len(events) < self._snapshot_interval * 2:
                return

            # Take snapshot at current state
            machine = SagaStateMachine.from_events(saga_id, events)
            snapshot = SagaSnapshot.from_state_machine(machine)
            self.save_snapshot(snapshot)

            # Keep only events after snapshot
            events_file = self._events_file(saga_id)
            with open(events_file, "w", encoding="utf-8") as f:
                # Write nothing - events up to snapshot version are compacted
                pass

            logger.info(f"Compacted saga {saga_id}: {len(events)} events → snapshot")

    def iter_events(
        self,
        saga_id: str,
        from_version: int = 0,
    ) -> Iterator[SagaEvent]:
        """Iterate over events without loading all into memory.

        Args:
            saga_id: Saga identifier.
            from_version: Start from this version.

        Yields:
            Events one by one.
        """
        events_file = self._events_file(saga_id)

        if not events_file.exists():
            return

        with open(events_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= from_version:
                    event_data = json.loads(line.strip())
                    yield SagaEvent.from_dict(event_data)
