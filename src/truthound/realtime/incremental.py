"""Incremental validation for streaming data.

Provides stateful validation that accumulates results
across batches and supports checkpointing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar
import json
import threading
from pathlib import Path
from collections import defaultdict
import hashlib

import polars as pl

from truthound.realtime.base import (
    StreamingConfig,
    BatchResult,
    WindowResult,
    WindowType,
    WindowConfig,
)


# =============================================================================
# State Store Interface
# =============================================================================


StateT = TypeVar("StateT")


class StateStore(ABC, Generic[StateT]):
    """Abstract base class for state storage.

    Provides interface for storing and retrieving validation state.
    """

    @abstractmethod
    def get(self, key: str) -> StateT | None:
        """Get state by key."""
        ...

    @abstractmethod
    def set(self, key: str, value: StateT) -> None:
        """Set state for key."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete state for key."""
        ...

    @abstractmethod
    def keys(self) -> list[str]:
        """Get all keys."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all state."""
        ...


class MemoryStateStore(StateStore[Any]):
    """In-memory state store.

    Simple thread-safe in-memory storage.
    Suitable for testing and small-scale use.
    """

    def __init__(self):
        self._state: dict[str, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._state.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            if key in self._state:
                del self._state[key]

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._state.keys())

    def clear(self) -> None:
        with self._lock:
            self._state.clear()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def from_dict(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._state = dict(data)


# =============================================================================
# Checkpoint Manager
# =============================================================================


@dataclass
class Checkpoint:
    """A validation checkpoint.

    Stores the state at a specific point in time
    for recovery purposes.
    """

    checkpoint_id: str
    created_at: datetime
    batch_count: int
    total_records: int
    total_issues: int
    state_snapshot: dict[str, Any]
    position: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "batch_count": self.batch_count,
            "total_records": self.total_records,
            "total_issues": self.total_issues,
            "state_snapshot": self.state_snapshot,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            batch_count=data["batch_count"],
            total_records=data["total_records"],
            total_issues=data["total_issues"],
            state_snapshot=data.get("state_snapshot", {}),
            position=data.get("position", {}),
        )


class CheckpointManager:
    """Manage validation checkpoints.

    Provides methods to:
    - Create checkpoints
    - Restore from checkpoints
    - List available checkpoints
    - Clean up old checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        max_checkpoints: int = 10,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum checkpoints to keep
        """
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._max_checkpoints = max_checkpoints
        self._checkpoints: list[Checkpoint] = []
        self._lock = threading.RLock()

        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._load_checkpoints()

    def _load_checkpoints(self) -> None:
        """Load existing checkpoints from disk."""
        if not self._checkpoint_dir:
            return

        for file in sorted(self._checkpoint_dir.glob("checkpoint_*.json")):
            try:
                with open(file) as f:
                    data = json.load(f)
                    self._checkpoints.append(Checkpoint.from_dict(data))
            except Exception:
                pass

    def create_checkpoint(
        self,
        state: StateStore,
        batch_count: int,
        total_records: int,
        total_issues: int,
        position: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            state: Current state store
            batch_count: Number of batches processed
            total_records: Total records processed
            total_issues: Total issues found
            position: Stream position for recovery

        Returns:
            Created checkpoint
        """
        import uuid

        checkpoint = Checkpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            created_at=datetime.now(),
            batch_count=batch_count,
            total_records=total_records,
            total_issues=total_issues,
            state_snapshot=state.to_dict() if hasattr(state, "to_dict") else {},
            position=position or {},
        )

        with self._lock:
            self._checkpoints.append(checkpoint)

            # Save to disk
            if self._checkpoint_dir:
                filename = f"checkpoint_{checkpoint.checkpoint_id}.json"
                with open(self._checkpoint_dir / filename, "w") as f:
                    json.dump(checkpoint.to_dict(), f, indent=2)

            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()

        return checkpoint

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max limit."""
        while len(self._checkpoints) > self._max_checkpoints:
            oldest = self._checkpoints.pop(0)

            if self._checkpoint_dir:
                filename = f"checkpoint_{oldest.checkpoint_id}.json"
                filepath = self._checkpoint_dir / filename
                if filepath.exists():
                    filepath.unlink()

    def get_latest(self) -> Checkpoint | None:
        """Get the most recent checkpoint."""
        with self._lock:
            if self._checkpoints:
                return self._checkpoints[-1]
            return None

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a checkpoint by ID."""
        with self._lock:
            for cp in self._checkpoints:
                if cp.checkpoint_id == checkpoint_id:
                    return cp
            return None

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints."""
        with self._lock:
            return list(self._checkpoints)

    def restore(
        self,
        checkpoint_id: str,
        state: StateStore,
    ) -> Checkpoint | None:
        """Restore state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore
            state: State store to restore into

        Returns:
            Restored checkpoint or None
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return None

        if hasattr(state, "from_dict"):
            state.from_dict(checkpoint.state_snapshot)

        return checkpoint


# =============================================================================
# Incremental Validator
# =============================================================================


@dataclass
class IncrementalState:
    """State for incremental validation.

    Tracks aggregated statistics across batches.
    """

    column_null_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    column_value_counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    column_min: dict[str, float] = field(default_factory=dict)
    column_max: dict[str, float] = field(default_factory=dict)
    column_sum: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    column_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    issue_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    seen_hashes: set[str] = field(default_factory=set)


class IncrementalValidator:
    """Stateful incremental validation.

    Accumulates validation state across batches for:
    - Running aggregates (min, max, sum, count)
    - Duplicate detection across batches
    - Trend analysis
    - Anomaly detection

    Example:
        >>> validator = IncrementalValidator(
        ...     validators=["null", "range"],
        ...     checkpoint_interval=100,
        ... )
        >>>
        >>> for batch in data_stream:
        ...     result = validator.validate_batch(batch)
        ...     if validator.batch_count % 100 == 0:
        ...         validator.checkpoint()
    """

    def __init__(
        self,
        validators: list[str] | None = None,
        config: StreamingConfig | None = None,
        window_config: WindowConfig | None = None,
        state_store: StateStore | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        track_duplicates: bool = False,
        duplicate_columns: list[str] | None = None,
    ):
        """Initialize incremental validator.

        Args:
            validators: List of validators to use
            config: Streaming configuration
            window_config: Window configuration
            state_store: State store for persistence
            checkpoint_manager: Checkpoint manager
            track_duplicates: Whether to track cross-batch duplicates
            duplicate_columns: Columns to use for duplicate detection
        """
        self._validators = validators or []
        self._config = config or StreamingConfig()
        self._window_config = window_config or WindowConfig()
        self._state_store = state_store or MemoryStateStore()
        self._checkpoint_manager = checkpoint_manager
        self._track_duplicates = track_duplicates
        self._duplicate_columns = duplicate_columns

        self._state = IncrementalState()
        self._batch_count = 0
        self._total_records = 0
        self._total_issues = 0
        self._current_window: WindowResult | None = None
        self._window_batches: list[BatchResult] = []
        self._window_start: datetime | None = None
        self._lock = threading.RLock()

    @property
    def batch_count(self) -> int:
        return self._batch_count

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def total_issues(self) -> int:
        return self._total_issues

    def validate_batch(
        self,
        batch: pl.DataFrame,
        batch_id: str | None = None,
    ) -> BatchResult:
        """Validate a batch and update incremental state.

        Args:
            batch: DataFrame batch
            batch_id: Optional batch identifier

        Returns:
            BatchResult with validation results
        """
        import time
        import uuid

        start = time.perf_counter()
        batch_id = batch_id or str(uuid.uuid4())[:8]

        # Update column statistics
        self._update_column_stats(batch)

        # Check for cross-batch duplicates
        duplicate_count = 0
        if self._track_duplicates:
            duplicate_count = self._check_duplicates(batch)

        # Run validators
        from truthound.api import check

        issues = []
        try:
            report = check(batch, validators=self._validators)
            issues = list(report.issues)
        except Exception:
            pass

        # Update issue counts
        for issue in issues:
            self._state.issue_counts[issue.issue_type] += issue.count

        elapsed = (time.perf_counter() - start) * 1000

        result = BatchResult(
            batch_id=batch_id,
            record_count=len(batch),
            issue_count=len(issues),
            issues=issues,
            processing_time_ms=elapsed,
            metadata={
                "duplicate_count": duplicate_count,
                "cumulative_records": self._total_records + len(batch),
            },
        )

        # Update counters
        with self._lock:
            self._batch_count += 1
            self._total_records += len(batch)
            self._total_issues += len(issues)
            self._window_batches.append(result)

        # Check window completion
        self._check_window_completion()

        return result

    def _update_column_stats(self, batch: pl.DataFrame) -> None:
        """Update running column statistics."""
        schema = batch.schema
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }

        for col in batch.columns:
            # Null counts
            null_count = batch[col].null_count()
            self._state.column_null_counts[col] += null_count

            # Numeric statistics
            if type(schema[col]) in numeric_types:
                stats = batch.select([
                    pl.col(col).min().alias("min"),
                    pl.col(col).max().alias("max"),
                    pl.col(col).sum().alias("sum"),
                    pl.col(col).count().alias("count"),
                ]).row(0)

                if stats[0] is not None:
                    if col not in self._state.column_min or stats[0] < self._state.column_min[col]:
                        self._state.column_min[col] = stats[0]
                if stats[1] is not None:
                    if col not in self._state.column_max or stats[1] > self._state.column_max[col]:
                        self._state.column_max[col] = stats[1]
                if stats[2] is not None:
                    self._state.column_sum[col] += stats[2]
                if stats[3] is not None:
                    self._state.column_count[col] += stats[3]

    def _check_duplicates(self, batch: pl.DataFrame) -> int:
        """Check for duplicates across batches.

        Returns:
            Number of duplicates found
        """
        columns = self._duplicate_columns or batch.columns
        columns = [c for c in columns if c in batch.columns]

        if not columns:
            return 0

        duplicate_count = 0
        for row in batch.select(columns).iter_rows():
            row_hash = hashlib.md5(str(row).encode()).hexdigest()
            if row_hash in self._state.seen_hashes:
                duplicate_count += 1
            else:
                self._state.seen_hashes.add(row_hash)

        return duplicate_count

    def _check_window_completion(self) -> None:
        """Check if current window is complete."""
        if not self._window_batches:
            return

        window_type = self._window_config.window_type
        window_size = self._window_config.window_size

        should_close = False

        if window_type == WindowType.TUMBLING:
            # Close window after window_size records or time
            total_records = sum(b.record_count for b in self._window_batches)
            if total_records >= window_size:
                should_close = True

        elif window_type == WindowType.GLOBAL:
            # Never auto-close
            pass

        if should_close:
            self._close_window()

    def _close_window(self) -> None:
        """Close current window and create result."""
        if not self._window_batches:
            return

        import uuid

        now = datetime.now()
        start = self._window_start or self._window_batches[0].timestamp

        total_records = sum(b.record_count for b in self._window_batches)
        total_issues = sum(b.issue_count for b in self._window_batches)

        # Aggregate issues by type
        aggregate_issues: dict[str, int] = defaultdict(int)
        for batch in self._window_batches:
            for issue in batch.issues:
                aggregate_issues[issue.issue_type] += issue.count

        self._current_window = WindowResult(
            window_id=str(uuid.uuid4())[:8],
            window_start=start,
            window_end=now,
            total_records=total_records,
            total_issues=total_issues,
            batch_results=list(self._window_batches),
            aggregate_issues=dict(aggregate_issues),
        )

        # Reset for next window
        self._window_batches = []
        self._window_start = now

    def get_current_window(self) -> WindowResult | None:
        """Get the current (or last closed) window result."""
        return self._current_window

    def close_window(self) -> WindowResult | None:
        """Force close current window and return result."""
        self._close_window()
        return self._current_window

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all batches."""
        with self._lock:
            mean_values = {}
            for col in self._state.column_sum:
                count = self._state.column_count.get(col, 0)
                if count > 0:
                    mean_values[col] = self._state.column_sum[col] / count

            return {
                "batch_count": self._batch_count,
                "total_records": self._total_records,
                "total_issues": self._total_issues,
                "issue_rate": self._total_issues / self._total_records if self._total_records > 0 else 0,
                "column_null_counts": dict(self._state.column_null_counts),
                "column_ranges": {
                    col: {
                        "min": self._state.column_min.get(col),
                        "max": self._state.column_max.get(col),
                        "mean": mean_values.get(col),
                    }
                    for col in self._state.column_min
                },
                "issue_counts_by_type": dict(self._state.issue_counts),
                "unique_records_seen": len(self._state.seen_hashes) if self._track_duplicates else None,
            }

    def checkpoint(self) -> Checkpoint | None:
        """Create a checkpoint of current state.

        Returns:
            Created checkpoint or None if no manager
        """
        if not self._checkpoint_manager:
            return None

        return self._checkpoint_manager.create_checkpoint(
            state=self._state_store,
            batch_count=self._batch_count,
            total_records=self._total_records,
            total_issues=self._total_issues,
        )

    def restore(self, checkpoint_id: str) -> bool:
        """Restore from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if successful
        """
        if not self._checkpoint_manager:
            return False

        checkpoint = self._checkpoint_manager.restore(
            checkpoint_id,
            self._state_store,
        )

        if checkpoint:
            self._batch_count = checkpoint.batch_count
            self._total_records = checkpoint.total_records
            self._total_issues = checkpoint.total_issues
            return True

        return False

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._state = IncrementalState()
            self._batch_count = 0
            self._total_records = 0
            self._total_issues = 0
            self._window_batches = []
            self._window_start = None
            self._current_window = None
            self._state_store.clear()
