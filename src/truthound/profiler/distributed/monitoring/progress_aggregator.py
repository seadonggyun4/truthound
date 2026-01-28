"""Progress aggregator for distributed monitoring.

This module provides progress aggregation from multiple distributed tasks/partitions.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from truthound.profiler.distributed.monitoring.protocols import (
    AggregatedProgress,
    EventSeverity,
    IProgressAggregator,
    MonitorEvent,
    MonitorEventType,
)


logger = logging.getLogger(__name__)


@dataclass
class PartitionState:
    """State of a single partition."""

    partition_id: int
    progress: float = 0.0
    rows_processed: int = 0
    total_rows: int = 0
    is_complete: bool = False
    is_failed: bool = False
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DistributedProgressAggregator(IProgressAggregator):
    """Aggregates progress from multiple distributed partitions.

    Thread-safe implementation that tracks progress across all partitions
    and provides overall progress estimation with ETA calculation.

    Example:
        aggregator = DistributedProgressAggregator(
            on_progress=lambda p: print(f"Overall: {p.percent:.1f}%"),
            milestone_interval=10,  # Emit milestone every 10%
        )

        aggregator.set_total_partitions(10)
        aggregator.update_partition(0, progress=0.5, rows_processed=500)
        aggregator.complete_partition(0, rows_processed=1000)

        progress = aggregator.get_progress()
        print(f"Completed: {progress.completed_partitions}/{progress.total_partitions}")
    """

    def __init__(
        self,
        on_progress: Callable[[AggregatedProgress], None] | None = None,
        on_event: Callable[[MonitorEvent], None] | None = None,
        milestone_interval: int = 10,
        emit_interval_seconds: float = 1.0,
    ) -> None:
        """Initialize progress aggregator.

        Args:
            on_progress: Callback for progress updates
            on_event: Callback for events
            milestone_interval: Emit milestone event every N percent
            emit_interval_seconds: Minimum interval between progress emissions
        """
        self._on_progress = on_progress
        self._on_event = on_event
        self._milestone_interval = milestone_interval
        self._emit_interval_seconds = emit_interval_seconds

        # State
        self._total_partitions = 0
        self._partitions: dict[int, PartitionState] = {}
        self._lock = threading.RLock()

        # Timing
        self._start_time: datetime | None = None
        self._last_emit_time: datetime | None = None
        self._last_milestone = -1

        # Cumulative stats
        self._total_rows = 0
        self._processed_rows = 0

    def set_total_partitions(self, count: int) -> None:
        """Set total partition count.

        Args:
            count: Number of partitions
        """
        with self._lock:
            self._total_partitions = count
            self._partitions.clear()
            self._start_time = datetime.now()
            self._last_milestone = -1
            self._total_rows = 0
            self._processed_rows = 0

            self._emit_event(
                MonitorEventType.PROGRESS_UPDATE,
                f"Starting distributed processing with {count} partitions",
            )

    def set_partition_rows(self, partition_id: int, total_rows: int) -> None:
        """Set total rows for a partition.

        Args:
            partition_id: Partition identifier
            total_rows: Total rows in partition
        """
        with self._lock:
            partition = self._get_or_create_partition(partition_id)
            old_total = partition.total_rows
            partition.total_rows = total_rows
            self._total_rows += total_rows - old_total

    def start_partition(self, partition_id: int, total_rows: int = 0) -> None:
        """Mark partition as started.

        Args:
            partition_id: Partition identifier
            total_rows: Total rows in partition
        """
        with self._lock:
            partition = self._get_or_create_partition(partition_id)
            partition.started_at = datetime.now()
            if total_rows > 0:
                old_total = partition.total_rows
                partition.total_rows = total_rows
                self._total_rows += total_rows - old_total

            self._emit_event(
                MonitorEventType.PARTITION_START,
                f"Partition {partition_id} started ({total_rows} rows)",
                partition_id=partition_id,
            )

    def update_partition(
        self,
        partition_id: int,
        progress: float,
        rows_processed: int = 0,
    ) -> None:
        """Update partition progress.

        Args:
            partition_id: Partition identifier
            progress: Progress ratio (0.0 to 1.0)
            rows_processed: Rows processed
        """
        with self._lock:
            partition = self._get_or_create_partition(partition_id)

            # Update rows
            if rows_processed > 0:
                delta = rows_processed - partition.rows_processed
                if delta > 0:
                    self._processed_rows += delta
                partition.rows_processed = rows_processed

            partition.progress = min(1.0, max(0.0, progress))

            self._maybe_emit_progress()

    def complete_partition(
        self,
        partition_id: int,
        rows_processed: int = 0,
    ) -> None:
        """Mark partition as complete.

        Args:
            partition_id: Partition identifier
            rows_processed: Final row count
        """
        with self._lock:
            partition = self._get_or_create_partition(partition_id)

            if rows_processed > 0:
                delta = rows_processed - partition.rows_processed
                if delta > 0:
                    self._processed_rows += delta
                partition.rows_processed = rows_processed

            partition.progress = 1.0
            partition.is_complete = True
            partition.completed_at = datetime.now()

            duration = 0.0
            if partition.started_at:
                duration = (partition.completed_at - partition.started_at).total_seconds()

            self._emit_event(
                MonitorEventType.PARTITION_COMPLETE,
                f"Partition {partition_id} completed ({partition.rows_processed} rows, {duration:.2f}s)",
                partition_id=partition_id,
                progress=1.0,
            )

            self._maybe_emit_progress(force=True)
            self._check_overall_completion()

    def fail_partition(
        self,
        partition_id: int,
        error: str,
    ) -> None:
        """Mark partition as failed.

        Args:
            partition_id: Partition identifier
            error: Error message
        """
        with self._lock:
            partition = self._get_or_create_partition(partition_id)
            partition.is_failed = True
            partition.error_message = error
            partition.completed_at = datetime.now()

            self._emit_event(
                MonitorEventType.PARTITION_ERROR,
                f"Partition {partition_id} failed: {error}",
                partition_id=partition_id,
                severity=EventSeverity.ERROR,
                metadata={"error": error},
            )

            self._maybe_emit_progress(force=True)
            self._check_overall_completion()

    def get_progress(self) -> AggregatedProgress:
        """Get aggregated progress.

        Returns:
            Aggregated progress data
        """
        with self._lock:
            completed = sum(1 for p in self._partitions.values() if p.is_complete)
            failed = sum(1 for p in self._partitions.values() if p.is_failed)
            in_progress = sum(
                1
                for p in self._partitions.values()
                if not p.is_complete and not p.is_failed and p.started_at is not None
            )

            # Calculate weighted progress
            if self._total_partitions == 0:
                overall_progress = 0.0
            else:
                total_progress = sum(p.progress for p in self._partitions.values())
                overall_progress = total_progress / self._total_partitions

            # Calculate timing
            elapsed = 0.0
            if self._start_time:
                elapsed = (datetime.now() - self._start_time).total_seconds()

            # Calculate throughput
            rows_per_second = self._processed_rows / elapsed if elapsed > 0 else 0.0

            # Estimate remaining time
            estimated_remaining = None
            if overall_progress > 0 and elapsed > 0:
                total_estimated = elapsed / overall_progress
                estimated_remaining = max(0, total_estimated - elapsed)

            # Build partition progress map
            partition_progress = {p.partition_id: p.progress for p in self._partitions.values()}

            return AggregatedProgress(
                total_partitions=self._total_partitions,
                completed_partitions=completed,
                failed_partitions=failed,
                in_progress_partitions=in_progress,
                overall_progress=overall_progress,
                total_rows=self._total_rows,
                processed_rows=self._processed_rows,
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=estimated_remaining,
                rows_per_second=rows_per_second,
                partition_progress=partition_progress,
            )

    def get_partition_progress(self, partition_id: int) -> float:
        """Get progress for a specific partition.

        Args:
            partition_id: Partition identifier

        Returns:
            Progress ratio (0.0 to 1.0)
        """
        with self._lock:
            partition = self._partitions.get(partition_id)
            if partition is None:
                return 0.0
            return partition.progress

    def get_slowest_partitions(self, count: int = 5) -> list[tuple[int, float]]:
        """Get the slowest partitions by progress.

        Args:
            count: Number of partitions to return

        Returns:
            List of (partition_id, progress) tuples
        """
        with self._lock:
            active = [
                (p.partition_id, p.progress)
                for p in self._partitions.values()
                if not p.is_complete and not p.is_failed
            ]
            active.sort(key=lambda x: x[1])
            return active[:count]

    def get_failed_partitions(self) -> list[tuple[int, str]]:
        """Get failed partitions with error messages.

        Returns:
            List of (partition_id, error_message) tuples
        """
        with self._lock:
            return [
                (p.partition_id, p.error_message or "Unknown error")
                for p in self._partitions.values()
                if p.is_failed
            ]

    def reset(self) -> None:
        """Reset aggregator state."""
        with self._lock:
            self._total_partitions = 0
            self._partitions.clear()
            self._start_time = None
            self._last_emit_time = None
            self._last_milestone = -1
            self._total_rows = 0
            self._processed_rows = 0

    def _get_or_create_partition(self, partition_id: int) -> PartitionState:
        """Get or create partition state.

        Args:
            partition_id: Partition identifier

        Returns:
            Partition state
        """
        if partition_id not in self._partitions:
            self._partitions[partition_id] = PartitionState(partition_id=partition_id)
        return self._partitions[partition_id]

    def _maybe_emit_progress(self, force: bool = False) -> None:
        """Emit progress update if conditions are met.

        Args:
            force: Force emission regardless of interval
        """
        now = datetime.now()

        # Check emission interval
        if not force and self._last_emit_time is not None:
            elapsed = (now - self._last_emit_time).total_seconds()
            if elapsed < self._emit_interval_seconds:
                return

        self._last_emit_time = now

        progress = self.get_progress()

        # Emit progress callback
        if self._on_progress:
            try:
                self._on_progress(progress)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")

        # Check for milestones
        current_milestone = int(progress.percent // self._milestone_interval) * self._milestone_interval
        if current_milestone > self._last_milestone and current_milestone > 0:
            self._last_milestone = current_milestone
            self._emit_event(
                MonitorEventType.PROGRESS_MILESTONE,
                f"Progress milestone: {current_milestone}% ({progress.completed_partitions}/{progress.total_partitions} partitions)",
                progress=progress.overall_progress,
                severity=EventSeverity.INFO,
            )

    def _check_overall_completion(self) -> None:
        """Check if all partitions are complete."""
        progress = self.get_progress()
        if progress.is_complete:
            elapsed = progress.elapsed_seconds
            success_rate = progress.success_rate * 100

            self._emit_event(
                MonitorEventType.AGGREGATION_COMPLETE,
                f"All partitions complete: {progress.completed_partitions} succeeded, "
                f"{progress.failed_partitions} failed ({success_rate:.1f}% success rate) in {elapsed:.2f}s",
                progress=1.0,
            )

    def _emit_event(
        self,
        event_type: MonitorEventType,
        message: str,
        partition_id: int | None = None,
        progress: float = 0.0,
        severity: EventSeverity = EventSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a monitoring event.

        Args:
            event_type: Event type
            message: Event message
            partition_id: Partition identifier
            progress: Current progress
            severity: Event severity
            metadata: Additional metadata
        """
        if self._on_event is None:
            return

        event = MonitorEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            partition_id=partition_id,
            progress=progress,
            metadata=metadata or {},
        )

        try:
            self._on_event(event)
        except Exception as e:
            logger.warning(f"Error in progress aggregator event callback: {e}")


class StreamingProgressAggregator(DistributedProgressAggregator):
    """Progress aggregator optimized for streaming updates.

    Provides smoother progress updates for real-time UI display
    with interpolation between actual updates.
    """

    def __init__(
        self,
        *args: Any,
        interpolation_interval_ms: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize streaming progress aggregator.

        Args:
            *args: Parent class arguments
            interpolation_interval_ms: Interval for interpolated updates
            **kwargs: Parent class keyword arguments
        """
        super().__init__(*args, **kwargs)
        self._interpolation_interval_ms = interpolation_interval_ms
        self._last_reported_progress: dict[int, float] = {}

    def update_partition(
        self,
        partition_id: int,
        progress: float,
        rows_processed: int = 0,
    ) -> None:
        """Update partition progress with interpolation.

        Args:
            partition_id: Partition identifier
            progress: Progress ratio (0.0 to 1.0)
            rows_processed: Rows processed
        """
        with self._lock:
            last_progress = self._last_reported_progress.get(partition_id, 0.0)

            # Only update if progress increased
            if progress > last_progress:
                self._last_reported_progress[partition_id] = progress
                super().update_partition(partition_id, progress, rows_processed)

    def reset(self) -> None:
        """Reset aggregator state."""
        super().reset()
        self._last_reported_progress.clear()
