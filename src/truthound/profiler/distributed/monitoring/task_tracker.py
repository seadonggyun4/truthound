"""Task tracker implementation for distributed monitoring.

This module provides task lifecycle tracking for distributed profiling operations.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable

from truthound.profiler.distributed.monitoring.config import TaskTrackerConfig
from truthound.profiler.distributed.monitoring.protocols import (
    EventSeverity,
    ITaskTracker,
    MonitorEvent,
    MonitorEventType,
    TaskInfo,
    TaskState,
)


logger = logging.getLogger(__name__)


class TaskTracker(ITaskTracker):
    """Tracks distributed task lifecycle and progress.

    Thread-safe implementation for tracking tasks across multiple workers.
    Maintains a limited history of completed tasks and emits events for
    task state changes.

    Example:
        tracker = TaskTracker(
            config=TaskTrackerConfig(max_history_size=1000),
            on_event=lambda e: print(e.message),
        )

        tracker.submit_task("task-1", partition_id=0, total_rows=1000)
        tracker.start_task("task-1", worker_id="worker-1")
        tracker.update_progress("task-1", progress=0.5, rows_processed=500)
        tracker.complete_task("task-1", rows_processed=1000)
    """

    def __init__(
        self,
        config: TaskTrackerConfig | None = None,
        on_event: Callable[[MonitorEvent], None] | None = None,
    ) -> None:
        """Initialize task tracker.

        Args:
            config: Tracker configuration
            on_event: Callback for task events
        """
        self._config = config or TaskTrackerConfig()
        self._on_event = on_event

        # Task storage
        self._tasks: dict[str, TaskInfo] = {}
        self._completed_tasks: OrderedDict[str, TaskInfo] = OrderedDict()
        self._lock = threading.RLock()

        # Progress throttling
        self._last_progress_update: dict[str, datetime] = {}

        # Statistics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_retried = 0

    def submit_task(
        self,
        task_id: str,
        partition_id: int,
        total_rows: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record task submission.

        Args:
            task_id: Unique task identifier
            partition_id: Partition being processed
            total_rows: Total rows in partition
            metadata: Additional task metadata
        """
        with self._lock:
            now = datetime.now()

            task = TaskInfo(
                task_id=task_id,
                partition_id=partition_id,
                state=TaskState.SUBMITTED,
                total_rows=total_rows,
                submitted_at=now,
                metadata=metadata or {},
            )

            self._tasks[task_id] = task
            self._total_submitted += 1

            self._emit_event(
                MonitorEventType.TASK_SUBMITTED,
                f"Task {task_id} submitted for partition {partition_id}",
                task_id=task_id,
                partition_id=partition_id,
            )

    def start_task(
        self,
        task_id: str,
        worker_id: str,
    ) -> None:
        """Record task start.

        Args:
            task_id: Task identifier
            worker_id: Worker executing the task
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning(f"Start called for unknown task: {task_id}")
                return

            now = datetime.now()
            task.started_at = now
            task.worker_id = worker_id
            task.state = TaskState.RUNNING

            self._emit_event(
                MonitorEventType.TASK_STARTED,
                f"Task {task_id} started on worker {worker_id}",
                task_id=task_id,
                partition_id=task.partition_id,
                worker_id=worker_id,
            )

    def update_progress(
        self,
        task_id: str,
        progress: float,
        rows_processed: int = 0,
    ) -> None:
        """Update task progress.

        Args:
            task_id: Task identifier
            progress: Progress ratio (0.0 to 1.0)
            rows_processed: Rows processed so far
        """
        if not self._config.enable_progress_tracking:
            return

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            # Throttle progress updates
            now = datetime.now()
            last_update = self._last_progress_update.get(task_id)
            if last_update is not None:
                elapsed = (now - last_update).total_seconds()
                if elapsed < self._config.progress_update_interval_seconds:
                    return

            self._last_progress_update[task_id] = now

            task.progress = min(1.0, max(0.0, progress))
            task.rows_processed = rows_processed

            self._emit_event(
                MonitorEventType.TASK_PROGRESS,
                f"Task {task_id} progress: {progress * 100:.1f}%",
                task_id=task_id,
                partition_id=task.partition_id,
                worker_id=task.worker_id,
                progress=progress,
                severity=EventSeverity.DEBUG,
            )

    def complete_task(
        self,
        task_id: str,
        rows_processed: int = 0,
    ) -> None:
        """Record task completion.

        Args:
            task_id: Task identifier
            rows_processed: Final row count
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning(f"Complete called for unknown task: {task_id}")
                return

            now = datetime.now()
            task.completed_at = now
            task.state = TaskState.COMPLETED
            task.progress = 1.0
            task.rows_processed = rows_processed if rows_processed else task.rows_processed

            self._total_completed += 1

            # Move to completed history
            self._move_to_history(task_id)

            duration = task.duration_seconds or 0.0
            self._emit_event(
                MonitorEventType.TASK_COMPLETED,
                f"Task {task_id} completed in {duration:.2f}s ({task.rows_processed} rows)",
                task_id=task_id,
                partition_id=task.partition_id,
                worker_id=task.worker_id,
                progress=1.0,
            )

    def fail_task(
        self,
        task_id: str,
        error: str,
        retry: bool = False,
    ) -> None:
        """Record task failure.

        Args:
            task_id: Task identifier
            error: Error message
            retry: Whether task will be retried
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning(f"Fail called for unknown task: {task_id}")
                return

            now = datetime.now()

            if retry and task.retry_count < self._config.max_retries:
                task.retry_count += 1
                task.state = TaskState.RETRYING
                task.error_message = error
                self._total_retried += 1

                self._emit_event(
                    MonitorEventType.TASK_RETRYING,
                    f"Task {task_id} will retry (attempt {task.retry_count}/{self._config.max_retries}): {error}",
                    task_id=task_id,
                    partition_id=task.partition_id,
                    worker_id=task.worker_id,
                    severity=EventSeverity.WARNING,
                    metadata={"error": error, "retry_count": task.retry_count},
                )
            else:
                task.completed_at = now
                task.state = TaskState.FAILED
                task.error_message = error
                self._total_failed += 1

                # Move to completed history
                self._move_to_history(task_id)

                self._emit_event(
                    MonitorEventType.TASK_FAILED,
                    f"Task {task_id} failed: {error}",
                    task_id=task_id,
                    partition_id=task.partition_id,
                    worker_id=task.worker_id,
                    severity=EventSeverity.ERROR,
                    metadata={"error": error},
                )

    def cancel_task(self, task_id: str) -> None:
        """Record task cancellation.

        Args:
            task_id: Task identifier
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning(f"Cancel called for unknown task: {task_id}")
                return

            task.completed_at = datetime.now()
            task.state = TaskState.CANCELLED

            # Move to completed history
            self._move_to_history(task_id)

            self._emit_event(
                MonitorEventType.TASK_CANCELLED,
                f"Task {task_id} cancelled",
                task_id=task_id,
                partition_id=task.partition_id,
                worker_id=task.worker_id,
                severity=EventSeverity.WARNING,
            )

    def timeout_task(self, task_id: str) -> None:
        """Record task timeout.

        Args:
            task_id: Task identifier
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning(f"Timeout called for unknown task: {task_id}")
                return

            task.completed_at = datetime.now()
            task.state = TaskState.TIMEOUT
            self._total_failed += 1

            # Move to completed history
            self._move_to_history(task_id)

            self._emit_event(
                MonitorEventType.TASK_TIMEOUT,
                f"Task {task_id} timed out after {self._config.timeout_seconds}s",
                task_id=task_id,
                partition_id=task.partition_id,
                worker_id=task.worker_id,
                severity=EventSeverity.ERROR,
            )

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task information.

        Args:
            task_id: Task identifier

        Returns:
            Task info or None if not found
        """
        with self._lock:
            # Check active tasks first
            if task_id in self._tasks:
                return self._tasks[task_id]
            # Then check history
            return self._completed_tasks.get(task_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all tracked tasks.

        Returns:
            List of all task info (active + recent history)
        """
        with self._lock:
            active = list(self._tasks.values())
            completed = list(self._completed_tasks.values())
            return active + completed

    def get_active_tasks(self) -> list[TaskInfo]:
        """Get all active (non-terminal) tasks.

        Returns:
            List of active task info
        """
        with self._lock:
            return [t for t in self._tasks.values() if not t.is_terminal]

    def get_running_tasks(self) -> list[TaskInfo]:
        """Get all running tasks.

        Returns:
            List of running task info
        """
        with self._lock:
            return [t for t in self._tasks.values() if t.state == TaskState.RUNNING]

    def get_pending_tasks(self) -> list[TaskInfo]:
        """Get all pending tasks.

        Returns:
            List of pending task info
        """
        with self._lock:
            return [
                t
                for t in self._tasks.values()
                if t.state in {TaskState.PENDING, TaskState.SUBMITTED}
            ]

    def get_failed_tasks(self) -> list[TaskInfo]:
        """Get all failed tasks.

        Returns:
            List of failed task info
        """
        with self._lock:
            all_tasks = list(self._tasks.values()) + list(self._completed_tasks.values())
            return [t for t in all_tasks if t.state == TaskState.FAILED]

    def get_tasks_by_worker(self, worker_id: str) -> list[TaskInfo]:
        """Get tasks for a specific worker.

        Args:
            worker_id: Worker identifier

        Returns:
            List of task info for the worker
        """
        with self._lock:
            all_tasks = list(self._tasks.values()) + list(self._completed_tasks.values())
            return [t for t in all_tasks if t.worker_id == worker_id]

    def get_tasks_by_partition(self, partition_id: int) -> list[TaskInfo]:
        """Get tasks for a specific partition.

        Args:
            partition_id: Partition identifier

        Returns:
            List of task info for the partition
        """
        with self._lock:
            all_tasks = list(self._tasks.values()) + list(self._completed_tasks.values())
            return [t for t in all_tasks if t.partition_id == partition_id]

    def get_statistics(self) -> dict[str, Any]:
        """Get tracker statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            active_tasks = list(self._tasks.values())
            running = sum(1 for t in active_tasks if t.state == TaskState.RUNNING)
            pending = sum(
                1 for t in active_tasks if t.state in {TaskState.PENDING, TaskState.SUBMITTED}
            )

            return {
                "total_submitted": self._total_submitted,
                "total_completed": self._total_completed,
                "total_failed": self._total_failed,
                "total_retried": self._total_retried,
                "active_tasks": len(active_tasks),
                "running_tasks": running,
                "pending_tasks": pending,
                "history_size": len(self._completed_tasks),
            }

    def reset(self) -> None:
        """Reset tracker state."""
        with self._lock:
            self._tasks.clear()
            self._completed_tasks.clear()
            self._last_progress_update.clear()
            self._total_submitted = 0
            self._total_completed = 0
            self._total_failed = 0
            self._total_retried = 0

    def check_timeouts(self) -> list[str]:
        """Check for timed-out tasks.

        Returns:
            List of task IDs that timed out
        """
        if self._config.timeout_seconds <= 0:
            return []

        with self._lock:
            timed_out = []
            now = datetime.now()

            for task in list(self._tasks.values()):
                if task.state != TaskState.RUNNING:
                    continue

                if task.started_at is None:
                    continue

                elapsed = (now - task.started_at).total_seconds()
                if elapsed > self._config.timeout_seconds:
                    timed_out.append(task.task_id)

            # Timeout the tasks
            for task_id in timed_out:
                self.timeout_task(task_id)

            return timed_out

    def generate_task_id(self, prefix: str = "task") -> str:
        """Generate a unique task ID.

        Args:
            prefix: ID prefix

        Returns:
            Unique task ID
        """
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _move_to_history(self, task_id: str) -> None:
        """Move task from active to history.

        Args:
            task_id: Task identifier
        """
        task = self._tasks.pop(task_id, None)
        if task is None:
            return

        # Add to history
        self._completed_tasks[task_id] = task

        # Clean up progress tracking
        self._last_progress_update.pop(task_id, None)

        # Trim history if needed
        while len(self._completed_tasks) > self._config.max_history_size:
            self._completed_tasks.popitem(last=False)

    def _emit_event(
        self,
        event_type: MonitorEventType,
        message: str,
        task_id: str | None = None,
        partition_id: int | None = None,
        worker_id: str | None = None,
        progress: float = 0.0,
        severity: EventSeverity = EventSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a monitoring event.

        Args:
            event_type: Event type
            message: Event message
            task_id: Task identifier
            partition_id: Partition identifier
            worker_id: Worker identifier
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
            task_id=task_id,
            partition_id=partition_id,
            worker_id=worker_id,
            progress=progress,
            metadata=metadata or {},
        )

        try:
            self._on_event(event)
        except Exception as e:
            logger.warning(f"Error in task tracker event callback: {e}")
