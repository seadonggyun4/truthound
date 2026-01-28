"""Tests for TaskTracker."""

from datetime import datetime, timedelta

import pytest

from truthound.profiler.distributed.monitoring.config import TaskTrackerConfig
from truthound.profiler.distributed.monitoring.protocols import (
    MonitorEvent,
    MonitorEventType,
    TaskState,
)
from truthound.profiler.distributed.monitoring.task_tracker import TaskTracker


class TestTaskTracker:
    """Tests for TaskTracker."""

    def test_submit_task(self) -> None:
        """Test task submission."""
        tracker = TaskTracker()

        tracker.submit_task(
            task_id="task-1",
            partition_id=0,
            total_rows=1000,
        )

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.task_id == "task-1"
        assert task.partition_id == 0
        assert task.total_rows == 1000
        assert task.state == TaskState.SUBMITTED

    def test_start_task(self) -> None:
        """Test task start."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)

        tracker.start_task("task-1", worker_id="worker-1")

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.state == TaskState.RUNNING
        assert task.worker_id == "worker-1"
        assert task.started_at is not None

    def test_update_progress(self) -> None:
        """Test progress update."""
        config = TaskTrackerConfig(progress_update_interval_seconds=0)
        tracker = TaskTracker(config=config)
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="worker-1")

        tracker.update_progress("task-1", progress=0.5, rows_processed=500)

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.progress == 0.5
        assert task.rows_processed == 500

    def test_complete_task(self) -> None:
        """Test task completion."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="worker-1")

        tracker.complete_task("task-1", rows_processed=1000)

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.state == TaskState.COMPLETED
        assert task.progress == 1.0
        assert task.rows_processed == 1000
        assert task.completed_at is not None

    def test_fail_task(self) -> None:
        """Test task failure."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="worker-1")

        tracker.fail_task("task-1", error="Test error")

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.state == TaskState.FAILED
        assert task.error_message == "Test error"

    def test_fail_task_with_retry(self) -> None:
        """Test task failure with retry."""
        config = TaskTrackerConfig(max_retries=3)
        tracker = TaskTracker(config=config)
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="worker-1")

        tracker.fail_task("task-1", error="Test error", retry=True)

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.state == TaskState.RETRYING
        assert task.retry_count == 1

    def test_fail_task_max_retries_exceeded(self) -> None:
        """Test task failure after max retries."""
        config = TaskTrackerConfig(max_retries=2)
        tracker = TaskTracker(config=config)
        tracker.submit_task("task-1", partition_id=0)

        # Exhaust retries
        for _ in range(2):
            tracker.start_task("task-1", worker_id="worker-1")
            tracker.fail_task("task-1", error="Error", retry=True)

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.retry_count == 2

        # Next failure should not retry
        tracker.start_task("task-1", worker_id="worker-1")
        tracker.fail_task("task-1", error="Final error", retry=True)

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.state == TaskState.FAILED

    def test_cancel_task(self) -> None:
        """Test task cancellation."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="worker-1")

        tracker.cancel_task("task-1")

        task = tracker.get_task("task-1")
        assert task is not None
        assert task.state == TaskState.CANCELLED

    def test_get_all_tasks(self) -> None:
        """Test getting all tasks."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.submit_task("task-2", partition_id=1)
        tracker.submit_task("task-3", partition_id=2)

        tasks = tracker.get_all_tasks()
        assert len(tasks) == 3

    def test_get_active_tasks(self) -> None:
        """Test getting active tasks."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.submit_task("task-2", partition_id=1)
        tracker.start_task("task-2", worker_id="w1")
        tracker.submit_task("task-3", partition_id=2)
        tracker.start_task("task-3", worker_id="w1")
        tracker.complete_task("task-3")

        active = tracker.get_active_tasks()
        assert len(active) == 2  # task-1 (submitted) and task-2 (running)

    def test_get_running_tasks(self) -> None:
        """Test getting running tasks."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.submit_task("task-2", partition_id=1)
        tracker.start_task("task-1", worker_id="w1")

        running = tracker.get_running_tasks()
        assert len(running) == 1
        assert running[0].task_id == "task-1"

    def test_get_failed_tasks(self) -> None:
        """Test getting failed tasks."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="w1")
        tracker.fail_task("task-1", error="Error")

        tracker.submit_task("task-2", partition_id=1)
        tracker.start_task("task-2", worker_id="w1")
        tracker.complete_task("task-2")

        failed = tracker.get_failed_tasks()
        assert len(failed) == 1
        assert failed[0].task_id == "task-1"

    def test_get_tasks_by_worker(self) -> None:
        """Test getting tasks by worker."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="worker-1")
        tracker.complete_task("task-1")

        tracker.submit_task("task-2", partition_id=1)
        tracker.start_task("task-2", worker_id="worker-2")
        tracker.complete_task("task-2")

        tracker.submit_task("task-3", partition_id=2)
        tracker.start_task("task-3", worker_id="worker-1")

        worker1_tasks = tracker.get_tasks_by_worker("worker-1")
        assert len(worker1_tasks) == 2

    def test_get_tasks_by_partition(self) -> None:
        """Test getting tasks by partition."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.submit_task("task-2", partition_id=0)  # Retry
        tracker.submit_task("task-3", partition_id=1)

        partition0_tasks = tracker.get_tasks_by_partition(0)
        assert len(partition0_tasks) == 2

    def test_history_size_limit(self) -> None:
        """Test history size limit."""
        config = TaskTrackerConfig(max_history_size=3)
        tracker = TaskTracker(config=config)

        # Complete more tasks than history limit
        for i in range(5):
            task_id = f"task-{i}"
            tracker.submit_task(task_id, partition_id=i)
            tracker.start_task(task_id, worker_id="w1")
            tracker.complete_task(task_id)

        # Should only keep 3 in history
        stats = tracker.get_statistics()
        assert stats["history_size"] == 3

    def test_event_callback(self) -> None:
        """Test event callback."""
        events: list[MonitorEvent] = []

        tracker = TaskTracker(on_event=lambda e: events.append(e))
        tracker.submit_task("task-1", partition_id=0)

        assert len(events) == 1
        assert events[0].event_type == MonitorEventType.TASK_SUBMITTED

    def test_generate_task_id(self) -> None:
        """Test task ID generation."""
        tracker = TaskTracker()

        id1 = tracker.generate_task_id()
        id2 = tracker.generate_task_id()

        assert id1.startswith("task_")
        assert id2.startswith("task_")
        assert id1 != id2

    def test_generate_task_id_with_prefix(self) -> None:
        """Test task ID generation with custom prefix."""
        tracker = TaskTracker()

        task_id = tracker.generate_task_id(prefix="partition")

        assert task_id.startswith("partition_")

    def test_get_statistics(self) -> None:
        """Test statistics."""
        tracker = TaskTracker()

        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="w1")
        tracker.complete_task("task-1")

        tracker.submit_task("task-2", partition_id=1)
        tracker.start_task("task-2", worker_id="w1")
        tracker.fail_task("task-2", error="Error")

        tracker.submit_task("task-3", partition_id=2)

        stats = tracker.get_statistics()
        assert stats["total_submitted"] == 3
        assert stats["total_completed"] == 1
        assert stats["total_failed"] == 1
        assert stats["pending_tasks"] == 1

    def test_reset(self) -> None:
        """Test reset."""
        tracker = TaskTracker()
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="w1")
        tracker.complete_task("task-1")

        tracker.reset()

        assert tracker.get_all_tasks() == []
        stats = tracker.get_statistics()
        assert stats["total_submitted"] == 0

    def test_progress_throttling(self) -> None:
        """Test progress update throttling."""
        config = TaskTrackerConfig(progress_update_interval_seconds=1.0)
        tracker = TaskTracker(config=config)
        tracker.submit_task("task-1", partition_id=0)
        tracker.start_task("task-1", worker_id="w1")

        # First update should work
        tracker.update_progress("task-1", progress=0.25)
        task = tracker.get_task("task-1")
        assert task is not None
        assert task.progress == 0.25

        # Immediate second update should be throttled
        tracker.update_progress("task-1", progress=0.5)
        task = tracker.get_task("task-1")
        assert task is not None
        assert task.progress == 0.25  # Still 0.25 due to throttling

    def test_unknown_task_handling(self) -> None:
        """Test handling of unknown task IDs."""
        tracker = TaskTracker()

        # These should not raise
        tracker.start_task("unknown", worker_id="w1")
        tracker.update_progress("unknown", progress=0.5)
        tracker.complete_task("unknown")
        tracker.fail_task("unknown", error="Error")
        tracker.cancel_task("unknown")

        # Get should return None
        assert tracker.get_task("unknown") is None
