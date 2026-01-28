"""Tests for monitoring protocols and types."""

from datetime import datetime

import pytest

from truthound.profiler.distributed.monitoring.protocols import (
    AggregatedProgress,
    EventSeverity,
    HealthStatus,
    MonitorEvent,
    MonitorEventType,
    MonitorMetrics,
    TaskInfo,
    TaskState,
    WorkerHealth,
)


class TestMonitorEvent:
    """Tests for MonitorEvent."""

    def test_create_event(self) -> None:
        """Test event creation."""
        event = MonitorEvent(
            event_type=MonitorEventType.TASK_STARTED,
            message="Task started",
        )

        assert event.event_type == MonitorEventType.TASK_STARTED
        assert event.message == "Task started"
        assert event.severity == EventSeverity.INFO
        assert event.progress == 0.0

    def test_event_with_all_fields(self) -> None:
        """Test event with all fields."""
        event = MonitorEvent(
            event_type=MonitorEventType.TASK_COMPLETED,
            severity=EventSeverity.INFO,
            message="Task completed",
            task_id="task-1",
            partition_id=5,
            worker_id="worker-1",
            progress=1.0,
            metadata={"duration": 2.5},
        )

        assert event.task_id == "task-1"
        assert event.partition_id == 5
        assert event.worker_id == "worker-1"
        assert event.progress == 1.0
        assert event.metadata["duration"] == 2.5

    def test_is_error_property(self) -> None:
        """Test is_error property."""
        info_event = MonitorEvent(
            event_type=MonitorEventType.TASK_STARTED,
            severity=EventSeverity.INFO,
        )
        error_event = MonitorEvent(
            event_type=MonitorEventType.TASK_FAILED,
            severity=EventSeverity.ERROR,
        )
        critical_event = MonitorEvent(
            event_type=MonitorEventType.HEALTH_CRITICAL,
            severity=EventSeverity.CRITICAL,
        )

        assert not info_event.is_error
        assert error_event.is_error
        assert critical_event.is_error

    def test_is_task_event_property(self) -> None:
        """Test is_task_event property."""
        task_event = MonitorEvent(event_type=MonitorEventType.TASK_STARTED)
        worker_event = MonitorEvent(event_type=MonitorEventType.WORKER_REGISTERED)
        progress_event = MonitorEvent(event_type=MonitorEventType.PROGRESS_UPDATE)

        assert task_event.is_task_event
        assert not worker_event.is_task_event
        assert not progress_event.is_task_event

    def test_is_worker_event_property(self) -> None:
        """Test is_worker_event property."""
        task_event = MonitorEvent(event_type=MonitorEventType.TASK_STARTED)
        worker_event = MonitorEvent(event_type=MonitorEventType.WORKER_REGISTERED)

        assert not task_event.is_worker_event
        assert worker_event.is_worker_event

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        event = MonitorEvent(
            event_type=MonitorEventType.TASK_COMPLETED,
            severity=EventSeverity.INFO,
            message="Done",
            task_id="task-1",
            progress=1.0,
        )

        d = event.to_dict()

        assert d["event_type"] == "task_completed"
        assert d["severity"] == "INFO"
        assert d["message"] == "Done"
        assert d["task_id"] == "task-1"
        assert d["progress"] == 1.0
        assert "timestamp" in d


class TestTaskInfo:
    """Tests for TaskInfo."""

    def test_create_task_info(self) -> None:
        """Test task info creation."""
        task = TaskInfo(
            task_id="task-1",
            partition_id=0,
        )

        assert task.task_id == "task-1"
        assert task.partition_id == 0
        assert task.state == TaskState.PENDING
        assert task.progress == 0.0

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        now = datetime.now()
        task = TaskInfo(
            task_id="task-1",
            partition_id=0,
            started_at=now,
            completed_at=now,
        )

        # Same time should be ~0
        assert task.duration_seconds is not None
        assert task.duration_seconds >= 0

    def test_duration_seconds_not_started(self) -> None:
        """Test duration when not started."""
        task = TaskInfo(task_id="task-1", partition_id=0)

        assert task.duration_seconds is None

    def test_wait_time_seconds(self) -> None:
        """Test wait time calculation."""
        now = datetime.now()
        task = TaskInfo(
            task_id="task-1",
            partition_id=0,
            submitted_at=now,
            started_at=now,
        )

        assert task.wait_time_seconds is not None
        assert task.wait_time_seconds >= 0

    def test_is_terminal(self) -> None:
        """Test is_terminal property."""
        pending = TaskInfo(task_id="t1", partition_id=0, state=TaskState.PENDING)
        running = TaskInfo(task_id="t2", partition_id=0, state=TaskState.RUNNING)
        completed = TaskInfo(task_id="t3", partition_id=0, state=TaskState.COMPLETED)
        failed = TaskInfo(task_id="t4", partition_id=0, state=TaskState.FAILED)

        assert not pending.is_terminal
        assert not running.is_terminal
        assert completed.is_terminal
        assert failed.is_terminal

    def test_rows_per_second(self) -> None:
        """Test throughput calculation."""
        now = datetime.now()
        task = TaskInfo(
            task_id="task-1",
            partition_id=0,
            started_at=now,
            completed_at=now,
            rows_processed=1000,
        )

        # With near-zero duration, should handle gracefully
        assert task.rows_per_second >= 0

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        task = TaskInfo(
            task_id="task-1",
            partition_id=0,
            state=TaskState.RUNNING,
            progress=0.5,
            rows_processed=500,
        )

        d = task.to_dict()

        assert d["task_id"] == "task-1"
        assert d["partition_id"] == 0
        assert d["state"] == "running"
        assert d["progress"] == 0.5
        assert d["rows_processed"] == 500


class TestWorkerHealth:
    """Tests for WorkerHealth."""

    def test_create_worker_health(self) -> None:
        """Test worker health creation."""
        health = WorkerHealth(
            worker_id="worker-1",
            status=HealthStatus.HEALTHY,
        )

        assert health.worker_id == "worker-1"
        assert health.status == HealthStatus.HEALTHY
        assert health.is_healthy

    def test_total_tasks(self) -> None:
        """Test total_tasks property."""
        health = WorkerHealth(
            worker_id="worker-1",
            completed_tasks=10,
            failed_tasks=2,
        )

        assert health.total_tasks == 12

    def test_is_healthy(self) -> None:
        """Test is_healthy property."""
        healthy = WorkerHealth(worker_id="w1", status=HealthStatus.HEALTHY)
        degraded = WorkerHealth(worker_id="w2", status=HealthStatus.DEGRADED)
        unhealthy = WorkerHealth(worker_id="w3", status=HealthStatus.UNHEALTHY)

        assert healthy.is_healthy
        assert not degraded.is_healthy
        assert not unhealthy.is_healthy

    def test_heartbeat_age_seconds(self) -> None:
        """Test heartbeat age calculation."""
        health = WorkerHealth(
            worker_id="worker-1",
            last_heartbeat=datetime.now(),
        )

        assert health.heartbeat_age_seconds is not None
        assert health.heartbeat_age_seconds >= 0

    def test_heartbeat_age_no_heartbeat(self) -> None:
        """Test heartbeat age when no heartbeat."""
        health = WorkerHealth(worker_id="worker-1")

        assert health.heartbeat_age_seconds is None

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        health = WorkerHealth(
            worker_id="worker-1",
            status=HealthStatus.HEALTHY,
            cpu_percent=50.0,
            memory_percent=60.0,
        )

        d = health.to_dict()

        assert d["worker_id"] == "worker-1"
        assert d["status"] == "healthy"
        assert d["cpu_percent"] == 50.0
        assert d["memory_percent"] == 60.0


class TestAggregatedProgress:
    """Tests for AggregatedProgress."""

    def test_create_progress(self) -> None:
        """Test progress creation."""
        progress = AggregatedProgress(
            total_partitions=10,
            completed_partitions=5,
        )

        assert progress.total_partitions == 10
        assert progress.completed_partitions == 5
        assert progress.percent == 0.0  # overall_progress is 0

    def test_percent_property(self) -> None:
        """Test percent property."""
        progress = AggregatedProgress(
            total_partitions=10,
            overall_progress=0.75,
        )

        assert progress.percent == 75.0

    def test_pending_partitions(self) -> None:
        """Test pending_partitions calculation."""
        progress = AggregatedProgress(
            total_partitions=10,
            completed_partitions=3,
            failed_partitions=1,
            in_progress_partitions=2,
        )

        assert progress.pending_partitions == 4

    def test_is_complete(self) -> None:
        """Test is_complete property."""
        in_progress = AggregatedProgress(
            total_partitions=10,
            completed_partitions=5,
        )
        all_complete = AggregatedProgress(
            total_partitions=10,
            completed_partitions=10,
        )
        with_failures = AggregatedProgress(
            total_partitions=10,
            completed_partitions=8,
            failed_partitions=2,
        )

        assert not in_progress.is_complete
        assert all_complete.is_complete
        assert with_failures.is_complete

    def test_success_rate(self) -> None:
        """Test success_rate property."""
        progress = AggregatedProgress(
            total_partitions=10,
            completed_partitions=8,
            failed_partitions=2,
        )

        assert progress.success_rate == 0.8

    def test_success_rate_no_completions(self) -> None:
        """Test success_rate with no completions."""
        progress = AggregatedProgress(total_partitions=10)

        assert progress.success_rate == 1.0

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        progress = AggregatedProgress(
            total_partitions=10,
            completed_partitions=5,
            overall_progress=0.5,
        )

        d = progress.to_dict()

        assert d["total_partitions"] == 10
        assert d["completed_partitions"] == 5
        assert d["overall_progress"] == 0.5
        assert d["percent"] == 50.0
        assert "is_complete" in d
        assert "success_rate" in d


class TestMonitorMetrics:
    """Tests for MonitorMetrics."""

    def test_create_metrics(self) -> None:
        """Test metrics creation."""
        metrics = MonitorMetrics(
            tasks_total=100,
            tasks_completed=80,
            tasks_failed=5,
        )

        assert metrics.tasks_total == 100
        assert metrics.tasks_completed == 80
        assert metrics.tasks_failed == 5

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        metrics = MonitorMetrics(
            tasks_total=100,
            tasks_completed=80,
            tasks_failed=5,
            workers_total=4,
            workers_healthy=3,
        )

        d = metrics.to_dict()

        assert d["tasks"]["total"] == 100
        assert d["tasks"]["completed"] == 80
        assert d["tasks"]["failed"] == 5
        assert d["workers"]["total"] == 4
        assert d["workers"]["healthy"] == 3
