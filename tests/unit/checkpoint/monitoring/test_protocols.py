"""Tests for monitoring protocols and data classes."""

import pytest
from datetime import datetime

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
    MonitoringEvent,
    MonitoringEventType,
    AlertSeverity,
    MetricType,
)


class TestQueueMetrics:
    """Tests for QueueMetrics data class."""

    def test_default_values(self) -> None:
        """Test default values for QueueMetrics."""
        metrics = QueueMetrics(queue_name="test_queue")

        assert metrics.queue_name == "test_queue"
        assert metrics.pending_count == 0
        assert metrics.running_count == 0
        assert metrics.completed_count == 0
        assert metrics.failed_count == 0
        assert metrics.total_tasks == 0

    def test_total_tasks_calculation(self) -> None:
        """Test total_tasks property calculation."""
        metrics = QueueMetrics(
            queue_name="test",
            pending_count=10,
            running_count=5,
            completed_count=100,
            failed_count=3,
        )

        assert metrics.total_tasks == 118

    def test_success_rate_all_success(self) -> None:
        """Test success_rate when all tasks succeed."""
        metrics = QueueMetrics(
            queue_name="test",
            completed_count=100,
            failed_count=0,
        )

        assert metrics.success_rate == 1.0

    def test_success_rate_with_failures(self) -> None:
        """Test success_rate with some failures."""
        metrics = QueueMetrics(
            queue_name="test",
            completed_count=90,
            failed_count=10,
        )

        assert metrics.success_rate == 0.9

    def test_success_rate_no_finished(self) -> None:
        """Test success_rate when no tasks are finished."""
        metrics = QueueMetrics(
            queue_name="test",
            pending_count=100,
        )

        assert metrics.success_rate == 1.0  # Default when no finished tasks

    def test_is_healthy(self) -> None:
        """Test is_healthy property."""
        healthy = QueueMetrics(
            queue_name="test",
            pending_count=100,
            completed_count=95,
            failed_count=5,
        )
        assert healthy.is_healthy is True

        unhealthy_backlog = QueueMetrics(
            queue_name="test",
            pending_count=2000,  # Too many pending
        )
        assert unhealthy_backlog.is_healthy is False

        unhealthy_failures = QueueMetrics(
            queue_name="test",
            pending_count=10,
            completed_count=80,
            failed_count=20,  # High failure rate
        )
        assert unhealthy_failures.is_healthy is False

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        metrics = QueueMetrics(
            queue_name="test",
            pending_count=10,
            completed_count=90,
        )

        result = metrics.to_dict()

        assert result["queue_name"] == "test"
        assert result["pending_count"] == 10
        assert result["completed_count"] == 90
        assert "timestamp" in result
        assert "success_rate" in result


class TestWorkerMetrics:
    """Tests for WorkerMetrics data class."""

    def test_default_values(self) -> None:
        """Test default values."""
        metrics = WorkerMetrics(worker_id="worker-1")

        assert metrics.worker_id == "worker-1"
        assert metrics.state == "unknown"
        assert metrics.current_tasks == 0
        assert metrics.max_concurrency == 1

    def test_load_factor(self) -> None:
        """Test load_factor calculation."""
        metrics = WorkerMetrics(
            worker_id="worker-1",
            current_tasks=5,
            max_concurrency=10,
        )

        assert metrics.load_factor == 0.5

    def test_load_factor_zero_concurrency(self) -> None:
        """Test load_factor with zero max concurrency."""
        metrics = WorkerMetrics(
            worker_id="worker-1",
            max_concurrency=0,
        )

        assert metrics.load_factor == 1.0  # Fully loaded

    def test_is_available(self) -> None:
        """Test is_available property."""
        available = WorkerMetrics(
            worker_id="worker-1",
            state="online",
            current_tasks=5,
            max_concurrency=10,
        )
        assert available.is_available is True

        offline = WorkerMetrics(
            worker_id="worker-1",
            state="offline",
            current_tasks=0,
            max_concurrency=10,
        )
        assert offline.is_available is False

        full = WorkerMetrics(
            worker_id="worker-1",
            state="online",
            current_tasks=10,
            max_concurrency=10,
        )
        assert full.is_available is False

    def test_available_slots(self) -> None:
        """Test available_slots calculation."""
        metrics = WorkerMetrics(
            worker_id="worker-1",
            state="online",
            current_tasks=3,
            max_concurrency=10,
        )

        assert metrics.available_slots == 7

        offline = WorkerMetrics(
            worker_id="worker-1",
            state="offline",
            current_tasks=0,
            max_concurrency=10,
        )
        assert offline.available_slots == 0


class TestTaskMetrics:
    """Tests for TaskMetrics data class."""

    def test_wait_time_calculation(self) -> None:
        """Test wait_time_ms calculation."""
        submitted = datetime(2024, 1, 1, 10, 0, 0)
        started = datetime(2024, 1, 1, 10, 0, 5)  # 5 seconds later

        metrics = TaskMetrics(
            task_id="task-1",
            checkpoint_name="test",
            submitted_at=submitted,
            started_at=started,
        )

        assert metrics.wait_time_ms == 5000.0

    def test_execution_time_calculation(self) -> None:
        """Test execution_time_ms calculation."""
        started = datetime(2024, 1, 1, 10, 0, 0)
        completed = datetime(2024, 1, 1, 10, 0, 10)  # 10 seconds later

        metrics = TaskMetrics(
            task_id="task-1",
            checkpoint_name="test",
            started_at=started,
            completed_at=completed,
        )

        assert metrics.execution_time_ms == 10000.0

    def test_total_time_calculation(self) -> None:
        """Test total_time_ms calculation."""
        submitted = datetime(2024, 1, 1, 10, 0, 0)
        completed = datetime(2024, 1, 1, 10, 0, 15)

        metrics = TaskMetrics(
            task_id="task-1",
            checkpoint_name="test",
            submitted_at=submitted,
            completed_at=completed,
        )

        assert metrics.total_time_ms == 15000.0


class TestMonitoringEvent:
    """Tests for MonitoringEvent data class."""

    def test_creation(self) -> None:
        """Test event creation."""
        event = MonitoringEvent(
            event_type=MonitoringEventType.TASK_COMPLETED,
            source="test_collector",
            data={"task_id": "task-1"},
        )

        assert event.event_type == MonitoringEventType.TASK_COMPLETED
        assert event.source == "test_collector"
        assert event.data["task_id"] == "task-1"
        assert event.severity == AlertSeverity.INFO

    def test_to_dict(self) -> None:
        """Test to_dict serialization."""
        event = MonitoringEvent(
            event_type=MonitoringEventType.ALERT_TRIGGERED,
            source="aggregator",
            severity=AlertSeverity.WARNING,
        )

        result = event.to_dict()

        assert result["event_type"] == "alert_triggered"
        assert result["source"] == "aggregator"
        assert result["severity"] == "warning"


class TestEnums:
    """Tests for enum classes."""

    def test_metric_type_str(self) -> None:
        """Test MetricType string conversion."""
        assert str(MetricType.COUNTER) == "counter"
        assert str(MetricType.GAUGE) == "gauge"

    def test_monitoring_event_type_str(self) -> None:
        """Test MonitoringEventType string conversion."""
        assert str(MonitoringEventType.TASK_COMPLETED) == "task_completed"
        assert str(MonitoringEventType.WORKER_REGISTERED) == "worker_registered"

    def test_alert_severity_str(self) -> None:
        """Test AlertSeverity string conversion."""
        assert str(AlertSeverity.CRITICAL) == "critical"
        assert str(AlertSeverity.WARNING) == "warning"
