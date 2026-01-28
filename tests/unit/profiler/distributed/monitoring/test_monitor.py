"""Tests for DistributedMonitor."""

import time

import pytest

from truthound.profiler.distributed.monitoring.callbacks import (
    ConsoleMonitorCallback,
    LoggingMonitorCallback,
)
from truthound.profiler.distributed.monitoring.config import MonitorConfig
from truthound.profiler.distributed.monitoring.monitor import DistributedMonitor
from truthound.profiler.distributed.monitoring.protocols import (
    AggregatedProgress,
    HealthStatus,
    MonitorEvent,
    TaskState,
)


class TestDistributedMonitor:
    """Tests for DistributedMonitor."""

    def test_create_with_default_config(self) -> None:
        """Test creation with default config."""
        monitor = DistributedMonitor()

        assert monitor is not None
        assert not monitor._started

    def test_create_with_config(self) -> None:
        """Test creation with custom config."""
        config = MonitorConfig.minimal()
        monitor = DistributedMonitor(config=config)

        assert monitor is not None
        assert monitor._config == config

    def test_invalid_config_raises(self) -> None:
        """Test that invalid config raises error."""
        config = MonitorConfig()
        config.task_tracker.max_history_size = -1

        with pytest.raises(ValueError, match="Invalid configuration"):
            DistributedMonitor(config=config)

    def test_start_stop(self) -> None:
        """Test start and stop."""
        monitor = DistributedMonitor()

        monitor.start(total_partitions=5)
        assert monitor._started

        monitor.stop()
        assert not monitor._started

    def test_submit_and_get_task(self) -> None:
        """Test task submission and retrieval."""
        monitor = DistributedMonitor()
        monitor.start()

        task_id = monitor.submit_task(partition_id=0, total_rows=1000)

        task = monitor.get_task(task_id)
        assert task is not None
        assert task.partition_id == 0
        assert task.total_rows == 1000
        assert task.state == TaskState.SUBMITTED

        monitor.stop()

    def test_task_lifecycle(self) -> None:
        """Test full task lifecycle."""
        monitor = DistributedMonitor()
        monitor.start()

        # Submit
        task_id = monitor.submit_task(partition_id=0, total_rows=1000)

        # Start
        monitor.start_task(task_id, worker_id="worker-1")
        task = monitor.get_task(task_id)
        assert task is not None
        assert task.state == TaskState.RUNNING

        # Update progress
        monitor.update_progress(task_id, progress=0.5, rows_processed=500)

        # Complete
        monitor.complete_task(task_id, rows_processed=1000)
        task = monitor.get_task(task_id)
        assert task is not None
        assert task.state == TaskState.COMPLETED

        monitor.stop()

    def test_task_failure(self) -> None:
        """Test task failure."""
        monitor = DistributedMonitor()
        monitor.start()

        task_id = monitor.submit_task(partition_id=0)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.fail_task(task_id, error="Test error")

        task = monitor.get_task(task_id)
        assert task is not None
        assert task.state == TaskState.FAILED

        monitor.stop()

    def test_task_cancellation(self) -> None:
        """Test task cancellation."""
        monitor = DistributedMonitor()
        monitor.start()

        task_id = monitor.submit_task(partition_id=0)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.cancel_task(task_id)

        task = monitor.get_task(task_id)
        assert task is not None
        assert task.state == TaskState.CANCELLED

        monitor.stop()

    def test_get_all_tasks(self) -> None:
        """Test getting all tasks."""
        monitor = DistributedMonitor()
        monitor.start()

        monitor.submit_task(partition_id=0)
        monitor.submit_task(partition_id=1)
        monitor.submit_task(partition_id=2)

        tasks = monitor.get_all_tasks()
        assert len(tasks) == 3

        monitor.stop()

    def test_progress_aggregation(self) -> None:
        """Test progress aggregation."""
        monitor = DistributedMonitor()
        monitor.start(total_partitions=2)

        task_id = monitor.submit_task(partition_id=0, total_rows=1000)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.complete_task(task_id, rows_processed=1000)

        progress = monitor.get_progress()
        assert progress.completed_partitions == 1
        assert progress.total_partitions == 2

        monitor.stop()

    def test_worker_health_tracking(self) -> None:
        """Test worker health tracking."""
        config = MonitorConfig(enable_health_monitoring=True)
        monitor = DistributedMonitor(config=config)
        monitor.start()

        monitor.register_worker("worker-1", metadata={"cpu_count": 8})
        monitor.record_heartbeat(
            "worker-1",
            cpu_percent=50.0,
            memory_percent=60.0,
            active_tasks=2,
        )

        health = monitor.get_worker_health("worker-1")
        assert health is not None
        assert health.cpu_percent == 50.0
        assert health.memory_percent == 60.0

        monitor.stop()

    def test_get_all_workers(self) -> None:
        """Test getting all workers."""
        config = MonitorConfig(enable_health_monitoring=True)
        monitor = DistributedMonitor(config=config)
        monitor.start()

        monitor.register_worker("worker-1")
        monitor.register_worker("worker-2")
        monitor.record_heartbeat("worker-1", cpu_percent=50)
        monitor.record_heartbeat("worker-2", cpu_percent=60)

        workers = monitor.get_all_workers()
        assert len(workers) == 2

        monitor.stop()

    def test_overall_health(self) -> None:
        """Test overall system health."""
        config = MonitorConfig(enable_health_monitoring=True)
        monitor = DistributedMonitor(config=config)
        monitor.start()

        monitor.register_worker("worker-1")
        monitor.record_heartbeat("worker-1", cpu_percent=50, memory_percent=60)

        health = monitor.get_overall_health()
        assert health == HealthStatus.HEALTHY

        monitor.stop()

    def test_metrics(self) -> None:
        """Test metrics collection."""
        config = MonitorConfig(enable_metrics_collection=True)
        monitor = DistributedMonitor(config=config)
        monitor.start()

        task_id = monitor.submit_task(partition_id=0)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.complete_task(task_id, rows_processed=1000)

        metrics = monitor.get_metrics()
        assert metrics.tasks_total == 1
        assert metrics.tasks_completed == 1

        monitor.stop()

    def test_summary(self) -> None:
        """Test summary generation."""
        monitor = DistributedMonitor()
        monitor.start(total_partitions=2)

        task_id = monitor.submit_task(partition_id=0)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.complete_task(task_id)

        summary = monitor.get_summary()

        assert "tasks" in summary
        assert "progress" in summary
        assert summary["started"]

        monitor.stop()

    def test_add_callback(self) -> None:
        """Test adding callbacks."""
        events: list[MonitorEvent] = []
        monitor = DistributedMonitor()

        callback = ConsoleMonitorCallback(enabled=False)  # Don't print
        monitor.add_callback(callback)

        monitor.start()
        monitor.stop()

        # Callback should be in chain
        assert len(monitor._callback_chain) == 1

    def test_progress_callback(self) -> None:
        """Test progress callback."""
        progress_updates: list[AggregatedProgress] = []
        monitor = DistributedMonitor(
            on_progress=lambda p: progress_updates.append(p)
        )
        monitor.start(total_partitions=2)

        task_id = monitor.submit_task(partition_id=0)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.complete_task(task_id)

        # Should have progress updates
        assert len(progress_updates) >= 1

        monitor.stop()

    def test_reset(self) -> None:
        """Test reset."""
        monitor = DistributedMonitor()
        monitor.start(total_partitions=2)

        task_id = monitor.submit_task(partition_id=0)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.complete_task(task_id)

        monitor.reset()

        assert monitor.get_all_tasks() == []
        progress = monitor.get_progress()
        assert progress.total_partitions == 0

        monitor.stop()

    def test_disabled_features(self) -> None:
        """Test with features disabled."""
        config = MonitorConfig(
            enable_task_tracking=False,
            enable_progress_aggregation=False,
            enable_health_monitoring=False,
            enable_metrics_collection=False,
        )
        monitor = DistributedMonitor(config=config)
        monitor.start()

        # These should not error
        task_id = monitor.submit_task(partition_id=0)
        assert task_id == ""

        monitor.register_worker("worker-1")
        assert monitor.get_worker_health("worker-1") is None

        metrics = monitor.get_metrics()
        assert metrics.tasks_total == 0

        monitor.stop()


class TestMonitorPresets:
    """Tests for monitor configuration presets."""

    def test_minimal_preset(self) -> None:
        """Test minimal preset."""
        config = MonitorConfig.minimal()

        assert config.enable_monitoring
        assert config.enable_task_tracking
        assert not config.enable_health_monitoring
        assert not config.enable_metrics_collection

    def test_standard_preset(self) -> None:
        """Test standard preset."""
        config = MonitorConfig.standard()

        assert config.enable_monitoring
        assert config.enable_task_tracking
        assert config.enable_progress_aggregation
        assert config.enable_health_monitoring
        assert config.enable_metrics_collection

    def test_full_preset(self) -> None:
        """Test full preset."""
        config = MonitorConfig.full()

        assert config.enable_monitoring
        assert config.callbacks.enable_callbacks
        assert config.callbacks.async_dispatch
        assert config.metrics.enable_prometheus

    def test_production_preset(self) -> None:
        """Test production preset."""
        config = MonitorConfig.production()

        assert config.enable_monitoring
        assert config.task_tracker.timeout_seconds > 0
        assert config.task_tracker.max_retries > 0
