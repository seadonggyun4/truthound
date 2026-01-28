"""Main DistributedMonitor implementation.

This module provides the main orchestrator for distributed profiling monitoring.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable

from truthound.profiler.distributed.monitoring.callbacks import (
    CallbackChain,
    MonitorCallbackAdapter,
)
from truthound.profiler.distributed.monitoring.config import MonitorConfig
from truthound.profiler.distributed.monitoring.health_monitor import WorkerHealthMonitor
from truthound.profiler.distributed.monitoring.metrics_collector import (
    DistributedMetricsCollector,
)
from truthound.profiler.distributed.monitoring.progress_aggregator import (
    DistributedProgressAggregator,
)
from truthound.profiler.distributed.monitoring.protocols import (
    AggregatedProgress,
    EventSeverity,
    HealthStatus,
    IMonitorCallback,
    MonitorEvent,
    MonitorEventType,
    MonitorMetrics,
    TaskInfo,
    WorkerHealth,
)
from truthound.profiler.distributed.monitoring.task_tracker import TaskTracker


logger = logging.getLogger(__name__)


class DistributedMonitor:
    """Main orchestrator for distributed profiling monitoring.

    Coordinates task tracking, progress aggregation, health monitoring,
    and metrics collection for distributed profiling operations.

    Example:
        # Create monitor with configuration
        monitor = DistributedMonitor(
            config=MonitorConfig.production(),
        )

        # Add callbacks
        monitor.add_callback(ConsoleMonitorCallback())
        monitor.add_callback(LoggingMonitorCallback())

        # Start monitoring
        monitor.start(total_partitions=10)

        # Track task lifecycle
        task_id = monitor.submit_task(partition_id=0, total_rows=1000)
        monitor.start_task(task_id, worker_id="worker-1")
        monitor.update_progress(task_id, progress=0.5, rows_processed=500)
        monitor.complete_task(task_id, rows_processed=1000)

        # Get summary
        summary = monitor.get_summary()
        print(f"Completed: {summary['progress']['completed_partitions']}")

        # Stop monitoring
        monitor.stop()
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        on_progress: Callable[[AggregatedProgress], None] | None = None,
    ) -> None:
        """Initialize distributed monitor.

        Args:
            config: Monitor configuration
            on_progress: Callback for progress updates
        """
        self._config = config or MonitorConfig()
        self._on_progress = on_progress

        # Validate configuration
        errors = self._config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")

        # Initialize components
        self._callback_chain = CallbackChain()

        # Task tracker
        self._task_tracker: TaskTracker | None = None
        if self._config.enable_task_tracking:
            self._task_tracker = TaskTracker(
                config=self._config.task_tracker,
                on_event=self._dispatch_event,
            )

        # Progress aggregator
        self._progress_aggregator: DistributedProgressAggregator | None = None
        if self._config.enable_progress_aggregation:
            self._progress_aggregator = DistributedProgressAggregator(
                on_progress=self._handle_progress_update,
                on_event=self._dispatch_event,
            )

        # Health monitor
        self._health_monitor: WorkerHealthMonitor | None = None
        if self._config.enable_health_monitoring:
            self._health_monitor = WorkerHealthMonitor(
                config=self._config.health_check,
                on_event=self._dispatch_event,
            )

        # Metrics collector
        self._metrics_collector: DistributedMetricsCollector | None = None
        if self._config.enable_metrics_collection:
            self._metrics_collector = DistributedMetricsCollector(
                config=self._config.metrics,
                on_event=self._dispatch_event,
            )

        # State
        self._started = False
        self._start_time: datetime | None = None
        self._lock = threading.RLock()

        # Background tasks
        self._background_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self, total_partitions: int = 0) -> None:
        """Start monitoring.

        Args:
            total_partitions: Total number of partitions to monitor
        """
        with self._lock:
            if self._started:
                return

            self._started = True
            self._start_time = datetime.now()

            # Initialize progress aggregator
            if self._progress_aggregator and total_partitions > 0:
                self._progress_aggregator.set_total_partitions(total_partitions)

            # Start metrics collector
            if self._metrics_collector:
                self._metrics_collector.start()

            # Start callbacks
            self._callback_chain.start()

            # Start background tasks
            if self._config.enable_health_monitoring:
                self._stop_event.clear()
                self._background_thread = threading.Thread(
                    target=self._background_loop, daemon=True
                )
                self._background_thread.start()

            self._dispatch_event(MonitorEvent(
                event_type=MonitorEventType.MONITOR_START,
                severity=EventSeverity.INFO,
                message=f"Distributed monitoring started ({total_partitions} partitions)",
                metadata={"total_partitions": total_partitions},
            ))

    def stop(self) -> None:
        """Stop monitoring."""
        with self._lock:
            if not self._started:
                return

            # Stop background tasks
            self._stop_event.set()
            if self._background_thread:
                self._background_thread.join(timeout=5.0)
                self._background_thread = None

            # Get final summary
            summary = self.get_summary()

            # Stop callbacks
            self._callback_chain.stop()

            self._dispatch_event(MonitorEvent(
                event_type=MonitorEventType.MONITOR_STOP,
                severity=EventSeverity.INFO,
                message="Distributed monitoring stopped",
                metadata=summary,
            ))

            self._started = False

    def reset(self) -> None:
        """Reset monitor state."""
        with self._lock:
            if self._task_tracker:
                self._task_tracker.reset()
            if self._progress_aggregator:
                self._progress_aggregator.reset()
            if self._health_monitor:
                self._health_monitor.reset()
            if self._metrics_collector:
                self._metrics_collector.reset()

    # =========================================================================
    # Callback Management
    # =========================================================================

    def add_callback(self, callback: MonitorCallbackAdapter | IMonitorCallback) -> None:
        """Add a monitoring callback.

        Args:
            callback: Callback to add
        """
        if isinstance(callback, MonitorCallbackAdapter):
            self._callback_chain.add(callback)
        else:
            # Wrap simple callbacks
            wrapper = _SimpleCallbackWrapper(callback)
            self._callback_chain.add(wrapper)

    def remove_callback(self, callback: MonitorCallbackAdapter) -> bool:
        """Remove a monitoring callback.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was removed
        """
        return self._callback_chain.remove(callback)

    def set_progress_callback(
        self,
        callback: Callable[[AggregatedProgress], None],
    ) -> None:
        """Set progress callback.

        Args:
            callback: Callback for progress updates
        """
        self._on_progress = callback

    # =========================================================================
    # Task Tracking
    # =========================================================================

    def submit_task(
        self,
        partition_id: int,
        total_rows: int = 0,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a task for tracking.

        Args:
            partition_id: Partition being processed
            total_rows: Total rows in partition
            task_id: Task ID (auto-generated if not provided)
            metadata: Additional task metadata

        Returns:
            Task ID
        """
        if self._task_tracker is None:
            return ""

        if task_id is None:
            task_id = self._task_tracker.generate_task_id()

        self._task_tracker.submit_task(
            task_id=task_id,
            partition_id=partition_id,
            total_rows=total_rows,
            metadata=metadata,
        )

        if self._metrics_collector:
            self._metrics_collector.record_task_submitted()

        if self._progress_aggregator:
            self._progress_aggregator.set_partition_rows(partition_id, total_rows)

        return task_id

    def start_task(self, task_id: str, worker_id: str) -> None:
        """Record task start.

        Args:
            task_id: Task identifier
            worker_id: Worker executing the task
        """
        if self._task_tracker is None:
            return

        task = self._task_tracker.get_task(task_id)
        wait_time = 0.0
        if task and task.submitted_at:
            wait_time = (datetime.now() - task.submitted_at).total_seconds()

        self._task_tracker.start_task(task_id, worker_id)

        if self._metrics_collector:
            self._metrics_collector.record_task_started(wait_time)

        if self._progress_aggregator and task:
            self._progress_aggregator.start_partition(task.partition_id)

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
        if self._task_tracker is None:
            return

        self._task_tracker.update_progress(task_id, progress, rows_processed)

        task = self._task_tracker.get_task(task_id)
        if task and self._progress_aggregator:
            self._progress_aggregator.update_partition(
                task.partition_id, progress, rows_processed
            )

    def complete_task(self, task_id: str, rows_processed: int = 0) -> None:
        """Record task completion.

        Args:
            task_id: Task identifier
            rows_processed: Final row count
        """
        if self._task_tracker is None:
            return

        task = self._task_tracker.get_task(task_id)
        if task is None:
            return

        self._task_tracker.complete_task(task_id, rows_processed)

        if self._metrics_collector:
            duration = task.duration_seconds or 0.0
            self._metrics_collector.record_task_completed(duration, rows_processed)

        if self._health_monitor and task.worker_id:
            duration = task.duration_seconds or 0.0
            self._health_monitor.record_task_complete(task.worker_id, duration, success=True)

        if self._progress_aggregator:
            self._progress_aggregator.complete_partition(task.partition_id, rows_processed)

    def fail_task(self, task_id: str, error: str, retry: bool = False) -> None:
        """Record task failure.

        Args:
            task_id: Task identifier
            error: Error message
            retry: Whether task will be retried
        """
        if self._task_tracker is None:
            return

        task = self._task_tracker.get_task(task_id)
        if task is None:
            return

        self._task_tracker.fail_task(task_id, error, retry)

        duration = task.duration_seconds or 0.0

        if retry:
            if self._metrics_collector:
                self._metrics_collector.record_task_retried()
        else:
            if self._metrics_collector:
                self._metrics_collector.record_task_failed(duration)

            if self._health_monitor and task.worker_id:
                self._health_monitor.record_task_complete(task.worker_id, duration, success=False)

            if self._progress_aggregator:
                self._progress_aggregator.fail_partition(task.partition_id, error)

    def cancel_task(self, task_id: str) -> None:
        """Cancel a task.

        Args:
            task_id: Task identifier
        """
        if self._task_tracker:
            self._task_tracker.cancel_task(task_id)

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task information.

        Args:
            task_id: Task identifier

        Returns:
            Task info or None
        """
        if self._task_tracker is None:
            return None
        return self._task_tracker.get_task(task_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all tracked tasks.

        Returns:
            List of task info
        """
        if self._task_tracker is None:
            return []
        return self._task_tracker.get_all_tasks()

    # =========================================================================
    # Worker Health
    # =========================================================================

    def register_worker(
        self,
        worker_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a worker.

        Args:
            worker_id: Worker identifier
            metadata: Worker metadata
        """
        if self._health_monitor:
            self._health_monitor.register_worker(worker_id, metadata)

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker.

        Args:
            worker_id: Worker identifier
        """
        if self._health_monitor:
            self._health_monitor.unregister_worker(worker_id)

    def record_heartbeat(
        self,
        worker_id: str,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        memory_used_mb: float = 0.0,
        active_tasks: int = 0,
    ) -> None:
        """Record worker heartbeat.

        Args:
            worker_id: Worker identifier
            cpu_percent: CPU usage
            memory_percent: Memory usage percent
            memory_used_mb: Memory used in MB
            active_tasks: Active task count
        """
        if self._health_monitor:
            self._health_monitor.record_heartbeat(
                worker_id, cpu_percent, memory_percent, memory_used_mb, active_tasks
            )

        if self._metrics_collector:
            self._metrics_collector.record_resource_usage(memory_used_mb, cpu_percent)

    def get_worker_health(self, worker_id: str) -> WorkerHealth | None:
        """Get worker health.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker health or None
        """
        if self._health_monitor is None:
            return None
        return self._health_monitor.get_worker_health(worker_id)

    def get_all_workers(self) -> list[WorkerHealth]:
        """Get all worker health.

        Returns:
            List of worker health
        """
        if self._health_monitor is None:
            return []
        return self._health_monitor.get_all_workers()

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health.

        Returns:
            Overall health status
        """
        if self._health_monitor is None:
            return HealthStatus.UNKNOWN
        return self._health_monitor.get_overall_health()

    # =========================================================================
    # Progress
    # =========================================================================

    def get_progress(self) -> AggregatedProgress:
        """Get aggregated progress.

        Returns:
            Aggregated progress
        """
        if self._progress_aggregator is None:
            return AggregatedProgress()
        return self._progress_aggregator.get_progress()

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> MonitorMetrics:
        """Get current metrics.

        Returns:
            Current metrics
        """
        if self._metrics_collector is None:
            return MonitorMetrics()
        return self._metrics_collector.get_metrics()

    # =========================================================================
    # Summary
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """Get monitoring summary.

        Returns:
            Summary dictionary with all monitoring data
        """
        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()

        summary: dict[str, Any] = {
            "started": self._started,
            "elapsed_seconds": elapsed,
        }

        if self._task_tracker:
            summary["tasks"] = self._task_tracker.get_statistics()

        if self._progress_aggregator:
            summary["progress"] = self._progress_aggregator.get_progress().to_dict()

        if self._health_monitor:
            summary["health"] = self._health_monitor.get_statistics()

        if self._metrics_collector:
            summary["metrics"] = self._metrics_collector.get_metrics().to_dict()

        return summary

    # =========================================================================
    # Internal
    # =========================================================================

    def _dispatch_event(self, event: MonitorEvent) -> None:
        """Dispatch event to all callbacks.

        Args:
            event: Event to dispatch
        """
        if not self._config.callbacks.enable_callbacks:
            return

        try:
            self._callback_chain.on_event(event)
        except Exception as e:
            if self._config.callbacks.error_handling == "raise":
                raise
            elif self._config.callbacks.error_handling == "log":
                logger.warning(f"Error in monitor callback: {e}")

    def _handle_progress_update(self, progress: AggregatedProgress) -> None:
        """Handle progress update from aggregator.

        Args:
            progress: Aggregated progress
        """
        # Update metrics collector with worker counts
        if self._metrics_collector and self._health_monitor:
            workers = self._health_monitor.get_all_workers()
            healthy = sum(1 for w in workers if w.is_healthy)
            self._metrics_collector.record_worker_count(len(workers), healthy)

        # Dispatch to external callback
        if self._on_progress:
            try:
                self._on_progress(progress)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")

    def _background_loop(self) -> None:
        """Background loop for periodic tasks."""
        health_check_interval = self._config.health_check.heartbeat_interval_seconds
        metrics_interval = self._config.metrics.collection_interval_seconds

        last_health_check = time.time()
        last_metrics_emit = time.time()

        while not self._stop_event.is_set():
            now = time.time()

            # Health check
            if self._health_monitor and now - last_health_check >= health_check_interval:
                self._health_monitor.check_stalled_workers()
                last_health_check = now

            # Task timeout check
            if self._task_tracker:
                self._task_tracker.check_timeouts()

            # Metrics snapshot
            if self._metrics_collector and now - last_metrics_emit >= metrics_interval:
                self._metrics_collector.emit_metrics_snapshot()
                last_metrics_emit = now

            # Sleep briefly
            self._stop_event.wait(timeout=1.0)


class _SimpleCallbackWrapper(MonitorCallbackAdapter):
    """Wrapper for simple IMonitorCallback implementations."""

    def __init__(self, callback: IMonitorCallback) -> None:
        super().__init__()
        self._callback = callback

    def _handle_event(self, event: MonitorEvent) -> None:
        self._callback.on_event(event)
