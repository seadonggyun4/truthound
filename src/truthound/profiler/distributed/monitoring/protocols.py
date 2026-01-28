"""Protocols and types for distributed monitoring.

This module defines the core abstractions for the distributed monitoring system,
following a Protocol-based design for maximum extensibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Protocol, runtime_checkable


# =============================================================================
# Enums and Constants
# =============================================================================


class MonitorEventType(str, Enum):
    """Types of monitoring events."""

    # Lifecycle events
    MONITOR_START = "monitor_start"
    MONITOR_STOP = "monitor_stop"

    # Task events
    TASK_SUBMITTED = "task_submitted"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_RETRYING = "task_retrying"
    TASK_TIMEOUT = "task_timeout"

    # Partition events
    PARTITION_START = "partition_start"
    PARTITION_PROGRESS = "partition_progress"
    PARTITION_COMPLETE = "partition_complete"
    PARTITION_ERROR = "partition_error"

    # Worker events
    WORKER_REGISTERED = "worker_registered"
    WORKER_UNREGISTERED = "worker_unregistered"
    WORKER_HEALTHY = "worker_healthy"
    WORKER_UNHEALTHY = "worker_unhealthy"
    WORKER_STALLED = "worker_stalled"
    WORKER_RECOVERED = "worker_recovered"

    # Progress events
    PROGRESS_UPDATE = "progress_update"
    PROGRESS_MILESTONE = "progress_milestone"

    # Metrics events
    METRICS_SNAPSHOT = "metrics_snapshot"
    METRICS_ANOMALY = "metrics_anomaly"

    # Health events
    HEALTH_CHECK = "health_check"
    HEALTH_DEGRADED = "health_degraded"
    HEALTH_CRITICAL = "health_critical"

    # Resource events
    MEMORY_WARNING = "memory_warning"
    MEMORY_CRITICAL = "memory_critical"
    CPU_WARNING = "cpu_warning"
    BACKPRESSURE_DETECTED = "backpressure_detected"

    # Aggregation events
    AGGREGATION_START = "aggregation_start"
    AGGREGATION_COMPLETE = "aggregation_complete"


class TaskState(str, Enum):
    """States of a distributed task."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class EventSeverity(Enum):
    """Severity levels for events."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class MonitorEvent:
    """Immutable event from the monitoring system.

    Attributes:
        event_type: Type of the event
        severity: Event severity level
        timestamp: When the event occurred
        message: Human-readable description
        task_id: Associated task ID (if applicable)
        partition_id: Associated partition ID (if applicable)
        worker_id: Associated worker ID (if applicable)
        progress: Current progress (0.0 to 1.0)
        metadata: Additional event-specific data
    """

    event_type: MonitorEventType
    severity: EventSeverity = EventSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    task_id: str | None = None
    partition_id: int | None = None
    worker_id: str | None = None
    progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Check if this is an error event."""
        return self.severity in {EventSeverity.ERROR, EventSeverity.CRITICAL}

    @property
    def is_task_event(self) -> bool:
        """Check if this is a task-related event."""
        return self.event_type.value.startswith("task_")

    @property
    def is_worker_event(self) -> bool:
        """Check if this is a worker-related event."""
        return self.event_type.value.startswith("worker_")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "task_id": self.task_id,
            "partition_id": self.partition_id,
            "worker_id": self.worker_id,
            "progress": self.progress,
            "metadata": self.metadata,
        }


@dataclass
class TaskInfo:
    """Information about a tracked task.

    Attributes:
        task_id: Unique task identifier
        partition_id: Associated partition
        worker_id: Worker executing the task
        state: Current task state
        progress: Completion progress (0.0 to 1.0)
        submitted_at: When task was submitted
        started_at: When task started execution
        completed_at: When task completed
        rows_processed: Number of rows processed
        total_rows: Total rows to process
        error_message: Error message if failed
        retry_count: Number of retries attempted
        metadata: Additional task data
    """

    task_id: str
    partition_id: int
    worker_id: str | None = None
    state: TaskState = TaskState.PENDING
    progress: float = 0.0
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    rows_processed: int = 0
    total_rows: int = 0
    error_message: str | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Get task duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def wait_time_seconds(self) -> float | None:
        """Get time spent waiting before execution."""
        if self.submitted_at is None or self.started_at is None:
            return None
        return (self.started_at - self.submitted_at).total_seconds()

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.state in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.TIMEOUT,
        }

    @property
    def rows_per_second(self) -> float:
        """Calculate processing throughput."""
        duration = self.duration_seconds
        if duration is None or duration <= 0:
            return 0.0
        return self.rows_processed / duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "partition_id": self.partition_id,
            "worker_id": self.worker_id,
            "state": self.state.value,
            "progress": self.progress,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "rows_processed": self.rows_processed,
            "total_rows": self.total_rows,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }


@dataclass
class WorkerHealth:
    """Health information for a worker.

    Attributes:
        worker_id: Worker identifier
        status: Current health status
        last_heartbeat: Last heartbeat time
        cpu_percent: CPU usage percentage
        memory_percent: Memory usage percentage
        memory_used_mb: Memory used in MB
        active_tasks: Number of active tasks
        completed_tasks: Total completed tasks
        failed_tasks: Total failed tasks
        avg_task_time_seconds: Average task duration
        error_rate: Recent error rate
        metadata: Additional health data
    """

    worker_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_heartbeat: datetime | None = None
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_time_seconds: float = 0.0
    error_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tasks(self) -> int:
        """Get total tasks processed."""
        return self.completed_tasks + self.failed_tasks

    @property
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def heartbeat_age_seconds(self) -> float | None:
        """Get seconds since last heartbeat."""
        if self.last_heartbeat is None:
            return None
        return (datetime.now() - self.last_heartbeat).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "last_heartbeat": (
                self.last_heartbeat.isoformat() if self.last_heartbeat else None
            ),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "avg_task_time_seconds": self.avg_task_time_seconds,
            "error_rate": self.error_rate,
        }


@dataclass
class AggregatedProgress:
    """Aggregated progress from multiple tasks/partitions.

    Attributes:
        total_partitions: Total number of partitions
        completed_partitions: Number of completed partitions
        failed_partitions: Number of failed partitions
        in_progress_partitions: Number of partitions in progress
        overall_progress: Weighted overall progress (0.0 to 1.0)
        total_rows: Total rows across all partitions
        processed_rows: Rows processed so far
        elapsed_seconds: Time elapsed since start
        estimated_remaining_seconds: Estimated time remaining
        rows_per_second: Current throughput
        partition_progress: Per-partition progress
    """

    total_partitions: int = 0
    completed_partitions: int = 0
    failed_partitions: int = 0
    in_progress_partitions: int = 0
    overall_progress: float = 0.0
    total_rows: int = 0
    processed_rows: int = 0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    rows_per_second: float = 0.0
    partition_progress: dict[int, float] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        """Get progress as percentage."""
        return self.overall_progress * 100

    @property
    def pending_partitions(self) -> int:
        """Get number of pending partitions."""
        return (
            self.total_partitions
            - self.completed_partitions
            - self.failed_partitions
            - self.in_progress_partitions
        )

    @property
    def is_complete(self) -> bool:
        """Check if all partitions are complete."""
        return (
            self.completed_partitions + self.failed_partitions >= self.total_partitions
        )

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        total = self.completed_partitions + self.failed_partitions
        if total == 0:
            return 1.0
        return self.completed_partitions / total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_partitions": self.total_partitions,
            "completed_partitions": self.completed_partitions,
            "failed_partitions": self.failed_partitions,
            "in_progress_partitions": self.in_progress_partitions,
            "pending_partitions": self.pending_partitions,
            "overall_progress": self.overall_progress,
            "percent": self.percent,
            "total_rows": self.total_rows,
            "processed_rows": self.processed_rows,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "rows_per_second": self.rows_per_second,
            "is_complete": self.is_complete,
            "success_rate": self.success_rate,
        }


@dataclass
class MonitorMetrics:
    """Aggregated metrics from the monitoring system.

    Attributes:
        timestamp: When metrics were collected
        tasks_total: Total tasks tracked
        tasks_completed: Completed tasks
        tasks_failed: Failed tasks
        tasks_running: Currently running tasks
        tasks_pending: Pending tasks
        avg_task_duration_seconds: Average task duration
        p50_task_duration_seconds: Median task duration
        p95_task_duration_seconds: 95th percentile duration
        p99_task_duration_seconds: 99th percentile duration
        total_rows_processed: Total rows processed
        rows_per_second: Current throughput
        workers_total: Total workers
        workers_healthy: Healthy workers
        workers_unhealthy: Unhealthy workers
        memory_used_total_mb: Total memory used
        cpu_utilization_percent: Average CPU usage
        error_rate: Overall error rate
        retry_rate: Retry rate
    """

    timestamp: datetime = field(default_factory=datetime.now)
    tasks_total: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_running: int = 0
    tasks_pending: int = 0
    avg_task_duration_seconds: float = 0.0
    p50_task_duration_seconds: float = 0.0
    p95_task_duration_seconds: float = 0.0
    p99_task_duration_seconds: float = 0.0
    total_rows_processed: int = 0
    rows_per_second: float = 0.0
    workers_total: int = 0
    workers_healthy: int = 0
    workers_unhealthy: int = 0
    memory_used_total_mb: float = 0.0
    cpu_utilization_percent: float = 0.0
    error_rate: float = 0.0
    retry_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "tasks": {
                "total": self.tasks_total,
                "completed": self.tasks_completed,
                "failed": self.tasks_failed,
                "running": self.tasks_running,
                "pending": self.tasks_pending,
            },
            "duration": {
                "avg": self.avg_task_duration_seconds,
                "p50": self.p50_task_duration_seconds,
                "p95": self.p95_task_duration_seconds,
                "p99": self.p99_task_duration_seconds,
            },
            "throughput": {
                "total_rows": self.total_rows_processed,
                "rows_per_second": self.rows_per_second,
            },
            "workers": {
                "total": self.workers_total,
                "healthy": self.workers_healthy,
                "unhealthy": self.workers_unhealthy,
            },
            "resources": {
                "memory_mb": self.memory_used_total_mb,
                "cpu_percent": self.cpu_utilization_percent,
            },
            "rates": {
                "error_rate": self.error_rate,
                "retry_rate": self.retry_rate,
            },
        }


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class IMonitorCallback(Protocol):
    """Protocol for monitor event callbacks.

    Implement this to receive monitoring events from the distributed system.
    """

    def on_event(self, event: MonitorEvent) -> None:
        """Handle a monitoring event.

        Args:
            event: The monitoring event
        """
        ...


@runtime_checkable
class ITaskTracker(Protocol):
    """Protocol for tracking distributed tasks.

    Implementations track the lifecycle and progress of distributed tasks.
    """

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

    def cancel_task(self, task_id: str) -> None:
        """Record task cancellation.

        Args:
            task_id: Task identifier
        """
        ...

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task information.

        Args:
            task_id: Task identifier

        Returns:
            Task info or None if not found
        """
        ...

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all tracked tasks.

        Returns:
            List of all task info
        """
        ...


@runtime_checkable
class IProgressAggregator(Protocol):
    """Protocol for aggregating progress across partitions.

    Implementations aggregate progress from multiple distributed tasks.
    """

    def set_total_partitions(self, count: int) -> None:
        """Set total partition count.

        Args:
            count: Number of partitions
        """
        ...

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
        ...

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
        ...

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
        ...

    def get_progress(self) -> AggregatedProgress:
        """Get aggregated progress.

        Returns:
            Aggregated progress data
        """
        ...

    def reset(self) -> None:
        """Reset aggregator state."""
        ...


@runtime_checkable
class IHealthMonitor(Protocol):
    """Protocol for monitoring worker health.

    Implementations track worker health and availability.
    """

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
        ...

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker.

        Args:
            worker_id: Worker identifier
        """
        ...

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
        ...

    def record_task_complete(
        self,
        worker_id: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record task completion for a worker.

        Args:
            worker_id: Worker identifier
            duration_seconds: Task duration
            success: Whether task succeeded
        """
        ...

    def get_worker_health(self, worker_id: str) -> WorkerHealth | None:
        """Get worker health.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker health or None if not found
        """
        ...

    def get_all_workers(self) -> list[WorkerHealth]:
        """Get all worker health.

        Returns:
            List of all worker health
        """
        ...

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health.

        Returns:
            Overall health status
        """
        ...


@runtime_checkable
class IMetricsCollector(Protocol):
    """Protocol for collecting monitoring metrics.

    Implementations collect and aggregate performance metrics.
    """

    def record_task_submitted(self) -> None:
        """Record task submission."""
        ...

    def record_task_started(self, wait_time_seconds: float = 0.0) -> None:
        """Record task start.

        Args:
            wait_time_seconds: Time spent waiting
        """
        ...

    def record_task_completed(
        self,
        duration_seconds: float,
        rows_processed: int = 0,
    ) -> None:
        """Record task completion.

        Args:
            duration_seconds: Task duration
            rows_processed: Rows processed
        """
        ...

    def record_task_failed(self, duration_seconds: float = 0.0) -> None:
        """Record task failure.

        Args:
            duration_seconds: Time until failure
        """
        ...

    def record_task_retried(self) -> None:
        """Record task retry."""
        ...

    def record_rows_processed(self, count: int) -> None:
        """Record rows processed.

        Args:
            count: Number of rows
        """
        ...

    def get_metrics(self) -> MonitorMetrics:
        """Get current metrics.

        Returns:
            Current metrics snapshot
        """
        ...

    def reset(self) -> None:
        """Reset metrics."""
        ...


# =============================================================================
# Type Aliases
# =============================================================================

# Callback function type
MonitorCallbackFn = Callable[[MonitorEvent], None]

# Progress callback function type
ProgressCallbackFn = Callable[[AggregatedProgress], None]
