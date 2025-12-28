"""Core protocols and data classes for Job Queue Monitoring.

This module defines the foundational abstractions for the monitoring system,
following Protocol-First design principles for loose coupling and extensibility.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from truthound.checkpoint.distributed.protocols import TaskState, WorkerState


# =============================================================================
# Exceptions
# =============================================================================


class MonitoringError(Exception):
    """Base exception for monitoring errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CollectorError(MonitoringError):
    """Raised when metric collection fails."""

    def __init__(
        self,
        message: str,
        collector_name: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(message, {"collector": collector_name, "reason": reason})
        self.collector_name = collector_name
        self.reason = reason


class AggregatorError(MonitoringError):
    """Raised when metric aggregation fails."""

    def __init__(
        self,
        message: str,
        aggregator_name: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(message, {"aggregator": aggregator_name, "reason": reason})
        self.aggregator_name = aggregator_name
        self.reason = reason


# =============================================================================
# Enums
# =============================================================================


class MetricType(str, Enum):
    """Type of metric being collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

    def __str__(self) -> str:
        return self.value


class MonitoringEventType(str, Enum):
    """Types of monitoring events."""

    # Queue events
    QUEUE_CREATED = "queue_created"
    QUEUE_DELETED = "queue_deleted"
    QUEUE_SIZE_CHANGED = "queue_size_changed"
    QUEUE_THRESHOLD_EXCEEDED = "queue_threshold_exceeded"

    # Worker events
    WORKER_REGISTERED = "worker_registered"
    WORKER_UNREGISTERED = "worker_unregistered"
    WORKER_STATE_CHANGED = "worker_state_changed"
    WORKER_HEARTBEAT = "worker_heartbeat"
    WORKER_OVERLOADED = "worker_overloaded"

    # Task events
    TASK_SUBMITTED = "task_submitted"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRYING = "task_retrying"
    TASK_TIMEOUT = "task_timeout"
    TASK_CANCELLED = "task_cancelled"

    # System events
    METRICS_COLLECTED = "metrics_collected"
    ALERT_TRIGGERED = "alert_triggered"
    THRESHOLD_CROSSED = "threshold_crossed"

    def __str__(self) -> str:
        return self.value


class AlertSeverity(str, Enum):
    """Severity levels for monitoring alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class QueueMetrics:
    """Metrics for a task queue.

    Attributes:
        queue_name: Name of the queue.
        pending_count: Number of tasks waiting to be processed.
        running_count: Number of tasks currently running.
        completed_count: Number of completed tasks.
        failed_count: Number of failed tasks.
        avg_wait_time_ms: Average time tasks wait in queue.
        avg_execution_time_ms: Average task execution time.
        throughput_per_second: Tasks processed per second.
        timestamp: When metrics were collected.
        labels: Additional labels for the queue.
    """

    queue_name: str
    pending_count: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    avg_wait_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def total_tasks(self) -> int:
        """Get total number of tasks across all states."""
        return self.pending_count + self.running_count + self.completed_count + self.failed_count

    @property
    def success_rate(self) -> float:
        """Get success rate (0.0 to 1.0)."""
        total_finished = self.completed_count + self.failed_count
        if total_finished == 0:
            return 1.0
        return self.completed_count / total_finished

    @property
    def is_healthy(self) -> bool:
        """Check if queue is healthy (not backing up)."""
        # Healthy if pending tasks aren't growing excessively
        return self.pending_count < 1000 and self.success_rate > 0.9

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "queue_name": self.queue_name,
            "pending_count": self.pending_count,
            "running_count": self.running_count,
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "throughput_per_second": self.throughput_per_second,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "total_tasks": self.total_tasks,
            "success_rate": self.success_rate,
            "is_healthy": self.is_healthy,
        }


@dataclass(frozen=True)
class WorkerMetrics:
    """Metrics for a worker node.

    Attributes:
        worker_id: Unique worker identifier.
        state: Current worker state.
        current_tasks: Number of tasks currently running.
        completed_tasks: Number of completed tasks.
        failed_tasks: Number of failed tasks.
        cpu_percent: CPU utilization percentage.
        memory_mb: Memory usage in megabytes.
        uptime_seconds: Worker uptime in seconds.
        last_heartbeat: Last heartbeat timestamp.
        hostname: Worker hostname.
        tags: Worker tags for routing.
        metadata: Additional metadata.
    """

    worker_id: str
    state: str = "unknown"
    current_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    hostname: str = ""
    max_concurrency: int = 1
    tags: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def load_factor(self) -> float:
        """Get worker load factor (0.0 to 1.0)."""
        if self.max_concurrency == 0:
            return 1.0
        return self.current_tasks / self.max_concurrency

    @property
    def is_available(self) -> bool:
        """Check if worker can accept new tasks."""
        return self.state == "online" and self.current_tasks < self.max_concurrency

    @property
    def available_slots(self) -> int:
        """Get number of available task slots."""
        if self.state != "online":
            return 0
        return max(0, self.max_concurrency - self.current_tasks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "state": self.state,
            "current_tasks": self.current_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "uptime_seconds": self.uptime_seconds,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "hostname": self.hostname,
            "max_concurrency": self.max_concurrency,
            "load_factor": self.load_factor,
            "is_available": self.is_available,
            "available_slots": self.available_slots,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskMetrics:
    """Metrics for a specific task.

    Attributes:
        task_id: Unique task identifier.
        checkpoint_name: Name of the checkpoint.
        state: Current task state.
        queue_name: Queue the task is in.
        worker_id: Worker executing the task.
        submitted_at: When task was submitted.
        started_at: When task started.
        completed_at: When task completed.
        retries: Number of retry attempts.
        error: Error message if failed.
    """

    task_id: str
    checkpoint_name: str
    state: str = "pending"
    queue_name: str = "default"
    worker_id: str | None = None
    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retries: int = 0
    error: str | None = None

    @property
    def wait_time_ms(self) -> float | None:
        """Get time spent waiting in queue."""
        if self.started_at and self.submitted_at:
            return (self.started_at - self.submitted_at).total_seconds() * 1000
        return None

    @property
    def execution_time_ms(self) -> float | None:
        """Get execution time."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def total_time_ms(self) -> float | None:
        """Get total time from submission to completion."""
        if self.completed_at and self.submitted_at:
            return (self.completed_at - self.submitted_at).total_seconds() * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "checkpoint_name": self.checkpoint_name,
            "state": self.state,
            "queue_name": self.queue_name,
            "worker_id": self.worker_id,
            "submitted_at": self.submitted_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "error": self.error,
            "wait_time_ms": self.wait_time_ms,
            "execution_time_ms": self.execution_time_ms,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class MonitoringEvent:
    """Event emitted by the monitoring system.

    Attributes:
        event_type: Type of event.
        source: Source of the event (collector, aggregator, etc.).
        timestamp: When event occurred.
        data: Event-specific data.
        severity: Alert severity if applicable.
        labels: Additional labels.
    """

    event_type: MonitoringEventType
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: dict[str, Any] = field(default_factory=dict)
    severity: AlertSeverity = AlertSeverity.INFO
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "severity": self.severity.value,
            "labels": self.labels,
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window.

    Attributes:
        window_start: Start of the aggregation window.
        window_end: End of the aggregation window.
        sample_count: Number of samples in the window.
        queue_metrics: Aggregated queue metrics.
        worker_metrics: Aggregated worker metrics.
        task_summary: Task summary for the window.
    """

    window_start: datetime
    window_end: datetime
    sample_count: int = 0
    queue_metrics: list[QueueMetrics] = field(default_factory=list)
    worker_metrics: list[WorkerMetrics] = field(default_factory=list)
    task_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def window_duration_seconds(self) -> float:
        """Get window duration in seconds."""
        return (self.window_end - self.window_start).total_seconds()


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class MetricCollectorProtocol(Protocol):
    """Protocol for metric collectors.

    Collectors are responsible for gathering metrics from various sources
    (in-memory, Redis, Prometheus, etc.) and exposing them in a unified format.
    """

    @property
    def name(self) -> str:
        """Get collector name."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if collector is connected to its data source."""
        ...

    async def connect(self) -> None:
        """Connect to the data source."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        ...

    async def collect_queue_metrics(self) -> list[QueueMetrics]:
        """Collect current queue metrics.

        Returns:
            List of queue metrics.

        Raises:
            CollectorError: If collection fails.
        """
        ...

    async def collect_worker_metrics(self) -> list[WorkerMetrics]:
        """Collect current worker metrics.

        Returns:
            List of worker metrics.

        Raises:
            CollectorError: If collection fails.
        """
        ...

    async def collect_task_metrics(self, task_ids: list[str] | None = None) -> list[TaskMetrics]:
        """Collect task metrics.

        Args:
            task_ids: Optional list of specific task IDs to collect.
                      If None, collects all active tasks.

        Returns:
            List of task metrics.

        Raises:
            CollectorError: If collection fails.
        """
        ...

    async def subscribe(self) -> AsyncIterator[MonitoringEvent]:
        """Subscribe to real-time metric updates.

        Yields:
            Monitoring events as they occur.
        """
        ...

    async def health_check(self) -> bool:
        """Check collector health.

        Returns:
            True if healthy, False otherwise.
        """
        ...


@runtime_checkable
class MetricAggregatorProtocol(Protocol):
    """Protocol for metric aggregators.

    Aggregators process raw metrics to compute statistics,
    detect trends, and generate alerts.
    """

    @property
    def name(self) -> str:
        """Get aggregator name."""
        ...

    def aggregate_queues(self, metrics: list[QueueMetrics]) -> QueueMetrics:
        """Aggregate multiple queue metrics into a single summary.

        Args:
            metrics: List of queue metrics to aggregate.

        Returns:
            Aggregated queue metrics.
        """
        ...

    def aggregate_workers(self, metrics: list[WorkerMetrics]) -> dict[str, Any]:
        """Aggregate worker metrics into a summary.

        Args:
            metrics: List of worker metrics to aggregate.

        Returns:
            Worker summary dictionary.
        """
        ...

    def calculate_rate(
        self,
        current: int,
        previous: int,
        interval_seconds: float,
    ) -> float:
        """Calculate rate of change.

        Args:
            current: Current value.
            previous: Previous value.
            interval_seconds: Time interval.

        Returns:
            Rate per second.
        """
        ...

    def detect_anomalies(
        self,
        metrics: list[QueueMetrics],
        threshold: float = 2.0,
    ) -> list[MonitoringEvent]:
        """Detect anomalies in metrics.

        Args:
            metrics: List of metrics to analyze.
            threshold: Z-score threshold for anomaly detection.

        Returns:
            List of anomaly events.
        """
        ...


@runtime_checkable
class MonitoringViewProtocol(Protocol):
    """Protocol for monitoring views.

    Views render metrics in various formats for different consumers
    (API, CLI, dashboard, etc.).
    """

    @property
    def name(self) -> str:
        """Get view name."""
        ...

    def render(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> dict[str, Any]:
        """Render metrics as a dictionary.

        Args:
            metrics: Metrics to render.

        Returns:
            Rendered view as dictionary.
        """
        ...

    def render_summary(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
    ) -> dict[str, Any]:
        """Render a summary view.

        Args:
            queue_metrics: Queue metrics.
            worker_metrics: Worker metrics.

        Returns:
            Summary view as dictionary.
        """
        ...

    def format_for_api(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> dict[str, Any]:
        """Format metrics for REST API response.

        Args:
            metrics: Metrics to format.

        Returns:
            API-formatted dictionary.
        """
        ...

    def format_for_cli(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> str:
        """Format metrics for CLI display.

        Args:
            metrics: Metrics to format.

        Returns:
            CLI-formatted string.
        """
        ...


# =============================================================================
# Type Aliases
# =============================================================================

MetricT = TypeVar("MetricT", QueueMetrics, WorkerMetrics, TaskMetrics)
CollectorT = TypeVar("CollectorT", bound=MetricCollectorProtocol)
AggregatorT = TypeVar("AggregatorT", bound=MetricAggregatorProtocol)
ViewT = TypeVar("ViewT", bound=MonitoringViewProtocol)
