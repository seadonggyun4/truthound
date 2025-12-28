"""In-memory metric collector.

Collects metrics from in-memory data structures, useful for local
development, testing, and single-process deployments.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
    MonitoringEvent,
    MonitoringEventType,
    AlertSeverity,
    CollectorError,
)
from truthound.checkpoint.monitoring.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


@dataclass
class InMemoryTask:
    """In-memory task representation."""

    task_id: str
    checkpoint_name: str
    queue_name: str
    state: str = "pending"
    worker_id: str | None = None
    submitted_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retries: int = 0
    error: str | None = None
    priority: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InMemoryWorker:
    """In-memory worker representation."""

    worker_id: str
    hostname: str
    state: str = "online"
    max_concurrency: int = 4
    current_tasks: set[str] = field(default_factory=set)
    completed_count: int = 0
    failed_count: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    tags: set[str] = field(default_factory=set)


@dataclass
class InMemoryQueue:
    """In-memory queue representation."""

    queue_name: str
    pending_tasks: list[str] = field(default_factory=list)
    running_tasks: list[str] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    total_wait_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)


class InMemoryCollector(BaseCollector):
    """In-memory metric collector.

    Stores task, worker, and queue state in memory and collects
    metrics from these data structures. Thread-safe for concurrent access.

    Example:
        >>> collector = InMemoryCollector()
        >>> await collector.connect()
        >>>
        >>> # Register a worker
        >>> collector.register_worker("worker-1", "localhost")
        >>>
        >>> # Submit a task
        >>> task_id = collector.submit_task("my_checkpoint", "default")
        >>>
        >>> # Get metrics
        >>> queue_metrics = await collector.collect_queue_metrics()
        >>> print(queue_metrics[0].pending_count)
        1
    """

    def __init__(
        self,
        name: str = "in_memory",
        collect_interval_seconds: float = 1.0,
        cache_ttl_seconds: float = 0.5,
    ) -> None:
        """Initialize in-memory collector.

        Args:
            name: Collector name.
            collect_interval_seconds: Collection interval.
            cache_ttl_seconds: Cache TTL.
        """
        super().__init__(
            name=name,
            collect_interval_seconds=collect_interval_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._lock = threading.RLock()
        self._tasks: dict[str, InMemoryTask] = {}
        self._workers: dict[str, InMemoryWorker] = {}
        self._queues: dict[str, InMemoryQueue] = {}

        # Timing tracking
        self._collection_start: datetime | None = None
        self._metrics_window_start: datetime | None = None

    async def connect(self) -> None:
        """Initialize the collector."""
        await super().connect()
        self._collection_start = datetime.now()
        self._metrics_window_start = datetime.now()

        # Create default queue if none exists
        self.ensure_queue("default")

    def ensure_queue(self, queue_name: str, labels: dict[str, str] | None = None) -> InMemoryQueue:
        """Ensure a queue exists.

        Args:
            queue_name: Queue name.
            labels: Optional queue labels.

        Returns:
            The queue.
        """
        with self._lock:
            if queue_name not in self._queues:
                self._queues[queue_name] = InMemoryQueue(
                    queue_name=queue_name,
                    labels=labels or {},
                )
                self._emit_event(MonitoringEvent(
                    event_type=MonitoringEventType.QUEUE_CREATED,
                    source=self.name,
                    data={"queue_name": queue_name},
                ))
            return self._queues[queue_name]

    def register_worker(
        self,
        worker_id: str,
        hostname: str,
        max_concurrency: int = 4,
        tags: set[str] | None = None,
    ) -> InMemoryWorker:
        """Register a new worker.

        Args:
            worker_id: Unique worker ID.
            hostname: Worker hostname.
            max_concurrency: Maximum concurrent tasks.
            tags: Worker tags.

        Returns:
            The registered worker.
        """
        with self._lock:
            worker = InMemoryWorker(
                worker_id=worker_id,
                hostname=hostname,
                max_concurrency=max_concurrency,
                tags=tags or set(),
            )
            self._workers[worker_id] = worker

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.WORKER_REGISTERED,
                source=self.name,
                data={
                    "worker_id": worker_id,
                    "hostname": hostname,
                    "max_concurrency": max_concurrency,
                },
            ))

            return worker

    def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker.

        Args:
            worker_id: Worker to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        with self._lock:
            worker = self._workers.pop(worker_id, None)
            if worker is None:
                return False

            # Fail any running tasks
            for task_id in list(worker.current_tasks):
                self.fail_task(task_id, "Worker unregistered")

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.WORKER_UNREGISTERED,
                source=self.name,
                data={"worker_id": worker_id},
            ))

            return True

    def update_worker_heartbeat(
        self,
        worker_id: str,
        cpu_percent: float = 0.0,
        memory_mb: float = 0.0,
    ) -> bool:
        """Update worker heartbeat.

        Args:
            worker_id: Worker ID.
            cpu_percent: Current CPU usage.
            memory_mb: Current memory usage.

        Returns:
            True if updated, False if worker not found.
        """
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return False

            worker.last_heartbeat = datetime.now()
            worker.cpu_percent = cpu_percent
            worker.memory_mb = memory_mb

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.WORKER_HEARTBEAT,
                source=self.name,
                data={
                    "worker_id": worker_id,
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                },
            ))

            return True

    def submit_task(
        self,
        checkpoint_name: str,
        queue_name: str = "default",
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a new task.

        Args:
            checkpoint_name: Checkpoint to run.
            queue_name: Queue to submit to.
            priority: Task priority (0-10).
            metadata: Task metadata.

        Returns:
            Task ID.
        """
        task_id = f"task-{uuid4().hex[:16]}"

        with self._lock:
            # Ensure queue exists
            queue = self.ensure_queue(queue_name)

            task = InMemoryTask(
                task_id=task_id,
                checkpoint_name=checkpoint_name,
                queue_name=queue_name,
                priority=priority,
                metadata=metadata or {},
            )
            self._tasks[task_id] = task

            # Add to queue
            queue.pending_tasks.append(task_id)

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.TASK_SUBMITTED,
                source=self.name,
                data={
                    "task_id": task_id,
                    "checkpoint_name": checkpoint_name,
                    "queue_name": queue_name,
                },
            ))

        return task_id

    def start_task(self, task_id: str, worker_id: str) -> bool:
        """Start a task on a worker.

        Args:
            task_id: Task to start.
            worker_id: Worker to run on.

        Returns:
            True if started, False otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            worker = self._workers.get(worker_id)

            if task is None or worker is None:
                return False

            if task.state != "pending":
                return False

            if len(worker.current_tasks) >= worker.max_concurrency:
                return False

            # Update task state
            task.state = "running"
            task.worker_id = worker_id
            task.started_at = datetime.now()

            # Update queue
            queue = self._queues.get(task.queue_name)
            if queue:
                if task_id in queue.pending_tasks:
                    queue.pending_tasks.remove(task_id)
                queue.running_tasks.append(task_id)

                # Update wait time
                wait_time = (task.started_at - task.submitted_at).total_seconds() * 1000
                queue.total_wait_time_ms += wait_time

            # Update worker
            worker.current_tasks.add(task_id)

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.TASK_STARTED,
                source=self.name,
                data={
                    "task_id": task_id,
                    "worker_id": worker_id,
                    "checkpoint_name": task.checkpoint_name,
                },
            ))

            return True

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task to complete.

        Returns:
            True if completed, False otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state != "running":
                return False

            task.state = "succeeded"
            task.completed_at = datetime.now()

            # Update queue
            queue = self._queues.get(task.queue_name)
            if queue:
                if task_id in queue.running_tasks:
                    queue.running_tasks.remove(task_id)
                queue.completed_count += 1

                # Update execution time
                if task.started_at:
                    exec_time = (task.completed_at - task.started_at).total_seconds() * 1000
                    queue.total_execution_time_ms += exec_time

            # Update worker
            if task.worker_id:
                worker = self._workers.get(task.worker_id)
                if worker:
                    worker.current_tasks.discard(task_id)
                    worker.completed_count += 1

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.TASK_COMPLETED,
                source=self.name,
                data={
                    "task_id": task_id,
                    "checkpoint_name": task.checkpoint_name,
                },
            ))

            return True

    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed.

        Args:
            task_id: Task to fail.
            error: Error message.

        Returns:
            True if failed, False otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            task.state = "failed"
            task.error = error
            task.completed_at = datetime.now()

            # Update queue
            queue = self._queues.get(task.queue_name)
            if queue:
                if task_id in queue.pending_tasks:
                    queue.pending_tasks.remove(task_id)
                if task_id in queue.running_tasks:
                    queue.running_tasks.remove(task_id)
                queue.failed_count += 1

            # Update worker
            if task.worker_id:
                worker = self._workers.get(task.worker_id)
                if worker:
                    worker.current_tasks.discard(task_id)
                    worker.failed_count += 1

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.TASK_FAILED,
                source=self.name,
                data={
                    "task_id": task_id,
                    "checkpoint_name": task.checkpoint_name,
                    "error": error,
                },
                severity=AlertSeverity.ERROR,
            ))

            return True

    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task.

        Args:
            task_id: Task to retry.

        Returns:
            True if retrying, False otherwise.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state not in ("failed", "timeout"):
                return False

            task.state = "pending"
            task.retries += 1
            task.error = None
            task.worker_id = None
            task.started_at = None
            task.completed_at = None

            # Re-add to queue
            queue = self.ensure_queue(task.queue_name)
            queue.pending_tasks.append(task_id)

            self._emit_event(MonitoringEvent(
                event_type=MonitoringEventType.TASK_RETRYING,
                source=self.name,
                data={
                    "task_id": task_id,
                    "retries": task.retries,
                },
            ))

            return True

    async def collect_queue_metrics(self) -> list[QueueMetrics]:
        """Collect queue metrics."""
        metrics = []

        with self._lock:
            window_seconds = 60.0  # Calculate rates over last minute
            if self._metrics_window_start:
                window_seconds = max(
                    1.0,
                    (datetime.now() - self._metrics_window_start).total_seconds(),
                )

            for queue in self._queues.values():
                pending = len(queue.pending_tasks)
                running = len(queue.running_tasks)
                completed = queue.completed_count
                failed = queue.failed_count

                # Calculate averages
                total_finished = completed + failed
                avg_wait = queue.total_wait_time_ms / max(1, total_finished)
                avg_exec = queue.total_execution_time_ms / max(1, total_finished)

                # Calculate throughput
                throughput = total_finished / window_seconds

                metrics.append(QueueMetrics(
                    queue_name=queue.queue_name,
                    pending_count=pending,
                    running_count=running,
                    completed_count=completed,
                    failed_count=failed,
                    avg_wait_time_ms=avg_wait,
                    avg_execution_time_ms=avg_exec,
                    throughput_per_second=throughput,
                    labels=queue.labels,
                ))

        return metrics

    async def collect_worker_metrics(self) -> list[WorkerMetrics]:
        """Collect worker metrics."""
        metrics = []

        with self._lock:
            for worker in self._workers.values():
                uptime = (datetime.now() - worker.registered_at).total_seconds()

                metrics.append(WorkerMetrics(
                    worker_id=worker.worker_id,
                    state=worker.state,
                    current_tasks=len(worker.current_tasks),
                    completed_tasks=worker.completed_count,
                    failed_tasks=worker.failed_count,
                    cpu_percent=worker.cpu_percent,
                    memory_mb=worker.memory_mb,
                    uptime_seconds=uptime,
                    last_heartbeat=worker.last_heartbeat,
                    hostname=worker.hostname,
                    max_concurrency=worker.max_concurrency,
                    tags=frozenset(worker.tags),
                ))

        return metrics

    async def collect_task_metrics(
        self,
        task_ids: list[str] | None = None,
    ) -> list[TaskMetrics]:
        """Collect task metrics."""
        metrics = []

        with self._lock:
            if task_ids is not None:
                tasks = [self._tasks[tid] for tid in task_ids if tid in self._tasks]
            else:
                # Return active tasks only by default
                tasks = [
                    t for t in self._tasks.values()
                    if t.state in ("pending", "running", "retrying")
                ]

            for task in tasks:
                metrics.append(TaskMetrics(
                    task_id=task.task_id,
                    checkpoint_name=task.checkpoint_name,
                    state=task.state,
                    queue_name=task.queue_name,
                    worker_id=task.worker_id,
                    submitted_at=task.submitted_at,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                    retries=task.retries,
                    error=task.error,
                ))

        return metrics

    def get_task(self, task_id: str) -> InMemoryTask | None:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_worker(self, worker_id: str) -> InMemoryWorker | None:
        """Get a worker by ID."""
        with self._lock:
            return self._workers.get(worker_id)

    def get_queue(self, queue_name: str) -> InMemoryQueue | None:
        """Get a queue by name."""
        with self._lock:
            return self._queues.get(queue_name)

    def clear(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._tasks.clear()
            self._workers.clear()
            self._queues.clear()
            self._metrics_window_start = datetime.now()
