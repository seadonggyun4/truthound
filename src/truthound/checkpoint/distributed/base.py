"""Base Classes for Distributed Checkpoint Orchestration.

This module provides abstract base classes that backends can extend
to implement distributed checkpoint execution.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, TypeVar
from queue import Queue, Empty

from truthound.checkpoint.distributed.protocols import (
    BackendCapability,
    ClusterState,
    DistributedBackendProtocol,
    DistributedConfig,
    DistributedError,
    DistributedTask,
    DistributedTaskProtocol,
    DistributedTaskResult,
    TaskCancelledError,
    TaskPriority,
    TaskState,
    TaskSubmissionError,
    TaskTimeoutError,
    WorkerInfo,
    WorkerState,
    WorkerNotAvailableError,
    BackendNotAvailableError,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import Checkpoint, CheckpointResult


logger = logging.getLogger(__name__)


T = TypeVar("T")


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class TaskMetrics:
    """Metrics for task execution.

    Attributes:
        tasks_submitted: Total tasks submitted.
        tasks_succeeded: Tasks that succeeded.
        tasks_failed: Tasks that failed.
        tasks_cancelled: Tasks that were cancelled.
        tasks_timed_out: Tasks that timed out.
        total_execution_time_ms: Total execution time in milliseconds.
        total_queue_time_ms: Total time spent in queue.
        retries: Total retry attempts.
    """

    tasks_submitted: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    tasks_timed_out: int = 0
    total_execution_time_ms: float = 0.0
    total_queue_time_ms: float = 0.0
    retries: int = 0

    @property
    def tasks_completed(self) -> int:
        """Total completed tasks (success + failed)."""
        return self.tasks_succeeded + self.tasks_failed

    @property
    def success_rate(self) -> float:
        """Success rate (0.0 to 1.0)."""
        if self.tasks_completed == 0:
            return 0.0
        return self.tasks_succeeded / self.tasks_completed

    @property
    def average_execution_time_ms(self) -> float:
        """Average execution time in milliseconds."""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_execution_time_ms / self.tasks_completed

    @property
    def average_queue_time_ms(self) -> float:
        """Average queue time in milliseconds."""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_queue_time_ms / self.tasks_completed

    def record_submission(self) -> None:
        """Record a task submission."""
        self.tasks_submitted += 1

    def record_completion(self, result: DistributedTaskResult) -> None:
        """Record a task completion."""
        if result.state == TaskState.SUCCEEDED:
            self.tasks_succeeded += 1
        elif result.state == TaskState.FAILED:
            self.tasks_failed += 1
        elif result.state == TaskState.CANCELLED:
            self.tasks_cancelled += 1
        elif result.state == TaskState.TIMEOUT:
            self.tasks_timed_out += 1

        if result.duration_ms:
            self.total_execution_time_ms += result.duration_ms
        if result.queue_time_ms:
            self.total_queue_time_ms += result.queue_time_ms
        self.retries += result.retries

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tasks_submitted": self.tasks_submitted,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "tasks_cancelled": self.tasks_cancelled,
            "tasks_timed_out": self.tasks_timed_out,
            "tasks_completed": self.tasks_completed,
            "success_rate": self.success_rate,
            "average_execution_time_ms": self.average_execution_time_ms,
            "average_queue_time_ms": self.average_queue_time_ms,
            "total_retries": self.retries,
        }

    def to_prometheus(self) -> str:
        """Export as Prometheus metrics."""
        lines = [
            f"truthound_distributed_tasks_submitted_total {self.tasks_submitted}",
            f"truthound_distributed_tasks_succeeded_total {self.tasks_succeeded}",
            f"truthound_distributed_tasks_failed_total {self.tasks_failed}",
            f"truthound_distributed_tasks_cancelled_total {self.tasks_cancelled}",
            f"truthound_distributed_tasks_timed_out_total {self.tasks_timed_out}",
            f"truthound_distributed_task_execution_time_ms_total {self.total_execution_time_ms}",
            f"truthound_distributed_task_queue_time_ms_total {self.total_queue_time_ms}",
            f"truthound_distributed_task_retries_total {self.retries}",
        ]
        return "\n".join(lines)


# =============================================================================
# Base Backend Implementation
# =============================================================================


class BaseDistributedBackend(ABC):
    """Abstract base class for distributed backends.

    This provides common functionality that all backends can use,
    including metrics collection, retry logic, and task tracking.

    Subclasses must implement:
        - _do_submit(): Actual task submission logic
        - _do_connect(): Backend connection logic
        - _do_disconnect(): Backend disconnection logic
        - _do_get_cluster_state(): Get cluster state
        - _do_get_workers(): Get worker list
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the backend.

        Args:
            config: Distributed configuration.
            **kwargs: Additional backend-specific options.
        """
        self._config = config or DistributedConfig()

        # Apply kwargs to backend_options
        for key, value in kwargs.items():
            self._config.backend_options[key] = value

        self._connected = False
        self._lock = threading.RLock()
        self._tasks: dict[str, DistributedTask[Any]] = {}
        self._metrics = TaskMetrics()

        # Callbacks
        self._on_task_complete: list[Callable[[DistributedTaskResult], None]] = []
        self._on_task_error: list[Callable[[str, Exception], None]] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Get backend name."""
        ...

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapability:
        """Get backend capabilities."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._connected

    @property
    def config(self) -> DistributedConfig:
        """Get backend configuration."""
        return self._config

    @property
    def metrics(self) -> TaskMetrics:
        """Get task metrics."""
        return self._metrics

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self, **kwargs: Any) -> None:
        """Connect to the backend."""
        with self._lock:
            if self._connected:
                return

            try:
                self._do_connect(**kwargs)
                self._connected = True
                logger.info(f"Connected to {self.name} backend")
            except Exception as e:
                raise BackendNotAvailableError(
                    self.name,
                    reason=str(e),
                )

    def disconnect(self) -> None:
        """Disconnect from the backend."""
        with self._lock:
            if not self._connected:
                return

            try:
                self._do_disconnect()
            finally:
                self._connected = False
                logger.info(f"Disconnected from {self.name} backend")

    @abstractmethod
    def _do_connect(self, **kwargs: Any) -> None:
        """Perform actual connection (implemented by subclass)."""
        ...

    @abstractmethod
    def _do_disconnect(self) -> None:
        """Perform actual disconnection (implemented by subclass)."""
        ...

    @contextmanager
    def connection(self, **kwargs: Any) -> Iterator[None]:
        """Context manager for connection lifecycle."""
        self.connect(**kwargs)
        try:
            yield
        finally:
            self.disconnect()

    # -------------------------------------------------------------------------
    # Task Submission
    # -------------------------------------------------------------------------

    def submit(
        self,
        checkpoint: "Checkpoint",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistributedTaskProtocol["CheckpointResult"]:
        """Submit a checkpoint for execution."""
        if not self._connected:
            raise TaskSubmissionError(
                "Backend not connected",
                checkpoint_name=checkpoint.name,
                reason="Call connect() first",
            )

        # Create task
        task: DistributedTask["CheckpointResult"] = DistributedTask.create(
            checkpoint=checkpoint,
            backend=self,
        )
        task._metadata["priority"] = priority.value
        task._metadata["context"] = context or {}

        timeout = timeout or self._config.task_timeout_seconds

        with self._lock:
            self._tasks[task.task_id] = task
            self._metrics.record_submission()

        try:
            # Submit to backend
            task._set_state(TaskState.SUBMITTED)
            self._do_submit(task, checkpoint, priority, timeout, context, **kwargs)
            task._set_state(TaskState.QUEUED)
            logger.debug(f"Task {task.task_id} submitted for checkpoint {checkpoint.name}")
            return task

        except Exception as e:
            task._set_error(str(e), e)
            raise TaskSubmissionError(
                f"Failed to submit task: {e}",
                checkpoint_name=checkpoint.name,
                reason=str(e),
            )

    def submit_batch(
        self,
        checkpoints: list["Checkpoint"],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[DistributedTaskProtocol["CheckpointResult"]]:
        """Submit multiple checkpoints for execution."""
        # Check capability
        if not (self.capabilities & BackendCapability.BATCH_SUBMIT):
            # Fall back to sequential submission
            return [
                self.submit(cp, priority, timeout, context, **kwargs)
                for cp in checkpoints
            ]

        tasks = []
        for checkpoint in checkpoints:
            try:
                task = self.submit(checkpoint, priority, timeout, context, **kwargs)
                tasks.append(task)
            except TaskSubmissionError as e:
                logger.error(f"Failed to submit {checkpoint.name}: {e}")
                # Continue with other checkpoints

        return tasks

    @abstractmethod
    def _do_submit(
        self,
        task: DistributedTask["CheckpointResult"],
        checkpoint: "Checkpoint",
        priority: TaskPriority,
        timeout: float,
        context: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Perform actual task submission (implemented by subclass)."""
        ...

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    def get_task(self, task_id: str) -> DistributedTaskProtocol["CheckpointResult"] | None:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def cancel_task(self, task_id: str, terminate: bool = False) -> bool:
        """Cancel a task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            return task.cancel(terminate=terminate)

    def _complete_task(
        self,
        task_id: str,
        result: "CheckpointResult | None" = None,
        error: str | None = None,
        exception: Exception | None = None,
    ) -> None:
        """Mark a task as complete (internal use)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return

            if error:
                task._set_error(error, exception)
            elif result:
                task._set_result(result)

            # Record metrics
            self._metrics.record_completion(task.to_result())

            # Invoke callbacks
            task_result = task.to_result()
            for callback in self._on_task_complete:
                try:
                    callback(task_result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

    # -------------------------------------------------------------------------
    # Cluster State
    # -------------------------------------------------------------------------

    def get_cluster_state(self) -> ClusterState:
        """Get current cluster state."""
        if not self._connected:
            return ClusterState(
                backend_name=self.name,
                is_healthy=False,
            )
        return self._do_get_cluster_state()

    @abstractmethod
    def _do_get_cluster_state(self) -> ClusterState:
        """Get cluster state (implemented by subclass)."""
        ...

    def get_workers(self) -> list[WorkerInfo]:
        """Get list of workers."""
        if not self._connected:
            return []
        return self._do_get_workers()

    @abstractmethod
    def _do_get_workers(self) -> list[WorkerInfo]:
        """Get workers (implemented by subclass)."""
        ...

    def scale_workers(self, count: int) -> bool:
        """Scale the number of workers."""
        if not (self.capabilities & BackendCapability.WORKER_SCALING):
            return False
        return self._do_scale_workers(count)

    def _do_scale_workers(self, count: int) -> bool:
        """Scale workers (override if supported)."""
        return False

    def health_check(self) -> bool:
        """Perform health check."""
        if not self._connected:
            return False
        try:
            state = self.get_cluster_state()
            return state.is_healthy
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_task_complete(
        self,
        callback: Callable[[DistributedTaskResult], None],
    ) -> None:
        """Register a task completion callback."""
        self._on_task_complete.append(callback)

    def on_task_error(
        self,
        callback: Callable[[str, Exception], None],
    ) -> None:
        """Register a task error callback."""
        self._on_task_error.append(callback)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = self._config.retry_delay_seconds * (
            self._config.retry_backoff_multiplier ** attempt
        )
        return min(delay, self._config.retry_max_delay_seconds)


# =============================================================================
# Base Orchestrator Implementation
# =============================================================================


class BaseDistributedOrchestrator:
    """Base class for distributed checkpoint orchestrators.

    The orchestrator provides a high-level API for submitting and
    managing distributed checkpoint execution across multiple backends.
    """

    def __init__(
        self,
        backend: BaseDistributedBackend,
        config: DistributedConfig | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            backend: The distributed backend to use.
            config: Configuration options.
        """
        self._backend = backend
        self._config = config or backend.config
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None

    @property
    def backend(self) -> BaseDistributedBackend:
        """Get the backend."""
        return self._backend

    @property
    def is_connected(self) -> bool:
        """Check if connected to backend."""
        return self._backend.is_connected

    @property
    def metrics(self) -> TaskMetrics:
        """Get execution metrics."""
        return self._backend.metrics

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def connect(self, **kwargs: Any) -> "BaseDistributedOrchestrator":
        """Connect to the backend."""
        self._backend.connect(**kwargs)
        return self

    def disconnect(self) -> None:
        """Disconnect from the backend."""
        self._backend.disconnect()
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __enter__(self) -> "BaseDistributedOrchestrator":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()

    # -------------------------------------------------------------------------
    # Task Submission
    # -------------------------------------------------------------------------

    def submit(
        self,
        checkpoint: "Checkpoint",
        priority: int | TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistributedTaskProtocol["CheckpointResult"]:
        """Submit a checkpoint for distributed execution.

        Args:
            checkpoint: Checkpoint to execute.
            priority: Task priority (0-10 or TaskPriority).
            timeout: Task timeout in seconds.
            context: Additional context for the run.
            **kwargs: Backend-specific options.

        Returns:
            Task handle for tracking execution.
        """
        if isinstance(priority, int):
            priority = TaskPriority.from_int(priority)

        return self._backend.submit(
            checkpoint=checkpoint,
            priority=priority,
            timeout=timeout,
            context=context,
            **kwargs,
        )

    def submit_batch(
        self,
        checkpoints: list["Checkpoint"],
        priority: int | TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[DistributedTaskProtocol["CheckpointResult"]]:
        """Submit multiple checkpoints for execution.

        Args:
            checkpoints: List of checkpoints to execute.
            priority: Task priority.
            timeout: Task timeout in seconds.
            context: Additional context for all runs.
            **kwargs: Backend-specific options.

        Returns:
            List of task handles.
        """
        if isinstance(priority, int):
            priority = TaskPriority.from_int(priority)

        return self._backend.submit_batch(
            checkpoints=checkpoints,
            priority=priority,
            timeout=timeout,
            context=context,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Result Collection
    # -------------------------------------------------------------------------

    def gather(
        self,
        tasks: list[DistributedTaskProtocol["CheckpointResult"]],
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list["CheckpointResult | Exception"]:
        """Wait for all tasks to complete and return results.

        Args:
            tasks: List of tasks to wait for.
            timeout: Maximum time to wait.
            return_exceptions: If True, return exceptions instead of raising.

        Returns:
            List of results (or exceptions if return_exceptions=True).
        """
        results: list["CheckpointResult | Exception"] = []
        deadline = time.time() + timeout if timeout else None

        for task in tasks:
            remaining = None
            if deadline:
                remaining = max(0, deadline - time.time())
                if remaining == 0:
                    if return_exceptions:
                        results.append(
                            TaskTimeoutError("Gather timeout", task.task_id, timeout or 0)
                        )
                        continue
                    raise TaskTimeoutError("Gather timeout", task.task_id, timeout or 0)

            try:
                result = task.result(timeout=remaining)
                results.append(result)
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise

        return results

    async def gather_async(
        self,
        tasks: list[DistributedTaskProtocol["CheckpointResult"]],
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> list["CheckpointResult | Exception"]:
        """Async version of gather()."""
        results: list["CheckpointResult | Exception"] = []

        async def get_result(
            task: DistributedTaskProtocol["CheckpointResult"],
        ) -> "CheckpointResult | Exception":
            try:
                return await task.result_async(timeout=timeout)
            except Exception as e:
                if return_exceptions:
                    return e
                raise

        gathered = await asyncio.gather(
            *[get_result(t) for t in tasks],
            return_exceptions=return_exceptions,
        )
        return list(gathered)

    def as_completed(
        self,
        tasks: list[DistributedTaskProtocol["CheckpointResult"]],
        timeout: float | None = None,
    ) -> Iterator[DistributedTaskProtocol["CheckpointResult"]]:
        """Yield tasks as they complete.

        Args:
            tasks: List of tasks to monitor.
            timeout: Maximum time to wait for each task.

        Yields:
            Tasks as they complete.
        """
        pending = set(tasks)
        deadline = time.time() + timeout if timeout else None

        while pending:
            completed = None
            for task in pending:
                if task.is_ready():
                    completed = task
                    break

            if completed:
                pending.remove(completed)
                yield completed
            else:
                if deadline and time.time() > deadline:
                    return
                time.sleep(0.05)

    # -------------------------------------------------------------------------
    # Cluster Management
    # -------------------------------------------------------------------------

    def get_cluster_state(self) -> ClusterState:
        """Get current cluster state."""
        return self._backend.get_cluster_state()

    def get_workers(self) -> list[WorkerInfo]:
        """Get list of workers."""
        return self._backend.get_workers()

    def scale_workers(self, count: int) -> bool:
        """Scale the number of workers."""
        return self._backend.scale_workers(count)

    def health_check(self) -> bool:
        """Check cluster health."""
        return self._backend.health_check()

    def wait_for_workers(
        self,
        min_workers: int = 1,
        timeout: float = 60.0,
    ) -> bool:
        """Wait for minimum number of workers to be available.

        Args:
            min_workers: Minimum number of workers required.
            timeout: Maximum time to wait.

        Returns:
            True if workers are available, False if timeout.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            workers = self.get_workers()
            available = len([w for w in workers if w.is_available])
            if available >= min_workers:
                return True
            time.sleep(1.0)

        return False
