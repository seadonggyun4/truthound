"""Protocols and Data Classes for Distributed Checkpoint Orchestration.

This module defines the core abstractions for distributed checkpoint execution,
including backend protocols, task states, and configuration options.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, Flag, auto
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Generator,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from uuid import uuid4

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import Checkpoint, CheckpointResult


logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class DistributedError(Exception):
    """Base exception for distributed checkpoint errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TaskSubmissionError(DistributedError):
    """Raised when task submission fails."""

    def __init__(
        self,
        message: str,
        checkpoint_name: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message, {"checkpoint_name": checkpoint_name, "reason": reason})
        self.checkpoint_name = checkpoint_name
        self.reason = reason


class TaskTimeoutError(DistributedError):
    """Raised when task execution times out."""

    def __init__(
        self,
        message: str,
        task_id: str,
        timeout_seconds: float,
    ) -> None:
        super().__init__(message, {"task_id": task_id, "timeout_seconds": timeout_seconds})
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds


class TaskCancelledError(DistributedError):
    """Raised when task is cancelled."""

    def __init__(self, message: str, task_id: str, reason: str | None = None) -> None:
        super().__init__(message, {"task_id": task_id, "reason": reason})
        self.task_id = task_id
        self.reason = reason


class WorkerNotAvailableError(DistributedError):
    """Raised when no workers are available."""

    def __init__(
        self,
        message: str = "No workers available",
        required_workers: int = 1,
        available_workers: int = 0,
    ) -> None:
        super().__init__(
            message,
            {"required_workers": required_workers, "available_workers": available_workers},
        )
        self.required_workers = required_workers
        self.available_workers = available_workers


class BackendNotAvailableError(DistributedError):
    """Raised when backend is not available."""

    def __init__(
        self,
        backend_name: str,
        reason: str | None = None,
        install_hint: str | None = None,
    ) -> None:
        message = f"Backend '{backend_name}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"backend_name": backend_name, "install_hint": install_hint})
        self.backend_name = backend_name
        self.reason = reason
        self.install_hint = install_hint


# =============================================================================
# Enums
# =============================================================================


class TaskState(str, Enum):
    """State of a distributed task."""

    PENDING = "pending"          # Task created but not yet submitted
    SUBMITTED = "submitted"      # Task submitted to backend
    QUEUED = "queued"           # Task in backend queue
    RUNNING = "running"         # Task currently executing
    SUCCEEDED = "succeeded"     # Task completed successfully
    FAILED = "failed"           # Task failed with error
    CANCELLED = "cancelled"     # Task was cancelled
    REVOKED = "revoked"         # Task was revoked (Celery term)
    RETRYING = "retrying"       # Task is being retried
    TIMEOUT = "timeout"         # Task timed out

    @property
    def is_terminal(self) -> bool:
        """Check if state is terminal (no further transitions)."""
        return self in {
            TaskState.SUCCEEDED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.REVOKED,
            TaskState.TIMEOUT,
        }

    @property
    def is_active(self) -> bool:
        """Check if task is actively processing."""
        return self in {TaskState.RUNNING, TaskState.RETRYING}


class TaskPriority(int, Enum):
    """Priority levels for task scheduling."""

    LOWEST = 0
    LOW = 3
    NORMAL = 5
    HIGH = 7
    HIGHEST = 9
    CRITICAL = 10

    @classmethod
    def from_int(cls, value: int) -> "TaskPriority":
        """Convert integer to priority."""
        if value <= 0:
            return cls.LOWEST
        elif value <= 3:
            return cls.LOW
        elif value <= 5:
            return cls.NORMAL
        elif value <= 7:
            return cls.HIGH
        elif value <= 9:
            return cls.HIGHEST
        return cls.CRITICAL


class WorkerState(str, Enum):
    """State of a worker node."""

    ONLINE = "online"           # Worker is online and accepting tasks
    OFFLINE = "offline"         # Worker is offline
    BUSY = "busy"               # Worker is at capacity
    DRAINING = "draining"       # Worker is draining (no new tasks)
    MAINTENANCE = "maintenance" # Worker is in maintenance mode
    UNKNOWN = "unknown"         # Worker state unknown


class BackendCapability(Flag):
    """Capabilities supported by a backend."""

    NONE = 0
    ASYNC_SUBMIT = auto()       # Async task submission
    BATCH_SUBMIT = auto()       # Batch task submission
    PRIORITY_QUEUE = auto()     # Priority-based scheduling
    RETRY_POLICY = auto()       # Automatic retry on failure
    RATE_LIMITING = auto()      # Task rate limiting
    TASK_ROUTING = auto()       # Route tasks to specific workers
    RESULT_BACKEND = auto()     # Store task results
    TASK_REVOKE = auto()        # Revoke/cancel running tasks
    PROGRESS_TRACKING = auto()  # Track task progress
    CHAIN_TASKS = auto()        # Chain tasks together
    GROUP_TASKS = auto()        # Execute tasks in groups
    BROADCAST = auto()          # Broadcast to all workers
    SCHEDULED_TASKS = auto()    # Schedule tasks for later execution
    WORKER_SCALING = auto()     # Dynamic worker scaling
    HEALTH_CHECK = auto()       # Worker health checks
    METRICS = auto()            # Collect execution metrics

    # Common capability sets
    BASIC = ASYNC_SUBMIT | RESULT_BACKEND
    STANDARD = BASIC | BATCH_SUBMIT | RETRY_POLICY | TASK_REVOKE
    ADVANCED = STANDARD | PRIORITY_QUEUE | PROGRESS_TRACKING | CHAIN_TASKS | GROUP_TASKS
    FULL = ADVANCED | RATE_LIMITING | TASK_ROUTING | BROADCAST | SCHEDULED_TASKS | WORKER_SCALING | HEALTH_CHECK | METRICS


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DistributedConfig:
    """Configuration for distributed checkpoint orchestration.

    Attributes:
        backend: Backend type ("celery", "ray", "kubernetes", "local").
        max_workers: Maximum number of concurrent workers.
        max_retries: Maximum retry attempts on failure.
        retry_delay_seconds: Delay between retries.
        task_timeout_seconds: Default task timeout.
        result_ttl_seconds: How long to keep results.
        heartbeat_interval_seconds: Worker heartbeat interval.
        enable_metrics: Enable metrics collection.
        enable_tracing: Enable distributed tracing.
        backend_options: Backend-specific configuration.
    """

    backend: str = "local"
    max_workers: int = 4
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff_multiplier: float = 2.0
    retry_max_delay_seconds: float = 300.0
    task_timeout_seconds: float = 3600.0
    result_ttl_seconds: float = 86400.0  # 24 hours
    heartbeat_interval_seconds: float = 30.0
    enable_metrics: bool = True
    enable_tracing: bool = False
    backend_options: dict[str, Any] = field(default_factory=dict)

    def with_backend(self, backend: str, **options: Any) -> "DistributedConfig":
        """Create new config with different backend."""
        new_options = {**self.backend_options, **options}
        return DistributedConfig(
            backend=backend,
            max_workers=self.max_workers,
            max_retries=self.max_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            retry_backoff_multiplier=self.retry_backoff_multiplier,
            retry_max_delay_seconds=self.retry_max_delay_seconds,
            task_timeout_seconds=self.task_timeout_seconds,
            result_ttl_seconds=self.result_ttl_seconds,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            enable_metrics=self.enable_metrics,
            enable_tracing=self.enable_tracing,
            backend_options=new_options,
        )


@dataclass
class WorkerInfo:
    """Information about a worker node.

    Attributes:
        worker_id: Unique worker identifier.
        hostname: Worker hostname.
        state: Current worker state.
        current_tasks: Number of tasks currently running.
        max_concurrency: Maximum concurrent tasks.
        registered_at: When worker registered.
        last_heartbeat: Last heartbeat timestamp.
        tags: Worker tags for routing.
        metadata: Additional worker metadata.
    """

    worker_id: str
    hostname: str
    state: WorkerState = WorkerState.UNKNOWN
    current_tasks: int = 0
    max_concurrency: int = 1
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if worker can accept new tasks."""
        return (
            self.state == WorkerState.ONLINE
            and self.current_tasks < self.max_concurrency
        )

    @property
    def available_slots(self) -> int:
        """Get number of available task slots."""
        if self.state != WorkerState.ONLINE:
            return 0
        return max(0, self.max_concurrency - self.current_tasks)

    @property
    def load_factor(self) -> float:
        """Get worker load factor (0.0 to 1.0)."""
        if self.max_concurrency == 0:
            return 1.0
        return self.current_tasks / self.max_concurrency


@dataclass
class ClusterState:
    """State of the distributed cluster.

    Attributes:
        workers: List of worker info.
        total_capacity: Total task capacity.
        current_load: Current number of running tasks.
        pending_tasks: Number of pending tasks.
        backend_name: Name of the backend.
        backend_version: Version of the backend.
        is_healthy: Whether cluster is healthy.
        last_updated: When state was last updated.
    """

    workers: list[WorkerInfo] = field(default_factory=list)
    total_capacity: int = 0
    current_load: int = 0
    pending_tasks: int = 0
    backend_name: str = ""
    backend_version: str = ""
    is_healthy: bool = True
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        return self.total_capacity - self.current_load

    @property
    def online_workers(self) -> list[WorkerInfo]:
        """Get list of online workers."""
        return [w for w in self.workers if w.state == WorkerState.ONLINE]

    @property
    def utilization(self) -> float:
        """Get cluster utilization (0.0 to 1.0)."""
        if self.total_capacity == 0:
            return 0.0
        return self.current_load / self.total_capacity


@dataclass
class DistributedTaskResult:
    """Result of a distributed task execution.

    Attributes:
        task_id: Unique task identifier.
        checkpoint_name: Name of the checkpoint that was run.
        state: Final task state.
        result: The CheckpointResult if successful.
        error: Error message if failed.
        exception: Exception if failed.
        worker_id: ID of worker that executed the task.
        submitted_at: When task was submitted.
        started_at: When task started executing.
        completed_at: When task completed.
        retries: Number of retry attempts.
        metadata: Additional result metadata.
    """

    task_id: str
    checkpoint_name: str
    state: TaskState
    result: "CheckpointResult | None" = None
    error: str | None = None
    exception: Exception | None = None
    worker_id: str | None = None
    submitted_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if task succeeded."""
        return self.state == TaskState.SUCCEEDED

    @property
    def duration_ms(self) -> float | None:
        """Get execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def queue_time_ms(self) -> float | None:
        """Get time spent in queue in milliseconds."""
        if self.submitted_at and self.started_at:
            return (self.started_at - self.submitted_at).total_seconds() * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "checkpoint_name": self.checkpoint_name,
            "state": self.state.value,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "worker_id": self.worker_id,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "duration_ms": self.duration_ms,
            "queue_time_ms": self.queue_time_ms,
            "metadata": self.metadata,
        }


# =============================================================================
# Protocols
# =============================================================================


T = TypeVar("T")


@runtime_checkable
class DistributedTaskProtocol(Protocol[T]):
    """Protocol for a distributed task handle.

    This represents a submitted task that can be awaited, cancelled, or queried.
    """

    @property
    def task_id(self) -> str:
        """Get unique task identifier."""
        ...

    @property
    def state(self) -> TaskState:
        """Get current task state."""
        ...

    @property
    def checkpoint_name(self) -> str:
        """Get name of the checkpoint being executed."""
        ...

    def result(self, timeout: float | None = None) -> T:
        """Wait for and return the task result.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Task result.

        Raises:
            TaskTimeoutError: If timeout exceeded.
            TaskCancelledError: If task was cancelled.
            DistributedError: If task failed.
        """
        ...

    async def result_async(self, timeout: float | None = None) -> T:
        """Async version of result().

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            Task result.
        """
        ...

    def cancel(self, terminate: bool = False) -> bool:
        """Cancel the task.

        Args:
            terminate: If True, terminate even if running.

        Returns:
            True if cancelled, False if already completed.
        """
        ...

    def is_ready(self) -> bool:
        """Check if task has completed."""
        ...

    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        ...

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for task to complete.

        Args:
            timeout: Maximum time to wait.

        Returns:
            True if completed, False if timeout.
        """
        ...


@runtime_checkable
class DistributedBackendProtocol(Protocol):
    """Protocol for distributed execution backends.

    Implementations must provide methods for task submission, worker
    management, and cluster state queries.
    """

    @property
    def name(self) -> str:
        """Get backend name."""
        ...

    @property
    def capabilities(self) -> BackendCapability:
        """Get backend capabilities."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        ...

    def connect(self, **kwargs: Any) -> None:
        """Connect to the backend.

        Args:
            **kwargs: Backend-specific connection options.

        Raises:
            BackendNotAvailableError: If connection fails.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the backend."""
        ...

    def submit(
        self,
        checkpoint: "Checkpoint",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DistributedTaskProtocol["CheckpointResult"]:
        """Submit a checkpoint for execution.

        Args:
            checkpoint: Checkpoint to execute.
            priority: Task priority.
            timeout: Task timeout in seconds.
            context: Additional context for the run.
            **kwargs: Backend-specific options.

        Returns:
            Task handle for tracking execution.

        Raises:
            TaskSubmissionError: If submission fails.
        """
        ...

    def submit_batch(
        self,
        checkpoints: list["Checkpoint"],
        priority: TaskPriority = TaskPriority.NORMAL,
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

        Raises:
            TaskSubmissionError: If submission fails.
        """
        ...

    def get_task(self, task_id: str) -> DistributedTaskProtocol["CheckpointResult"] | None:
        """Get a task by ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task handle if found, None otherwise.
        """
        ...

    def get_cluster_state(self) -> ClusterState:
        """Get current cluster state.

        Returns:
            ClusterState with worker and capacity info.
        """
        ...

    def get_workers(self) -> list[WorkerInfo]:
        """Get list of workers.

        Returns:
            List of worker info.
        """
        ...

    def scale_workers(self, count: int) -> bool:
        """Scale the number of workers.

        Args:
            count: Desired worker count.

        Returns:
            True if scaling initiated, False if not supported.
        """
        ...

    def health_check(self) -> bool:
        """Perform health check.

        Returns:
            True if healthy, False otherwise.
        """
        ...


@dataclass
class DistributedTask(Generic[T]):
    """Concrete implementation of a distributed task handle.

    This provides a standard implementation that backends can use or extend.
    """

    task_id: str
    checkpoint_name: str
    _state: TaskState = TaskState.PENDING
    _result: T | None = None
    _error: str | None = None
    _exception: Exception | None = None
    _backend: Any = None  # Reference to backend for operations
    _submitted_at: datetime = field(default_factory=datetime.now)
    _started_at: datetime | None = None
    _completed_at: datetime | None = None
    _worker_id: str | None = None
    _retries: int = 0
    _metadata: dict[str, Any] = field(default_factory=dict)
    _callbacks: list[Callable[[T | None, Exception | None], None]] = field(
        default_factory=list
    )

    @classmethod
    def create(
        cls,
        checkpoint: "Checkpoint",
        backend: Any = None,
    ) -> "DistributedTask[T]":
        """Create a new task for a checkpoint."""
        return cls(
            task_id=f"task-{uuid4().hex[:16]}",
            checkpoint_name=checkpoint.name,
            _state=TaskState.PENDING,
            _backend=backend,
            _submitted_at=datetime.now(),
        )

    @property
    def state(self) -> TaskState:
        """Get current task state."""
        return self._state

    def _set_state(self, state: TaskState) -> None:
        """Set task state (internal use)."""
        self._state = state

    def _set_result(self, result: T) -> None:
        """Set task result (internal use)."""
        self._result = result
        self._state = TaskState.SUCCEEDED
        self._completed_at = datetime.now()
        self._invoke_callbacks()

    def _set_error(self, error: str, exception: Exception | None = None) -> None:
        """Set task error (internal use)."""
        self._error = error
        self._exception = exception
        self._state = TaskState.FAILED
        self._completed_at = datetime.now()
        self._invoke_callbacks()

    def _invoke_callbacks(self) -> None:
        """Invoke registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(self._result, self._exception)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def result(self, timeout: float | None = None) -> T:
        """Wait for and return the task result."""
        import time

        start = time.time()
        while not self.is_ready():
            if timeout and (time.time() - start) > timeout:
                raise TaskTimeoutError(
                    f"Task {self.task_id} timed out",
                    self.task_id,
                    timeout,
                )
            time.sleep(0.1)

        if self._state == TaskState.CANCELLED:
            raise TaskCancelledError(f"Task {self.task_id} was cancelled", self.task_id)

        if self._state == TaskState.FAILED:
            raise DistributedError(
                f"Task {self.task_id} failed: {self._error}",
                {"exception": self._exception},
            )

        return self._result  # type: ignore

    async def result_async(self, timeout: float | None = None) -> T:
        """Async version of result()."""
        import asyncio

        start = asyncio.get_event_loop().time()
        while not self.is_ready():
            if timeout and (asyncio.get_event_loop().time() - start) > timeout:
                raise TaskTimeoutError(
                    f"Task {self.task_id} timed out",
                    self.task_id,
                    timeout,
                )
            await asyncio.sleep(0.1)

        if self._state == TaskState.CANCELLED:
            raise TaskCancelledError(f"Task {self.task_id} was cancelled", self.task_id)

        if self._state == TaskState.FAILED:
            raise DistributedError(
                f"Task {self.task_id} failed: {self._error}",
                {"exception": self._exception},
            )

        return self._result  # type: ignore

    def cancel(self, terminate: bool = False) -> bool:
        """Cancel the task."""
        if self.is_ready():
            return False

        self._state = TaskState.CANCELLED
        self._completed_at = datetime.now()

        # Notify backend if available
        if self._backend and hasattr(self._backend, "cancel_task"):
            try:
                self._backend.cancel_task(self.task_id, terminate=terminate)
            except Exception as e:
                logger.warning(f"Backend cancel failed: {e}")

        return True

    def is_ready(self) -> bool:
        """Check if task has completed."""
        return self._state.is_terminal

    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self._state == TaskState.SUCCEEDED

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for task to complete."""
        try:
            self.result(timeout=timeout)
            return True
        except TaskTimeoutError:
            return False
        except Exception:
            return True  # Task completed (with error)

    def add_callback(
        self,
        callback: Callable[[T | None, Exception | None], None],
    ) -> "DistributedTask[T]":
        """Add a completion callback."""
        self._callbacks.append(callback)
        # If already complete, invoke immediately
        if self.is_ready():
            callback(self._result, self._exception)
        return self

    def to_result(self) -> DistributedTaskResult:
        """Convert to DistributedTaskResult."""
        return DistributedTaskResult(
            task_id=self.task_id,
            checkpoint_name=self.checkpoint_name,
            state=self._state,
            result=self._result,  # type: ignore
            error=self._error,
            exception=self._exception,
            worker_id=self._worker_id,
            submitted_at=self._submitted_at,
            started_at=self._started_at,
            completed_at=self._completed_at,
            retries=self._retries,
            metadata=self._metadata,
        )
