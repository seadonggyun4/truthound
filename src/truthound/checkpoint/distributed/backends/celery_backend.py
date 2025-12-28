"""Celery Backend for Distributed Checkpoints.

This backend provides distributed execution using Celery,
a production-ready distributed task queue.

Requirements:
    pip install celery[redis]
    # or
    pip install celery[rabbitmq]
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from truthound.checkpoint.distributed.base import BaseDistributedBackend
from truthound.checkpoint.distributed.protocols import (
    BackendCapability,
    BackendNotAvailableError,
    ClusterState,
    DistributedConfig,
    DistributedTask,
    DistributedTaskProtocol,
    TaskPriority,
    TaskState,
    WorkerInfo,
    WorkerState,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import Checkpoint, CheckpointResult


logger = logging.getLogger(__name__)


# Check if Celery is available
try:
    from celery import Celery
    from celery.result import AsyncResult
    from celery.app.control import Inspect
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None
    AsyncResult = None
    Inspect = None


@dataclass
class CeleryBackendConfig:
    """Configuration for Celery backend.

    Attributes:
        broker_url: Celery broker URL (redis://, amqp://, etc.).
        result_backend: Result backend URL.
        task_queue: Queue name for checkpoint tasks.
        task_serializer: Serializer for tasks (json, pickle, etc.).
        result_serializer: Serializer for results.
        task_acks_late: Acknowledge tasks after completion.
        task_reject_on_worker_lost: Reject task if worker dies.
        worker_prefetch_multiplier: Prefetch multiplier.
        task_time_limit: Hard time limit for tasks.
        task_soft_time_limit: Soft time limit for tasks.
    """

    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    task_queue: str = "truthound.checkpoints"
    task_serializer: str = "json"
    result_serializer: str = "json"
    task_acks_late: bool = True
    task_reject_on_worker_lost: bool = True
    worker_prefetch_multiplier: int = 1
    task_time_limit: int = 3600
    task_soft_time_limit: int = 3500


class CeleryTask(DistributedTask["CheckpointResult"]):
    """Celery-specific task wrapper."""

    def __init__(
        self,
        checkpoint_name: str,
        async_result: "AsyncResult",
        backend: "CeleryBackend",
    ) -> None:
        super().__init__(
            task_id=async_result.id,
            checkpoint_name=checkpoint_name,
            _backend=backend,
        )
        self._async_result = async_result

    @property
    def state(self) -> TaskState:
        """Map Celery state to TaskState."""
        celery_state = self._async_result.state

        state_map = {
            "PENDING": TaskState.QUEUED,
            "RECEIVED": TaskState.QUEUED,
            "STARTED": TaskState.RUNNING,
            "SUCCESS": TaskState.SUCCEEDED,
            "FAILURE": TaskState.FAILED,
            "REVOKED": TaskState.REVOKED,
            "RETRY": TaskState.RETRYING,
        }

        return state_map.get(celery_state, TaskState.PENDING)

    def result(self, timeout: float | None = None) -> "CheckpointResult":
        """Wait for and return the result."""
        from truthound.checkpoint.checkpoint import CheckpointResult

        try:
            result_dict = self._async_result.get(timeout=timeout)
            return CheckpointResult.from_dict(result_dict)
        except Exception as e:
            if "TimeoutError" in type(e).__name__:
                from truthound.checkpoint.distributed.protocols import TaskTimeoutError
                raise TaskTimeoutError(
                    f"Task {self.task_id} timed out",
                    self.task_id,
                    timeout or 0,
                )
            raise

    async def result_async(self, timeout: float | None = None) -> "CheckpointResult":
        """Async version of result()."""
        import asyncio
        from truthound.checkpoint.checkpoint import CheckpointResult

        loop = asyncio.get_event_loop()

        def get_result() -> dict[str, Any]:
            return self._async_result.get(timeout=timeout)

        result_dict = await loop.run_in_executor(None, get_result)
        return CheckpointResult.from_dict(result_dict)

    def cancel(self, terminate: bool = False) -> bool:
        """Revoke the Celery task."""
        try:
            self._async_result.revoke(terminate=terminate)
            return True
        except Exception as e:
            logger.error(f"Failed to revoke task: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if task is complete."""
        return self._async_result.ready()

    def is_successful(self) -> bool:
        """Check if task succeeded."""
        return self._async_result.successful()


class CeleryBackend(BaseDistributedBackend):
    """Celery-based distributed backend.

    This backend uses Celery for distributed task execution, supporting
    Redis, RabbitMQ, and other message brokers.

    Example:
        >>> from truthound.checkpoint.distributed import CeleryBackend
        >>>
        >>> backend = CeleryBackend(
        ...     broker_url="redis://localhost:6379/0",
        ...     result_backend="redis://localhost:6379/0",
        ... )
        >>>
        >>> with backend.connection():
        ...     task = backend.submit(checkpoint)
        ...     result = task.result(timeout=300)

    Note:
        You need to run Celery workers to execute tasks:

        $ celery -A truthound.checkpoint.distributed.backends.celery_backend worker -l info
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        broker_url: str | None = None,
        result_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Celery backend.

        Args:
            config: Distributed configuration.
            broker_url: Celery broker URL.
            result_backend: Result backend URL.
            **kwargs: Additional Celery configuration.
        """
        if not CELERY_AVAILABLE:
            raise BackendNotAvailableError(
                "celery",
                reason="Celery is not installed",
                install_hint="pip install celery[redis]",
            )

        super().__init__(config, **kwargs)

        # Build Celery config
        self._celery_config = CeleryBackendConfig(
            broker_url=broker_url or self._config.backend_options.get(
                "broker_url", "redis://localhost:6379/0"
            ),
            result_backend=result_backend or self._config.backend_options.get(
                "result_backend", "redis://localhost:6379/0"
            ),
        )

        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(self._celery_config, key):
                setattr(self._celery_config, key, value)

        self._app: Celery | None = None
        self._task: Any = None  # Celery task function

    @property
    def name(self) -> str:
        return "celery"

    @property
    def capabilities(self) -> BackendCapability:
        return (
            BackendCapability.ASYNC_SUBMIT
            | BackendCapability.BATCH_SUBMIT
            | BackendCapability.PRIORITY_QUEUE
            | BackendCapability.RETRY_POLICY
            | BackendCapability.RATE_LIMITING
            | BackendCapability.TASK_ROUTING
            | BackendCapability.RESULT_BACKEND
            | BackendCapability.TASK_REVOKE
            | BackendCapability.PROGRESS_TRACKING
            | BackendCapability.CHAIN_TASKS
            | BackendCapability.GROUP_TASKS
            | BackendCapability.BROADCAST
            | BackendCapability.SCHEDULED_TASKS
            | BackendCapability.HEALTH_CHECK
            | BackendCapability.METRICS
        )

    @property
    def app(self) -> Celery:
        """Get the Celery app instance."""
        if self._app is None:
            raise RuntimeError("Backend not connected")
        return self._app

    def _do_connect(self, **kwargs: Any) -> None:
        """Initialize Celery app and register task."""
        # Create Celery app
        self._app = Celery(
            "truthound.distributed",
            broker=self._celery_config.broker_url,
            backend=self._celery_config.result_backend,
        )

        # Configure Celery
        self._app.conf.update(
            task_serializer=self._celery_config.task_serializer,
            result_serializer=self._celery_config.result_serializer,
            task_acks_late=self._celery_config.task_acks_late,
            task_reject_on_worker_lost=self._celery_config.task_reject_on_worker_lost,
            worker_prefetch_multiplier=self._celery_config.worker_prefetch_multiplier,
            task_time_limit=self._celery_config.task_time_limit,
            task_soft_time_limit=self._celery_config.task_soft_time_limit,
            task_routes={
                "truthound.execute_checkpoint": {
                    "queue": self._celery_config.task_queue,
                },
            },
        )

        # Register task
        @self._app.task(
            name="truthound.execute_checkpoint",
            bind=True,
            max_retries=self._config.max_retries,
            default_retry_delay=self._config.retry_delay_seconds,
        )
        def execute_checkpoint(
            self_task: Any,
            checkpoint_dict: dict[str, Any],
            context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Execute a checkpoint task."""
            from truthound.checkpoint.checkpoint import (
                Checkpoint,
                CheckpointConfig,
            )

            try:
                # Reconstruct checkpoint
                config = CheckpointConfig(**checkpoint_dict.get("config", {}))
                checkpoint = Checkpoint(config=config)

                # Run
                result = checkpoint.run(context=context)

                return result.to_dict()

            except Exception as exc:
                # Retry with exponential backoff
                retry_delay = self._config.retry_delay_seconds * (
                    self._config.retry_backoff_multiplier ** self_task.request.retries
                )
                raise self_task.retry(exc=exc, countdown=retry_delay)

        self._task = execute_checkpoint

        logger.info(f"Celery backend connected to {self._celery_config.broker_url}")

    def _do_disconnect(self) -> None:
        """Close Celery connections."""
        if self._app:
            self._app.close()
            self._app = None
        self._task = None

    def _do_submit(
        self,
        task: DistributedTask["CheckpointResult"],
        checkpoint: "Checkpoint",
        priority: TaskPriority,
        timeout: float,
        context: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Submit task to Celery."""
        if self._task is None:
            raise RuntimeError("Backend not connected")

        # Serialize checkpoint
        checkpoint_dict = {
            "config": {
                "name": checkpoint.config.name,
                "data_source": str(checkpoint.config.data_source),
                "validators": checkpoint.config.validators,
                "min_severity": checkpoint.config.min_severity,
                "fail_on_critical": checkpoint.config.fail_on_critical,
                "fail_on_high": checkpoint.config.fail_on_high,
                "timeout_seconds": checkpoint.config.timeout_seconds,
                "sample_size": checkpoint.config.sample_size,
            }
        }

        # Submit to Celery
        async_result = self._task.apply_async(
            args=[checkpoint_dict, context],
            queue=self._celery_config.task_queue,
            priority=priority.value,
            time_limit=timeout,
            soft_time_limit=timeout - 10 if timeout > 10 else timeout,
            **kwargs,
        )

        # Replace task with Celery-specific wrapper
        celery_task = CeleryTask(
            checkpoint_name=checkpoint.name,
            async_result=async_result,
            backend=self,
        )

        # Copy state to original task
        task.task_id = celery_task.task_id
        task._async_result = async_result  # type: ignore

        # Update task tracking
        with self._lock:
            self._tasks[task.task_id] = task

    def _do_get_cluster_state(self) -> ClusterState:
        """Get Celery cluster state."""
        if self._app is None:
            return ClusterState(backend_name=self.name, is_healthy=False)

        inspect = self._app.control.inspect()
        workers = []

        try:
            # Get active workers
            active_tasks = inspect.active() or {}
            stats = inspect.stats() or {}
            ping = inspect.ping() or {}

            for worker_name, worker_stats in stats.items():
                concurrency = worker_stats.get("pool", {}).get("max-concurrency", 1)
                active_count = len(active_tasks.get(worker_name, []))

                workers.append(WorkerInfo(
                    worker_id=worker_name,
                    hostname=worker_stats.get("broker", {}).get("hostname", "unknown"),
                    state=WorkerState.ONLINE if worker_name in ping else WorkerState.OFFLINE,
                    current_tasks=active_count,
                    max_concurrency=concurrency,
                    metadata={
                        "pid": worker_stats.get("pid"),
                        "software": worker_stats.get("software"),
                    },
                ))

            total_capacity = sum(w.max_concurrency for w in workers)
            current_load = sum(w.current_tasks for w in workers)

            # Get queue length
            reserved = inspect.reserved() or {}
            pending = sum(len(tasks) for tasks in reserved.values())

            return ClusterState(
                workers=workers,
                total_capacity=total_capacity,
                current_load=current_load,
                pending_tasks=pending,
                backend_name=self.name,
                backend_version=self._app.VERSION if hasattr(self._app, 'VERSION') else "unknown",
                is_healthy=len(workers) > 0,
            )

        except Exception as e:
            logger.error(f"Failed to get cluster state: {e}")
            return ClusterState(
                backend_name=self.name,
                is_healthy=False,
            )

    def _do_get_workers(self) -> list[WorkerInfo]:
        """Get list of Celery workers."""
        state = self._do_get_cluster_state()
        return state.workers

    def broadcast(
        self,
        command: str,
        arguments: dict[str, Any] | None = None,
        destination: list[str] | None = None,
    ) -> dict[str, Any]:
        """Broadcast a command to workers.

        Args:
            command: Command to broadcast (e.g., "shutdown", "pool_shrink").
            arguments: Command arguments.
            destination: List of worker names (None = all workers).

        Returns:
            Response from workers.
        """
        if self._app is None:
            raise RuntimeError("Backend not connected")

        return self._app.control.broadcast(
            command,
            arguments=arguments or {},
            destination=destination,
            reply=True,
        )

    def purge_queue(self) -> int:
        """Purge all pending tasks from the queue.

        Returns:
            Number of tasks purged.
        """
        if self._app is None:
            raise RuntimeError("Backend not connected")

        return self._app.control.purge()


# Create a standalone Celery app for workers
# This can be used to run workers: celery -A celery_backend worker
if CELERY_AVAILABLE:
    celery_app = Celery("truthound.distributed")

    @celery_app.task(name="truthound.execute_checkpoint", bind=True)
    def execute_checkpoint_worker(
        self: Any,
        checkpoint_dict: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Worker task for checkpoint execution."""
        from truthound.checkpoint.checkpoint import (
            Checkpoint,
            CheckpointConfig,
        )

        config = CheckpointConfig(**checkpoint_dict.get("config", {}))
        checkpoint = Checkpoint(config=config)
        result = checkpoint.run(context=context)

        return result.to_dict()
