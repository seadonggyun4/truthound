"""Local Multi-Process Backend for Distributed Checkpoints.

This backend provides distributed execution using local processes,
suitable for development, testing, and single-machine deployment.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    Future,
    as_completed,
)
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from truthound.checkpoint.distributed.base import BaseDistributedBackend
from truthound.checkpoint.distributed.protocols import (
    BackendCapability,
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


@dataclass
class LocalBackendConfig:
    """Configuration for local backend.

    Attributes:
        max_workers: Maximum number of worker processes/threads.
        use_processes: Use processes instead of threads.
        worker_timeout: Timeout for worker operations.
    """

    max_workers: int = 4
    use_processes: bool = False
    worker_timeout: float = 3600.0


def _execute_checkpoint_task(
    checkpoint_dict: dict[str, Any],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Execute a checkpoint task in a worker.

    This function is designed to be picklable for process execution.

    Args:
        checkpoint_dict: Serialized checkpoint configuration.
        context: Execution context.

    Returns:
        Serialized CheckpointResult.
    """
    # Import here to avoid circular imports in worker processes
    from truthound.checkpoint.checkpoint import (
        Checkpoint,
        CheckpointConfig,
        CheckpointResult,
    )

    # Reconstruct checkpoint from dict
    config = CheckpointConfig(**checkpoint_dict.get("config", {}))
    checkpoint = Checkpoint(config=config)

    # Run checkpoint
    result = checkpoint.run(context=context)

    # Serialize result for return
    return result.to_dict()


class LocalBackend(BaseDistributedBackend):
    """Local multi-process/thread backend for distributed checkpoints.

    This backend executes checkpoints using a pool of local workers,
    providing a simple way to parallelize validation without external
    dependencies.

    Example:
        >>> from truthound.checkpoint.distributed import LocalBackend
        >>> from truthound.checkpoint import Checkpoint
        >>>
        >>> backend = LocalBackend(max_workers=4)
        >>> backend.connect()
        >>>
        >>> checkpoint = Checkpoint(name="test", data_source="data.csv")
        >>> task = backend.submit(checkpoint)
        >>> result = task.result(timeout=60)
        >>>
        >>> backend.disconnect()
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        max_workers: int = 4,
        use_processes: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize local backend.

        Args:
            config: Distributed configuration.
            max_workers: Maximum number of workers.
            use_processes: Use processes instead of threads.
            **kwargs: Additional options.
        """
        super().__init__(config, **kwargs)

        self._local_config = LocalBackendConfig(
            max_workers=max_workers,
            use_processes=use_processes,
        )

        self._executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None
        self._futures: dict[str, Future[dict[str, Any]]] = {}
        self._workers: list[WorkerInfo] = []

    @property
    def name(self) -> str:
        return "local"

    @property
    def capabilities(self) -> BackendCapability:
        return (
            BackendCapability.ASYNC_SUBMIT
            | BackendCapability.BATCH_SUBMIT
            | BackendCapability.RESULT_BACKEND
            | BackendCapability.TASK_REVOKE
            | BackendCapability.HEALTH_CHECK
            | BackendCapability.METRICS
        )

    def _do_connect(self, **kwargs: Any) -> None:
        """Initialize the executor pool."""
        max_workers = kwargs.get("max_workers", self._local_config.max_workers)
        use_processes = kwargs.get("use_processes", self._local_config.use_processes)

        if use_processes:
            self._executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Create virtual workers
        self._workers = [
            WorkerInfo(
                worker_id=f"local-worker-{i}",
                hostname=os.uname().nodename,
                state=WorkerState.ONLINE,
                max_concurrency=1,
            )
            for i in range(max_workers)
        ]

        logger.info(
            f"Local backend connected with {max_workers} "
            f"{'process' if use_processes else 'thread'} workers"
        )

    def _do_disconnect(self) -> None:
        """Shutdown the executor pool."""
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._futures.clear()
        self._workers.clear()

    def _do_submit(
        self,
        task: DistributedTask["CheckpointResult"],
        checkpoint: "Checkpoint",
        priority: TaskPriority,
        timeout: float,
        context: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Submit task to the executor pool."""
        if self._executor is None:
            raise RuntimeError("Backend not connected")

        # Serialize checkpoint for process execution
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

        # Submit to executor
        future = self._executor.submit(
            _execute_checkpoint_task,
            checkpoint_dict,
            context,
        )

        self._futures[task.task_id] = future

        # Set up completion callback
        def on_complete(f: Future[dict[str, Any]]) -> None:
            try:
                result_dict = f.result()
                # Deserialize result
                from truthound.checkpoint.checkpoint import CheckpointResult
                result = CheckpointResult.from_dict(result_dict)
                self._complete_task(task.task_id, result=result)
            except Exception as e:
                self._complete_task(task.task_id, error=str(e), exception=e)

        future.add_done_callback(on_complete)

        # Mark as running
        task._set_state(TaskState.RUNNING)
        task._started_at = datetime.now()

    def _do_get_cluster_state(self) -> ClusterState:
        """Get local cluster state."""
        # Count running tasks
        running = sum(1 for f in self._futures.values() if f.running())

        return ClusterState(
            workers=self._workers.copy(),
            total_capacity=len(self._workers),
            current_load=running,
            pending_tasks=sum(1 for f in self._futures.values() if not f.done()),
            backend_name=self.name,
            backend_version="1.0.0",
            is_healthy=self._executor is not None,
        )

    def _do_get_workers(self) -> list[WorkerInfo]:
        """Get list of workers."""
        return self._workers.copy()

    def cancel_task(self, task_id: str, terminate: bool = False) -> bool:
        """Cancel a running task."""
        future = self._futures.get(task_id)
        if future is None:
            return False

        if future.done():
            return False

        cancelled = future.cancel()
        if cancelled:
            task = self._tasks.get(task_id)
            if task:
                task._set_state(TaskState.CANCELLED)
                task._completed_at = datetime.now()

        return cancelled


class LocalBackendWithPriority(LocalBackend):
    """Local backend with priority queue support.

    This extends LocalBackend to support task prioritization using
    a priority queue instead of direct executor submission.
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        max_workers: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, max_workers=max_workers, use_processes=False, **kwargs)

        from queue import PriorityQueue

        self._priority_queue: PriorityQueue[tuple[int, str, Any]] = PriorityQueue()
        self._dispatcher_thread: threading.Thread | None = None
        self._dispatcher_running = False

    @property
    def capabilities(self) -> BackendCapability:
        return super().capabilities | BackendCapability.PRIORITY_QUEUE

    def _do_connect(self, **kwargs: Any) -> None:
        super()._do_connect(**kwargs)

        # Start dispatcher thread
        self._dispatcher_running = True
        self._dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            daemon=True,
            name="local-priority-dispatcher",
        )
        self._dispatcher_thread.start()

    def _do_disconnect(self) -> None:
        self._dispatcher_running = False
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5.0)
            self._dispatcher_thread = None
        super()._do_disconnect()

    def _do_submit(
        self,
        task: DistributedTask["CheckpointResult"],
        checkpoint: "Checkpoint",
        priority: TaskPriority,
        timeout: float,
        context: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Add task to priority queue."""
        # Negate priority so higher values are processed first
        priority_value = -priority.value

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

        self._priority_queue.put((
            priority_value,
            task.task_id,
            {
                "task": task,
                "checkpoint_dict": checkpoint_dict,
                "context": context,
                "timeout": timeout,
            },
        ))

    def _dispatcher_loop(self) -> None:
        """Dispatch tasks from priority queue to executor."""
        while self._dispatcher_running:
            try:
                # Get next task from queue
                priority, task_id, data = self._priority_queue.get(timeout=1.0)

                task = data["task"]
                checkpoint_dict = data["checkpoint_dict"]
                context = data["context"]

                # Submit to executor
                if self._executor:
                    future = self._executor.submit(
                        _execute_checkpoint_task,
                        checkpoint_dict,
                        context,
                    )

                    self._futures[task_id] = future

                    def on_complete(f: Future[dict[str, Any]], tid: str = task_id) -> None:
                        try:
                            result_dict = f.result()
                            from truthound.checkpoint.checkpoint import CheckpointResult
                            result = CheckpointResult.from_dict(result_dict)
                            self._complete_task(tid, result=result)
                        except Exception as e:
                            self._complete_task(tid, error=str(e), exception=e)

                    future.add_done_callback(on_complete)

                    task._set_state(TaskState.RUNNING)
                    task._started_at = datetime.now()

            except Exception:
                # Queue empty or other error, continue
                pass
