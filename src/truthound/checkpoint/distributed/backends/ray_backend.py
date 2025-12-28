"""Ray Backend for Distributed Checkpoints.

This backend provides distributed execution using Ray,
a unified framework for scaling AI and Python applications.

Requirements:
    pip install ray
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
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


# Check if Ray is available
try:
    import ray
    from ray.exceptions import RayTaskError, GetTimeoutError
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None


@dataclass
class RayBackendConfig:
    """Configuration for Ray backend.

    Attributes:
        address: Ray cluster address (None = start local cluster).
        num_cpus: Number of CPUs for local cluster.
        num_gpus: Number of GPUs for local cluster.
        runtime_env: Ray runtime environment.
        namespace: Ray namespace.
        object_store_memory: Object store memory limit.
        dashboard_host: Dashboard host.
        dashboard_port: Dashboard port.
    """

    address: str | None = None
    num_cpus: int | None = None
    num_gpus: int | None = None
    runtime_env: dict[str, Any] | None = None
    namespace: str = "truthound"
    object_store_memory: int | None = None
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8265


class RayTask(DistributedTask["CheckpointResult"]):
    """Ray-specific task wrapper."""

    def __init__(
        self,
        task_id: str,
        checkpoint_name: str,
        object_ref: "ray.ObjectRef",
        backend: "RayBackend",
    ) -> None:
        super().__init__(
            task_id=task_id,
            checkpoint_name=checkpoint_name,
            _backend=backend,
        )
        self._object_ref = object_ref
        self._result_cache: "CheckpointResult | None" = None
        self._error_cache: Exception | None = None

    @property
    def state(self) -> TaskState:
        """Get task state by checking Ray object status."""
        if self._result_cache is not None:
            return TaskState.SUCCEEDED
        if self._error_cache is not None:
            return TaskState.FAILED

        # Check if ready
        ready, _ = ray.wait([self._object_ref], timeout=0)
        if ready:
            try:
                self._result_cache = ray.get(self._object_ref, timeout=0)
                return TaskState.SUCCEEDED
            except Exception as e:
                self._error_cache = e
                return TaskState.FAILED

        return TaskState.RUNNING

    def result(self, timeout: float | None = None) -> "CheckpointResult":
        """Wait for and return the result."""
        if self._result_cache is not None:
            return self._result_cache

        try:
            result_dict = ray.get(self._object_ref, timeout=timeout)
            from truthound.checkpoint.checkpoint import CheckpointResult
            self._result_cache = CheckpointResult.from_dict(result_dict)
            return self._result_cache
        except GetTimeoutError:
            from truthound.checkpoint.distributed.protocols import TaskTimeoutError
            raise TaskTimeoutError(
                f"Task {self.task_id} timed out",
                self.task_id,
                timeout or 0,
            )
        except RayTaskError as e:
            self._error_cache = e
            raise

    async def result_async(self, timeout: float | None = None) -> "CheckpointResult":
        """Async version of result()."""
        import asyncio

        if self._result_cache is not None:
            return self._result_cache

        try:
            # Ray doesn't have native async get, use executor
            loop = asyncio.get_event_loop()
            result_dict = await loop.run_in_executor(
                None,
                lambda: ray.get(self._object_ref, timeout=timeout),
            )
            from truthound.checkpoint.checkpoint import CheckpointResult
            self._result_cache = CheckpointResult.from_dict(result_dict)
            return self._result_cache
        except GetTimeoutError:
            from truthound.checkpoint.distributed.protocols import TaskTimeoutError
            raise TaskTimeoutError(
                f"Task {self.task_id} timed out",
                self.task_id,
                timeout or 0,
            )

    def cancel(self, terminate: bool = False) -> bool:
        """Cancel the Ray task."""
        try:
            ray.cancel(self._object_ref, force=terminate)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if task is complete."""
        if self._result_cache or self._error_cache:
            return True
        ready, _ = ray.wait([self._object_ref], timeout=0)
        return len(ready) > 0

    def is_successful(self) -> bool:
        """Check if task succeeded."""
        if self._error_cache:
            return False
        if self._result_cache:
            return True
        if self.is_ready():
            try:
                self.result(timeout=0)
                return True
            except Exception:
                return False
        return False


class RayBackend(BaseDistributedBackend):
    """Ray-based distributed backend.

    This backend uses Ray for distributed task execution, providing
    automatic scaling and fault tolerance.

    Example:
        >>> from truthound.checkpoint.distributed import RayBackend
        >>>
        >>> # Connect to existing cluster
        >>> backend = RayBackend(address="ray://localhost:10001")
        >>>
        >>> # Or start local cluster
        >>> backend = RayBackend(num_cpus=4)
        >>>
        >>> with backend.connection():
        ...     task = backend.submit(checkpoint)
        ...     result = task.result(timeout=300)
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        address: str | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Ray backend.

        Args:
            config: Distributed configuration.
            address: Ray cluster address.
            num_cpus: Number of CPUs for local cluster.
            num_gpus: Number of GPUs for local cluster.
            **kwargs: Additional Ray configuration.
        """
        if not RAY_AVAILABLE:
            raise BackendNotAvailableError(
                "ray",
                reason="Ray is not installed",
                install_hint="pip install ray",
            )

        super().__init__(config, **kwargs)

        self._ray_config = RayBackendConfig(
            address=address,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(self._ray_config, key):
                setattr(self._ray_config, key, value)

        self._initialized_ray = False

    @property
    def name(self) -> str:
        return "ray"

    @property
    def capabilities(self) -> BackendCapability:
        return (
            BackendCapability.ASYNC_SUBMIT
            | BackendCapability.BATCH_SUBMIT
            | BackendCapability.RESULT_BACKEND
            | BackendCapability.TASK_REVOKE
            | BackendCapability.WORKER_SCALING
            | BackendCapability.HEALTH_CHECK
            | BackendCapability.METRICS
        )

    def _do_connect(self, **kwargs: Any) -> None:
        """Initialize Ray runtime."""
        if ray.is_initialized():
            logger.info("Ray already initialized, reusing existing context")
            return

        init_kwargs = {
            "namespace": self._ray_config.namespace,
        }

        if self._ray_config.address:
            init_kwargs["address"] = self._ray_config.address
        else:
            # Local cluster
            if self._ray_config.num_cpus:
                init_kwargs["num_cpus"] = self._ray_config.num_cpus
            if self._ray_config.num_gpus:
                init_kwargs["num_gpus"] = self._ray_config.num_gpus
            if self._ray_config.object_store_memory:
                init_kwargs["object_store_memory"] = self._ray_config.object_store_memory

        if self._ray_config.runtime_env:
            init_kwargs["runtime_env"] = self._ray_config.runtime_env

        # Apply override kwargs
        init_kwargs.update(kwargs)

        ray.init(**init_kwargs)
        self._initialized_ray = True

        logger.info(f"Ray backend connected: {ray.get_runtime_context().get_node_id()}")

    def _do_disconnect(self) -> None:
        """Shutdown Ray if we initialized it."""
        if self._initialized_ray and ray.is_initialized():
            ray.shutdown()
            self._initialized_ray = False
            logger.info("Ray backend disconnected")

    def _do_submit(
        self,
        task: DistributedTask["CheckpointResult"],
        checkpoint: "Checkpoint",
        priority: TaskPriority,
        timeout: float,
        context: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Submit task to Ray."""
        # Create remote function
        @ray.remote
        def execute_checkpoint(
            checkpoint_dict: dict[str, Any],
            context: dict[str, Any] | None,
        ) -> dict[str, Any]:
            """Execute checkpoint in Ray worker."""
            from truthound.checkpoint.checkpoint import (
                Checkpoint,
                CheckpointConfig,
            )

            config = CheckpointConfig(**checkpoint_dict.get("config", {}))
            cp = Checkpoint(config=config)
            result = cp.run(context=context)

            return result.to_dict()

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

        # Determine resources
        num_cpus = kwargs.get("num_cpus", 1)
        num_gpus = kwargs.get("num_gpus", 0)

        # Submit to Ray
        object_ref = execute_checkpoint.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        ).remote(checkpoint_dict, context)

        # Create Ray task wrapper
        ray_task = RayTask(
            task_id=f"ray-{object_ref.hex()[:16]}",
            checkpoint_name=checkpoint.name,
            object_ref=object_ref,
            backend=self,
        )

        # Update original task
        task.task_id = ray_task.task_id
        task._object_ref = object_ref  # type: ignore
        task._set_state(TaskState.RUNNING)
        task._started_at = datetime.now()

        # Update tracking
        with self._lock:
            self._tasks[task.task_id] = task

    def submit_batch(
        self,
        checkpoints: list["Checkpoint"],
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[DistributedTaskProtocol["CheckpointResult"]]:
        """Submit multiple checkpoints efficiently using Ray's parallelism."""
        tasks = []

        # Create remote function once
        @ray.remote
        def execute_checkpoint_batch(
            checkpoint_dict: dict[str, Any],
            ctx: dict[str, Any] | None,
        ) -> dict[str, Any]:
            from truthound.checkpoint.checkpoint import (
                Checkpoint,
                CheckpointConfig,
            )

            config = CheckpointConfig(**checkpoint_dict.get("config", {}))
            cp = Checkpoint(config=config)
            result = cp.run(context=ctx)

            return result.to_dict()

        # Submit all at once
        object_refs = []
        for checkpoint in checkpoints:
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

            ref = execute_checkpoint_batch.remote(checkpoint_dict, context)
            object_refs.append((checkpoint, ref))

        # Create task wrappers
        for checkpoint, ref in object_refs:
            task: DistributedTask["CheckpointResult"] = DistributedTask.create(
                checkpoint=checkpoint,
                backend=self,
            )

            ray_task = RayTask(
                task_id=f"ray-{ref.hex()[:16]}",
                checkpoint_name=checkpoint.name,
                object_ref=ref,
                backend=self,
            )

            task.task_id = ray_task.task_id
            task._object_ref = ref  # type: ignore
            task._set_state(TaskState.RUNNING)
            task._started_at = datetime.now()

            with self._lock:
                self._tasks[task.task_id] = task

            tasks.append(task)
            self._metrics.record_submission()

        return tasks

    def _do_get_cluster_state(self) -> ClusterState:
        """Get Ray cluster state."""
        if not ray.is_initialized():
            return ClusterState(backend_name=self.name, is_healthy=False)

        try:
            nodes = ray.nodes()
            workers = []

            for node in nodes:
                if not node.get("Alive", False):
                    continue

                resources = node.get("Resources", {})
                resources_used = node.get("Resources_Used", {})

                total_cpus = resources.get("CPU", 0)
                used_cpus = resources_used.get("CPU", 0)

                workers.append(WorkerInfo(
                    worker_id=node.get("NodeID", "unknown"),
                    hostname=node.get("NodeManagerAddress", "unknown"),
                    state=WorkerState.ONLINE if node.get("Alive") else WorkerState.OFFLINE,
                    current_tasks=int(used_cpus),
                    max_concurrency=int(total_cpus),
                    metadata={
                        "object_store_memory": node.get("ObjectStoreMemory"),
                        "ray_version": node.get("RayVersion"),
                    },
                ))

            total_capacity = sum(w.max_concurrency for w in workers)
            current_load = sum(w.current_tasks for w in workers)

            return ClusterState(
                workers=workers,
                total_capacity=total_capacity,
                current_load=current_load,
                pending_tasks=0,  # Ray doesn't expose pending task count easily
                backend_name=self.name,
                backend_version=ray.__version__,
                is_healthy=len(workers) > 0,
            )

        except Exception as e:
            logger.error(f"Failed to get cluster state: {e}")
            return ClusterState(backend_name=self.name, is_healthy=False)

    def _do_get_workers(self) -> list[WorkerInfo]:
        """Get list of Ray nodes."""
        state = self._do_get_cluster_state()
        return state.workers

    def _do_scale_workers(self, count: int) -> bool:
        """Scale Ray cluster (requires Ray autoscaler)."""
        # Ray autoscaling is typically managed externally
        # This would require Ray cluster configuration
        logger.warning("Ray autoscaling requires external cluster management")
        return False


class RayActorBackend(RayBackend):
    """Ray backend using persistent actors for better performance.

    This variant uses Ray actors to maintain checkpoint execution context,
    reducing initialization overhead for repeated executions.
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        num_actors: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._num_actors = num_actors
        self._actors: list[Any] = []
        self._actor_index = 0
        self._actor_lock = None

    def _do_connect(self, **kwargs: Any) -> None:
        """Initialize Ray and create actor pool."""
        import threading
        super()._do_connect(**kwargs)

        self._actor_lock = threading.Lock()

        # Create actor class
        @ray.remote
        class CheckpointExecutor:
            def __init__(self) -> None:
                self._execution_count = 0

            def execute(
                self,
                checkpoint_dict: dict[str, Any],
                context: dict[str, Any] | None,
            ) -> dict[str, Any]:
                from truthound.checkpoint.checkpoint import (
                    Checkpoint,
                    CheckpointConfig,
                )

                config = CheckpointConfig(**checkpoint_dict.get("config", {}))
                checkpoint = Checkpoint(config=config)
                result = checkpoint.run(context=context)

                self._execution_count += 1

                return result.to_dict()

            def get_stats(self) -> dict[str, Any]:
                return {"execution_count": self._execution_count}

        # Create actor pool
        self._actors = [CheckpointExecutor.remote() for _ in range(self._num_actors)]

        logger.info(f"Created {self._num_actors} Ray checkpoint executor actors")

    def _do_disconnect(self) -> None:
        """Shutdown actors and Ray."""
        for actor in self._actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self._actors.clear()
        super()._do_disconnect()

    def _get_next_actor(self) -> Any:
        """Get next actor in round-robin fashion."""
        with self._actor_lock:
            actor = self._actors[self._actor_index]
            self._actor_index = (self._actor_index + 1) % len(self._actors)
            return actor

    def _do_submit(
        self,
        task: DistributedTask["CheckpointResult"],
        checkpoint: "Checkpoint",
        priority: TaskPriority,
        timeout: float,
        context: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Submit task to an actor."""
        actor = self._get_next_actor()

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

        # Submit to actor
        object_ref = actor.execute.remote(checkpoint_dict, context)

        ray_task = RayTask(
            task_id=f"ray-actor-{object_ref.hex()[:16]}",
            checkpoint_name=checkpoint.name,
            object_ref=object_ref,
            backend=self,
        )

        task.task_id = ray_task.task_id
        task._object_ref = object_ref  # type: ignore
        task._set_state(TaskState.RUNNING)
        task._started_at = datetime.now()

        with self._lock:
            self._tasks[task.task_id] = task
