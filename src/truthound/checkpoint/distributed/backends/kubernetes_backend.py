"""Kubernetes Backend for Distributed Checkpoints.

This backend provides distributed execution using Kubernetes Jobs,
suitable for cloud-native deployments.

Requirements:
    pip install kubernetes
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

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


# Check if kubernetes client is available
try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    client = None
    config = None
    ApiException = Exception


@dataclass
class KubernetesBackendConfig:
    """Configuration for Kubernetes backend.

    Attributes:
        namespace: Kubernetes namespace for jobs.
        image: Container image for checkpoint execution.
        image_pull_policy: Image pull policy.
        service_account: Service account name.
        ttl_seconds_after_finished: TTL for completed jobs.
        active_deadline_seconds: Maximum job duration.
        backoff_limit: Number of retries for failed jobs.
        cpu_request: CPU request per job.
        cpu_limit: CPU limit per job.
        memory_request: Memory request per job.
        memory_limit: Memory limit per job.
        labels: Labels to apply to jobs.
        annotations: Annotations to apply to jobs.
        env_from_secrets: Secrets to mount as environment variables.
        env_from_configmaps: ConfigMaps to mount as environment variables.
        volumes: Volume specifications.
        volume_mounts: Volume mount specifications.
    """

    namespace: str = "default"
    image: str = "truthound/checkpoint-executor:latest"
    image_pull_policy: str = "IfNotPresent"
    service_account: str | None = None
    ttl_seconds_after_finished: int = 3600
    active_deadline_seconds: int = 3600
    backoff_limit: int = 3
    cpu_request: str = "100m"
    cpu_limit: str = "1"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    env_from_secrets: list[str] = field(default_factory=list)
    env_from_configmaps: list[str] = field(default_factory=list)
    volumes: list[dict[str, Any]] = field(default_factory=list)
    volume_mounts: list[dict[str, Any]] = field(default_factory=list)


class KubernetesTask(DistributedTask["CheckpointResult"]):
    """Kubernetes-specific task wrapper."""

    def __init__(
        self,
        task_id: str,
        job_name: str,
        checkpoint_name: str,
        backend: "KubernetesBackend",
    ) -> None:
        super().__init__(
            task_id=task_id,
            checkpoint_name=checkpoint_name,
            _backend=backend,
        )
        self._job_name = job_name
        self._result_cache: "CheckpointResult | None" = None
        self._error_cache: str | None = None

    @property
    def state(self) -> TaskState:
        """Get task state by querying Kubernetes."""
        if self._result_cache is not None:
            return TaskState.SUCCEEDED
        if self._error_cache is not None:
            return TaskState.FAILED

        try:
            status = self._backend._get_job_status(self._job_name)
            return status
        except Exception:
            return TaskState.PENDING

    def result(self, timeout: float | None = None) -> "CheckpointResult":
        """Wait for and return the result."""
        if self._result_cache is not None:
            return self._result_cache

        deadline = time.time() + timeout if timeout else None

        while True:
            state = self.state
            if state == TaskState.SUCCEEDED:
                # Get result from job annotation or ConfigMap
                result_data = self._backend._get_job_result(self._job_name)
                from truthound.checkpoint.checkpoint import CheckpointResult
                self._result_cache = CheckpointResult.from_dict(result_data)
                return self._result_cache

            if state == TaskState.FAILED:
                error = self._backend._get_job_error(self._job_name)
                self._error_cache = error
                raise Exception(f"Job failed: {error}")

            if deadline and time.time() > deadline:
                from truthound.checkpoint.distributed.protocols import TaskTimeoutError
                raise TaskTimeoutError(
                    f"Task {self.task_id} timed out",
                    self.task_id,
                    timeout or 0,
                )

            time.sleep(2.0)

    async def result_async(self, timeout: float | None = None) -> "CheckpointResult":
        """Async version of result()."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.result(timeout=timeout),
        )

    def cancel(self, terminate: bool = False) -> bool:
        """Delete the Kubernetes job."""
        try:
            self._backend._delete_job(self._job_name)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if job is complete."""
        state = self.state
        return state in {TaskState.SUCCEEDED, TaskState.FAILED}

    def is_successful(self) -> bool:
        """Check if job succeeded."""
        return self.state == TaskState.SUCCEEDED


class KubernetesBackend(BaseDistributedBackend):
    """Kubernetes-based distributed backend.

    This backend executes checkpoints as Kubernetes Jobs, providing
    cloud-native scaling and fault tolerance.

    Example:
        >>> from truthound.checkpoint.distributed import KubernetesBackend
        >>>
        >>> backend = KubernetesBackend(
        ...     namespace="truthound",
        ...     image="my-registry/truthound:latest",
        ... )
        >>>
        >>> with backend.connection():
        ...     task = backend.submit(checkpoint)
        ...     result = task.result(timeout=300)

    Note:
        Requires proper Kubernetes RBAC permissions:
        - jobs: create, get, list, watch, delete
        - pods: get, list, watch
        - configmaps: create, get, delete (for result storage)
    """

    def __init__(
        self,
        config: DistributedConfig | None = None,
        namespace: str = "default",
        image: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Kubernetes backend.

        Args:
            config: Distributed configuration.
            namespace: Kubernetes namespace.
            image: Container image for execution.
            **kwargs: Additional Kubernetes configuration.
        """
        if not K8S_AVAILABLE:
            raise BackendNotAvailableError(
                "kubernetes",
                reason="kubernetes client is not installed",
                install_hint="pip install kubernetes",
            )

        super().__init__(config, **kwargs)

        self._k8s_config = KubernetesBackendConfig(
            namespace=namespace,
            image=image or "truthound/checkpoint-executor:latest",
        )

        # Apply additional kwargs
        for key, value in kwargs.items():
            if hasattr(self._k8s_config, key):
                setattr(self._k8s_config, key, value)

        self._batch_v1: Any = None
        self._core_v1: Any = None
        self._watcher_thread: threading.Thread | None = None
        self._watcher_running = False

    @property
    def name(self) -> str:
        return "kubernetes"

    @property
    def capabilities(self) -> BackendCapability:
        return (
            BackendCapability.ASYNC_SUBMIT
            | BackendCapability.BATCH_SUBMIT
            | BackendCapability.RETRY_POLICY
            | BackendCapability.RESULT_BACKEND
            | BackendCapability.TASK_REVOKE
            | BackendCapability.WORKER_SCALING
            | BackendCapability.HEALTH_CHECK
            | BackendCapability.METRICS
        )

    def _do_connect(self, **kwargs: Any) -> None:
        """Initialize Kubernetes client."""
        try:
            # Try in-cluster config first
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except Exception:
            try:
                # Fall back to kubeconfig
                config.load_kube_config()
                logger.info("Using kubeconfig Kubernetes configuration")
            except Exception as e:
                raise BackendNotAvailableError(
                    "kubernetes",
                    reason=f"Failed to load Kubernetes config: {e}",
                )

        self._batch_v1 = client.BatchV1Api()
        self._core_v1 = client.CoreV1Api()

        # Verify namespace exists
        try:
            self._core_v1.read_namespace(self._k8s_config.namespace)
        except ApiException as e:
            if e.status == 404:
                logger.warning(
                    f"Namespace {self._k8s_config.namespace} not found, "
                    "it will be created on first job submission"
                )

        logger.info(f"Kubernetes backend connected to namespace {self._k8s_config.namespace}")

    def _do_disconnect(self) -> None:
        """Cleanup Kubernetes resources."""
        self._watcher_running = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5.0)
            self._watcher_thread = None
        self._batch_v1 = None
        self._core_v1 = None

    def _do_submit(
        self,
        task: DistributedTask["CheckpointResult"],
        checkpoint: "Checkpoint",
        priority: TaskPriority,
        timeout: float,
        context: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Submit task as Kubernetes Job."""
        job_name = f"truthound-cp-{uuid4().hex[:8]}"

        # Serialize checkpoint to JSON
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
            },
            "context": context or {},
        }
        checkpoint_json = json.dumps(checkpoint_dict)
        checkpoint_b64 = base64.b64encode(checkpoint_json.encode()).decode()

        # Build Job spec
        job = self._build_job_spec(
            job_name=job_name,
            checkpoint_b64=checkpoint_b64,
            priority=priority,
            timeout=timeout,
            **kwargs,
        )

        # Create Job
        try:
            self._batch_v1.create_namespaced_job(
                namespace=self._k8s_config.namespace,
                body=job,
            )
        except ApiException as e:
            raise Exception(f"Failed to create Kubernetes job: {e}")

        # Create task wrapper
        k8s_task = KubernetesTask(
            task_id=job_name,
            job_name=job_name,
            checkpoint_name=checkpoint.name,
            backend=self,
        )

        task.task_id = k8s_task.task_id
        task._job_name = job_name  # type: ignore
        task._set_state(TaskState.QUEUED)

        with self._lock:
            self._tasks[task.task_id] = task

        logger.info(f"Created Kubernetes job: {job_name}")

    def _build_job_spec(
        self,
        job_name: str,
        checkpoint_b64: str,
        priority: TaskPriority,
        timeout: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build Kubernetes Job specification."""
        labels = {
            "app": "truthound",
            "component": "checkpoint-executor",
            "job-name": job_name,
            **self._k8s_config.labels,
        }

        annotations = {
            "truthound.io/checkpoint-priority": str(priority.value),
            **self._k8s_config.annotations,
        }

        # Environment variables
        env = [
            client.V1EnvVar(name="TRUTHOUND_CHECKPOINT_DATA", value=checkpoint_b64),
            client.V1EnvVar(name="TRUTHOUND_JOB_NAME", value=job_name),
            client.V1EnvVar(name="TRUTHOUND_NAMESPACE", value=self._k8s_config.namespace),
        ]

        # Add env from secrets
        env_from = []
        for secret_name in self._k8s_config.env_from_secrets:
            env_from.append(
                client.V1EnvFromSource(
                    secret_ref=client.V1SecretEnvSource(name=secret_name)
                )
            )
        for configmap_name in self._k8s_config.env_from_configmaps:
            env_from.append(
                client.V1EnvFromSource(
                    config_map_ref=client.V1ConfigMapEnvSource(name=configmap_name)
                )
            )

        # Resource requirements
        resources = client.V1ResourceRequirements(
            requests={
                "cpu": kwargs.get("cpu_request", self._k8s_config.cpu_request),
                "memory": kwargs.get("memory_request", self._k8s_config.memory_request),
            },
            limits={
                "cpu": kwargs.get("cpu_limit", self._k8s_config.cpu_limit),
                "memory": kwargs.get("memory_limit", self._k8s_config.memory_limit),
            },
        )

        # Container spec
        container = client.V1Container(
            name="checkpoint-executor",
            image=self._k8s_config.image,
            image_pull_policy=self._k8s_config.image_pull_policy,
            command=["python", "-m", "truthound.checkpoint.distributed.k8s_executor"],
            env=env,
            env_from=env_from if env_from else None,
            resources=resources,
            volume_mounts=[
                client.V1VolumeMount(**vm) for vm in self._k8s_config.volume_mounts
            ] if self._k8s_config.volume_mounts else None,
        )

        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            service_account_name=self._k8s_config.service_account,
            volumes=[
                client.V1Volume(**v) for v in self._k8s_config.volumes
            ] if self._k8s_config.volumes else None,
        )

        # Pod template
        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels, annotations=annotations),
            spec=pod_spec,
        )

        # Job spec
        job_spec = client.V1JobSpec(
            template=template,
            backoff_limit=self._k8s_config.backoff_limit,
            active_deadline_seconds=int(timeout),
            ttl_seconds_after_finished=self._k8s_config.ttl_seconds_after_finished,
        )

        # Job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self._k8s_config.namespace,
                labels=labels,
                annotations=annotations,
            ),
            spec=job_spec,
        )

        return job

    def _get_job_status(self, job_name: str) -> TaskState:
        """Get job status from Kubernetes."""
        try:
            job = self._batch_v1.read_namespaced_job(
                name=job_name,
                namespace=self._k8s_config.namespace,
            )

            status = job.status

            if status.succeeded and status.succeeded > 0:
                return TaskState.SUCCEEDED
            if status.failed and status.failed > 0:
                return TaskState.FAILED
            if status.active and status.active > 0:
                return TaskState.RUNNING

            return TaskState.QUEUED

        except ApiException as e:
            if e.status == 404:
                return TaskState.CANCELLED
            raise

    def _get_job_result(self, job_name: str) -> dict[str, Any]:
        """Get job result from ConfigMap."""
        result_cm_name = f"{job_name}-result"
        try:
            cm = self._core_v1.read_namespaced_config_map(
                name=result_cm_name,
                namespace=self._k8s_config.namespace,
            )
            result_json = cm.data.get("result", "{}")
            return json.loads(result_json)
        except ApiException as e:
            if e.status == 404:
                # Result not yet available or job didn't store result
                return {}
            raise

    def _get_job_error(self, job_name: str) -> str:
        """Get job error message from pod logs."""
        try:
            # Get pods for this job
            pods = self._core_v1.list_namespaced_pod(
                namespace=self._k8s_config.namespace,
                label_selector=f"job-name={job_name}",
            )

            if pods.items:
                pod = pods.items[0]
                try:
                    logs = self._core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace=self._k8s_config.namespace,
                        tail_lines=50,
                    )
                    return logs
                except Exception:
                    pass

                # Check container status
                if pod.status and pod.status.container_statuses:
                    for cs in pod.status.container_statuses:
                        if cs.state and cs.state.terminated:
                            return cs.state.terminated.message or "Unknown error"

            return "Unknown error"
        except Exception as e:
            return str(e)

    def _delete_job(self, job_name: str) -> None:
        """Delete a Kubernetes job."""
        try:
            self._batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self._k8s_config.namespace,
                propagation_policy="Foreground",
            )
        except ApiException as e:
            if e.status != 404:
                raise

        # Also delete result ConfigMap
        try:
            self._core_v1.delete_namespaced_config_map(
                name=f"{job_name}-result",
                namespace=self._k8s_config.namespace,
            )
        except Exception:
            pass

    def _do_get_cluster_state(self) -> ClusterState:
        """Get Kubernetes cluster state."""
        if self._batch_v1 is None:
            return ClusterState(backend_name=self.name, is_healthy=False)

        try:
            # Get nodes
            nodes = self._core_v1.list_node()
            workers = []

            for node in nodes.items:
                # Check node conditions
                is_ready = False
                for condition in node.status.conditions or []:
                    if condition.type == "Ready":
                        is_ready = condition.status == "True"
                        break

                # Get allocatable resources
                allocatable = node.status.allocatable or {}
                cpu = allocatable.get("cpu", "0")
                # Parse CPU (e.g., "4" or "4000m")
                if cpu.endswith("m"):
                    cpu_count = int(cpu[:-1]) // 1000
                else:
                    cpu_count = int(cpu)

                workers.append(WorkerInfo(
                    worker_id=node.metadata.name,
                    hostname=node.metadata.name,
                    state=WorkerState.ONLINE if is_ready else WorkerState.OFFLINE,
                    max_concurrency=cpu_count,
                    metadata={
                        "labels": node.metadata.labels,
                        "cpu": cpu,
                        "memory": allocatable.get("memory"),
                    },
                ))

            # Get active jobs
            jobs = self._batch_v1.list_namespaced_job(
                namespace=self._k8s_config.namespace,
                label_selector="app=truthound",
            )

            active_count = sum(
                1 for job in jobs.items
                if job.status.active and job.status.active > 0
            )
            pending_count = sum(
                1 for job in jobs.items
                if not job.status.active
                and not job.status.succeeded
                and not job.status.failed
            )

            total_capacity = sum(w.max_concurrency for w in workers)

            return ClusterState(
                workers=workers,
                total_capacity=total_capacity,
                current_load=active_count,
                pending_tasks=pending_count,
                backend_name=self.name,
                is_healthy=any(w.state == WorkerState.ONLINE for w in workers),
            )

        except Exception as e:
            logger.error(f"Failed to get cluster state: {e}")
            return ClusterState(backend_name=self.name, is_healthy=False)

    def _do_get_workers(self) -> list[WorkerInfo]:
        """Get list of Kubernetes nodes."""
        state = self._do_get_cluster_state()
        return state.workers

    def _do_scale_workers(self, count: int) -> bool:
        """Scale workers (requires cluster autoscaler)."""
        # Kubernetes node scaling is typically managed by cluster autoscaler
        # or node pool management in cloud providers
        logger.warning("Kubernetes node scaling requires external cluster management")
        return False

    def cleanup_completed_jobs(self, max_age_seconds: int = 3600) -> int:
        """Clean up completed jobs older than max_age.

        Args:
            max_age_seconds: Maximum age in seconds.

        Returns:
            Number of jobs cleaned up.
        """
        if self._batch_v1 is None:
            return 0

        jobs = self._batch_v1.list_namespaced_job(
            namespace=self._k8s_config.namespace,
            label_selector="app=truthound",
        )

        cleaned = 0
        now = datetime.now()

        for job in jobs.items:
            # Check if completed
            if not (job.status.succeeded or job.status.failed):
                continue

            # Check age
            completion_time = job.status.completion_time
            if completion_time:
                age = (now - completion_time.replace(tzinfo=None)).total_seconds()
                if age > max_age_seconds:
                    try:
                        self._delete_job(job.metadata.name)
                        cleaned += 1
                    except Exception as e:
                        logger.error(f"Failed to cleanup job {job.metadata.name}: {e}")

        return cleaned
