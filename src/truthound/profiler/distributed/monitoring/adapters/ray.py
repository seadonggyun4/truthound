"""Ray backend monitoring adapter.

This module provides monitoring for the Ray distributed backend.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from truthound.profiler.distributed.monitoring.adapters.base import (
    BackendMetrics,
    BackendMonitorAdapter,
    BackendWorkerInfo,
)
from truthound.profiler.distributed.monitoring.protocols import (
    EventSeverity,
    HealthStatus,
    MonitorEvent,
    MonitorEventType,
    WorkerHealth,
)


logger = logging.getLogger(__name__)


class RayMonitorAdapter(BackendMonitorAdapter):
    """Monitoring adapter for Ray distributed backend.

    Monitors Ray workers and collects cluster metrics.

    Example:
        import ray

        ray.init()
        adapter = RayMonitorAdapter()
        adapter.connect()

        workers = adapter.get_workers()
        metrics = adapter.get_metrics()

        adapter.disconnect()
    """

    def __init__(
        self,
        address: str | None = None,
        namespace: str | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """Initialize Ray adapter.

        Args:
            address: Ray cluster address
            namespace: Ray namespace
            poll_interval_seconds: Polling interval
        """
        super().__init__(poll_interval_seconds)

        self._address = address
        self._namespace = namespace
        self._initialized_ray = False

        # Worker tracking
        self._workers: dict[str, BackendWorkerInfo] = {}
        self._worker_health: dict[str, WorkerHealth] = {}

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "ray"

    def connect(self) -> None:
        """Connect to Ray cluster."""
        if self._connected:
            return

        try:
            import ray
        except ImportError:
            raise RuntimeError("ray is required for Ray monitoring")

        # Initialize Ray if not already
        if not ray.is_initialized():
            init_args: dict[str, Any] = {}
            if self._address:
                init_args["address"] = self._address
            if self._namespace:
                init_args["namespace"] = self._namespace

            ray.init(**init_args)
            self._initialized_ray = True

        self._refresh_workers()
        self._connected = True

        cluster_info = ray.cluster_resources()
        self._emit_event(MonitorEvent(
            event_type=MonitorEventType.MONITOR_START,
            message=f"Connected to Ray cluster ({cluster_info.get('CPU', 0)} CPUs)",
            metadata={"resources": cluster_info},
        ))

    def disconnect(self) -> None:
        """Disconnect from Ray cluster."""
        if not self._connected:
            return

        # Only shutdown if we initialized
        if self._initialized_ray:
            try:
                import ray

                ray.shutdown()
            except Exception:
                pass
            self._initialized_ray = False

        self._workers.clear()
        self._worker_health.clear()
        self._connected = False

    def get_workers(self) -> list[BackendWorkerInfo]:
        """Get all workers.

        Returns:
            List of worker info
        """
        self._refresh_workers()
        return list(self._workers.values())

    def get_worker_health(self, worker_id: str) -> WorkerHealth | None:
        """Get worker health.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker health or None
        """
        self._refresh_workers()
        return self._worker_health.get(worker_id)

    def get_metrics(self) -> BackendMetrics:
        """Get backend metrics.

        Returns:
            Backend metrics
        """
        try:
            import ray

            if not ray.is_initialized():
                return BackendMetrics()

            # Get cluster resources
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()

            # Get nodes
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n.get("Alive", False)]

            # Calculate metrics
            total_cpu = cluster_resources.get("CPU", 0)
            used_cpu = total_cpu - available_resources.get("CPU", 0)
            cpu_percent = (used_cpu / total_cpu * 100) if total_cpu > 0 else 0

            total_memory = cluster_resources.get("memory", 0) / (1024 * 1024)
            used_memory = total_memory - available_resources.get("memory", 0) / (1024 * 1024)

            return BackendMetrics(
                timestamp=datetime.now(),
                workers_total=len(alive_nodes),
                workers_active=len(alive_nodes),  # All alive nodes are active
                tasks_running=0,  # Would need to query Ray state
                tasks_pending=0,
                memory_used_mb=used_memory,
                memory_available_mb=total_memory - used_memory,
                cpu_utilization_percent=cpu_percent,
                metadata={
                    "total_cpu": total_cpu,
                    "available_cpu": available_resources.get("CPU", 0),
                    "total_gpu": cluster_resources.get("GPU", 0),
                    "available_gpu": available_resources.get("GPU", 0),
                    "object_store_memory": cluster_resources.get("object_store_memory", 0),
                },
            )
        except Exception as e:
            logger.warning(f"Error getting Ray metrics: {e}")
            return BackendMetrics()

    def _refresh_workers(self) -> None:
        """Refresh worker information from cluster."""
        try:
            import ray

            if not ray.is_initialized():
                return

            nodes = ray.nodes()
            current_workers = set()

            for node in nodes:
                if not node.get("Alive", False):
                    continue

                node_id = node.get("NodeID", "")
                current_workers.add(node_id)

                # Parse address
                address = node.get("NodeManagerAddress", "")
                host = address
                port = node.get("NodeManagerPort", 0)

                # Get resources for this node
                resources = node.get("Resources", {})
                cpu_count = int(resources.get("CPU", 1))
                memory_mb = resources.get("memory", 0) / (1024 * 1024)
                object_store_mb = resources.get("object_store_memory", 0) / (1024 * 1024)

                self._workers[node_id] = BackendWorkerInfo(
                    worker_id=node_id,
                    worker_type="ray_node",
                    host=host,
                    port=port,
                    cpu_count=cpu_count,
                    memory_mb=memory_mb,
                    started_at=None,  # Not easily available
                    metadata={
                        "object_store_mb": object_store_mb,
                        "gpu_count": resources.get("GPU", 0),
                        "is_head": node.get("IsHeadNode", False),
                        "raylet_pid": node.get("RayletPid", 0),
                    },
                )

                # Calculate health
                # Note: Ray doesn't easily expose per-node memory usage
                status = HealthStatus.HEALTHY

                # Check if node has been alive for a while
                raylet_pid = node.get("RayletPid", 0)
                if raylet_pid == 0:
                    status = HealthStatus.UNKNOWN

                self._worker_health[node_id] = WorkerHealth(
                    worker_id=node_id,
                    status=status,
                    last_heartbeat=datetime.now(),
                    cpu_percent=0.0,  # Not available per node
                    memory_percent=0.0,
                    memory_used_mb=0.0,
                    active_tasks=0,
                    metadata={
                        "is_head": node.get("IsHeadNode", False),
                    },
                )

            # Remove workers that are no longer in cluster
            for worker_id in list(self._workers.keys()):
                if worker_id not in current_workers:
                    del self._workers[worker_id]
                    self._worker_health.pop(worker_id, None)

                    self._emit_event(MonitorEvent(
                        event_type=MonitorEventType.WORKER_UNREGISTERED,
                        severity=EventSeverity.WARNING,
                        message=f"Ray node {worker_id[:8]}... removed from cluster",
                        worker_id=worker_id,
                    ))

        except Exception as e:
            logger.warning(f"Error refreshing Ray workers: {e}")

    def get_task_status(self, object_ref: Any) -> str:
        """Get status of a Ray task.

        Args:
            object_ref: Ray ObjectRef

        Returns:
            Task status string
        """
        try:
            import ray

            ready, _ = ray.wait([object_ref], timeout=0)
            if ready:
                return "completed"
            return "running"
        except Exception:
            return "unknown"
