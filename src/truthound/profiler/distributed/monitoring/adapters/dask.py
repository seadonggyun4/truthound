"""Dask backend monitoring adapter.

This module provides monitoring for the Dask distributed backend.
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


class DaskMonitorAdapter(BackendMonitorAdapter):
    """Monitoring adapter for Dask distributed backend.

    Monitors Dask workers and collects cluster metrics.

    Example:
        from dask.distributed import Client

        client = Client()
        adapter = DaskMonitorAdapter(client=client)
        adapter.connect()

        workers = adapter.get_workers()
        metrics = adapter.get_metrics()

        adapter.disconnect()
    """

    def __init__(
        self,
        client: Any = None,
        address: str | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """Initialize Dask adapter.

        Args:
            client: Existing Dask Client
            address: Scheduler address (if no client provided)
            poll_interval_seconds: Polling interval
        """
        super().__init__(poll_interval_seconds)

        self._client = client
        self._address = address
        self._owns_client = client is None

        # Worker tracking
        self._workers: dict[str, BackendWorkerInfo] = {}
        self._worker_health: dict[str, WorkerHealth] = {}

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "dask"

    def connect(self) -> None:
        """Connect to Dask cluster."""
        if self._connected:
            return

        try:
            from dask.distributed import Client
        except ImportError:
            raise RuntimeError("dask[distributed] is required for Dask monitoring")

        if self._owns_client:
            if self._address:
                self._client = Client(self._address)
            else:
                self._client = Client()

        self._refresh_workers()
        self._connected = True

        self._emit_event(MonitorEvent(
            event_type=MonitorEventType.MONITOR_START,
            message=f"Connected to Dask cluster at {self._client.scheduler_info()['address']}",
        ))

    def disconnect(self) -> None:
        """Disconnect from Dask cluster."""
        if not self._connected:
            return

        if self._owns_client and self._client:
            self._client.close()
            self._client = None

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
        if not self._client:
            return BackendMetrics()

        try:
            scheduler_info = self._client.scheduler_info()
            workers_info = scheduler_info.get("workers", {})

            # Aggregate metrics
            total_memory = 0.0
            used_memory = 0.0
            total_tasks = 0
            pending_tasks = 0

            for worker_data in workers_info.values():
                total_memory += worker_data.get("memory_limit", 0) / (1024 * 1024)
                used_memory += worker_data.get("metrics", {}).get("memory", 0) / (1024 * 1024)
                total_tasks += worker_data.get("metrics", {}).get("executing", 0)
                pending_tasks += len(worker_data.get("processing", {}))

            return BackendMetrics(
                timestamp=datetime.now(),
                workers_total=len(workers_info),
                workers_active=sum(
                    1
                    for w in workers_info.values()
                    if w.get("metrics", {}).get("executing", 0) > 0
                ),
                tasks_running=total_tasks,
                tasks_pending=pending_tasks,
                memory_used_mb=used_memory,
                memory_available_mb=total_memory - used_memory,
                metadata={
                    "scheduler_address": scheduler_info.get("address", ""),
                    "scheduler_type": scheduler_info.get("type", ""),
                },
            )
        except Exception as e:
            logger.warning(f"Error getting Dask metrics: {e}")
            return BackendMetrics()

    def get_client(self) -> Any:
        """Get the Dask client.

        Returns:
            Dask Client or None
        """
        return self._client

    def _refresh_workers(self) -> None:
        """Refresh worker information from cluster."""
        if not self._client:
            return

        try:
            scheduler_info = self._client.scheduler_info()
            workers_info = scheduler_info.get("workers", {})

            current_workers = set()

            for worker_addr, worker_data in workers_info.items():
                worker_id = worker_data.get("id", worker_addr)
                current_workers.add(worker_id)

                # Parse address
                host = ""
                port = 0
                try:
                    if "://" in worker_addr:
                        addr_part = worker_addr.split("://")[1]
                        if ":" in addr_part:
                            host, port_str = addr_part.rsplit(":", 1)
                            port = int(port_str)
                        else:
                            host = addr_part
                except Exception:
                    pass

                # Create or update worker info
                self._workers[worker_id] = BackendWorkerInfo(
                    worker_id=worker_id,
                    worker_type="dask_worker",
                    host=host,
                    port=port,
                    cpu_count=worker_data.get("nthreads", 1),
                    memory_mb=worker_data.get("memory_limit", 0) / (1024 * 1024),
                    started_at=datetime.now(),  # Dask doesn't expose start time easily
                    metadata={
                        "address": worker_addr,
                        "local_directory": worker_data.get("local_directory", ""),
                        "services": worker_data.get("services", {}),
                    },
                )

                # Update health
                metrics = worker_data.get("metrics", {})
                memory_used = metrics.get("memory", 0) / (1024 * 1024)
                memory_limit = worker_data.get("memory_limit", 1) / (1024 * 1024)
                memory_percent = (memory_used / memory_limit * 100) if memory_limit > 0 else 0

                # Determine health status
                status = HealthStatus.HEALTHY
                if memory_percent > 95:
                    status = HealthStatus.CRITICAL
                elif memory_percent > 80:
                    status = HealthStatus.DEGRADED

                executing = metrics.get("executing", 0)
                ready = metrics.get("ready", 0)

                self._worker_health[worker_id] = WorkerHealth(
                    worker_id=worker_id,
                    status=status,
                    last_heartbeat=datetime.now(),
                    cpu_percent=metrics.get("cpu", 0),
                    memory_percent=memory_percent,
                    memory_used_mb=memory_used,
                    active_tasks=executing + ready,
                    metadata={
                        "bandwidth": metrics.get("bandwidth", {}),
                        "num_fds": metrics.get("num_fds", 0),
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
                        message=f"Dask worker {worker_id} removed from cluster",
                        worker_id=worker_id,
                    ))

        except Exception as e:
            logger.warning(f"Error refreshing Dask workers: {e}")
