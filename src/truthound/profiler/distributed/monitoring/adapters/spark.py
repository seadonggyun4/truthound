"""Spark backend monitoring adapter.

This module provides monitoring for the Apache Spark backend.
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


class SparkMonitorAdapter(BackendMonitorAdapter):
    """Monitoring adapter for Apache Spark backend.

    Monitors Spark executors and collects cluster metrics.

    Example:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        adapter = SparkMonitorAdapter(spark_session=spark)
        adapter.connect()

        workers = adapter.get_workers()
        metrics = adapter.get_metrics()

        adapter.disconnect()
    """

    def __init__(
        self,
        spark_session: Any = None,
        app_name: str = "truthound",
        master: str = "local[*]",
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """Initialize Spark adapter.

        Args:
            spark_session: Existing SparkSession
            app_name: Application name (if creating session)
            master: Master URL (if creating session)
            poll_interval_seconds: Polling interval
        """
        super().__init__(poll_interval_seconds)

        self._spark = spark_session
        self._app_name = app_name
        self._master = master
        self._owns_session = spark_session is None

        # Worker tracking
        self._workers: dict[str, BackendWorkerInfo] = {}
        self._worker_health: dict[str, WorkerHealth] = {}

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "spark"

    def connect(self) -> None:
        """Connect to Spark cluster."""
        if self._connected:
            return

        try:
            from pyspark.sql import SparkSession
        except ImportError:
            raise RuntimeError("pyspark is required for Spark monitoring")

        if self._owns_session:
            self._spark = (
                SparkSession.builder.appName(self._app_name)
                .master(self._master)
                .getOrCreate()
            )

        self._refresh_workers()
        self._connected = True

        app_id = self._spark.sparkContext.applicationId
        self._emit_event(MonitorEvent(
            event_type=MonitorEventType.MONITOR_START,
            message=f"Connected to Spark cluster (app: {app_id})",
            metadata={"app_id": app_id, "master": self._master},
        ))

    def disconnect(self) -> None:
        """Disconnect from Spark cluster."""
        if not self._connected:
            return

        if self._owns_session and self._spark:
            self._spark.stop()
            self._spark = None

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
        if not self._spark:
            return BackendMetrics()

        try:
            sc = self._spark.sparkContext

            # Get executor info via REST API or status tracker
            status_tracker = sc.statusTracker()
            executor_ids = status_tracker.getExecutorInfos()

            # Aggregate metrics
            total_memory = 0.0
            total_cores = 0
            active_tasks = 0

            for exec_info in executor_ids:
                total_memory += exec_info.totalOnHeapStorageMemory() / (1024 * 1024)
                total_cores += exec_info.totalCores()

            # Get active stages
            active_stages = status_tracker.getActiveStageIds()
            for stage_id in active_stages:
                stage_info = status_tracker.getStageInfo(stage_id)
                if stage_info:
                    active_tasks += stage_info.numActiveTasks()

            return BackendMetrics(
                timestamp=datetime.now(),
                workers_total=len(executor_ids),
                workers_active=len([e for e in executor_ids if e.totalCores() > 0]),
                tasks_running=active_tasks,
                tasks_pending=0,  # Would need to query scheduler
                memory_used_mb=0.0,  # Not easily available
                memory_available_mb=total_memory,
                metadata={
                    "app_id": sc.applicationId,
                    "app_name": sc.appName,
                    "master": sc.master,
                    "total_cores": total_cores,
                    "active_stages": len(active_stages),
                },
            )
        except Exception as e:
            logger.warning(f"Error getting Spark metrics: {e}")
            return self._get_basic_metrics()

    def get_spark_session(self) -> Any:
        """Get the SparkSession.

        Returns:
            SparkSession or None
        """
        return self._spark

    def _refresh_workers(self) -> None:
        """Refresh worker information from cluster."""
        if not self._spark:
            return

        try:
            sc = self._spark.sparkContext
            status_tracker = sc.statusTracker()

            executor_infos = status_tracker.getExecutorInfos()
            current_workers = set()

            for exec_info in executor_infos:
                executor_id = exec_info.executorId()
                current_workers.add(executor_id)

                # Parse host from executor address
                host = exec_info.hostPort()
                port = 0
                if ":" in host:
                    host, port_str = host.rsplit(":", 1)
                    try:
                        port = int(port_str)
                    except ValueError:
                        pass

                self._workers[executor_id] = BackendWorkerInfo(
                    worker_id=executor_id,
                    worker_type="spark_executor",
                    host=host,
                    port=port,
                    cpu_count=exec_info.totalCores(),
                    memory_mb=exec_info.totalOnHeapStorageMemory() / (1024 * 1024),
                    started_at=None,  # Not available from StatusTracker
                    metadata={
                        "used_memory_mb": exec_info.usedOnHeapStorageMemory() / (1024 * 1024),
                        "used_off_heap_mb": exec_info.usedOffHeapStorageMemory() / (1024 * 1024),
                    },
                )

                # Calculate health
                total_mem = exec_info.totalOnHeapStorageMemory()
                used_mem = exec_info.usedOnHeapStorageMemory()
                memory_percent = (used_mem / total_mem * 100) if total_mem > 0 else 0

                status = HealthStatus.HEALTHY
                if memory_percent > 95:
                    status = HealthStatus.CRITICAL
                elif memory_percent > 80:
                    status = HealthStatus.DEGRADED

                self._worker_health[executor_id] = WorkerHealth(
                    worker_id=executor_id,
                    status=status,
                    last_heartbeat=datetime.now(),
                    memory_percent=memory_percent,
                    memory_used_mb=used_mem / (1024 * 1024),
                    active_tasks=0,  # Not available per executor
                )

            # Remove workers that are no longer in cluster
            for worker_id in list(self._workers.keys()):
                if worker_id not in current_workers:
                    del self._workers[worker_id]
                    self._worker_health.pop(worker_id, None)

                    self._emit_event(MonitorEvent(
                        event_type=MonitorEventType.WORKER_UNREGISTERED,
                        severity=EventSeverity.WARNING,
                        message=f"Spark executor {worker_id} removed from cluster",
                        worker_id=worker_id,
                    ))

        except Exception as e:
            logger.warning(f"Error refreshing Spark workers: {e}")

    def _get_basic_metrics(self) -> BackendMetrics:
        """Get basic metrics when detailed info is not available."""
        if not self._spark:
            return BackendMetrics()

        sc = self._spark.sparkContext
        return BackendMetrics(
            timestamp=datetime.now(),
            workers_total=sc.defaultParallelism,
            workers_active=sc.defaultParallelism,
            metadata={
                "app_id": sc.applicationId,
                "app_name": sc.appName,
                "master": sc.master,
                "default_parallelism": sc.defaultParallelism,
            },
        )
