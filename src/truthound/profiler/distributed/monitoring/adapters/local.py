"""Local backend monitoring adapter.

This module provides monitoring for the local (ThreadPoolExecutor) backend.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from truthound.profiler.distributed.monitoring.adapters.base import (
    BackendMetrics,
    BackendMonitorAdapter,
    BackendWorkerInfo,
)
from truthound.profiler.distributed.monitoring.protocols import (
    HealthStatus,
    WorkerHealth,
)


logger = logging.getLogger(__name__)


class LocalMonitorAdapter(BackendMonitorAdapter):
    """Monitoring adapter for local ThreadPoolExecutor backend.

    Monitors thread workers and collects system metrics.

    Example:
        adapter = LocalMonitorAdapter(max_workers=8)
        adapter.connect()

        workers = adapter.get_workers()
        metrics = adapter.get_metrics()

        adapter.disconnect()
    """

    def __init__(
        self,
        max_workers: int | None = None,
        executor: ThreadPoolExecutor | None = None,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """Initialize local adapter.

        Args:
            max_workers: Maximum worker threads
            executor: Existing executor (optional)
            poll_interval_seconds: Polling interval
        """
        super().__init__(poll_interval_seconds)

        self._max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        self._executor = executor
        self._owns_executor = executor is None

        # Worker tracking
        self._workers: dict[str, BackendWorkerInfo] = {}
        self._worker_health: dict[str, WorkerHealth] = {}
        self._active_tasks: dict[str, int] = {}  # worker_id -> active count
        self._lock = threading.RLock()

    @property
    def backend_name(self) -> str:
        """Get backend name."""
        return "local"

    def connect(self) -> None:
        """Connect to backend (start executor if needed)."""
        if self._connected:
            return

        if self._owns_executor:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        # Initialize workers
        for i in range(self._max_workers):
            worker_id = f"local_worker_{i}"
            self._workers[worker_id] = BackendWorkerInfo(
                worker_id=worker_id,
                worker_type="thread",
                host="localhost",
                cpu_count=1,
                memory_mb=self._get_available_memory() / self._max_workers,
                started_at=datetime.now(),
            )
            self._worker_health[worker_id] = WorkerHealth(
                worker_id=worker_id,
                status=HealthStatus.HEALTHY,
                last_heartbeat=datetime.now(),
            )
            self._active_tasks[worker_id] = 0

        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from backend (shutdown executor if owned)."""
        if not self._connected:
            return

        if self._owns_executor and self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._workers.clear()
        self._worker_health.clear()
        self._active_tasks.clear()
        self._connected = False

    def get_workers(self) -> list[BackendWorkerInfo]:
        """Get all workers.

        Returns:
            List of worker info
        """
        with self._lock:
            return list(self._workers.values())

    def get_worker_health(self, worker_id: str) -> WorkerHealth | None:
        """Get worker health.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker health or None
        """
        with self._lock:
            return self._worker_health.get(worker_id)

    def get_metrics(self) -> BackendMetrics:
        """Get backend metrics.

        Returns:
            Backend metrics
        """
        with self._lock:
            total_active = sum(self._active_tasks.values())
            memory_used = self._get_used_memory()
            memory_available = self._get_available_memory()
            cpu_percent = self._get_cpu_percent()

            return BackendMetrics(
                timestamp=datetime.now(),
                workers_total=len(self._workers),
                workers_active=sum(1 for c in self._active_tasks.values() if c > 0),
                tasks_running=total_active,
                tasks_pending=0,  # Local executor doesn't expose queue size
                memory_used_mb=memory_used,
                memory_available_mb=memory_available,
                cpu_utilization_percent=cpu_percent,
                metadata={
                    "max_workers": self._max_workers,
                    "executor_type": "ThreadPoolExecutor",
                },
            )

    def get_executor(self) -> ThreadPoolExecutor | None:
        """Get the thread pool executor.

        Returns:
            ThreadPoolExecutor or None
        """
        return self._executor

    def record_task_start(self, worker_id: str) -> None:
        """Record task start for a worker.

        Args:
            worker_id: Worker identifier
        """
        with self._lock:
            if worker_id in self._active_tasks:
                self._active_tasks[worker_id] += 1

            health = self._worker_health.get(worker_id)
            if health:
                health.active_tasks = self._active_tasks.get(worker_id, 0)
                health.last_heartbeat = datetime.now()

    def record_task_complete(
        self,
        worker_id: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record task completion for a worker.

        Args:
            worker_id: Worker identifier
            duration_seconds: Task duration
            success: Whether task succeeded
        """
        with self._lock:
            if worker_id in self._active_tasks:
                self._active_tasks[worker_id] = max(0, self._active_tasks[worker_id] - 1)

            health = self._worker_health.get(worker_id)
            if health:
                health.active_tasks = self._active_tasks.get(worker_id, 0)
                health.last_heartbeat = datetime.now()
                if success:
                    health.completed_tasks += 1
                else:
                    health.failed_tasks += 1

    def _get_available_memory(self) -> float:
        """Get available system memory in MB."""
        try:
            import psutil

            return psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            return 0.0

    def _get_used_memory(self) -> float:
        """Get used system memory in MB."""
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _get_cpu_percent(self) -> float:
        """Get CPU utilization percent."""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0
