"""Health monitor for distributed workers.

This module provides health monitoring for distributed profiling workers.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any, Callable

from truthound.profiler.distributed.monitoring.config import HealthCheckConfig
from truthound.profiler.distributed.monitoring.protocols import (
    EventSeverity,
    HealthStatus,
    IHealthMonitor,
    MonitorEvent,
    MonitorEventType,
    WorkerHealth,
)


logger = logging.getLogger(__name__)


class WorkerHealthMonitor(IHealthMonitor):
    """Monitors health and availability of distributed workers.

    Thread-safe implementation that tracks worker heartbeats, resource usage,
    and task performance to determine worker health status.

    Example:
        monitor = WorkerHealthMonitor(
            config=HealthCheckConfig(heartbeat_timeout_seconds=30),
            on_event=lambda e: print(e.message),
        )

        monitor.register_worker("worker-1")
        monitor.record_heartbeat("worker-1", cpu_percent=50, memory_percent=60)
        monitor.record_task_complete("worker-1", duration_seconds=2.5, success=True)

        health = monitor.get_worker_health("worker-1")
        print(f"Worker status: {health.status}")
    """

    def __init__(
        self,
        config: HealthCheckConfig | None = None,
        on_event: Callable[[MonitorEvent], None] | None = None,
    ) -> None:
        """Initialize health monitor.

        Args:
            config: Health check configuration
            on_event: Callback for health events
        """
        self._config = config or HealthCheckConfig()
        self._on_event = on_event

        # Worker state
        self._workers: dict[str, WorkerHealth] = {}
        self._lock = threading.RLock()

        # Task duration history per worker (for averages)
        self._task_durations: dict[str, deque[float]] = {}
        self._max_duration_history = 100

        # Previous status for change detection
        self._previous_status: dict[str, HealthStatus] = {}

    def register_worker(
        self,
        worker_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a worker.

        Args:
            worker_id: Worker identifier
            metadata: Worker metadata
        """
        with self._lock:
            if worker_id in self._workers:
                logger.warning(f"Worker {worker_id} already registered")
                return

            self._workers[worker_id] = WorkerHealth(
                worker_id=worker_id,
                status=HealthStatus.UNKNOWN,
                last_heartbeat=datetime.now(),
                metadata=metadata or {},
            )
            self._task_durations[worker_id] = deque(maxlen=self._max_duration_history)
            self._previous_status[worker_id] = HealthStatus.UNKNOWN

            self._emit_event(
                MonitorEventType.WORKER_REGISTERED,
                f"Worker {worker_id} registered",
                worker_id=worker_id,
            )

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker.

        Args:
            worker_id: Worker identifier
        """
        with self._lock:
            if worker_id not in self._workers:
                logger.warning(f"Worker {worker_id} not registered")
                return

            del self._workers[worker_id]
            self._task_durations.pop(worker_id, None)
            self._previous_status.pop(worker_id, None)

            self._emit_event(
                MonitorEventType.WORKER_UNREGISTERED,
                f"Worker {worker_id} unregistered",
                worker_id=worker_id,
            )

    def record_heartbeat(
        self,
        worker_id: str,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        memory_used_mb: float = 0.0,
        active_tasks: int = 0,
    ) -> None:
        """Record worker heartbeat.

        Args:
            worker_id: Worker identifier
            cpu_percent: CPU usage
            memory_percent: Memory usage percent
            memory_used_mb: Memory used in MB
            active_tasks: Active task count
        """
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                # Auto-register unknown workers
                self.register_worker(worker_id)
                worker = self._workers[worker_id]

            worker.last_heartbeat = datetime.now()
            worker.cpu_percent = cpu_percent
            worker.memory_percent = memory_percent
            worker.memory_used_mb = memory_used_mb
            worker.active_tasks = active_tasks

            # Update health status
            self._update_worker_status(worker)

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
            worker = self._workers.get(worker_id)
            if worker is None:
                logger.warning(f"Task complete for unknown worker: {worker_id}")
                return

            # Update task counts
            if success:
                worker.completed_tasks += 1
            else:
                worker.failed_tasks += 1

            # Record duration for average calculation
            durations = self._task_durations.get(worker_id)
            if durations is not None:
                durations.append(duration_seconds)
                worker.avg_task_time_seconds = sum(durations) / len(durations)

            # Update error rate
            total = worker.completed_tasks + worker.failed_tasks
            if total > 0:
                worker.error_rate = worker.failed_tasks / total

            # Update health status
            self._update_worker_status(worker)

    def get_worker_health(self, worker_id: str) -> WorkerHealth | None:
        """Get worker health.

        Args:
            worker_id: Worker identifier

        Returns:
            Worker health or None if not found
        """
        with self._lock:
            return self._workers.get(worker_id)

    def get_all_workers(self) -> list[WorkerHealth]:
        """Get all worker health.

        Returns:
            List of all worker health
        """
        with self._lock:
            return list(self._workers.values())

    def get_healthy_workers(self) -> list[WorkerHealth]:
        """Get healthy workers.

        Returns:
            List of healthy worker health
        """
        with self._lock:
            return [w for w in self._workers.values() if w.status == HealthStatus.HEALTHY]

    def get_unhealthy_workers(self) -> list[WorkerHealth]:
        """Get unhealthy workers.

        Returns:
            List of unhealthy worker health
        """
        with self._lock:
            return [
                w
                for w in self._workers.values()
                if w.status in {HealthStatus.UNHEALTHY, HealthStatus.CRITICAL}
            ]

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health.

        Returns:
            Overall health status
        """
        with self._lock:
            if not self._workers:
                return HealthStatus.UNKNOWN

            total = len(self._workers)
            healthy = sum(1 for w in self._workers.values() if w.status == HealthStatus.HEALTHY)
            critical = sum(1 for w in self._workers.values() if w.status == HealthStatus.CRITICAL)

            # If any critical workers, system is critical
            if critical > 0:
                return HealthStatus.CRITICAL

            # Check minimum healthy workers threshold
            healthy_percent = (healthy / total) * 100
            if healthy_percent < self._config.min_healthy_workers_percent:
                if healthy_percent < self._config.min_healthy_workers_percent / 2:
                    return HealthStatus.CRITICAL
                return HealthStatus.UNHEALTHY

            # If some degraded, system is degraded
            if healthy < total:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

    def check_stalled_workers(self) -> list[str]:
        """Check for stalled workers that haven't sent heartbeats.

        Returns:
            List of stalled worker IDs
        """
        with self._lock:
            stalled = []
            now = datetime.now()

            for worker in self._workers.values():
                if worker.last_heartbeat is None:
                    continue

                age = (now - worker.last_heartbeat).total_seconds()

                if age > self._config.stall_threshold_seconds:
                    if worker.status != HealthStatus.CRITICAL:
                        stalled.append(worker.worker_id)
                        worker.status = HealthStatus.CRITICAL
                        self._emit_status_change(worker, HealthStatus.CRITICAL)

            return stalled

    def get_statistics(self) -> dict[str, Any]:
        """Get health monitor statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            workers = list(self._workers.values())

            return {
                "total_workers": len(workers),
                "healthy_workers": sum(1 for w in workers if w.status == HealthStatus.HEALTHY),
                "degraded_workers": sum(1 for w in workers if w.status == HealthStatus.DEGRADED),
                "unhealthy_workers": sum(1 for w in workers if w.status == HealthStatus.UNHEALTHY),
                "critical_workers": sum(1 for w in workers if w.status == HealthStatus.CRITICAL),
                "overall_health": self.get_overall_health().value,
                "total_completed_tasks": sum(w.completed_tasks for w in workers),
                "total_failed_tasks": sum(w.failed_tasks for w in workers),
                "avg_cpu_percent": sum(w.cpu_percent for w in workers) / len(workers) if workers else 0,
                "avg_memory_percent": sum(w.memory_percent for w in workers) / len(workers) if workers else 0,
            }

    def reset(self) -> None:
        """Reset health monitor state."""
        with self._lock:
            self._workers.clear()
            self._task_durations.clear()
            self._previous_status.clear()

    def _update_worker_status(self, worker: WorkerHealth) -> None:
        """Update worker health status based on current metrics.

        Args:
            worker: Worker health to update
        """
        cfg = self._config
        now = datetime.now()

        # Check heartbeat age
        heartbeat_age = 0.0
        if worker.last_heartbeat:
            heartbeat_age = (now - worker.last_heartbeat).total_seconds()

        # Determine status
        new_status = HealthStatus.HEALTHY

        # Critical conditions
        if heartbeat_age > cfg.heartbeat_timeout_seconds:
            new_status = HealthStatus.UNHEALTHY
        elif heartbeat_age > cfg.stall_threshold_seconds:
            new_status = HealthStatus.CRITICAL
        elif worker.memory_percent >= cfg.memory_critical_percent:
            new_status = HealthStatus.CRITICAL
        elif worker.error_rate >= cfg.error_rate_critical:
            new_status = HealthStatus.CRITICAL

        # Warning conditions (only if not already critical/unhealthy)
        elif worker.memory_percent >= cfg.memory_warning_percent:
            new_status = HealthStatus.DEGRADED
        elif worker.cpu_percent >= cfg.cpu_warning_percent:
            new_status = HealthStatus.DEGRADED
        elif worker.error_rate >= cfg.error_rate_warning:
            new_status = HealthStatus.DEGRADED

        # Update status and emit event if changed
        if new_status != worker.status:
            old_status = worker.status
            worker.status = new_status
            self._emit_status_change(worker, old_status)

    def _emit_status_change(self, worker: WorkerHealth, old_status: HealthStatus) -> None:
        """Emit event for worker status change.

        Args:
            worker: Worker health
            old_status: Previous status
        """
        new_status = worker.status
        previous = self._previous_status.get(worker.worker_id)

        # Only emit if actually changed from last reported status
        if new_status == previous:
            return

        self._previous_status[worker.worker_id] = new_status

        # Determine event type and severity
        if new_status == HealthStatus.HEALTHY:
            event_type = MonitorEventType.WORKER_HEALTHY
            severity = EventSeverity.INFO
            if old_status in {HealthStatus.UNHEALTHY, HealthStatus.CRITICAL}:
                event_type = MonitorEventType.WORKER_RECOVERED
        elif new_status == HealthStatus.DEGRADED:
            event_type = MonitorEventType.WORKER_UNHEALTHY
            severity = EventSeverity.WARNING
        elif new_status == HealthStatus.UNHEALTHY:
            event_type = MonitorEventType.WORKER_UNHEALTHY
            severity = EventSeverity.ERROR
        elif new_status == HealthStatus.CRITICAL:
            event_type = MonitorEventType.WORKER_STALLED
            severity = EventSeverity.CRITICAL
        else:
            return

        self._emit_event(
            event_type,
            f"Worker {worker.worker_id} status changed: {old_status.value} -> {new_status.value}",
            worker_id=worker.worker_id,
            severity=severity,
            metadata={
                "old_status": old_status.value,
                "new_status": new_status.value,
                "cpu_percent": worker.cpu_percent,
                "memory_percent": worker.memory_percent,
                "error_rate": worker.error_rate,
            },
        )

    def _emit_event(
        self,
        event_type: MonitorEventType,
        message: str,
        worker_id: str | None = None,
        severity: EventSeverity = EventSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a monitoring event.

        Args:
            event_type: Event type
            message: Event message
            worker_id: Worker identifier
            severity: Event severity
            metadata: Additional metadata
        """
        if self._on_event is None:
            return

        event = MonitorEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            worker_id=worker_id,
            metadata=metadata or {},
        )

        try:
            self._on_event(event)
        except Exception as e:
            logger.warning(f"Error in health monitor event callback: {e}")
