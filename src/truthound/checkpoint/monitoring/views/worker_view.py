"""Worker status view.

Renders worker metrics for display in various formats.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
)
from truthound.checkpoint.monitoring.views.base import BaseView


class WorkerStatusView(BaseView):
    """View for rendering worker status.

    Example:
        >>> view = WorkerStatusView()
        >>>
        >>> # Render single worker
        >>> output = view.render(worker_metrics)
        >>>
        >>> # Render for CLI
        >>> print(view.format_for_cli(worker_metrics))
        Worker: worker-1
        ├── Status: online [+]
        ├── Hostname: localhost
        ├── Tasks: 3/4 (75%)
        ├── Completed: 1,234
        └── Uptime: 2.5h
    """

    def __init__(self, name: str = "worker_status") -> None:
        """Initialize worker status view."""
        super().__init__(name=name)

    def render(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> dict[str, Any]:
        """Render worker metrics as dictionary."""
        if not isinstance(metrics, WorkerMetrics):
            return {"error": "Expected WorkerMetrics"}

        return {
            "worker_id": metrics.worker_id,
            "hostname": metrics.hostname,
            "state": metrics.state,
            "is_available": metrics.is_available,
            "capacity": {
                "current_tasks": metrics.current_tasks,
                "max_concurrency": metrics.max_concurrency,
                "available_slots": metrics.available_slots,
                "load_factor": round(metrics.load_factor, 3),
            },
            "performance": {
                "completed_tasks": metrics.completed_tasks,
                "failed_tasks": metrics.failed_tasks,
                "success_rate": round(
                    metrics.completed_tasks / max(metrics.completed_tasks + metrics.failed_tasks, 1),
                    4,
                ),
            },
            "resources": {
                "cpu_percent": round(metrics.cpu_percent, 1),
                "memory_mb": round(metrics.memory_mb, 1),
            },
            "uptime_seconds": round(metrics.uptime_seconds, 0),
            "last_heartbeat": metrics.last_heartbeat.isoformat(),
            "tags": list(metrics.tags),
            "metadata": metrics.metadata,
        }

    def render_summary(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
    ) -> dict[str, Any]:
        """Render summary of all workers."""
        if not worker_metrics:
            return {
                "workers": [],
                "totals": {
                    "worker_count": 0,
                    "online": 0,
                    "offline": 0,
                    "total_capacity": 0,
                    "current_load": 0,
                },
                "health": {
                    "status": "no_data",
                    "avg_load_factor": 0,
                },
            }

        workers = [self.render(wm) for wm in worker_metrics]

        online_count = sum(1 for wm in worker_metrics if wm.state == "online")
        offline_count = len(worker_metrics) - online_count
        total_capacity = sum(wm.max_concurrency for wm in worker_metrics if wm.state == "online")
        current_load = sum(wm.current_tasks for wm in worker_metrics)
        avg_load = sum(wm.load_factor for wm in worker_metrics) / len(worker_metrics)
        avg_cpu = sum(wm.cpu_percent for wm in worker_metrics) / len(worker_metrics)
        avg_memory = sum(wm.memory_mb for wm in worker_metrics) / len(worker_metrics)

        # Determine health status
        if online_count == 0:
            status = "critical"
        elif offline_count > 0:
            status = "degraded"
        elif avg_load > 0.9:
            status = "warning"
        else:
            status = "healthy"

        return {
            "workers": workers,
            "totals": {
                "worker_count": len(worker_metrics),
                "online": online_count,
                "offline": offline_count,
                "total_capacity": total_capacity,
                "current_load": current_load,
                "available_capacity": total_capacity - current_load,
                "total_completed": sum(wm.completed_tasks for wm in worker_metrics),
                "total_failed": sum(wm.failed_tasks for wm in worker_metrics),
            },
            "health": {
                "status": status,
                "avg_load_factor": round(avg_load, 3),
                "avg_cpu_percent": round(avg_cpu, 1),
                "avg_memory_mb": round(avg_memory, 1),
            },
            "by_state": self._group_by_state(worker_metrics),
        }

    def _group_by_state(self, metrics: list[WorkerMetrics]) -> dict[str, int]:
        """Group workers by state."""
        counts: dict[str, int] = {}
        for m in metrics:
            counts[m.state] = counts.get(m.state, 0) + 1
        return counts

    def format_for_cli(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> str:
        """Format worker metrics for CLI display."""
        if not isinstance(metrics, WorkerMetrics):
            return "Error: Expected WorkerMetrics"

        status_icon = self._status_emoji(metrics.state)
        load_pct = f"{metrics.load_factor * 100:.0f}%"

        total_tasks = metrics.completed_tasks + metrics.failed_tasks
        success_rate = metrics.completed_tasks / max(total_tasks, 1)

        lines = [
            f"Worker: {metrics.worker_id}",
            f"├── Status: {metrics.state} {status_icon}",
            f"├── Hostname: {metrics.hostname}",
            f"├── Tasks: {metrics.current_tasks}/{metrics.max_concurrency} ({load_pct})",
            f"├── Completed: {self._format_count(metrics.completed_tasks)}",
            f"├── Failed: {self._format_count(metrics.failed_tasks)}",
            f"├── Success Rate: {self._format_percentage(success_rate)}",
            f"├── CPU: {metrics.cpu_percent:.1f}%",
            f"├── Memory: {metrics.memory_mb:.0f}MB",
            f"├── Uptime: {self._format_duration(metrics.uptime_seconds * 1000)}",
            f"└── Last Heartbeat: {self._format_time_ago(metrics.last_heartbeat)}",
        ]

        return "\n".join(lines)

    def _format_time_ago(self, dt: datetime) -> str:
        """Format datetime as 'X ago'."""
        delta = datetime.now() - dt
        seconds = delta.total_seconds()

        if seconds < 60:
            return f"{int(seconds)}s ago"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        else:
            return f"{int(seconds / 86400)}d ago"

    def format_summary_for_cli(
        self,
        worker_metrics: list[WorkerMetrics],
    ) -> str:
        """Format worker summary for CLI display."""
        if not worker_metrics:
            return "No workers found."

        lines = ["Worker Summary", "=" * 60]

        # Add each worker
        for wm in worker_metrics:
            icon = self._status_emoji(wm.state)
            load = f"{wm.load_factor * 100:.0f}%"

            line = (
                f"{icon} {wm.worker_id:20s} "
                f"{wm.hostname:15s} "
                f"Load:{load:4s} "
                f"C:{wm.completed_tasks:6d} "
                f"F:{wm.failed_tasks:4d} "
                f"CPU:{wm.cpu_percent:5.1f}%"
            )
            lines.append(line)

        # Add totals
        lines.append("-" * 60)

        online = sum(1 for wm in worker_metrics if wm.state == "online")
        total_capacity = sum(wm.max_concurrency for wm in worker_metrics if wm.state == "online")
        current_load = sum(wm.current_tasks for wm in worker_metrics)
        avg_load = sum(wm.load_factor for wm in worker_metrics) / len(worker_metrics)

        lines.append(
            f"Workers: {online}/{len(worker_metrics)} online | "
            f"Capacity: {current_load}/{total_capacity} | "
            f"Avg Load: {avg_load * 100:.1f}%"
        )

        return "\n".join(lines)

    def format_for_table(
        self,
        worker_metrics: list[WorkerMetrics],
    ) -> list[list[str]]:
        """Format worker metrics as table rows.

        Returns:
            List of rows, first row is headers.
        """
        headers = [
            "Worker", "Host", "State", "Tasks",
            "Load", "Completed", "Failed", "CPU", "Memory"
        ]

        rows = [headers]
        for wm in worker_metrics:
            row = [
                wm.worker_id,
                wm.hostname,
                wm.state,
                f"{wm.current_tasks}/{wm.max_concurrency}",
                f"{wm.load_factor * 100:.0f}%",
                str(wm.completed_tasks),
                str(wm.failed_tasks),
                f"{wm.cpu_percent:.1f}%",
                f"{wm.memory_mb:.0f}MB",
            ]
            rows.append(row)

        return rows
