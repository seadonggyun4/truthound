"""Queue status view.

Renders queue metrics for display in various formats.
"""

from __future__ import annotations

from typing import Any

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
)
from truthound.checkpoint.monitoring.views.base import BaseView


class QueueStatusView(BaseView):
    """View for rendering queue status.

    Example:
        >>> view = QueueStatusView()
        >>>
        >>> # Render single queue
        >>> output = view.render(queue_metrics)
        >>>
        >>> # Render for CLI
        >>> print(view.format_for_cli(queue_metrics))
        Queue: default
        ├── Status: healthy [v]
        ├── Pending: 5
        ├── Running: 3
        ├── Completed: 1,234
        └── Throughput: 2.5/s
    """

    def __init__(self, name: str = "queue_status") -> None:
        """Initialize queue status view."""
        super().__init__(name=name)

    def render(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> dict[str, Any]:
        """Render queue metrics as dictionary."""
        if not isinstance(metrics, QueueMetrics):
            return {"error": "Expected QueueMetrics"}

        return {
            "queue_name": metrics.queue_name,
            "status": "healthy" if metrics.is_healthy else "degraded",
            "counts": {
                "pending": metrics.pending_count,
                "running": metrics.running_count,
                "completed": metrics.completed_count,
                "failed": metrics.failed_count,
                "total": metrics.total_tasks,
            },
            "performance": {
                "avg_wait_time_ms": round(metrics.avg_wait_time_ms, 2),
                "avg_execution_time_ms": round(metrics.avg_execution_time_ms, 2),
                "throughput_per_second": round(metrics.throughput_per_second, 3),
                "success_rate": round(metrics.success_rate, 4),
            },
            "timestamp": metrics.timestamp.isoformat(),
            "labels": metrics.labels,
        }

    def render_summary(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
    ) -> dict[str, Any]:
        """Render summary of all queues."""
        if not queue_metrics:
            return {
                "queues": [],
                "totals": {
                    "queue_count": 0,
                    "pending": 0,
                    "running": 0,
                    "completed": 0,
                    "failed": 0,
                },
                "health": {
                    "healthy": 0,
                    "degraded": 0,
                    "status": "no_data",
                },
            }

        queues = [self.render(qm) for qm in queue_metrics]
        healthy_count = sum(1 for qm in queue_metrics if qm.is_healthy)
        degraded_count = len(queue_metrics) - healthy_count

        # Determine overall health status
        if degraded_count == 0:
            overall_status = "healthy"
        elif degraded_count < len(queue_metrics) / 2:
            overall_status = "degraded"
        else:
            overall_status = "critical"

        return {
            "queues": queues,
            "totals": {
                "queue_count": len(queue_metrics),
                "pending": sum(qm.pending_count for qm in queue_metrics),
                "running": sum(qm.running_count for qm in queue_metrics),
                "completed": sum(qm.completed_count for qm in queue_metrics),
                "failed": sum(qm.failed_count for qm in queue_metrics),
                "throughput": sum(qm.throughput_per_second for qm in queue_metrics),
            },
            "health": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "status": overall_status,
            },
        }

    def format_for_cli(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> str:
        """Format queue metrics for CLI display."""
        if not isinstance(metrics, QueueMetrics):
            return "Error: Expected QueueMetrics"

        status = "healthy" if metrics.is_healthy else "degraded"
        status_icon = self._status_emoji(status)

        lines = [
            f"Queue: {metrics.queue_name}",
            f"├── Status: {status} {status_icon}",
            f"├── Pending: {self._format_count(metrics.pending_count)}",
            f"├── Running: {self._format_count(metrics.running_count)}",
            f"├── Completed: {self._format_count(metrics.completed_count)}",
            f"├── Failed: {self._format_count(metrics.failed_count)}",
            f"├── Success Rate: {self._format_percentage(metrics.success_rate)}",
            f"├── Avg Wait: {self._format_duration(metrics.avg_wait_time_ms)}",
            f"├── Avg Exec: {self._format_duration(metrics.avg_execution_time_ms)}",
            f"└── Throughput: {self._format_rate(metrics.throughput_per_second)}",
        ]

        return "\n".join(lines)

    def format_summary_for_cli(
        self,
        queue_metrics: list[QueueMetrics],
    ) -> str:
        """Format queue summary for CLI display."""
        if not queue_metrics:
            return "No queues found."

        lines = ["Queue Summary", "=" * 50]

        # Add each queue
        for i, qm in enumerate(queue_metrics):
            status = "healthy" if qm.is_healthy else "degraded"
            icon = self._status_emoji(status)

            line = (
                f"{icon} {qm.queue_name:20s} "
                f"P:{qm.pending_count:4d} "
                f"R:{qm.running_count:4d} "
                f"C:{qm.completed_count:6d} "
                f"F:{qm.failed_count:4d} "
                f"({self._format_percentage(qm.success_rate)})"
            )
            lines.append(line)

        # Add totals
        lines.append("-" * 50)
        total_pending = sum(qm.pending_count for qm in queue_metrics)
        total_running = sum(qm.running_count for qm in queue_metrics)
        total_completed = sum(qm.completed_count for qm in queue_metrics)
        total_failed = sum(qm.failed_count for qm in queue_metrics)
        total_throughput = sum(qm.throughput_per_second for qm in queue_metrics)

        lines.append(
            f"{'Total':25s} "
            f"P:{total_pending:4d} "
            f"R:{total_running:4d} "
            f"C:{total_completed:6d} "
            f"F:{total_failed:4d}"
        )
        lines.append(f"Total Throughput: {self._format_rate(total_throughput)}")

        return "\n".join(lines)

    def format_for_table(
        self,
        queue_metrics: list[QueueMetrics],
    ) -> list[list[str]]:
        """Format queue metrics as table rows.

        Returns:
            List of rows, first row is headers.
        """
        headers = [
            "Queue", "Status", "Pending", "Running",
            "Completed", "Failed", "Success %", "Throughput"
        ]

        rows = [headers]
        for qm in queue_metrics:
            status = "healthy" if qm.is_healthy else "degraded"
            row = [
                qm.queue_name,
                status,
                str(qm.pending_count),
                str(qm.running_count),
                str(qm.completed_count),
                str(qm.failed_count),
                f"{qm.success_rate * 100:.1f}%",
                self._format_rate(qm.throughput_per_second),
            ]
            rows.append(row)

        return rows
