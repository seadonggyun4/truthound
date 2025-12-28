"""Base view implementation.

Provides abstract base class for monitoring views with common formatting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from truthound.checkpoint.monitoring.protocols import (
    MonitoringViewProtocol,
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
)


class BaseView(ABC):
    """Abstract base class for monitoring views.

    Provides common formatting functionality and enforces
    the view protocol interface.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize view.

        Args:
            name: View name.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Get view name."""
        return self._name

    @abstractmethod
    def render(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> dict[str, Any]:
        """Render metrics as a dictionary.

        Args:
            metrics: Metrics to render.

        Returns:
            Rendered view as dictionary.
        """
        pass

    @abstractmethod
    def render_summary(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
    ) -> dict[str, Any]:
        """Render a summary view.

        Args:
            queue_metrics: Queue metrics.
            worker_metrics: Worker metrics.

        Returns:
            Summary view as dictionary.
        """
        pass

    def format_for_api(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> dict[str, Any]:
        """Format metrics for REST API response.

        Default implementation uses render() and adds API metadata.

        Args:
            metrics: Metrics to format.

        Returns:
            API-formatted dictionary.
        """
        rendered = self.render(metrics)
        return {
            "data": rendered,
            "type": self._get_metrics_type(metrics),
            "view": self.name,
            "generated_at": datetime.now().isoformat(),
        }

    @abstractmethod
    def format_for_cli(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> str:
        """Format metrics for CLI display.

        Args:
            metrics: Metrics to format.

        Returns:
            CLI-formatted string.
        """
        pass

    def _get_metrics_type(
        self,
        metrics: QueueMetrics | WorkerMetrics | TaskMetrics,
    ) -> str:
        """Get the type name of the metrics."""
        if isinstance(metrics, QueueMetrics):
            return "queue"
        elif isinstance(metrics, WorkerMetrics):
            return "worker"
        elif isinstance(metrics, TaskMetrics):
            return "task"
        return "unknown"

    def _format_duration(self, milliseconds: float) -> str:
        """Format duration in human-readable form."""
        if milliseconds < 1000:
            return f"{milliseconds:.0f}ms"
        elif milliseconds < 60000:
            return f"{milliseconds / 1000:.1f}s"
        elif milliseconds < 3600000:
            return f"{milliseconds / 60000:.1f}m"
        else:
            return f"{milliseconds / 3600000:.1f}h"

    def _format_count(self, count: int) -> str:
        """Format large counts in human-readable form."""
        if count < 1000:
            return str(count)
        elif count < 1000000:
            return f"{count / 1000:.1f}K"
        elif count < 1000000000:
            return f"{count / 1000000:.1f}M"
        else:
            return f"{count / 1000000000:.1f}B"

    def _format_rate(self, rate: float) -> str:
        """Format rate in human-readable form."""
        if rate < 1:
            return f"{rate * 60:.1f}/min"
        elif rate < 100:
            return f"{rate:.1f}/s"
        else:
            return f"{rate:.0f}/s"

    def _format_percentage(self, value: float) -> str:
        """Format percentage value."""
        return f"{value * 100:.1f}%"

    def _status_emoji(self, status: str) -> str:
        """Get emoji for status (for CLI display)."""
        emoji_map = {
            "online": "[+]",
            "offline": "[-]",
            "busy": "[~]",
            "draining": "[!]",
            "pending": "[ ]",
            "running": "[>]",
            "succeeded": "[v]",
            "failed": "[x]",
            "healthy": "[v]",
            "degraded": "[!]",
            "critical": "[x]",
        }
        return emoji_map.get(status.lower(), "[?]")
