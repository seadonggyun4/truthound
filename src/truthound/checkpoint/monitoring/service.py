"""Monitoring service facade.

Provides a unified interface for the monitoring system,
coordinating collectors, aggregators, and views.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator

from truthound.checkpoint.monitoring.protocols import (
    MetricCollectorProtocol,
    MetricAggregatorProtocol,
    MonitoringViewProtocol,
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
    MonitoringEvent,
    MonitoringEventType,
    AggregatedMetrics,
    MonitoringError,
)
from truthound.checkpoint.monitoring.events import EventBus, get_event_bus
from truthound.checkpoint.monitoring.collectors import (
    InMemoryCollector,
    RedisCollector,
    PrometheusCollector,
)
from truthound.checkpoint.monitoring.aggregators import (
    RealtimeAggregator,
    SlidingWindowAggregator,
)
from truthound.checkpoint.monitoring.views import (
    QueueStatusView,
    WorkerStatusView,
    TaskDetailView,
)

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring service.

    Attributes:
        enabled: Whether monitoring is enabled.
        collector_type: Type of collector to use.
        aggregator_type: Type of aggregator to use.
        collect_interval_seconds: Metric collection interval.
        window_duration_minutes: Sliding window duration.
        alert_thresholds: Custom alert thresholds.
        redis_url: Redis URL (for Redis collector).
        prometheus_url: Prometheus URL (for Prometheus collector).
    """

    enabled: bool = True
    collector_type: str = "memory"  # "memory", "redis", "prometheus"
    aggregator_type: str = "realtime"  # "realtime", "window"
    collect_interval_seconds: float = 5.0
    window_duration_minutes: int = 15
    alert_thresholds: dict[str, Any] = field(default_factory=dict)
    redis_url: str = "redis://localhost:6379/0"
    prometheus_url: str = "http://localhost:9090"
    auto_start: bool = True


class MonitoringService:
    """Facade for the monitoring system.

    Coordinates collectors, aggregators, and views to provide
    a unified monitoring interface.

    Example:
        >>> service = MonitoringService()
        >>> await service.start()
        >>>
        >>> # Get current metrics
        >>> metrics = await service.get_queue_metrics()
        >>> print(service.format_queue_summary(metrics))
        >>>
        >>> # Check for alerts
        >>> alerts = service.check_alerts()
        >>> for alert in alerts:
        ...     print(f"Alert: {alert.data['message']}")
        >>>
        >>> # Subscribe to events
        >>> async for event in service.subscribe():
        ...     print(f"Event: {event.event_type}")
    """

    def __init__(
        self,
        config: MonitoringConfig | None = None,
    ) -> None:
        """Initialize monitoring service.

        Args:
            config: Service configuration.
        """
        self._config = config or MonitoringConfig()
        self._event_bus = get_event_bus()

        # Components (lazy initialized)
        self._collectors: list[MetricCollectorProtocol] = []
        self._aggregators: list[MetricAggregatorProtocol] = []
        self._views: dict[str, MonitoringViewProtocol] = {}

        # State
        self._started = False
        self._collect_task: asyncio.Task | None = None

        # Initialize default components
        self._setup_defaults()

    def _setup_defaults(self) -> None:
        """Set up default components based on config."""
        # Default views
        self._views["queue"] = QueueStatusView()
        self._views["worker"] = WorkerStatusView()
        self._views["task"] = TaskDetailView()

    def add_collector(self, collector: MetricCollectorProtocol) -> "MonitoringService":
        """Add a metric collector.

        Args:
            collector: Collector to add.

        Returns:
            Self for chaining.
        """
        self._collectors.append(collector)

        # Register with event bus
        if hasattr(collector, "set_event_bus"):
            collector.set_event_bus(self._event_bus)

        return self

    def add_aggregator(self, aggregator: MetricAggregatorProtocol) -> "MonitoringService":
        """Add a metric aggregator.

        Args:
            aggregator: Aggregator to add.

        Returns:
            Self for chaining.
        """
        self._aggregators.append(aggregator)
        return self

    def add_view(self, name: str, view: MonitoringViewProtocol) -> "MonitoringService":
        """Add a monitoring view.

        Args:
            name: View name.
            view: View to add.

        Returns:
            Self for chaining.
        """
        self._views[name] = view
        return self

    async def start(self) -> None:
        """Start the monitoring service."""
        if self._started:
            return

        if not self._config.enabled:
            logger.info("Monitoring is disabled")
            return

        # Create default collector if none added
        if not self._collectors:
            collector = await self._create_default_collector()
            self.add_collector(collector)

        # Create default aggregator if none added
        if not self._aggregators:
            aggregator = self._create_default_aggregator()
            self.add_aggregator(aggregator)

        # Connect all collectors
        for collector in self._collectors:
            try:
                await collector.connect()
            except Exception as e:
                logger.error(f"Failed to connect collector {collector.name}: {e}")

        # Start collection loop if using window aggregator
        if any(isinstance(a, SlidingWindowAggregator) for a in self._aggregators):
            self._collect_task = asyncio.create_task(self._collection_loop())

        self._started = True
        logger.info("Monitoring service started")

    async def stop(self) -> None:
        """Stop the monitoring service."""
        if not self._started:
            return

        # Stop collection loop
        if self._collect_task:
            self._collect_task.cancel()
            try:
                await self._collect_task
            except asyncio.CancelledError:
                pass
            self._collect_task = None

        # Disconnect all collectors
        for collector in self._collectors:
            try:
                await collector.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting collector {collector.name}: {e}")

        self._started = False
        logger.info("Monitoring service stopped")

    async def _create_default_collector(self) -> MetricCollectorProtocol:
        """Create the default collector based on config."""
        collector_type = self._config.collector_type.lower()

        if collector_type == "redis":
            return RedisCollector(
                redis_url=self._config.redis_url,
                collect_interval_seconds=self._config.collect_interval_seconds,
            )
        elif collector_type == "prometheus":
            return PrometheusCollector(
                prometheus_url=self._config.prometheus_url,
                collect_interval_seconds=self._config.collect_interval_seconds,
            )
        else:
            return InMemoryCollector(
                collect_interval_seconds=self._config.collect_interval_seconds,
            )

    def _create_default_aggregator(self) -> MetricAggregatorProtocol:
        """Create the default aggregator based on config."""
        aggregator_type = self._config.aggregator_type.lower()

        if aggregator_type == "window":
            return SlidingWindowAggregator(
                window_duration=timedelta(minutes=self._config.window_duration_minutes),
            )
        else:
            return RealtimeAggregator(
                **self._config.alert_thresholds,
            )

    async def _collection_loop(self) -> None:
        """Background loop for collecting metrics into sliding window."""
        while True:
            try:
                # Collect metrics
                queue_metrics = await self.get_queue_metrics()
                worker_metrics = await self.get_worker_metrics()

                # Add to window aggregators
                for aggregator in self._aggregators:
                    if isinstance(aggregator, SlidingWindowAggregator):
                        aggregator.add_sample(queue_metrics, worker_metrics)

            except Exception as e:
                logger.error(f"Collection loop error: {e}")

            await asyncio.sleep(self._config.collect_interval_seconds)

    async def get_queue_metrics(self) -> list[QueueMetrics]:
        """Get current queue metrics from all collectors.

        Returns:
            List of queue metrics.
        """
        all_metrics = []

        for collector in self._collectors:
            try:
                metrics = await collector.collect_queue_metrics()
                all_metrics.extend(metrics)
            except Exception as e:
                logger.error(f"Error collecting queue metrics from {collector.name}: {e}")

        return all_metrics

    async def get_worker_metrics(self) -> list[WorkerMetrics]:
        """Get current worker metrics from all collectors.

        Returns:
            List of worker metrics.
        """
        all_metrics = []

        for collector in self._collectors:
            try:
                metrics = await collector.collect_worker_metrics()
                all_metrics.extend(metrics)
            except Exception as e:
                logger.error(f"Error collecting worker metrics from {collector.name}: {e}")

        return all_metrics

    async def get_task_metrics(
        self,
        task_ids: list[str] | None = None,
    ) -> list[TaskMetrics]:
        """Get task metrics from all collectors.

        Args:
            task_ids: Optional list of specific task IDs.

        Returns:
            List of task metrics.
        """
        all_metrics = []

        for collector in self._collectors:
            try:
                metrics = await collector.collect_task_metrics(task_ids)
                all_metrics.extend(metrics)
            except Exception as e:
                logger.error(f"Error collecting task metrics from {collector.name}: {e}")

        return all_metrics

    def get_aggregated_metrics(
        self,
        window_duration: timedelta | None = None,
    ) -> AggregatedMetrics | None:
        """Get aggregated metrics from window aggregator.

        Args:
            window_duration: Optional window duration.

        Returns:
            Aggregated metrics or None if not available.
        """
        for aggregator in self._aggregators:
            if isinstance(aggregator, SlidingWindowAggregator):
                return aggregator.get_window_aggregate(window_duration)
        return None

    def check_alerts(
        self,
        queue_metrics: list[QueueMetrics] | None = None,
        worker_metrics: list[WorkerMetrics] | None = None,
    ) -> list[MonitoringEvent]:
        """Check metrics against thresholds and return alerts.

        Args:
            queue_metrics: Queue metrics to check (collects if not provided).
            worker_metrics: Worker metrics to check (collects if not provided).

        Returns:
            List of alert events.
        """
        alerts = []

        for aggregator in self._aggregators:
            if isinstance(aggregator, RealtimeAggregator):
                if queue_metrics or worker_metrics:
                    threshold_alerts = aggregator.check_thresholds(
                        queue_metrics or [],
                        worker_metrics,
                    )
                    alerts.extend(threshold_alerts)

                if queue_metrics:
                    trend_alerts = aggregator.detect_trends(queue_metrics)
                    alerts.extend(trend_alerts)

        return alerts

    async def get_realtime_summary(self) -> dict[str, Any]:
        """Get real-time system summary.

        Returns:
            Summary dictionary suitable for dashboards.
        """
        queue_metrics = await self.get_queue_metrics()
        worker_metrics = await self.get_worker_metrics()

        for aggregator in self._aggregators:
            if isinstance(aggregator, RealtimeAggregator):
                return aggregator.compute_realtime_summary(queue_metrics, worker_metrics)

        # Fallback simple summary
        return {
            "timestamp": datetime.now().isoformat(),
            "queues": {
                "count": len(queue_metrics),
                "total_pending": sum(q.pending_count for q in queue_metrics),
            },
            "workers": {
                "count": len(worker_metrics),
                "online": sum(1 for w in worker_metrics if w.state == "online"),
            },
        }

    async def subscribe(
        self,
        event_types: list[MonitoringEventType] | None = None,
    ) -> AsyncIterator[MonitoringEvent]:
        """Subscribe to monitoring events.

        Args:
            event_types: Optional filter for event types.

        Yields:
            Monitoring events.
        """
        for collector in self._collectors:
            async for event in collector.subscribe():
                if event_types is None or event.event_type in event_types:
                    yield event

    # View methods

    def format_queue_summary(self, metrics: list[QueueMetrics]) -> str:
        """Format queue metrics summary for CLI.

        Args:
            metrics: Queue metrics to format.

        Returns:
            CLI-formatted string.
        """
        view = self._views.get("queue")
        if view and isinstance(view, QueueStatusView):
            return view.format_summary_for_cli(metrics)
        return str(metrics)

    def format_worker_summary(self, metrics: list[WorkerMetrics]) -> str:
        """Format worker metrics summary for CLI.

        Args:
            metrics: Worker metrics to format.

        Returns:
            CLI-formatted string.
        """
        view = self._views.get("worker")
        if view and isinstance(view, WorkerStatusView):
            return view.format_summary_for_cli(metrics)
        return str(metrics)

    def format_task_list(
        self,
        tasks: list[TaskMetrics],
        show_completed: bool = False,
    ) -> str:
        """Format task list for CLI.

        Args:
            tasks: Task metrics to format.
            show_completed: Whether to show completed tasks.

        Returns:
            CLI-formatted string.
        """
        view = self._views.get("task")
        if view and isinstance(view, TaskDetailView):
            return view.format_task_list_for_cli(tasks, show_completed)
        return str(tasks)

    def render_for_api(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
    ) -> dict[str, Any]:
        """Render metrics for REST API response.

        Args:
            queue_metrics: Queue metrics.
            worker_metrics: Worker metrics.

        Returns:
            API-formatted dictionary.
        """
        queue_view = self._views.get("queue")
        worker_view = self._views.get("worker")

        return {
            "timestamp": datetime.now().isoformat(),
            "queues": (
                queue_view.render_summary(queue_metrics, worker_metrics)
                if queue_view else []
            ),
            "workers": (
                worker_view.render_summary(queue_metrics, worker_metrics)
                if worker_view else []
            ),
        }

    async def health_check(self) -> dict[str, Any]:
        """Check health of all monitoring components.

        Returns:
            Health check results.
        """
        results = {
            "status": "healthy",
            "collectors": {},
            "timestamp": datetime.now().isoformat(),
        }

        all_healthy = True
        for collector in self._collectors:
            try:
                healthy = await collector.health_check()
                results["collectors"][collector.name] = {
                    "connected": collector.is_connected,
                    "healthy": healthy,
                }
                if not healthy:
                    all_healthy = False
            except Exception as e:
                results["collectors"][collector.name] = {
                    "connected": False,
                    "healthy": False,
                    "error": str(e),
                }
                all_healthy = False

        results["status"] = "healthy" if all_healthy else "degraded"
        return results

    @property
    def is_started(self) -> bool:
        """Check if service is started."""
        return self._started

    @property
    def collectors(self) -> list[MetricCollectorProtocol]:
        """Get list of collectors."""
        return self._collectors.copy()

    @property
    def aggregators(self) -> list[MetricAggregatorProtocol]:
        """Get list of aggregators."""
        return self._aggregators.copy()


# Global service instance
_monitoring_service: MonitoringService | None = None


def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance.

    Returns:
        Global MonitoringService instance.
    """
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service


def configure_monitoring(config: MonitoringConfig) -> MonitoringService:
    """Configure and return the global monitoring service.

    Args:
        config: Service configuration.

    Returns:
        Configured MonitoringService instance.
    """
    global _monitoring_service
    _monitoring_service = MonitoringService(config)
    return _monitoring_service
