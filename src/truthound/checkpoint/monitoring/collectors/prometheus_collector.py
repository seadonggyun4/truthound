"""Prometheus metric collector.

Collects metrics from Prometheus server, useful for integrating
with existing Prometheus-based monitoring infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, AsyncIterator
from urllib.parse import urljoin

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    TaskMetrics,
    MonitoringEvent,
    MonitoringEventType,
    CollectorError,
)
from truthound.checkpoint.monitoring.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class PrometheusCollector(BaseCollector):
    """Prometheus metric collector.

    Queries a Prometheus server to collect metrics about checkpoints,
    queues, and workers. Assumes Truthound metrics are being exported
    with the standard naming convention.

    Metric naming conventions expected:
    - truthound_queue_pending_total{queue="..."}
    - truthound_queue_running_total{queue="..."}
    - truthound_queue_completed_total{queue="..."}
    - truthound_queue_failed_total{queue="..."}
    - truthound_worker_current_tasks{worker="...", hostname="..."}
    - truthound_worker_completed_total{worker="..."}
    - truthound_task_duration_seconds{checkpoint="..."}

    Example:
        >>> collector = PrometheusCollector(
        ...     prometheus_url="http://localhost:9090",
        ... )
        >>> await collector.connect()
        >>>
        >>> metrics = await collector.collect_queue_metrics()
    """

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        metric_prefix: str = "truthound_",
        name: str = "prometheus",
        collect_interval_seconds: float = 15.0,
        cache_ttl_seconds: float = 5.0,
        timeout_seconds: float = 10.0,
    ) -> None:
        """Initialize Prometheus collector.

        Args:
            prometheus_url: Prometheus server URL.
            metric_prefix: Prefix for Truthound metrics.
            name: Collector name.
            collect_interval_seconds: Collection interval.
            cache_ttl_seconds: Cache TTL.
            timeout_seconds: HTTP request timeout.
        """
        super().__init__(
            name=name,
            collect_interval_seconds=collect_interval_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        self._prometheus_url = prometheus_url.rstrip("/")
        self._metric_prefix = metric_prefix
        self._timeout = timeout_seconds
        self._http_client: Any = None

    async def connect(self) -> None:
        """Connect to Prometheus."""
        try:
            import httpx
        except ImportError:
            raise CollectorError(
                "httpx package not installed",
                self.name,
                "Install with: pip install httpx",
            )

        try:
            self._http_client = httpx.AsyncClient(
                base_url=self._prometheus_url,
                timeout=self._timeout,
            )

            # Test connection
            response = await self._http_client.get("/api/v1/status/config")
            if response.status_code != 200:
                raise CollectorError(
                    f"Prometheus returned status {response.status_code}",
                    self.name,
                )

            await super().connect()
            logger.info(f"Connected to Prometheus at {self._prometheus_url}")

        except Exception as e:
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            raise CollectorError(
                f"Failed to connect to Prometheus: {e}",
                self.name,
                str(e),
            )

    async def disconnect(self) -> None:
        """Disconnect from Prometheus."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        await super().disconnect()

    async def _query(self, query: str) -> list[dict[str, Any]]:
        """Execute a PromQL instant query.

        Args:
            query: PromQL query string.

        Returns:
            List of result vectors.
        """
        if self._http_client is None:
            raise CollectorError("Not connected to Prometheus", self.name)

        try:
            response = await self._http_client.get(
                "/api/v1/query",
                params={"query": query},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                raise CollectorError(
                    f"Prometheus query failed: {data.get('error', 'Unknown error')}",
                    self.name,
                )

            return data.get("data", {}).get("result", [])

        except Exception as e:
            raise CollectorError(
                f"Prometheus query error: {e}",
                self.name,
                str(e),
            )

    async def _query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "1m",
    ) -> list[dict[str, Any]]:
        """Execute a PromQL range query.

        Args:
            query: PromQL query string.
            start: Start time.
            end: End time.
            step: Query resolution.

        Returns:
            List of result matrices.
        """
        if self._http_client is None:
            raise CollectorError("Not connected to Prometheus", self.name)

        try:
            response = await self._http_client.get(
                "/api/v1/query_range",
                params={
                    "query": query,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "step": step,
                },
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                raise CollectorError(
                    f"Prometheus range query failed: {data.get('error', 'Unknown error')}",
                    self.name,
                )

            return data.get("data", {}).get("result", [])

        except Exception as e:
            raise CollectorError(
                f"Prometheus range query error: {e}",
                self.name,
                str(e),
            )

    async def collect_queue_metrics(self) -> list[QueueMetrics]:
        """Collect queue metrics from Prometheus."""
        metrics = []
        queue_data: dict[str, dict[str, float]] = {}

        # Define metric queries
        metric_queries = {
            "pending": f"{self._metric_prefix}queue_pending_total",
            "running": f"{self._metric_prefix}queue_running_total",
            "completed": f"{self._metric_prefix}queue_completed_total",
            "failed": f"{self._metric_prefix}queue_failed_total",
        }

        # Collect each metric
        for metric_name, query in metric_queries.items():
            try:
                results = await self._query(query)
                for result in results:
                    labels = result.get("metric", {})
                    queue_name = labels.get("queue", "default")
                    value = float(result.get("value", [0, 0])[1])

                    if queue_name not in queue_data:
                        queue_data[queue_name] = {}
                    queue_data[queue_name][metric_name] = value

            except CollectorError:
                # Metric might not exist yet
                pass

        # Collect rate metrics for throughput
        try:
            rate_query = f"rate({self._metric_prefix}queue_completed_total[1m])"
            results = await self._query(rate_query)
            for result in results:
                labels = result.get("metric", {})
                queue_name = labels.get("queue", "default")
                throughput = float(result.get("value", [0, 0])[1])

                if queue_name in queue_data:
                    queue_data[queue_name]["throughput"] = throughput
        except CollectorError:
            pass

        # Collect average execution time
        try:
            avg_query = (
                f"histogram_quantile(0.5, rate({self._metric_prefix}task_duration_seconds_bucket[5m]))"
            )
            results = await self._query(avg_query)
            for result in results:
                labels = result.get("metric", {})
                queue_name = labels.get("queue", "default")
                avg_duration = float(result.get("value", [0, 0])[1]) * 1000  # Convert to ms

                if queue_name in queue_data:
                    queue_data[queue_name]["avg_exec_ms"] = avg_duration
        except CollectorError:
            pass

        # Build QueueMetrics objects
        for queue_name, data in queue_data.items():
            metrics.append(QueueMetrics(
                queue_name=queue_name,
                pending_count=int(data.get("pending", 0)),
                running_count=int(data.get("running", 0)),
                completed_count=int(data.get("completed", 0)),
                failed_count=int(data.get("failed", 0)),
                avg_execution_time_ms=data.get("avg_exec_ms", 0.0),
                throughput_per_second=data.get("throughput", 0.0),
            ))

        return metrics

    async def collect_worker_metrics(self) -> list[WorkerMetrics]:
        """Collect worker metrics from Prometheus."""
        metrics = []
        worker_data: dict[str, dict[str, Any]] = {}

        # Define metric queries
        metric_queries = {
            "current_tasks": f"{self._metric_prefix}worker_current_tasks",
            "completed": f"{self._metric_prefix}worker_completed_total",
            "failed": f"{self._metric_prefix}worker_failed_total",
            "cpu_percent": f"{self._metric_prefix}worker_cpu_percent",
            "memory_mb": f"{self._metric_prefix}worker_memory_mb",
        }

        # Collect each metric
        for metric_name, query in metric_queries.items():
            try:
                results = await self._query(query)
                for result in results:
                    labels = result.get("metric", {})
                    worker_id = labels.get("worker", labels.get("instance", "unknown"))
                    hostname = labels.get("hostname", labels.get("instance", ""))
                    value = float(result.get("value", [0, 0])[1])

                    if worker_id not in worker_data:
                        worker_data[worker_id] = {"hostname": hostname}
                    worker_data[worker_id][metric_name] = value

            except CollectorError:
                pass

        # Check worker state (up metric)
        try:
            results = await self._query(f"{self._metric_prefix}worker_up")
            for result in results:
                labels = result.get("metric", {})
                worker_id = labels.get("worker", labels.get("instance", "unknown"))
                value = float(result.get("value", [0, 0])[1])

                if worker_id in worker_data:
                    worker_data[worker_id]["state"] = "online" if value == 1 else "offline"
        except CollectorError:
            pass

        # Build WorkerMetrics objects
        for worker_id, data in worker_data.items():
            metrics.append(WorkerMetrics(
                worker_id=worker_id,
                state=data.get("state", "unknown"),
                current_tasks=int(data.get("current_tasks", 0)),
                completed_tasks=int(data.get("completed", 0)),
                failed_tasks=int(data.get("failed", 0)),
                cpu_percent=data.get("cpu_percent", 0.0),
                memory_mb=data.get("memory_mb", 0.0),
                hostname=data.get("hostname", ""),
            ))

        return metrics

    async def collect_task_metrics(
        self,
        task_ids: list[str] | None = None,
    ) -> list[TaskMetrics]:
        """Collect task metrics from Prometheus.

        Note: Prometheus is not ideal for per-task metrics.
        This returns aggregate task information from labels.
        """
        metrics = []

        # Query task state metric
        try:
            query = f"{self._metric_prefix}task_state"
            if task_ids:
                # Build regex filter
                task_filter = "|".join(task_ids)
                query = f'{query}{{task_id=~"{task_filter}"}}'

            results = await self._query(query)
            for result in results:
                labels = result.get("metric", {})
                value = result.get("value", [0, 0])

                task_id = labels.get("task_id", "")
                if not task_id:
                    continue

                metrics.append(TaskMetrics(
                    task_id=task_id,
                    checkpoint_name=labels.get("checkpoint", ""),
                    state=labels.get("state", "unknown"),
                    queue_name=labels.get("queue", "default"),
                    worker_id=labels.get("worker"),
                ))

        except CollectorError:
            pass

        return metrics

    async def get_historical_metrics(
        self,
        metric_name: str,
        duration: timedelta = timedelta(hours=1),
        step: str = "1m",
    ) -> list[dict[str, Any]]:
        """Get historical metrics from Prometheus.

        Args:
            metric_name: Metric name without prefix.
            duration: Time range to query.
            step: Query resolution.

        Returns:
            List of time series data.
        """
        end = datetime.now()
        start = end - duration
        query = f"{self._metric_prefix}{metric_name}"

        return await self._query_range(query, start, end, step)

    async def subscribe(self) -> AsyncIterator[MonitoringEvent]:
        """Subscribe to Prometheus alerts.

        Note: This polls the Alertmanager API for active alerts.
        For real-time alerts, configure Alertmanager webhooks.
        """
        # For Prometheus, we poll for alerts instead of subscribing
        while self._running:
            try:
                # Check for active alerts
                response = await self._http_client.get("/api/v1/alerts")
                if response.status_code == 200:
                    data = response.json()
                    alerts = data.get("data", {}).get("alerts", [])

                    for alert in alerts:
                        if alert.get("state") == "firing":
                            event = MonitoringEvent(
                                event_type=MonitoringEventType.ALERT_TRIGGERED,
                                source=self.name,
                                data={
                                    "alert_name": alert.get("labels", {}).get("alertname"),
                                    "labels": alert.get("labels", {}),
                                    "annotations": alert.get("annotations", {}),
                                    "state": alert.get("state"),
                                },
                            )
                            yield event

            except Exception as e:
                logger.warning(f"Error checking Prometheus alerts: {e}")

            await asyncio.sleep(self._collect_interval)

    async def health_check(self) -> bool:
        """Check Prometheus connection health."""
        if self._http_client is None:
            return False

        try:
            response = await self._http_client.get("/-/healthy")
            return response.status_code == 200
        except Exception:
            return False
