"""Prometheus metric store.

Exports metrics to Prometheus for monitoring and alerting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import logging

from truthound.ml.monitoring.protocols import IMetricStore, ModelMetrics


logger = logging.getLogger(__name__)


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus store.

    Attributes:
        port: HTTP port for metrics endpoint
        path: Metrics endpoint path
        prefix: Metric name prefix
        push_gateway_url: Push gateway URL (optional)
        job_name: Job name for push gateway
    """

    port: int = 9090
    path: str = "/metrics"
    prefix: str = "truthound_ml_"
    push_gateway_url: str | None = None
    job_name: str = "truthound"


class PrometheusMetricStore(IMetricStore):
    """Prometheus metric store.

    Exposes metrics via HTTP endpoint and/or pushes to push gateway.
    Uses prometheus_client library.

    Example:
        >>> store = PrometheusMetricStore()
        >>> await store.start()  # Start HTTP server
        >>> await store.store(metrics)  # Update metrics

    Requires:
        pip install prometheus-client
    """

    def __init__(self, config: PrometheusConfig | None = None):
        self._config = config or PrometheusConfig()
        self._metrics_cache: dict[str, ModelMetrics] = {}
        self._gauges: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._counters: dict[str, Any] = {}
        self._server: Any = None
        self._registry: Any = None

    async def start(self) -> None:
        """Start Prometheus HTTP server."""
        try:
            from prometheus_client import (
                start_http_server,
                CollectorRegistry,
                Gauge,
                Histogram,
                Counter,
            )
        except ImportError:
            raise ImportError(
                "prometheus_client is required for Prometheus store. "
                "Install with: pip install prometheus-client"
            )

        self._registry = CollectorRegistry()

        # Create metrics
        prefix = self._config.prefix

        # Gauges for current values
        self._gauges["latency"] = Gauge(
            f"{prefix}latency_ms",
            "Model prediction latency in milliseconds",
            ["model_id"],
            registry=self._registry,
        )

        self._gauges["throughput"] = Gauge(
            f"{prefix}throughput_rps",
            "Model throughput in requests per second",
            ["model_id"],
            registry=self._registry,
        )

        self._gauges["prediction_drift"] = Gauge(
            f"{prefix}prediction_drift",
            "Prediction distribution drift score",
            ["model_id"],
            registry=self._registry,
        )

        self._gauges["feature_drift"] = Gauge(
            f"{prefix}feature_drift",
            "Feature drift score",
            ["model_id", "feature"],
            registry=self._registry,
        )

        self._gauges["accuracy"] = Gauge(
            f"{prefix}accuracy",
            "Model accuracy",
            ["model_id"],
            registry=self._registry,
        )

        self._gauges["precision"] = Gauge(
            f"{prefix}precision",
            "Model precision",
            ["model_id"],
            registry=self._registry,
        )

        self._gauges["recall"] = Gauge(
            f"{prefix}recall",
            "Model recall",
            ["model_id"],
            registry=self._registry,
        )

        self._gauges["f1"] = Gauge(
            f"{prefix}f1_score",
            "Model F1 score",
            ["model_id"],
            registry=self._registry,
        )

        self._gauges["custom"] = Gauge(
            f"{prefix}custom_metric",
            "Custom metric value",
            ["model_id", "metric_name"],
            registry=self._registry,
        )

        # Histogram for latency distribution
        self._histograms["latency"] = Histogram(
            f"{prefix}latency_histogram",
            "Latency histogram",
            ["model_id"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self._registry,
        )

        # Counter for predictions
        self._counters["predictions"] = Counter(
            f"{prefix}predictions_total",
            "Total predictions",
            ["model_id"],
            registry=self._registry,
        )

        # Start HTTP server
        start_http_server(self._config.port, registry=self._registry)
        logger.info(f"Prometheus metrics available at http://localhost:{self._config.port}{self._config.path}")

    async def stop(self) -> None:
        """Stop Prometheus server."""
        # Note: prometheus_client doesn't provide a clean shutdown
        pass

    async def store(self, metrics: ModelMetrics) -> None:
        """Store metrics by updating Prometheus gauges.

        Args:
            metrics: Metrics to store
        """
        model_id = metrics.model_id
        self._metrics_cache[model_id] = metrics

        # Update gauges
        if metrics.latency_ms:
            self._gauges["latency"].labels(model_id=model_id).set(metrics.latency_ms)
            self._histograms["latency"].labels(model_id=model_id).observe(metrics.latency_ms)

        if metrics.throughput_rps:
            self._gauges["throughput"].labels(model_id=model_id).set(metrics.throughput_rps)

        if metrics.prediction_drift is not None:
            self._gauges["prediction_drift"].labels(model_id=model_id).set(metrics.prediction_drift)

        for feature, drift in metrics.feature_drift.items():
            self._gauges["feature_drift"].labels(model_id=model_id, feature=feature).set(drift)

        if metrics.accuracy is not None:
            self._gauges["accuracy"].labels(model_id=model_id).set(metrics.accuracy)

        if metrics.precision is not None:
            self._gauges["precision"].labels(model_id=model_id).set(metrics.precision)

        if metrics.recall is not None:
            self._gauges["recall"].labels(model_id=model_id).set(metrics.recall)

        if metrics.f1_score is not None:
            self._gauges["f1"].labels(model_id=model_id).set(metrics.f1_score)

        for metric_name, value in metrics.custom_metrics.items():
            self._gauges["custom"].labels(model_id=model_id, metric_name=metric_name).set(value)

        # Push to gateway if configured
        if self._config.push_gateway_url:
            await self._push_to_gateway()

    async def _push_to_gateway(self) -> None:
        """Push metrics to Prometheus push gateway."""
        try:
            from prometheus_client import push_to_gateway
            push_to_gateway(
                self._config.push_gateway_url,
                job=self._config.job_name,
                registry=self._registry,
            )
        except Exception as e:
            logger.warning(f"Failed to push to gateway: {e}")

    async def query(
        self,
        model_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation: str = "mean",
    ) -> list[ModelMetrics]:
        """Query historical metrics.

        Note: Prometheus store doesn't keep history, returns cached values only.
        For historical queries, use PromQL directly.

        Args:
            model_id: Model identifier
            start_time: Query start time
            end_time: Query end time
            aggregation: Aggregation function

        Returns:
            List with current metrics only
        """
        metrics = self._metrics_cache.get(model_id)
        if metrics and start_time <= metrics.timestamp <= end_time:
            return [metrics]
        return []

    async def get_latest(self, model_id: str) -> ModelMetrics | None:
        """Get latest metrics for model.

        Args:
            model_id: Model identifier

        Returns:
            Latest metrics or None
        """
        return self._metrics_cache.get(model_id)

    async def delete(
        self,
        model_id: str,
        before: datetime | None = None,
    ) -> int:
        """Delete metrics (removes from cache only).

        Args:
            model_id: Model identifier
            before: Not used for Prometheus

        Returns:
            1 if deleted, 0 if not found
        """
        if model_id in self._metrics_cache:
            del self._metrics_cache[model_id]
            return 1
        return 0

    def get_scrape_target(self) -> str:
        """Get Prometheus scrape target URL.

        Returns:
            URL for Prometheus to scrape
        """
        return f"http://localhost:{self._config.port}{self._config.path}"

    async def __aenter__(self) -> "PrometheusMetricStore":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
