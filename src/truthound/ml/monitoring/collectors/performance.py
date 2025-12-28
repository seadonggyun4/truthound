"""Performance metric collector.

Collects latency and throughput metrics from predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import time
from collections import deque
import statistics

from truthound.ml.monitoring.protocols import (
    IMetricCollector,
    ModelMetrics,
    PredictionRecord,
)


@dataclass
class PerformanceConfig:
    """Configuration for performance collector.

    Attributes:
        window_size: Number of predictions to track
        percentiles: Latency percentiles to compute
        throughput_window_seconds: Window for throughput calculation
    """

    window_size: int = 1000
    percentiles: list[float] = field(default_factory=lambda: [0.5, 0.95, 0.99])
    throughput_window_seconds: int = 60


class PerformanceCollector(IMetricCollector):
    """Collects performance metrics (latency, throughput).

    Tracks:
    - Latency statistics (mean, median, percentiles)
    - Throughput (predictions per second)
    - Request counts and error rates

    Example:
        >>> collector = PerformanceCollector()
        >>> metrics = collector.collect("model-1", predictions)
        >>> print(f"Latency: {metrics.latency_ms}ms")
    """

    def __init__(self, config: PerformanceConfig | None = None):
        self._config = config or PerformanceConfig()
        self._latencies: dict[str, deque[float]] = {}
        self._timestamps: dict[str, deque[datetime]] = {}
        self._request_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "performance"

    def collect(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect performance metrics from predictions.

        Args:
            model_id: Model identifier
            predictions: Prediction records

        Returns:
            Performance metrics
        """
        if model_id not in self._latencies:
            self._latencies[model_id] = deque(maxlen=self._config.window_size)
            self._timestamps[model_id] = deque(maxlen=self._config.window_size)
            self._request_counts[model_id] = 0
            self._error_counts[model_id] = 0

        # Record predictions
        for pred in predictions:
            self._latencies[model_id].append(pred.latency_ms)
            self._timestamps[model_id].append(pred.timestamp)
            self._request_counts[model_id] += 1

            # Check for errors (prediction is None or contains error)
            if pred.prediction is None or pred.metadata.get("error"):
                self._error_counts[model_id] += 1

        # Compute metrics
        latencies = list(self._latencies[model_id])
        timestamps = list(self._timestamps[model_id])

        # Mean latency
        mean_latency = statistics.mean(latencies) if latencies else 0.0

        # Throughput (predictions in last window)
        throughput = self._compute_throughput(timestamps)

        # Custom metrics for percentiles
        custom_metrics: dict[str, float] = {}

        if latencies:
            sorted_latencies = sorted(latencies)
            for p in self._config.percentiles:
                idx = int(len(sorted_latencies) * p)
                if idx < len(sorted_latencies):
                    custom_metrics[f"latency_p{int(p*100)}"] = sorted_latencies[idx]

            # Also add min, max, std
            custom_metrics["latency_min"] = min(latencies)
            custom_metrics["latency_max"] = max(latencies)
            if len(latencies) > 1:
                custom_metrics["latency_std"] = statistics.stdev(latencies)

        # Error rate
        total = self._request_counts[model_id]
        errors = self._error_counts[model_id]
        if total > 0:
            custom_metrics["error_rate"] = errors / total

        custom_metrics["request_count"] = total

        return ModelMetrics(
            model_id=model_id,
            timestamp=datetime.now(timezone.utc),
            latency_ms=mean_latency,
            throughput_rps=throughput,
            custom_metrics=custom_metrics,
        )

    def _compute_throughput(self, timestamps: list[datetime]) -> float:
        """Compute throughput from timestamps."""
        if len(timestamps) < 2:
            return 0.0

        now = datetime.now(timezone.utc)
        window_start = now.timestamp() - self._config.throughput_window_seconds

        # Count predictions in window
        count = sum(1 for ts in timestamps if ts.timestamp() >= window_start)

        return count / self._config.throughput_window_seconds

    def reset(self) -> None:
        """Reset collector state."""
        self._latencies.clear()
        self._timestamps.clear()
        self._request_counts.clear()
        self._error_counts.clear()

    def get_latency_histogram(self, model_id: str, buckets: int = 20) -> dict[str, int]:
        """Get latency histogram for visualization.

        Args:
            model_id: Model identifier
            buckets: Number of histogram buckets

        Returns:
            Histogram as bucket_range -> count
        """
        latencies = list(self._latencies.get(model_id, []))
        if not latencies:
            return {}

        min_lat = min(latencies)
        max_lat = max(latencies)
        bucket_size = (max_lat - min_lat) / buckets if max_lat > min_lat else 1

        histogram: dict[str, int] = {}
        for i in range(buckets):
            lower = min_lat + i * bucket_size
            upper = lower + bucket_size
            count = sum(1 for lat in latencies if lower <= lat < upper)
            histogram[f"{lower:.1f}-{upper:.1f}"] = count

        return histogram
