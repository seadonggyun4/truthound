"""Base aggregator implementation.

Provides abstract base class with common aggregation functionality.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from truthound.checkpoint.monitoring.protocols import (
    MetricAggregatorProtocol,
    QueueMetrics,
    WorkerMetrics,
    MonitoringEvent,
    MonitoringEventType,
    AlertSeverity,
    AggregatedMetrics,
)

logger = logging.getLogger(__name__)


class BaseAggregator(ABC):
    """Abstract base class for metric aggregators.

    Provides common aggregation functionality including:
    - Rate calculation
    - Statistical aggregation
    - Anomaly detection (Z-score based)
    - Alert generation

    Subclasses can override methods for custom behavior.
    """

    def __init__(
        self,
        name: str | None = None,
        anomaly_threshold: float = 2.0,
    ) -> None:
        """Initialize aggregator.

        Args:
            name: Aggregator name.
            anomaly_threshold: Z-score threshold for anomaly detection.
        """
        self._name = name or self.__class__.__name__
        self._anomaly_threshold = anomaly_threshold

    @property
    def name(self) -> str:
        """Get aggregator name."""
        return self._name

    def aggregate_queues(self, metrics: list[QueueMetrics]) -> QueueMetrics:
        """Aggregate multiple queue metrics into a single summary.

        Sums counts and computes weighted averages for timing metrics.

        Args:
            metrics: List of queue metrics to aggregate.

        Returns:
            Aggregated queue metrics with name "aggregate".
        """
        if not metrics:
            return QueueMetrics(queue_name="aggregate")

        if len(metrics) == 1:
            return metrics[0]

        # Sum counts
        total_pending = sum(m.pending_count for m in metrics)
        total_running = sum(m.running_count for m in metrics)
        total_completed = sum(m.completed_count for m in metrics)
        total_failed = sum(m.failed_count for m in metrics)

        # Weighted average for timing (weight by completed count)
        total_weight = sum(m.completed_count for m in metrics)
        if total_weight > 0:
            avg_wait = sum(
                m.avg_wait_time_ms * m.completed_count for m in metrics
            ) / total_weight
            avg_exec = sum(
                m.avg_execution_time_ms * m.completed_count for m in metrics
            ) / total_weight
        else:
            avg_wait = 0.0
            avg_exec = 0.0

        # Sum throughput
        total_throughput = sum(m.throughput_per_second for m in metrics)

        # Latest timestamp
        latest_timestamp = max(m.timestamp for m in metrics)

        return QueueMetrics(
            queue_name="aggregate",
            pending_count=total_pending,
            running_count=total_running,
            completed_count=total_completed,
            failed_count=total_failed,
            avg_wait_time_ms=avg_wait,
            avg_execution_time_ms=avg_exec,
            throughput_per_second=total_throughput,
            timestamp=latest_timestamp,
            labels={"aggregated_from": ",".join(m.queue_name for m in metrics)},
        )

    def aggregate_workers(self, metrics: list[WorkerMetrics]) -> dict[str, Any]:
        """Aggregate worker metrics into a summary.

        Args:
            metrics: List of worker metrics to aggregate.

        Returns:
            Worker summary dictionary.
        """
        if not metrics:
            return {
                "total_workers": 0,
                "online_workers": 0,
                "total_capacity": 0,
                "current_load": 0,
                "avg_load_factor": 0.0,
                "avg_cpu_percent": 0.0,
                "avg_memory_mb": 0.0,
                "total_completed": 0,
                "total_failed": 0,
            }

        online_workers = [m for m in metrics if m.state == "online"]
        total_capacity = sum(m.max_concurrency for m in online_workers)
        current_load = sum(m.current_tasks for m in metrics)

        # Calculate averages
        avg_load = sum(m.load_factor for m in metrics) / len(metrics) if metrics else 0.0
        avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics) if metrics else 0.0
        avg_memory = sum(m.memory_mb for m in metrics) / len(metrics) if metrics else 0.0

        return {
            "total_workers": len(metrics),
            "online_workers": len(online_workers),
            "offline_workers": len(metrics) - len(online_workers),
            "total_capacity": total_capacity,
            "current_load": current_load,
            "available_capacity": total_capacity - current_load,
            "avg_load_factor": avg_load,
            "max_load_factor": max((m.load_factor for m in metrics), default=0.0),
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max((m.cpu_percent for m in metrics), default=0.0),
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max((m.memory_mb for m in metrics), default=0.0),
            "total_completed": sum(m.completed_tasks for m in metrics),
            "total_failed": sum(m.failed_tasks for m in metrics),
            "workers_by_state": self._group_by_state(metrics),
        }

    def _group_by_state(self, metrics: list[WorkerMetrics]) -> dict[str, int]:
        """Group workers by state."""
        state_counts: dict[str, int] = {}
        for m in metrics:
            state_counts[m.state] = state_counts.get(m.state, 0) + 1
        return state_counts

    def calculate_rate(
        self,
        current: int,
        previous: int,
        interval_seconds: float,
    ) -> float:
        """Calculate rate of change.

        Args:
            current: Current value.
            previous: Previous value.
            interval_seconds: Time interval.

        Returns:
            Rate per second.
        """
        if interval_seconds <= 0:
            return 0.0
        return (current - previous) / interval_seconds

    def detect_anomalies(
        self,
        metrics: list[QueueMetrics],
        threshold: float | None = None,
    ) -> list[MonitoringEvent]:
        """Detect anomalies in queue metrics using Z-score.

        Args:
            metrics: List of metrics to analyze.
            threshold: Z-score threshold (defaults to instance setting).

        Returns:
            List of anomaly events.
        """
        if len(metrics) < 3:
            # Need at least 3 samples for meaningful statistics
            return []

        threshold = threshold or self._anomaly_threshold
        events = []

        # Analyze pending count
        pending_anomalies = self._detect_numeric_anomalies(
            values=[m.pending_count for m in metrics],
            labels=[m.queue_name for m in metrics],
            metric_name="pending_count",
            threshold=threshold,
        )
        events.extend(pending_anomalies)

        # Analyze execution time
        exec_time_anomalies = self._detect_numeric_anomalies(
            values=[m.avg_execution_time_ms for m in metrics],
            labels=[m.queue_name for m in metrics],
            metric_name="avg_execution_time_ms",
            threshold=threshold,
        )
        events.extend(exec_time_anomalies)

        # Analyze success rate
        success_rates = [m.success_rate for m in metrics]
        success_anomalies = self._detect_low_value_anomalies(
            values=success_rates,
            labels=[m.queue_name for m in metrics],
            metric_name="success_rate",
            threshold=0.9,  # Alert if success rate drops below 90%
        )
        events.extend(success_anomalies)

        return events

    def _detect_numeric_anomalies(
        self,
        values: list[int | float],
        labels: list[str],
        metric_name: str,
        threshold: float,
    ) -> list[MonitoringEvent]:
        """Detect anomalies in numeric values using Z-score."""
        events = []

        # Calculate statistics
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev == 0:
            return events  # No variation, no anomalies

        # Check each value
        for i, (value, label) in enumerate(zip(values, labels)):
            z_score = abs(value - mean) / std_dev
            if z_score > threshold:
                events.append(MonitoringEvent(
                    event_type=MonitoringEventType.THRESHOLD_CROSSED,
                    source=self.name,
                    data={
                        "queue_name": label,
                        "metric_name": metric_name,
                        "value": value,
                        "mean": mean,
                        "std_dev": std_dev,
                        "z_score": z_score,
                        "threshold": threshold,
                    },
                    severity=AlertSeverity.WARNING if z_score < threshold * 1.5 else AlertSeverity.ERROR,
                    labels={"queue": label, "metric": metric_name},
                ))

        return events

    def _detect_low_value_anomalies(
        self,
        values: list[float],
        labels: list[str],
        metric_name: str,
        threshold: float,
    ) -> list[MonitoringEvent]:
        """Detect values below a threshold."""
        events = []

        for value, label in zip(values, labels):
            if value < threshold:
                events.append(MonitoringEvent(
                    event_type=MonitoringEventType.THRESHOLD_CROSSED,
                    source=self.name,
                    data={
                        "queue_name": label,
                        "metric_name": metric_name,
                        "value": value,
                        "threshold": threshold,
                    },
                    severity=AlertSeverity.WARNING if value > threshold * 0.5 else AlertSeverity.ERROR,
                    labels={"queue": label, "metric": metric_name},
                ))

        return events

    def compute_percentiles(
        self,
        values: list[float],
        percentiles: list[float] = [50, 90, 95, 99],
    ) -> dict[str, float]:
        """Compute percentiles for a list of values.

        Args:
            values: Values to analyze.
            percentiles: Percentiles to compute (0-100).

        Returns:
            Dictionary of percentile values.
        """
        if not values:
            return {f"p{int(p)}": 0.0 for p in percentiles}

        sorted_values = sorted(values)
        n = len(sorted_values)

        result = {}
        for p in percentiles:
            # Linear interpolation
            k = (n - 1) * (p / 100)
            f = math.floor(k)
            c = math.ceil(k)

            if f == c:
                result[f"p{int(p)}"] = sorted_values[int(k)]
            else:
                result[f"p{int(p)}"] = (
                    sorted_values[f] * (c - k) + sorted_values[c] * (k - f)
                )

        return result

    def compute_statistics(self, values: list[float]) -> dict[str, float]:
        """Compute basic statistics for a list of values.

        Args:
            values: Values to analyze.

        Returns:
            Dictionary with min, max, mean, std_dev, count.
        """
        if not values:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std_dev": 0.0,
                "count": 0,
            }

        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std_dev = math.sqrt(variance)

        return {
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "std_dev": std_dev,
            "count": n,
        }
