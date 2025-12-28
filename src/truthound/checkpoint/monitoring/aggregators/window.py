"""Sliding window aggregator.

Provides time-windowed aggregation of metrics for trend analysis
and historical comparisons.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    MonitoringEvent,
    MonitoringEventType,
    AlertSeverity,
    AggregatedMetrics,
)
from truthound.checkpoint.monitoring.aggregators.base import BaseAggregator

logger = logging.getLogger(__name__)


@dataclass
class WindowedSample:
    """A sample in the sliding window."""

    timestamp: datetime
    queue_metrics: list[QueueMetrics]
    worker_metrics: list[WorkerMetrics]


class SlidingWindowAggregator(BaseAggregator):
    """Sliding window metric aggregator.

    Maintains a time-based sliding window of metrics for
    computing moving averages and detecting trends.

    Features:
    - Configurable window duration
    - Multiple window sizes (1m, 5m, 15m, 1h)
    - Moving average calculation
    - Trend detection across window
    - Anomaly detection using historical baseline

    Example:
        >>> aggregator = SlidingWindowAggregator(
        ...     window_duration=timedelta(minutes=15),
        ...     sample_interval=timedelta(seconds=10),
        ... )
        >>>
        >>> # Add samples as they arrive
        >>> aggregator.add_sample(queue_metrics, worker_metrics)
        >>>
        >>> # Get aggregated metrics over window
        >>> aggregated = aggregator.get_window_aggregate()
    """

    def __init__(
        self,
        name: str = "sliding_window",
        window_duration: timedelta = timedelta(minutes=15),
        sample_interval: timedelta = timedelta(seconds=10),
        anomaly_threshold: float = 2.0,
        max_samples: int = 1000,
    ) -> None:
        """Initialize sliding window aggregator.

        Args:
            name: Aggregator name.
            window_duration: Duration of the sliding window.
            sample_interval: Expected interval between samples.
            anomaly_threshold: Z-score threshold for anomaly detection.
            max_samples: Maximum samples to keep (memory limit).
        """
        super().__init__(name=name, anomaly_threshold=anomaly_threshold)
        self._window_duration = window_duration
        self._sample_interval = sample_interval
        self._max_samples = max_samples

        # Sample storage
        self._samples: Deque[WindowedSample] = deque(maxlen=max_samples)

        # Precomputed aggregates for different windows
        self._window_caches: dict[str, tuple[datetime, AggregatedMetrics]] = {}

    def add_sample(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
        timestamp: datetime | None = None,
    ) -> None:
        """Add a sample to the window.

        Args:
            queue_metrics: Current queue metrics.
            worker_metrics: Current worker metrics.
            timestamp: Sample timestamp (defaults to now).
        """
        sample = WindowedSample(
            timestamp=timestamp or datetime.now(),
            queue_metrics=queue_metrics,
            worker_metrics=worker_metrics,
        )
        self._samples.append(sample)

        # Invalidate caches
        self._window_caches.clear()

        # Prune old samples
        self._prune_old_samples()

    def _prune_old_samples(self) -> None:
        """Remove samples outside the window."""
        cutoff = datetime.now() - self._window_duration

        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def get_window_aggregate(
        self,
        duration: timedelta | None = None,
    ) -> AggregatedMetrics:
        """Get aggregated metrics over the specified window.

        Args:
            duration: Window duration (defaults to full window).

        Returns:
            Aggregated metrics for the window.
        """
        duration = duration or self._window_duration
        cache_key = str(duration.total_seconds())

        # Check cache
        if cache_key in self._window_caches:
            cached_time, cached_result = self._window_caches[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 1.0:
                return cached_result

        cutoff = datetime.now() - duration
        samples_in_window = [s for s in self._samples if s.timestamp >= cutoff]

        if not samples_in_window:
            result = AggregatedMetrics(
                window_start=cutoff,
                window_end=datetime.now(),
                sample_count=0,
            )
            self._window_caches[cache_key] = (datetime.now(), result)
            return result

        # Aggregate queue metrics across samples
        all_queue_metrics = [
            qm
            for sample in samples_in_window
            for qm in sample.queue_metrics
        ]
        all_worker_metrics = [
            wm
            for sample in samples_in_window
            for wm in sample.worker_metrics
        ]

        # Compute aggregates per queue
        queue_aggregates = self._aggregate_by_queue(all_queue_metrics)
        worker_aggregates = self._aggregate_by_worker(all_worker_metrics)

        # Task summary
        task_summary = self._compute_task_summary(queue_aggregates)

        result = AggregatedMetrics(
            window_start=samples_in_window[0].timestamp,
            window_end=samples_in_window[-1].timestamp,
            sample_count=len(samples_in_window),
            queue_metrics=queue_aggregates,
            worker_metrics=worker_aggregates,
            task_summary=task_summary,
        )

        self._window_caches[cache_key] = (datetime.now(), result)
        return result

    def _aggregate_by_queue(
        self,
        metrics: list[QueueMetrics],
    ) -> list[QueueMetrics]:
        """Aggregate metrics by queue name."""
        by_queue: dict[str, list[QueueMetrics]] = {}

        for m in metrics:
            if m.queue_name not in by_queue:
                by_queue[m.queue_name] = []
            by_queue[m.queue_name].append(m)

        aggregates = []
        for queue_name, queue_metrics in by_queue.items():
            # Use the latest counts and average the rates
            latest = queue_metrics[-1]
            avg_wait = sum(m.avg_wait_time_ms for m in queue_metrics) / len(queue_metrics)
            avg_exec = sum(m.avg_execution_time_ms for m in queue_metrics) / len(queue_metrics)
            avg_throughput = sum(m.throughput_per_second for m in queue_metrics) / len(queue_metrics)

            aggregates.append(QueueMetrics(
                queue_name=queue_name,
                pending_count=latest.pending_count,
                running_count=latest.running_count,
                completed_count=latest.completed_count,
                failed_count=latest.failed_count,
                avg_wait_time_ms=avg_wait,
                avg_execution_time_ms=avg_exec,
                throughput_per_second=avg_throughput,
                timestamp=latest.timestamp,
                labels={
                    **latest.labels,
                    "sample_count": str(len(queue_metrics)),
                },
            ))

        return aggregates

    def _aggregate_by_worker(
        self,
        metrics: list[WorkerMetrics],
    ) -> list[WorkerMetrics]:
        """Aggregate metrics by worker ID."""
        by_worker: dict[str, list[WorkerMetrics]] = {}

        for m in metrics:
            if m.worker_id not in by_worker:
                by_worker[m.worker_id] = []
            by_worker[m.worker_id].append(m)

        aggregates = []
        for worker_id, worker_metrics in by_worker.items():
            # Use latest state, average the resource metrics
            latest = worker_metrics[-1]
            avg_cpu = sum(m.cpu_percent for m in worker_metrics) / len(worker_metrics)
            avg_memory = sum(m.memory_mb for m in worker_metrics) / len(worker_metrics)

            aggregates.append(WorkerMetrics(
                worker_id=worker_id,
                state=latest.state,
                current_tasks=latest.current_tasks,
                completed_tasks=latest.completed_tasks,
                failed_tasks=latest.failed_tasks,
                cpu_percent=avg_cpu,
                memory_mb=avg_memory,
                uptime_seconds=latest.uptime_seconds,
                last_heartbeat=latest.last_heartbeat,
                hostname=latest.hostname,
                max_concurrency=latest.max_concurrency,
                tags=latest.tags,
            ))

        return aggregates

    def _compute_task_summary(
        self,
        queue_metrics: list[QueueMetrics],
    ) -> dict[str, Any]:
        """Compute task summary from queue metrics."""
        total_pending = sum(q.pending_count for q in queue_metrics)
        total_running = sum(q.running_count for q in queue_metrics)
        total_completed = sum(q.completed_count for q in queue_metrics)
        total_failed = sum(q.failed_count for q in queue_metrics)
        total = total_completed + total_failed

        success_rate = total_completed / total if total > 0 else 1.0
        throughput = sum(q.throughput_per_second for q in queue_metrics)

        return {
            "total_pending": total_pending,
            "total_running": total_running,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "success_rate": success_rate,
            "throughput_per_second": throughput,
        }

    def get_moving_average(
        self,
        queue_name: str,
        metric: str = "throughput_per_second",
        windows: list[timedelta] | None = None,
    ) -> dict[str, float]:
        """Get moving averages for a queue metric.

        Args:
            queue_name: Queue name.
            metric: Metric to average.
            windows: Window durations (defaults to 1m, 5m, 15m).

        Returns:
            Dictionary of window duration -> average value.
        """
        windows = windows or [
            timedelta(minutes=1),
            timedelta(minutes=5),
            timedelta(minutes=15),
        ]

        results = {}
        now = datetime.now()

        for window in windows:
            cutoff = now - window
            samples = [
                s for s in self._samples
                if s.timestamp >= cutoff
            ]

            values = []
            for sample in samples:
                for qm in sample.queue_metrics:
                    if qm.queue_name == queue_name:
                        value = getattr(qm, metric, None)
                        if value is not None:
                            values.append(value)

            window_key = f"{int(window.total_seconds())}s"
            results[window_key] = sum(values) / len(values) if values else 0.0

        return results

    def detect_window_anomalies(
        self,
        current_metrics: list[QueueMetrics],
    ) -> list[MonitoringEvent]:
        """Detect anomalies by comparing current metrics to historical baseline.

        Args:
            current_metrics: Current queue metrics.

        Returns:
            List of anomaly events.
        """
        events = []

        # Get historical baseline from window
        aggregate = self.get_window_aggregate()
        if aggregate.sample_count < 5:
            return events  # Not enough history

        # Build baseline statistics
        for current in current_metrics:
            historical = [
                qm
                for qm in aggregate.queue_metrics
                if qm.queue_name == current.queue_name
            ]

            if not historical:
                continue

            baseline = historical[0]

            # Compare pending count
            if current.pending_count > baseline.pending_count * 2:
                events.append(MonitoringEvent(
                    event_type=MonitoringEventType.THRESHOLD_CROSSED,
                    source=self.name,
                    data={
                        "queue_name": current.queue_name,
                        "metric": "pending_count",
                        "current": current.pending_count,
                        "baseline": baseline.pending_count,
                        "ratio": current.pending_count / max(baseline.pending_count, 1),
                    },
                    severity=AlertSeverity.WARNING,
                    labels={"queue": current.queue_name, "anomaly": "pending_spike"},
                ))

            # Compare execution time
            if (
                current.avg_execution_time_ms > 0
                and baseline.avg_execution_time_ms > 0
                and current.avg_execution_time_ms > baseline.avg_execution_time_ms * 1.5
            ):
                events.append(MonitoringEvent(
                    event_type=MonitoringEventType.THRESHOLD_CROSSED,
                    source=self.name,
                    data={
                        "queue_name": current.queue_name,
                        "metric": "avg_execution_time_ms",
                        "current": current.avg_execution_time_ms,
                        "baseline": baseline.avg_execution_time_ms,
                        "ratio": current.avg_execution_time_ms / baseline.avg_execution_time_ms,
                    },
                    severity=AlertSeverity.WARNING,
                    labels={"queue": current.queue_name, "anomaly": "slowdown"},
                ))

        return events

    def get_trend(
        self,
        queue_name: str,
        metric: str = "throughput_per_second",
    ) -> dict[str, Any]:
        """Calculate trend for a queue metric.

        Args:
            queue_name: Queue name.
            metric: Metric to analyze.

        Returns:
            Trend information dictionary.
        """
        samples = list(self._samples)
        if len(samples) < 2:
            return {
                "direction": "stable",
                "slope": 0.0,
                "change_percent": 0.0,
            }

        # Extract values with timestamps
        data_points = []
        for sample in samples:
            for qm in sample.queue_metrics:
                if qm.queue_name == queue_name:
                    value = getattr(qm, metric, None)
                    if value is not None:
                        data_points.append((sample.timestamp, value))

        if len(data_points) < 2:
            return {
                "direction": "stable",
                "slope": 0.0,
                "change_percent": 0.0,
            }

        # Simple linear regression
        n = len(data_points)
        base_time = data_points[0][0]
        x_vals = [(t - base_time).total_seconds() for t, _ in data_points]
        y_vals = [v for _, v in data_points]

        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        slope = numerator / denominator if denominator != 0 else 0
        change_percent = ((y_vals[-1] - y_vals[0]) / y_vals[0] * 100) if y_vals[0] != 0 else 0

        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "direction": direction,
            "slope": slope,
            "change_percent": change_percent,
            "first_value": y_vals[0],
            "last_value": y_vals[-1],
            "sample_count": n,
        }

    @property
    def sample_count(self) -> int:
        """Get current sample count."""
        return len(self._samples)

    @property
    def window_coverage(self) -> float:
        """Get window coverage (0.0 to 1.0).

        Returns what fraction of the window has samples.
        """
        if not self._samples:
            return 0.0

        actual_duration = (
            self._samples[-1].timestamp - self._samples[0].timestamp
        ).total_seconds()
        expected_duration = self._window_duration.total_seconds()

        return min(actual_duration / expected_duration, 1.0)

    def clear(self) -> None:
        """Clear all samples and caches."""
        self._samples.clear()
        self._window_caches.clear()
