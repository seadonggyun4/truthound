"""Real-time metric aggregator.

Provides instant aggregation of metrics as they arrive,
without windowing or buffering.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from truthound.checkpoint.monitoring.protocols import (
    QueueMetrics,
    WorkerMetrics,
    MonitoringEvent,
    MonitoringEventType,
    AlertSeverity,
)
from truthound.checkpoint.monitoring.aggregators.base import BaseAggregator

logger = logging.getLogger(__name__)


class RealtimeAggregator(BaseAggregator):
    """Real-time metric aggregator.

    Provides instant aggregation without buffering, suitable for
    live dashboards and real-time monitoring.

    Features:
    - Instant aggregation of current metrics
    - Threshold-based alerting
    - Trend detection (comparing to previous values)

    Example:
        >>> aggregator = RealtimeAggregator()
        >>>
        >>> # Aggregate current queue metrics
        >>> summary = aggregator.aggregate_queues(queue_metrics)
        >>>
        >>> # Check thresholds and generate alerts
        >>> alerts = aggregator.check_thresholds(queue_metrics)
    """

    def __init__(
        self,
        name: str = "realtime",
        anomaly_threshold: float = 2.0,
        pending_alert_threshold: int = 100,
        success_rate_alert_threshold: float = 0.95,
        execution_time_alert_ms: float = 60000.0,
        load_factor_alert_threshold: float = 0.9,
    ) -> None:
        """Initialize real-time aggregator.

        Args:
            name: Aggregator name.
            anomaly_threshold: Z-score threshold for anomaly detection.
            pending_alert_threshold: Alert when pending count exceeds this.
            success_rate_alert_threshold: Alert when success rate drops below this.
            execution_time_alert_ms: Alert when avg execution time exceeds this.
            load_factor_alert_threshold: Alert when worker load factor exceeds this.
        """
        super().__init__(name=name, anomaly_threshold=anomaly_threshold)
        self._pending_threshold = pending_alert_threshold
        self._success_rate_threshold = success_rate_alert_threshold
        self._execution_time_threshold = execution_time_alert_ms
        self._load_factor_threshold = load_factor_alert_threshold

        # Track previous values for trend detection
        self._previous_queue_metrics: dict[str, QueueMetrics] = {}
        self._previous_worker_metrics: dict[str, WorkerMetrics] = {}

    def check_thresholds(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics] | None = None,
    ) -> list[MonitoringEvent]:
        """Check metrics against configured thresholds.

        Args:
            queue_metrics: Current queue metrics.
            worker_metrics: Current worker metrics (optional).

        Returns:
            List of alert events for threshold violations.
        """
        alerts = []

        # Check queue thresholds
        for qm in queue_metrics:
            # Check pending count
            if qm.pending_count > self._pending_threshold:
                alerts.append(self._create_alert(
                    "queue_pending_high",
                    f"Queue '{qm.queue_name}' has {qm.pending_count} pending tasks",
                    {"queue_name": qm.queue_name, "pending_count": qm.pending_count},
                    AlertSeverity.WARNING if qm.pending_count < self._pending_threshold * 2 else AlertSeverity.ERROR,
                ))

            # Check success rate
            if qm.success_rate < self._success_rate_threshold:
                alerts.append(self._create_alert(
                    "queue_success_rate_low",
                    f"Queue '{qm.queue_name}' success rate is {qm.success_rate:.1%}",
                    {"queue_name": qm.queue_name, "success_rate": qm.success_rate},
                    AlertSeverity.WARNING if qm.success_rate > 0.8 else AlertSeverity.ERROR,
                ))

            # Check execution time
            if qm.avg_execution_time_ms > self._execution_time_threshold:
                alerts.append(self._create_alert(
                    "queue_execution_time_high",
                    f"Queue '{qm.queue_name}' avg execution time is {qm.avg_execution_time_ms:.0f}ms",
                    {"queue_name": qm.queue_name, "avg_execution_time_ms": qm.avg_execution_time_ms},
                    AlertSeverity.WARNING,
                ))

        # Check worker thresholds
        if worker_metrics:
            for wm in worker_metrics:
                # Check load factor
                if wm.load_factor > self._load_factor_threshold:
                    alerts.append(self._create_alert(
                        "worker_overloaded",
                        f"Worker '{wm.worker_id}' load factor is {wm.load_factor:.1%}",
                        {"worker_id": wm.worker_id, "load_factor": wm.load_factor},
                        AlertSeverity.WARNING,
                    ))

                # Check worker state
                if wm.state != "online":
                    alerts.append(self._create_alert(
                        "worker_not_online",
                        f"Worker '{wm.worker_id}' is {wm.state}",
                        {"worker_id": wm.worker_id, "state": wm.state},
                        AlertSeverity.WARNING if wm.state == "draining" else AlertSeverity.ERROR,
                    ))

        return alerts

    def _create_alert(
        self,
        alert_name: str,
        message: str,
        data: dict[str, Any],
        severity: AlertSeverity,
    ) -> MonitoringEvent:
        """Create an alert event."""
        return MonitoringEvent(
            event_type=MonitoringEventType.ALERT_TRIGGERED,
            source=self.name,
            data={
                "alert_name": alert_name,
                "message": message,
                **data,
            },
            severity=severity,
            labels={"alert": alert_name},
        )

    def detect_trends(
        self,
        queue_metrics: list[QueueMetrics],
    ) -> list[MonitoringEvent]:
        """Detect trends by comparing to previous metrics.

        Args:
            queue_metrics: Current queue metrics.

        Returns:
            List of trend events.
        """
        events = []

        for qm in queue_metrics:
            previous = self._previous_queue_metrics.get(qm.queue_name)
            if previous is None:
                self._previous_queue_metrics[qm.queue_name] = qm
                continue

            # Calculate time difference
            time_diff = (qm.timestamp - previous.timestamp).total_seconds()
            if time_diff <= 0:
                continue

            # Check pending count trend
            pending_rate = self.calculate_rate(
                qm.pending_count,
                previous.pending_count,
                time_diff,
            )

            if pending_rate > 10:  # Growing by more than 10/second
                events.append(MonitoringEvent(
                    event_type=MonitoringEventType.QUEUE_THRESHOLD_EXCEEDED,
                    source=self.name,
                    data={
                        "queue_name": qm.queue_name,
                        "trend": "growing",
                        "rate_per_second": pending_rate,
                        "current_pending": qm.pending_count,
                        "previous_pending": previous.pending_count,
                    },
                    severity=AlertSeverity.WARNING,
                    labels={"queue": qm.queue_name, "trend": "growing"},
                ))

            # Check success rate trend
            rate_change = qm.success_rate - previous.success_rate
            if rate_change < -0.05:  # Dropped by more than 5%
                events.append(MonitoringEvent(
                    event_type=MonitoringEventType.THRESHOLD_CROSSED,
                    source=self.name,
                    data={
                        "queue_name": qm.queue_name,
                        "trend": "declining",
                        "metric": "success_rate",
                        "current": qm.success_rate,
                        "previous": previous.success_rate,
                        "change": rate_change,
                    },
                    severity=AlertSeverity.WARNING,
                    labels={"queue": qm.queue_name, "trend": "declining"},
                ))

            # Update previous
            self._previous_queue_metrics[qm.queue_name] = qm

        return events

    def compute_realtime_summary(
        self,
        queue_metrics: list[QueueMetrics],
        worker_metrics: list[WorkerMetrics],
    ) -> dict[str, Any]:
        """Compute a real-time summary of the system.

        Args:
            queue_metrics: Current queue metrics.
            worker_metrics: Current worker metrics.

        Returns:
            Summary dictionary suitable for dashboards.
        """
        queue_summary = self.aggregate_queues(queue_metrics)
        worker_summary = self.aggregate_workers(worker_metrics)

        # Calculate system health score (0-100)
        health_factors = []

        # Queue health: based on pending count and success rate
        if queue_metrics:
            avg_success_rate = sum(q.success_rate for q in queue_metrics) / len(queue_metrics)
            max_pending = max(q.pending_count for q in queue_metrics)
            queue_health = avg_success_rate * 100 * (1 - min(max_pending / 1000, 1) * 0.5)
            health_factors.append(queue_health)

        # Worker health: based on availability and load
        if worker_metrics:
            online_ratio = len([w for w in worker_metrics if w.state == "online"]) / len(worker_metrics)
            avg_load = sum(w.load_factor for w in worker_metrics) / len(worker_metrics)
            worker_health = online_ratio * 100 * (1 - avg_load * 0.3)
            health_factors.append(worker_health)

        overall_health = sum(health_factors) / len(health_factors) if health_factors else 100

        return {
            "timestamp": datetime.now().isoformat(),
            "health_score": round(overall_health, 1),
            "health_status": self._health_score_to_status(overall_health),
            "queues": {
                "total": len(queue_metrics),
                "healthy": len([q for q in queue_metrics if q.is_healthy]),
                "total_pending": queue_summary.pending_count,
                "total_running": queue_summary.running_count,
                "total_completed": queue_summary.completed_count,
                "total_failed": queue_summary.failed_count,
                "overall_success_rate": queue_summary.success_rate,
                "total_throughput": queue_summary.throughput_per_second,
            },
            "workers": worker_summary,
        }

    def _health_score_to_status(self, score: float) -> str:
        """Convert health score to status string."""
        if score >= 95:
            return "healthy"
        elif score >= 80:
            return "degraded"
        elif score >= 50:
            return "warning"
        else:
            return "critical"

    def clear_history(self) -> None:
        """Clear tracked previous metrics."""
        self._previous_queue_metrics.clear()
        self._previous_worker_metrics.clear()
