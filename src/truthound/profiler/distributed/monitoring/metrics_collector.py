"""Metrics collector for distributed monitoring.

This module provides metrics collection and aggregation for distributed profiling.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from truthound.profiler.distributed.monitoring.config import MetricsConfig
from truthound.profiler.distributed.monitoring.protocols import (
    EventSeverity,
    IMetricsCollector,
    MonitorEvent,
    MonitorEventType,
    MonitorMetrics,
)


logger = logging.getLogger(__name__)


@dataclass
class HistogramBucket:
    """Histogram bucket for duration distribution."""

    le: float  # Less than or equal to this value
    count: int = 0


class Histogram:
    """Simple histogram implementation for duration tracking.

    Provides count, sum, and bucket distributions for latency metrics.
    """

    def __init__(self, buckets: list[float] | None = None) -> None:
        """Initialize histogram.

        Args:
            buckets: Bucket boundaries (upper bounds)
        """
        if buckets is None:
            buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")]
        else:
            buckets = sorted(buckets + [float("inf")])

        self._buckets = [HistogramBucket(le=b) for b in buckets]
        self._count = 0
        self._sum = 0.0
        self._values: deque[float] = deque(maxlen=10000)  # For percentile calculation

    def observe(self, value: float) -> None:
        """Record an observation.

        Args:
            value: Value to record
        """
        self._count += 1
        self._sum += value
        self._values.append(value)

        for bucket in self._buckets:
            if value <= bucket.le:
                bucket.count += 1

    def get_count(self) -> int:
        """Get total count."""
        return self._count

    def get_sum(self) -> float:
        """Get total sum."""
        return self._sum

    def get_avg(self) -> float:
        """Get average value."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def get_percentile(self, p: float) -> float:
        """Get percentile value.

        Args:
            p: Percentile (0.0 to 1.0)

        Returns:
            Percentile value
        """
        if not self._values:
            return 0.0

        sorted_values = sorted(self._values)
        idx = int(math.ceil(p * len(sorted_values))) - 1
        idx = max(0, min(idx, len(sorted_values) - 1))
        return sorted_values[idx]

    def get_buckets(self) -> list[tuple[float, int]]:
        """Get bucket counts.

        Returns:
            List of (upper_bound, cumulative_count) tuples
        """
        return [(b.le, b.count) for b in self._buckets]

    def reset(self) -> None:
        """Reset histogram."""
        for bucket in self._buckets:
            bucket.count = 0
        self._count = 0
        self._sum = 0.0
        self._values.clear()


class Counter:
    """Thread-safe counter."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, value: int = 1) -> None:
        """Increment counter."""
        with self._lock:
            self._value += value

    def get(self) -> int:
        """Get counter value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset counter."""
        with self._lock:
            self._value = 0


class Gauge:
    """Thread-safe gauge."""

    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Get gauge value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset gauge."""
        with self._lock:
            self._value = 0.0


class DistributedMetricsCollector(IMetricsCollector):
    """Collects and aggregates metrics for distributed profiling.

    Thread-safe implementation that tracks task metrics, throughput,
    and resource usage across distributed workers.

    Example:
        collector = DistributedMetricsCollector(
            config=MetricsConfig(enable_percentiles=True),
            on_event=lambda e: print(e.message),
        )

        collector.record_task_submitted()
        collector.record_task_started(wait_time_seconds=0.1)
        collector.record_task_completed(duration_seconds=2.5, rows_processed=1000)

        metrics = collector.get_metrics()
        print(f"Completed: {metrics.tasks_completed}, Avg: {metrics.avg_task_duration_seconds:.2f}s")
    """

    def __init__(
        self,
        config: MetricsConfig | None = None,
        on_event: Callable[[MonitorEvent], None] | None = None,
    ) -> None:
        """Initialize metrics collector.

        Args:
            config: Metrics configuration
            on_event: Callback for metrics events
        """
        self._config = config or MetricsConfig()
        self._on_event = on_event
        self._lock = threading.RLock()

        # Task counters
        self._tasks_submitted = Counter()
        self._tasks_started = Counter()
        self._tasks_completed = Counter()
        self._tasks_failed = Counter()
        self._tasks_retried = Counter()

        # Active tasks gauge
        self._tasks_running = Gauge()
        self._tasks_pending = Gauge()

        # Duration histogram
        self._task_duration = Histogram(self._config.histogram_buckets)
        self._wait_time = Histogram(self._config.histogram_buckets)

        # Throughput tracking
        self._rows_processed = Counter()
        self._throughput_samples: deque[tuple[float, int]] = deque(maxlen=100)
        self._last_throughput_time = time.time()
        self._last_rows_count = 0

        # Resource gauges
        self._memory_used_mb = Gauge()
        self._cpu_utilization = Gauge()

        # Worker tracking
        self._workers_total = Gauge()
        self._workers_healthy = Gauge()

        # Start time for elapsed calculation
        self._start_time: datetime | None = None

    def start(self) -> None:
        """Start metrics collection."""
        self._start_time = datetime.now()

    def record_task_submitted(self) -> None:
        """Record task submission."""
        self._tasks_submitted.inc()
        self._tasks_pending.inc()

    def record_task_started(self, wait_time_seconds: float = 0.0) -> None:
        """Record task start.

        Args:
            wait_time_seconds: Time spent waiting
        """
        self._tasks_started.inc()
        self._tasks_pending.dec()
        self._tasks_running.inc()

        if wait_time_seconds > 0:
            self._wait_time.observe(wait_time_seconds)

    def record_task_completed(
        self,
        duration_seconds: float,
        rows_processed: int = 0,
    ) -> None:
        """Record task completion.

        Args:
            duration_seconds: Task duration
            rows_processed: Rows processed
        """
        self._tasks_completed.inc()
        self._tasks_running.dec()
        self._task_duration.observe(duration_seconds)

        if rows_processed > 0:
            self._rows_processed.inc(rows_processed)
            self._record_throughput_sample(rows_processed)

    def record_task_failed(self, duration_seconds: float = 0.0) -> None:
        """Record task failure.

        Args:
            duration_seconds: Time until failure
        """
        self._tasks_failed.inc()
        self._tasks_running.dec()

        if duration_seconds > 0:
            self._task_duration.observe(duration_seconds)

    def record_task_retried(self) -> None:
        """Record task retry."""
        self._tasks_retried.inc()

    def record_rows_processed(self, count: int) -> None:
        """Record rows processed.

        Args:
            count: Number of rows
        """
        self._rows_processed.inc(count)
        self._record_throughput_sample(count)

    def record_worker_count(self, total: int, healthy: int) -> None:
        """Record worker counts.

        Args:
            total: Total workers
            healthy: Healthy workers
        """
        self._workers_total.set(total)
        self._workers_healthy.set(healthy)

    def record_resource_usage(
        self,
        memory_mb: float,
        cpu_percent: float,
    ) -> None:
        """Record resource usage.

        Args:
            memory_mb: Memory used in MB
            cpu_percent: CPU utilization percent
        """
        self._memory_used_mb.set(memory_mb)
        self._cpu_utilization.set(cpu_percent)

    def get_metrics(self) -> MonitorMetrics:
        """Get current metrics.

        Returns:
            Current metrics snapshot
        """
        with self._lock:
            completed = self._tasks_completed.get()
            failed = self._tasks_failed.get()
            total = completed + failed

            # Calculate rates
            error_rate = failed / total if total > 0 else 0.0
            retry_rate = (
                self._tasks_retried.get() / self._tasks_submitted.get()
                if self._tasks_submitted.get() > 0
                else 0.0
            )

            # Get percentiles if enabled
            p50 = 0.0
            p95 = 0.0
            p99 = 0.0
            if self._config.enable_percentiles and self._task_duration.get_count() > 0:
                p50 = self._task_duration.get_percentile(0.50)
                p95 = self._task_duration.get_percentile(0.95)
                p99 = self._task_duration.get_percentile(0.99)

            return MonitorMetrics(
                timestamp=datetime.now(),
                tasks_total=self._tasks_submitted.get(),
                tasks_completed=completed,
                tasks_failed=failed,
                tasks_running=int(self._tasks_running.get()),
                tasks_pending=int(self._tasks_pending.get()),
                avg_task_duration_seconds=self._task_duration.get_avg(),
                p50_task_duration_seconds=p50,
                p95_task_duration_seconds=p95,
                p99_task_duration_seconds=p99,
                total_rows_processed=self._rows_processed.get(),
                rows_per_second=self._calculate_throughput(),
                workers_total=int(self._workers_total.get()),
                workers_healthy=int(self._workers_healthy.get()),
                workers_unhealthy=int(self._workers_total.get() - self._workers_healthy.get()),
                memory_used_total_mb=self._memory_used_mb.get(),
                cpu_utilization_percent=self._cpu_utilization.get(),
                error_rate=error_rate,
                retry_rate=retry_rate,
            )

    def get_task_duration_histogram(self) -> list[tuple[float, int]]:
        """Get task duration histogram buckets.

        Returns:
            List of (upper_bound, count) tuples
        """
        return self._task_duration.get_buckets()

    def get_wait_time_histogram(self) -> list[tuple[float, int]]:
        """Get wait time histogram buckets.

        Returns:
            List of (upper_bound, count) tuples
        """
        return self._wait_time.get_buckets()

    def get_elapsed_seconds(self) -> float:
        """Get elapsed time since start.

        Returns:
            Elapsed seconds
        """
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def reset(self) -> None:
        """Reset metrics."""
        with self._lock:
            self._tasks_submitted.reset()
            self._tasks_started.reset()
            self._tasks_completed.reset()
            self._tasks_failed.reset()
            self._tasks_retried.reset()
            self._tasks_running.reset()
            self._tasks_pending.reset()
            self._task_duration.reset()
            self._wait_time.reset()
            self._rows_processed.reset()
            self._throughput_samples.clear()
            self._memory_used_mb.reset()
            self._cpu_utilization.reset()
            self._workers_total.reset()
            self._workers_healthy.reset()
            self._start_time = None

    def emit_metrics_snapshot(self) -> None:
        """Emit current metrics as an event."""
        metrics = self.get_metrics()

        self._emit_event(
            MonitorEventType.METRICS_SNAPSHOT,
            f"Metrics snapshot: {metrics.tasks_completed}/{metrics.tasks_total} tasks, "
            f"{metrics.rows_per_second:.1f} rows/s",
            metadata=metrics.to_dict(),
        )

    def _record_throughput_sample(self, rows: int) -> None:
        """Record a throughput sample.

        Args:
            rows: Rows processed
        """
        now = time.time()
        self._throughput_samples.append((now, rows))

    def _calculate_throughput(self) -> float:
        """Calculate current throughput.

        Returns:
            Rows per second
        """
        if not self._throughput_samples:
            return 0.0

        now = time.time()
        window = 60.0  # 60 second window

        # Filter samples in window
        cutoff = now - window
        recent = [(t, r) for t, r in self._throughput_samples if t > cutoff]

        if not recent or len(recent) < 2:
            return 0.0

        # Calculate throughput over window
        total_rows = sum(r for _, r in recent)
        time_span = now - recent[0][0]

        if time_span <= 0:
            return 0.0

        return total_rows / time_span

    def _emit_event(
        self,
        event_type: MonitorEventType,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a monitoring event.

        Args:
            event_type: Event type
            message: Event message
            severity: Event severity
            metadata: Additional metadata
        """
        if self._on_event is None:
            return

        event = MonitorEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            metadata=metadata or {},
        )

        try:
            self._on_event(event)
        except Exception as e:
            logger.warning(f"Error in metrics collector event callback: {e}")


class PrometheusExporter:
    """Export metrics in Prometheus format.

    Generates Prometheus-compatible metrics output for scraping.
    """

    def __init__(self, collector: DistributedMetricsCollector, prefix: str = "truthound") -> None:
        """Initialize exporter.

        Args:
            collector: Metrics collector to export from
            prefix: Metric name prefix
        """
        self._collector = collector
        self._prefix = prefix

    def export(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus metrics text
        """
        metrics = self._collector.get_metrics()
        lines = []

        # Task counters
        lines.append(f"# HELP {self._prefix}_tasks_total Total tasks submitted")
        lines.append(f"# TYPE {self._prefix}_tasks_total counter")
        lines.append(f'{self._prefix}_tasks_total {metrics.tasks_total}')

        lines.append(f"# HELP {self._prefix}_tasks_completed_total Total tasks completed")
        lines.append(f"# TYPE {self._prefix}_tasks_completed_total counter")
        lines.append(f'{self._prefix}_tasks_completed_total {metrics.tasks_completed}')

        lines.append(f"# HELP {self._prefix}_tasks_failed_total Total tasks failed")
        lines.append(f"# TYPE {self._prefix}_tasks_failed_total counter")
        lines.append(f'{self._prefix}_tasks_failed_total {metrics.tasks_failed}')

        # Active tasks gauge
        lines.append(f"# HELP {self._prefix}_tasks_running Current running tasks")
        lines.append(f"# TYPE {self._prefix}_tasks_running gauge")
        lines.append(f'{self._prefix}_tasks_running {metrics.tasks_running}')

        lines.append(f"# HELP {self._prefix}_tasks_pending Current pending tasks")
        lines.append(f"# TYPE {self._prefix}_tasks_pending gauge")
        lines.append(f'{self._prefix}_tasks_pending {metrics.tasks_pending}')

        # Duration histogram
        lines.append(f"# HELP {self._prefix}_task_duration_seconds Task duration histogram")
        lines.append(f"# TYPE {self._prefix}_task_duration_seconds histogram")
        for le, count in self._collector.get_task_duration_histogram():
            le_str = "+Inf" if le == float("inf") else str(le)
            lines.append(f'{self._prefix}_task_duration_seconds_bucket{{le="{le_str}"}} {count}')
        lines.append(f'{self._prefix}_task_duration_seconds_count {metrics.tasks_completed}')
        lines.append(f'{self._prefix}_task_duration_seconds_sum {metrics.avg_task_duration_seconds * metrics.tasks_completed}')

        # Throughput
        lines.append(f"# HELP {self._prefix}_rows_processed_total Total rows processed")
        lines.append(f"# TYPE {self._prefix}_rows_processed_total counter")
        lines.append(f'{self._prefix}_rows_processed_total {metrics.total_rows_processed}')

        lines.append(f"# HELP {self._prefix}_rows_per_second Current throughput")
        lines.append(f"# TYPE {self._prefix}_rows_per_second gauge")
        lines.append(f'{self._prefix}_rows_per_second {metrics.rows_per_second:.2f}')

        # Workers
        lines.append(f"# HELP {self._prefix}_workers_total Total workers")
        lines.append(f"# TYPE {self._prefix}_workers_total gauge")
        lines.append(f'{self._prefix}_workers_total {metrics.workers_total}')

        lines.append(f"# HELP {self._prefix}_workers_healthy Healthy workers")
        lines.append(f"# TYPE {self._prefix}_workers_healthy gauge")
        lines.append(f'{self._prefix}_workers_healthy {metrics.workers_healthy}')

        # Error rate
        lines.append(f"# HELP {self._prefix}_error_rate Current error rate")
        lines.append(f"# TYPE {self._prefix}_error_rate gauge")
        lines.append(f'{self._prefix}_error_rate {metrics.error_rate:.4f}')

        return "\n".join(lines)
