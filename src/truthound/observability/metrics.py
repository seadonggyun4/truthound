"""Metrics collection system for Truthound.

This module provides a flexible metrics collection framework with support
for multiple backend exporters (Prometheus, StatsD, OpenTelemetry).

Metric Types:
    - Counter: Monotonically increasing value (e.g., request count)
    - Gauge: Point-in-time value (e.g., active connections)
    - Histogram: Distribution of values with buckets
    - Summary: Distribution with quantiles

Design Principles:
    1. Label-based: Dimensional metrics with key-value labels
    2. Backend agnostic: Same API for all exporters
    3. Lazy registration: Metrics created on first use
    4. Thread-safe: All operations are thread-safe
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator, TypeVar


# =============================================================================
# Metric Types
# =============================================================================


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass(frozen=True)
class MetricKey:
    """Unique identifier for a metric with labels."""

    name: str
    labels: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, name: str, labels: dict[str, str] | None = None) -> "MetricKey":
        """Create metric key from name and labels."""
        label_tuple = tuple(sorted((labels or {}).items()))
        return cls(name=name, labels=label_tuple)


@dataclass
class MetricMetadata:
    """Metadata about a metric."""

    name: str
    type: MetricType
    description: str
    unit: str = ""
    label_names: tuple[str, ...] = ()


# =============================================================================
# Metric Base Class
# =============================================================================


class Metric(ABC):
    """Abstract base class for metrics.

    All metrics support labels (dimensions) for multi-dimensional analysis.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        unit: str = "",
        label_names: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Initialize metric.

        Args:
            name: Metric name (should be lowercase with underscores).
            description: Human-readable description.
            unit: Unit of measurement.
            label_names: Names of labels for this metric.
        """
        self._name = name
        self._description = description
        self._unit = unit
        self._label_names = tuple(label_names)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get metric name."""
        return self._name

    @property
    def description(self) -> str:
        """Get metric description."""
        return self._description

    @property
    @abstractmethod
    def type(self) -> MetricType:
        """Get metric type."""
        pass

    @property
    def metadata(self) -> MetricMetadata:
        """Get metric metadata."""
        return MetricMetadata(
            name=self._name,
            type=self.type,
            description=self._description,
            unit=self._unit,
            label_names=self._label_names,
        )

    def _validate_labels(self, labels: dict[str, str]) -> None:
        """Validate label names match expected."""
        if set(labels.keys()) != set(self._label_names):
            expected = set(self._label_names)
            actual = set(labels.keys())
            raise ValueError(
                f"Label mismatch for '{self._name}': "
                f"expected {expected}, got {actual}"
            )

    @abstractmethod
    def collect(self) -> list[tuple[dict[str, str], float]]:
        """Collect all metric values with labels.

        Returns:
            List of (labels, value) tuples.
        """
        pass


# =============================================================================
# Counter
# =============================================================================


class Counter(Metric):
    """Monotonically increasing counter.

    Counters only go up (and reset to zero on restart).
    Use for: request counts, errors, completed tasks.

    Example:
        >>> requests = Counter("http_requests_total", "Total HTTP requests")
        >>> requests.inc()
        >>> requests.add(5)
        >>> requests.labels(method="GET", status="200").inc()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        labels: tuple[str, ...] | list[str] = (),
        **kwargs: Any,
    ) -> None:
        """Initialize counter.

        Args:
            name: Counter name.
            description: Human-readable description.
            labels: Label names for this counter.
            **kwargs: Additional arguments for Metric.
        """
        super().__init__(name, description, label_names=labels, **kwargs)
        self._values: dict[MetricKey, float] = {}

    @property
    def type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment counter.

        Args:
            value: Amount to increment (must be positive).
            **labels: Label values.
        """
        if value < 0:
            raise ValueError("Counter can only be incremented")
        self.add(value, **labels)

    def add(self, value: float, **labels: str) -> None:
        """Add to counter.

        Args:
            value: Amount to add (must be positive).
            **labels: Label values.
        """
        if value < 0:
            raise ValueError("Counter can only increase")

        if self._label_names:
            self._validate_labels(labels)

        key = MetricKey.create(self._name, labels)

        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def labels(self, **labels: str) -> "LabeledCounter":
        """Get counter with specific labels.

        Args:
            **labels: Label values.

        Returns:
            LabeledCounter for the specific label set.
        """
        return LabeledCounter(self, labels)

    def get(self, **labels: str) -> float:
        """Get current counter value.

        Args:
            **labels: Label values.

        Returns:
            Current value.
        """
        key = MetricKey.create(self._name, labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[tuple[dict[str, str], float]]:
        """Collect all counter values."""
        with self._lock:
            return [
                (dict(key.labels), value)
                for key, value in self._values.items()
            ]


class LabeledCounter:
    """Counter with pre-set labels."""

    def __init__(self, counter: Counter, labels: dict[str, str]) -> None:
        self._counter = counter
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        self._counter.inc(value, **self._labels)

    def add(self, value: float) -> None:
        """Add to counter."""
        self._counter.add(value, **self._labels)


# =============================================================================
# Gauge
# =============================================================================


class Gauge(Metric):
    """Point-in-time value that can go up or down.

    Use for: temperature, queue size, memory usage.

    Example:
        >>> temperature = Gauge("temperature_celsius", "Current temperature")
        >>> temperature.set(23.5)
        >>> temperature.inc()
        >>> temperature.dec(0.5)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        labels: tuple[str, ...] | list[str] = (),
        **kwargs: Any,
    ) -> None:
        """Initialize gauge."""
        super().__init__(name, description, label_names=labels, **kwargs)
        self._values: dict[MetricKey, float] = {}

    @property
    def type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float, **labels: str) -> None:
        """Set gauge value.

        Args:
            value: New value.
            **labels: Label values.
        """
        if self._label_names:
            self._validate_labels(labels)

        key = MetricKey.create(self._name, labels)

        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment gauge.

        Args:
            value: Amount to increment.
            **labels: Label values.
        """
        key = MetricKey.create(self._name, labels if self._label_names else {})

        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        """Decrement gauge.

        Args:
            value: Amount to decrement.
            **labels: Label values.
        """
        self.inc(-value, **labels)

    def get(self, **labels: str) -> float:
        """Get current gauge value."""
        key = MetricKey.create(self._name, labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def labels(self, **labels: str) -> "LabeledGauge":
        """Get gauge with specific labels."""
        return LabeledGauge(self, labels)

    @contextmanager
    def track_inprogress(self, **labels: str) -> Iterator[None]:
        """Track in-progress operations.

        Increments on entry, decrements on exit.
        """
        self.inc(**labels)
        try:
            yield
        finally:
            self.dec(**labels)

    def set_to_current_time(self, **labels: str) -> None:
        """Set gauge to current Unix timestamp."""
        self.set(time.time(), **labels)

    def collect(self) -> list[tuple[dict[str, str], float]]:
        """Collect all gauge values."""
        with self._lock:
            return [
                (dict(key.labels), value)
                for key, value in self._values.items()
            ]


class LabeledGauge:
    """Gauge with pre-set labels."""

    def __init__(self, gauge: Gauge, labels: dict[str, str]) -> None:
        self._gauge = gauge
        self._labels = labels

    def set(self, value: float) -> None:
        """Set gauge value."""
        self._gauge.set(value, **self._labels)

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        self._gauge.inc(value, **self._labels)

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        self._gauge.dec(value, **self._labels)


# =============================================================================
# Histogram
# =============================================================================


class Histogram(Metric):
    """Distribution of values with configurable buckets.

    Use for: request latency, response sizes.

    Example:
        >>> latency = Histogram(
        ...     "request_duration_seconds",
        ...     "Request latency",
        ...     buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        ... )
        >>> latency.observe(0.42)
        >>> with latency.time():
        ...     process_request()
    """

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
        0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        buckets: tuple[float, ...] | list[float] | None = None,
        labels: tuple[str, ...] | list[str] = (),
        **kwargs: Any,
    ) -> None:
        """Initialize histogram.

        Args:
            name: Histogram name.
            description: Human-readable description.
            buckets: Upper bounds for buckets.
            labels: Label names.
            **kwargs: Additional arguments.
        """
        super().__init__(name, description, label_names=labels, **kwargs)
        self._buckets = tuple(sorted(buckets or self.DEFAULT_BUCKETS))

        # Ensure +Inf is included
        if self._buckets[-1] != float("inf"):
            self._buckets = (*self._buckets, float("inf"))

        # Storage: key -> (bucket_counts, sum, count)
        self._data: dict[MetricKey, tuple[list[int], float, int]] = {}

    @property
    def type(self) -> MetricType:
        return MetricType.HISTOGRAM

    @property
    def buckets(self) -> tuple[float, ...]:
        """Get bucket boundaries."""
        return self._buckets

    def _get_data(self, key: MetricKey) -> tuple[list[int], float, int]:
        """Get or create data for a key."""
        if key not in self._data:
            self._data[key] = ([0] * len(self._buckets), 0.0, 0)
        return self._data[key]

    def observe(self, value: float, **labels: str) -> None:
        """Observe a value.

        Args:
            value: Observed value.
            **labels: Label values.
        """
        if self._label_names:
            self._validate_labels(labels)

        key = MetricKey.create(self._name, labels)

        with self._lock:
            bucket_counts, total_sum, count = self._get_data(key)

            # Find the bucket this value belongs to (first bucket where value <= bound)
            # Only increment that one bucket - cumulative is computed in collect()
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    bucket_counts[i] += 1
                    break

            # Update sum and count
            self._data[key] = (bucket_counts, total_sum + value, count + 1)

    @contextmanager
    def time(self, **labels: str) -> Iterator[None]:
        """Context manager to measure duration.

        Args:
            **labels: Label values.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, **labels)

    def labels(self, **labels: str) -> "LabeledHistogram":
        """Get histogram with specific labels."""
        return LabeledHistogram(self, labels)

    def collect(self) -> list[tuple[dict[str, str], dict[str, Any]]]:
        """Collect histogram data.

        Returns list of (labels, data) where data contains:
        - buckets: dict mapping bound to cumulative count
        - sum: total of all observed values
        - count: number of observations
        """
        with self._lock:
            results = []
            for key, (bucket_counts, total_sum, count) in self._data.items():
                # Calculate cumulative counts
                cumulative = []
                running = 0
                for c in bucket_counts:
                    running += c
                    cumulative.append(running)

                data = {
                    "buckets": {
                        str(bound): cum
                        for bound, cum in zip(self._buckets, cumulative)
                    },
                    "sum": total_sum,
                    "count": count,
                }
                results.append((dict(key.labels), data))
            return results


class LabeledHistogram:
    """Histogram with pre-set labels."""

    def __init__(self, histogram: Histogram, labels: dict[str, str]) -> None:
        self._histogram = histogram
        self._labels = labels

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._histogram.observe(value, **self._labels)

    @contextmanager
    def time(self) -> Iterator[None]:
        """Time a block of code."""
        with self._histogram.time(**self._labels):
            yield


# =============================================================================
# Summary
# =============================================================================


class Summary(Metric):
    """Summary with streaming quantiles.

    Similar to Histogram but calculates quantiles on the client side.
    Use when you need specific percentiles (p50, p90, p99).

    Note: This is a simplified implementation that keeps recent samples.
    For production use, consider a proper streaming quantile algorithm.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
        max_samples: int = 1000,
        labels: tuple[str, ...] | list[str] = (),
        **kwargs: Any,
    ) -> None:
        """Initialize summary.

        Args:
            name: Summary name.
            description: Human-readable description.
            quantiles: Quantiles to calculate (0.0-1.0).
            max_samples: Maximum samples to keep.
            labels: Label names.
            **kwargs: Additional arguments.
        """
        super().__init__(name, description, label_names=labels, **kwargs)
        self._quantiles = quantiles
        self._max_samples = max_samples
        self._data: dict[MetricKey, tuple[list[float], float, int]] = {}

    @property
    def type(self) -> MetricType:
        return MetricType.SUMMARY

    def observe(self, value: float, **labels: str) -> None:
        """Observe a value.

        Args:
            value: Observed value.
            **labels: Label values.
        """
        if self._label_names:
            self._validate_labels(labels)

        key = MetricKey.create(self._name, labels)

        with self._lock:
            if key not in self._data:
                self._data[key] = ([], 0.0, 0)

            samples, total_sum, count = self._data[key]

            # Add sample (circular buffer behavior)
            if len(samples) >= self._max_samples:
                samples.pop(0)
            samples.append(value)

            self._data[key] = (samples, total_sum + value, count + 1)

    @contextmanager
    def time(self, **labels: str) -> Iterator[None]:
        """Time a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start, **labels)

    def labels(self, **labels: str) -> "LabeledSummary":
        """Get summary with specific labels."""
        return LabeledSummary(self, labels)

    def collect(self) -> list[tuple[dict[str, str], dict[str, Any]]]:
        """Collect summary data with quantiles."""
        with self._lock:
            results = []
            for key, (samples, total_sum, count) in self._data.items():
                quantile_values = {}
                if samples:
                    sorted_samples = sorted(samples)
                    n = len(sorted_samples)
                    for q in self._quantiles:
                        idx = int(q * (n - 1))
                        quantile_values[f"p{int(q*100)}"] = sorted_samples[idx]

                data = {
                    "quantiles": quantile_values,
                    "sum": total_sum,
                    "count": count,
                }
                results.append((dict(key.labels), data))
            return results


class LabeledSummary:
    """Summary with pre-set labels."""

    def __init__(self, summary: Summary, labels: dict[str, str]) -> None:
        self._summary = summary
        self._labels = labels

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._summary.observe(value, **self._labels)


# =============================================================================
# Metrics Registry
# =============================================================================


class MetricsRegistry:
    """Central registry for all metrics.

    Ensures unique metric names and provides collection.
    """

    def __init__(self) -> None:
        """Initialize registry."""
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def register(self, metric: Metric) -> Metric:
        """Register a metric.

        Args:
            metric: Metric to register.

        Returns:
            The registered metric.

        Raises:
            ValueError: If metric name already registered with different type.
        """
        with self._lock:
            if metric.name in self._metrics:
                existing = self._metrics[metric.name]
                if existing.type != metric.type:
                    raise ValueError(
                        f"Metric '{metric.name}' already registered "
                        f"as {existing.type.value}"
                    )
                return existing
            self._metrics[metric.name] = metric
            return metric

    def get(self, name: str) -> Metric | None:
        """Get a registered metric by name."""
        return self._metrics.get(name)

    def unregister(self, name: str) -> bool:
        """Unregister a metric.

        Args:
            name: Metric name.

        Returns:
            True if unregistered, False if not found.
        """
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                return True
            return False

    def collect_all(self) -> dict[str, Any]:
        """Collect all metrics.

        Returns:
            Dictionary mapping metric names to their collected data.
        """
        with self._lock:
            result = {}
            for name, metric in self._metrics.items():
                result[name] = {
                    "type": metric.type.value,
                    "description": metric.description,
                    "data": metric.collect(),
                }
            return result

    def clear(self) -> None:
        """Clear all registered metrics."""
        with self._lock:
            self._metrics.clear()


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """High-level interface for creating and managing metrics.

    MetricsCollector provides a fluent API for creating metrics and
    manages their registration in the registry.

    Example:
        >>> collector = MetricsCollector()
        >>> requests = collector.counter(
        ...     "http_requests_total",
        ...     "Total HTTP requests",
        ...     labels=["method", "status"],
        ... )
        >>> latency = collector.histogram(
        ...     "request_duration_seconds",
        ...     "Request latency",
        ... )
    """

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        """Initialize collector.

        Args:
            registry: Metrics registry (creates new if None).
        """
        self._registry = registry or MetricsRegistry()

    @property
    def registry(self) -> MetricsRegistry:
        """Get the metrics registry."""
        return self._registry

    def counter(
        self,
        name: str,
        description: str = "",
        *,
        labels: list[str] | tuple[str, ...] = (),
        **kwargs: Any,
    ) -> Counter:
        """Create or get a counter.

        Args:
            name: Counter name.
            description: Human-readable description.
            labels: Label names.
            **kwargs: Additional arguments.

        Returns:
            Counter instance.
        """
        counter = Counter(name, description, labels=labels, **kwargs)
        return self._registry.register(counter)  # type: ignore

    def gauge(
        self,
        name: str,
        description: str = "",
        *,
        labels: list[str] | tuple[str, ...] = (),
        **kwargs: Any,
    ) -> Gauge:
        """Create or get a gauge.

        Args:
            name: Gauge name.
            description: Human-readable description.
            labels: Label names.
            **kwargs: Additional arguments.

        Returns:
            Gauge instance.
        """
        gauge = Gauge(name, description, labels=labels, **kwargs)
        return self._registry.register(gauge)  # type: ignore

    def histogram(
        self,
        name: str,
        description: str = "",
        *,
        buckets: list[float] | tuple[float, ...] | None = None,
        labels: list[str] | tuple[str, ...] = (),
        **kwargs: Any,
    ) -> Histogram:
        """Create or get a histogram.

        Args:
            name: Histogram name.
            description: Human-readable description.
            buckets: Bucket boundaries.
            labels: Label names.
            **kwargs: Additional arguments.

        Returns:
            Histogram instance.
        """
        histogram = Histogram(
            name, description, buckets=buckets, labels=labels, **kwargs
        )
        return self._registry.register(histogram)  # type: ignore

    def summary(
        self,
        name: str,
        description: str = "",
        *,
        quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
        labels: list[str] | tuple[str, ...] = (),
        **kwargs: Any,
    ) -> Summary:
        """Create or get a summary.

        Args:
            name: Summary name.
            description: Human-readable description.
            quantiles: Quantiles to calculate.
            labels: Label names.
            **kwargs: Additional arguments.

        Returns:
            Summary instance.
        """
        summary = Summary(
            name, description, quantiles=quantiles, labels=labels, **kwargs
        )
        return self._registry.register(summary)  # type: ignore


# =============================================================================
# Metrics Exporters
# =============================================================================


class MetricsExporter(ABC):
    """Abstract base class for metrics exporters."""

    @abstractmethod
    def export(self, registry: MetricsRegistry) -> str:
        """Export metrics from registry.

        Args:
            registry: Metrics registry to export.

        Returns:
            Exported metrics as string.
        """
        pass

    @abstractmethod
    def push(self, registry: MetricsRegistry, endpoint: str) -> bool:
        """Push metrics to remote endpoint.

        Args:
            registry: Metrics registry to push.
            endpoint: Remote endpoint URL.

        Returns:
            True if successful.
        """
        pass


class PrometheusExporter(MetricsExporter):
    """Export metrics in Prometheus text format.

    Generates output compatible with Prometheus text exposition format.

    Example output:
        # HELP http_requests_total Total HTTP requests
        # TYPE http_requests_total counter
        http_requests_total{method="GET",status="200"} 1234
    """

    def export(self, registry: MetricsRegistry) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        data = registry.collect_all()

        for name, info in data.items():
            metric_type = info["type"]
            description = info["description"]
            metric_data = info["data"]

            # HELP and TYPE lines
            lines.append(f"# HELP {name} {description}")
            lines.append(f"# TYPE {name} {metric_type}")

            # Metric lines
            for labels, value in metric_data:
                label_str = self._format_labels(labels)

                if metric_type == "histogram":
                    # Histogram has special format
                    for bound, count in value["buckets"].items():
                        bound_label = f'{label_str},le="{bound}"' if label_str else f'le="{bound}"'
                        lines.append(f"{name}_bucket{{{bound_label}}} {count}")
                    sum_labels = f"{{{label_str}}}" if label_str else ""
                    lines.append(f"{name}_sum{sum_labels} {value['sum']}")
                    lines.append(f"{name}_count{sum_labels} {value['count']}")

                elif metric_type == "summary":
                    for quantile_name, quantile_value in value.get("quantiles", {}).items():
                        q = quantile_name[1:]  # Remove 'p' prefix
                        q_label = f'{label_str},quantile="{int(q)/100}"' if label_str else f'quantile="{int(q)/100}"'
                        lines.append(f"{name}{{{q_label}}} {quantile_value}")
                    sum_labels = f"{{{label_str}}}" if label_str else ""
                    lines.append(f"{name}_sum{sum_labels} {value['sum']}")
                    lines.append(f"{name}_count{sum_labels} {value['count']}")

                else:
                    # Counter and Gauge
                    if label_str:
                        lines.append(f"{name}{{{label_str}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return ",".join(parts)

    def push(self, registry: MetricsRegistry, endpoint: str) -> bool:
        """Push metrics to Prometheus Pushgateway."""
        import urllib.request
        import urllib.error

        data = self.export(registry).encode("utf-8")

        try:
            request = urllib.request.Request(
                endpoint,
                data=data,
                method="POST",
                headers={"Content-Type": "text/plain"},
            )
            with urllib.request.urlopen(request, timeout=30):
                return True
        except urllib.error.URLError:
            return False


class StatsDExporter(MetricsExporter):
    """Export metrics in StatsD format.

    Sends metrics via UDP to a StatsD server.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8125,
        prefix: str = "",
    ) -> None:
        """Initialize StatsD exporter.

        Args:
            host: StatsD server host.
            port: StatsD server port.
            prefix: Metric name prefix.
        """
        self._host = host
        self._port = port
        self._prefix = prefix

    def export(self, registry: MetricsRegistry) -> str:
        """Export metrics in StatsD format."""
        lines = []
        data = registry.collect_all()

        for name, info in data.items():
            metric_type = info["type"]
            metric_data = info["data"]

            full_name = f"{self._prefix}{name}" if self._prefix else name

            for labels, value in metric_data:
                label_suffix = self._format_labels(labels)
                metric_name = f"{full_name}{label_suffix}"

                if metric_type == "counter":
                    lines.append(f"{metric_name}:{value}|c")
                elif metric_type == "gauge":
                    lines.append(f"{metric_name}:{value}|g")
                elif metric_type == "histogram":
                    # Send as timing
                    lines.append(f"{metric_name}:{value['sum']/max(value['count'], 1)*1000}|ms")
                elif metric_type == "summary":
                    lines.append(f"{metric_name}:{value['sum']/max(value['count'], 1)*1000}|ms")

        return "\n".join(lines)

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels as suffix."""
        if not labels:
            return ""
        parts = [f".{k}_{v}" for k, v in sorted(labels.items())]
        return "".join(parts)

    def push(self, registry: MetricsRegistry, endpoint: str | None = None) -> bool:
        """Send metrics to StatsD server via UDP."""
        import socket

        data = self.export(registry)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(data.encode("utf-8"), (self._host, self._port))
            sock.close()
            return True
        except socket.error:
            return False


class InMemoryExporter(MetricsExporter):
    """In-memory exporter for testing.

    Stores exported metrics for inspection.
    """

    def __init__(self) -> None:
        """Initialize in-memory exporter."""
        self._exports: list[dict[str, Any]] = []

    @property
    def exports(self) -> list[dict[str, Any]]:
        """Get all exports."""
        return self._exports

    def export(self, registry: MetricsRegistry) -> str:
        """Export metrics to memory."""
        import json
        data = registry.collect_all()
        self._exports.append(data)
        return json.dumps(data, default=str)

    def push(self, registry: MetricsRegistry, endpoint: str) -> bool:
        """Push to memory (always succeeds)."""
        self.export(registry)
        return True

    def clear(self) -> None:
        """Clear stored exports."""
        self._exports.clear()


# =============================================================================
# Global Metrics
# =============================================================================

_global_collector: MetricsCollector | None = None
_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector.

    Returns:
        Global MetricsCollector instance.
    """
    global _global_collector

    with _lock:
        if _global_collector is None:
            _global_collector = MetricsCollector()
        return _global_collector


def set_metrics(collector: MetricsCollector) -> None:
    """Set the global metrics collector.

    Args:
        collector: MetricsCollector to use globally.
    """
    global _global_collector

    with _lock:
        _global_collector = collector


def configure_metrics(
    *,
    exporter: MetricsExporter | None = None,
    registry: MetricsRegistry | None = None,
) -> MetricsCollector:
    """Configure global metrics.

    Args:
        exporter: Exporter to use.
        registry: Registry to use.

    Returns:
        Configured MetricsCollector.
    """
    collector = MetricsCollector(registry=registry)
    set_metrics(collector)
    return collector
