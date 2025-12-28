"""Prometheus metrics for store operations.

This module provides comprehensive metrics collection for stores with
Prometheus-compatible export. It supports multiple output formats and
includes both synchronous and asynchronous collection.

Features:
- Standard metric types (Counter, Gauge, Histogram, Summary)
- Prometheus text format export
- OpenMetrics format support
- HTTP endpoint for scraping
- Push gateway support
- Automatic label management
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Generator, TypeVar

from truthound.stores.observability.config import MetricsConfig, MetricsExportFormat

T = TypeVar("T")


@dataclass
class MetricValue:
    """A single metric value with labels."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def label_key(self) -> str:
        """Get a unique key for this label combination."""
        if not self.labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(self.labels.items()))


@dataclass
class HistogramValue:
    """Histogram metric value with buckets."""

    name: str
    labels: dict[str, str]
    buckets: dict[float, int]  # bucket boundary -> count
    sum: float
    count: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class SummaryValue:
    """Summary metric value with quantiles."""

    name: str
    labels: dict[str, str]
    quantiles: dict[float, float]  # quantile -> value
    sum: float
    count: int
    timestamp: float = field(default_factory=time.time)


class MetricType:
    """Metric type constants."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDescriptor:
    """Describes a metric's metadata."""

    name: str
    type: str
    help: str
    labels: list[str] = field(default_factory=list)


class BaseMetricsBackend(ABC):
    """Base class for metrics backends."""

    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        self.prefix = config.prefix
        self.default_labels = config.labels.copy()

    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        if self.prefix:
            return f"{self.prefix}_{name}"
        return name

    def _merge_labels(self, labels: dict[str, str] | None) -> dict[str, str]:
        """Merge provided labels with defaults."""
        result = self.default_labels.copy()
        if labels:
            result.update(labels)
        return result

    @abstractmethod
    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter."""
        ...

    @abstractmethod
    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        ...

    @abstractmethod
    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation."""
        ...

    @abstractmethod
    def summary(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a summary observation."""
        ...

    @contextmanager
    def timer(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Time an operation and record as histogram."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.histogram(name, duration, labels)

    @abstractmethod
    def export(self) -> str:
        """Export metrics in configured format."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all metrics."""
        ...


class InMemoryMetricsBackend(BaseMetricsBackend):
    """In-memory metrics backend for testing and development."""

    def __init__(self, config: MetricsConfig | None = None) -> None:
        super().__init__(config or MetricsConfig())
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: dict[str, dict[str, float]] = defaultdict(dict)
        self._histograms: dict[str, dict[str, HistogramValue]] = defaultdict(dict)
        self._summaries: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._descriptors: dict[str, MetricDescriptor] = {}

    def _register(self, name: str, type: str, help: str = "") -> None:
        """Register a metric descriptor."""
        full_name = self._full_name(name)
        if full_name not in self._descriptors:
            self._descriptors[full_name] = MetricDescriptor(
                name=full_name, type=type, help=help
            )

    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        label_key = MetricValue(name=name, value=value, labels=merged_labels).label_key()

        with self._lock:
            self._register(name, MetricType.COUNTER)
            self._counters[full_name][label_key] += value

    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        label_key = MetricValue(name=name, value=value, labels=merged_labels).label_key()

        with self._lock:
            self._register(name, MetricType.GAUGE)
            self._gauges[full_name][label_key] = value

    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        label_key = MetricValue(name=name, value=value, labels=merged_labels).label_key()

        with self._lock:
            self._register(name, MetricType.HISTOGRAM)
            if label_key not in self._histograms[full_name]:
                self._histograms[full_name][label_key] = HistogramValue(
                    name=full_name,
                    labels=merged_labels,
                    buckets={b: 0 for b in self.config.histogram_buckets},
                    sum=0.0,
                    count=0,
                )
            hist = self._histograms[full_name][label_key]
            hist.sum += value
            hist.count += 1
            for bucket in self.config.histogram_buckets:
                if value <= bucket:
                    hist.buckets[bucket] += 1

    def summary(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        full_name = self._full_name(name)
        merged_labels = self._merge_labels(labels)
        label_key = MetricValue(name=name, value=value, labels=merged_labels).label_key()

        with self._lock:
            self._register(name, MetricType.SUMMARY)
            self._summaries[full_name][label_key].append(value)

    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines: list[str] = []

        with self._lock:
            # Export counters
            for name, values in self._counters.items():
                desc = self._descriptors.get(name)
                if desc:
                    lines.append(f"# HELP {name} {desc.help}")
                    lines.append(f"# TYPE {name} counter")
                for label_key, value in values.items():
                    if label_key:
                        lines.append(f"{name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Export gauges
            for name, values in self._gauges.items():
                desc = self._descriptors.get(name)
                if desc:
                    lines.append(f"# HELP {name} {desc.help}")
                    lines.append(f"# TYPE {name} gauge")
                for label_key, value in values.items():
                    if label_key:
                        lines.append(f"{name}{{{label_key}}} {value}")
                    else:
                        lines.append(f"{name} {value}")

            # Export histograms
            for name, histograms in self._histograms.items():
                desc = self._descriptors.get(name)
                if desc:
                    lines.append(f"# HELP {name} {desc.help}")
                    lines.append(f"# TYPE {name} histogram")
                for label_key, hist in histograms.items():
                    base_labels = f"{{{label_key}}}" if label_key else ""
                    cumulative = 0
                    for bucket, count in sorted(hist.buckets.items()):
                        cumulative += count
                        bucket_labels = f'le="{bucket}"'
                        if label_key:
                            bucket_labels = f"{label_key},{bucket_labels}"
                        lines.append(f"{name}_bucket{{{bucket_labels}}} {cumulative}")
                    # +Inf bucket
                    inf_labels = 'le="+Inf"'
                    if label_key:
                        inf_labels = f"{label_key},{inf_labels}"
                    lines.append(f"{name}_bucket{{{inf_labels}}} {hist.count}")
                    lines.append(f"{name}_sum{base_labels} {hist.sum}")
                    lines.append(f"{name}_count{base_labels} {hist.count}")

            # Export summaries
            for name, summaries in self._summaries.items():
                desc = self._descriptors.get(name)
                if desc:
                    lines.append(f"# HELP {name} {desc.help}")
                    lines.append(f"# TYPE {name} summary")
                for label_key, values in summaries.items():
                    if values:
                        sorted_values = sorted(values)
                        count = len(values)
                        total = sum(values)
                        base_labels = f"{{{label_key}}}" if label_key else ""

                        # Calculate quantiles
                        for q in [0.5, 0.9, 0.95, 0.99]:
                            idx = int(q * count)
                            q_value = sorted_values[min(idx, count - 1)]
                            q_labels = f'quantile="{q}"'
                            if label_key:
                                q_labels = f"{label_key},{q_labels}"
                            lines.append(f"{name}{{{q_labels}}} {q_value}")

                        lines.append(f"{name}_sum{base_labels} {total}")
                        lines.append(f"{name}_count{base_labels} {count}")

        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._summaries.clear()

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> float:
        """Get current counter value (for testing)."""
        full_name = self._full_name(name)
        label_key = MetricValue(name=name, value=0, labels=labels or {}).label_key()
        with self._lock:
            return self._counters[full_name].get(label_key, 0.0)

    def get_gauge(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        """Get current gauge value (for testing)."""
        full_name = self._full_name(name)
        label_key = MetricValue(name=name, value=0, labels=labels or {}).label_key()
        with self._lock:
            return self._gauges[full_name].get(label_key)


class PrometheusMetricsBackend(InMemoryMetricsBackend):
    """Prometheus metrics backend with HTTP server and push gateway support."""

    def __init__(
        self,
        config: MetricsConfig | None = None,
        auto_start_server: bool = False,
    ) -> None:
        super().__init__(config or MetricsConfig())
        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None

        if auto_start_server and self.config.enable_http_server:
            self.start_http_server()

    def start_http_server(self) -> None:
        """Start HTTP server for metrics scraping."""
        if self._server is not None:
            return

        backend = self

        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == backend.config.http_path:
                    content = backend.export().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass  # Suppress logging

        self._server = HTTPServer(
            ("0.0.0.0", self.config.http_port),
            MetricsHandler,
        )
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._server_thread.start()

    def stop_http_server(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._server_thread = None

    def push_to_gateway(self, job: str = "truthound") -> bool:
        """Push metrics to Prometheus Push Gateway."""
        if not self.config.push_gateway_url:
            return False

        try:
            import urllib.request

            url = f"{self.config.push_gateway_url}/metrics/job/{job}"
            data = self.export().encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "text/plain")
            urllib.request.urlopen(req, timeout=5)
            return True
        except Exception:
            return False


class MetricsRegistry:
    """Registry for managing metrics across multiple stores."""

    _instance: "MetricsRegistry | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._backend: BaseMetricsBackend | None = None
        self._config: MetricsConfig | None = None
        self._initialized = True

    def configure(
        self,
        config: MetricsConfig | None = None,
        backend: BaseMetricsBackend | None = None,
    ) -> None:
        """Configure the global metrics registry."""
        self._config = config or MetricsConfig()
        if backend:
            self._backend = backend
        elif self._config.enabled:
            self._backend = PrometheusMetricsBackend(
                self._config,
                auto_start_server=self._config.enable_http_server,
            )

    @property
    def backend(self) -> BaseMetricsBackend | None:
        return self._backend

    def get_backend(self) -> BaseMetricsBackend:
        """Get the configured backend, creating a default if needed."""
        if self._backend is None:
            self.configure()
        if self._backend is None:
            self._backend = InMemoryMetricsBackend()
        return self._backend

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance and cls._instance._backend:
                cls._instance._backend.reset()
            cls._instance = None


@dataclass
class StoreMetrics:
    """Pre-defined metrics for store operations.

    This class provides convenient methods for recording common store metrics.
    """

    backend: BaseMetricsBackend
    store_type: str
    store_id: str

    def _labels(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Get base labels with optional extras."""
        labels = {"store_type": self.store_type, "store_id": self.store_id}
        if extra:
            labels.update(extra)
        return labels

    # Operation metrics
    def record_operation(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a store operation."""
        op_labels = self._labels({"operation": operation, **(labels or {})})

        self.backend.counter(
            "operations_total",
            labels=op_labels,
        )
        self.backend.histogram(
            "operation_duration_seconds",
            duration_seconds,
            labels=op_labels,
        )

        if success:
            self.backend.counter("operations_success_total", labels=op_labels)
        else:
            self.backend.counter("operations_failed_total", labels=op_labels)

    # CRUD metrics
    def record_save(self, duration_seconds: float, size_bytes: int = 0) -> None:
        """Record a save operation."""
        self.record_operation("save", duration_seconds)
        if size_bytes > 0:
            self.backend.histogram(
                "save_size_bytes",
                size_bytes,
                labels=self._labels(),
            )

    def record_get(self, duration_seconds: float, hit: bool = True) -> None:
        """Record a get operation."""
        self.record_operation("get", duration_seconds)
        if hit:
            self.backend.counter("cache_hits_total", labels=self._labels())
        else:
            self.backend.counter("cache_misses_total", labels=self._labels())

    def record_delete(self, duration_seconds: float) -> None:
        """Record a delete operation."""
        self.record_operation("delete", duration_seconds)

    def record_query(self, duration_seconds: float, result_count: int = 0) -> None:
        """Record a query operation."""
        self.record_operation("query", duration_seconds)
        self.backend.histogram(
            "query_result_count",
            result_count,
            labels=self._labels(),
        )

    # Resource metrics
    def set_item_count(self, count: int) -> None:
        """Set the current item count."""
        self.backend.gauge("items_total", count, labels=self._labels())

    def set_storage_size(self, size_bytes: int) -> None:
        """Set the current storage size."""
        self.backend.gauge("storage_bytes", size_bytes, labels=self._labels())

    # Connection metrics
    def record_connection(self, success: bool = True) -> None:
        """Record a connection attempt."""
        if success:
            self.backend.counter("connections_total", labels=self._labels())
        else:
            self.backend.counter("connection_errors_total", labels=self._labels())

    def set_active_connections(self, count: int) -> None:
        """Set the number of active connections."""
        self.backend.gauge("active_connections", count, labels=self._labels())

    # Error metrics
    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.backend.counter(
            "errors_total",
            labels=self._labels({"error_type": error_type}),
        )

    # Batch metrics
    def record_batch(
        self,
        operation: str,
        batch_size: int,
        duration_seconds: float,
    ) -> None:
        """Record a batch operation."""
        labels = self._labels({"operation": operation})
        self.backend.histogram("batch_size", batch_size, labels=labels)
        self.backend.histogram("batch_duration_seconds", duration_seconds, labels=labels)
        self.backend.counter("batch_operations_total", labels=labels)

    # Replication metrics
    def record_replication(
        self,
        target: str,
        duration_seconds: float,
        success: bool = True,
    ) -> None:
        """Record a replication event."""
        labels = self._labels({"target": target})
        self.backend.histogram("replication_duration_seconds", duration_seconds, labels=labels)
        if success:
            self.backend.counter("replications_total", labels=labels)
        else:
            self.backend.counter("replication_errors_total", labels=labels)

    def set_replication_lag(self, target: str, lag_seconds: float) -> None:
        """Set replication lag for a target."""
        self.backend.gauge(
            "replication_lag_seconds",
            lag_seconds,
            labels=self._labels({"target": target}),
        )
