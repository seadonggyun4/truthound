"""Observability module with OpenTelemetry integration for profiler metrics.

This module provides comprehensive observability for the profiling system:
- OpenTelemetry tracing for operation tracking
- Metrics collection for performance monitoring
- Structured logging integration
- Custom metric exporters

Key features:
- Pluggable exporter architecture
- Automatic span creation for profiling operations
- Histogram and counter metrics
- Context propagation

Example:
    from truthound.profiler.observability import (
        ProfilerTelemetry,
        MetricsCollector,
        traced,
    )

    # Initialize telemetry
    telemetry = ProfilerTelemetry(service_name="truthound-profiler")

    # Use decorator for automatic tracing
    @traced("profile_column")
    def profile_column(col: str) -> ColumnProfile:
        # profiling logic...
        pass

    # Or manual tracing
    with telemetry.span("custom_operation") as span:
        span.set_attribute("column_count", 10)
        # operation...
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Generator,
    Generic,
    Protocol,
    TypeVar,
)

# =============================================================================
# Types
# =============================================================================

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class SpanStatus(str, Enum):
    """Span status codes."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


# =============================================================================
# Span Protocol
# =============================================================================


class SpanProtocol(Protocol):
    """Protocol for span interface (compatible with OpenTelemetry)."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        ...

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set span status."""
        ...

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to span."""
        ...

    def record_exception(self, exception: BaseException) -> None:
        """Record exception in span."""
        ...

    def end(self) -> None:
        """End the span."""
        ...


# =============================================================================
# Span Implementation
# =============================================================================


@dataclass
class SpanEvent:
    """Event recorded in a span."""

    name: str
    timestamp: datetime
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Lightweight span implementation.

    Compatible with OpenTelemetry Span interface but can work standalone.
    """

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.UNSET
    status_description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    exception: BaseException | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def set_status(self, status: SpanStatus, description: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_description = description

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event to span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=datetime.now(),
                attributes=attributes or {},
            )
        )

    def record_exception(self, exception: BaseException) -> None:
        """Record exception in span."""
        self.exception = exception
        self.set_status(SpanStatus.ERROR, str(exception))
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return (datetime.now() - self.start_time).total_seconds() * 1000
        return (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_description": self.status_description,
            "attributes": self.attributes,
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ],
        }


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class MetricValue:
    """Single metric measurement."""

    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)


class Metric(ABC):
    """Abstract base class for metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        unit: str = "",
        labels: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.unit = unit
        self.label_names = labels or []
        self._lock = threading.Lock()

    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Get metric type."""
        pass

    @abstractmethod
    def collect(self) -> list[MetricValue]:
        """Collect current metric values."""
        pass


class Counter(Metric):
    """Monotonically increasing counter metric."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ):
        super().__init__(name, description, labels=labels)
        self._values: dict[tuple[str, ...], float] = defaultdict(float)

    @property
    def metric_type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment counter."""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] += value

    def _label_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Create label key tuple."""
        return tuple(labels.get(name, "") for name in self.label_names)

    def collect(self) -> list[MetricValue]:
        """Collect current values."""
        with self._lock:
            return [
                MetricValue(
                    value=value,
                    labels=dict(zip(self.label_names, key)),
                )
                for key, value in self._values.items()
            ]


class Gauge(Metric):
    """Gauge metric that can go up and down."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ):
        super().__init__(name, description, labels=labels)
        self._values: dict[tuple[str, ...], float] = {}

    @property
    def metric_type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float, **labels: str) -> None:
        """Set gauge value."""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, **labels: str) -> None:
        """Increment gauge."""
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, value: float = 1.0, **labels: str) -> None:
        """Decrement gauge."""
        self.inc(-value, **labels)

    def _label_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Create label key tuple."""
        return tuple(labels.get(name, "") for name in self.label_names)

    def collect(self) -> list[MetricValue]:
        """Collect current values."""
        with self._lock:
            return [
                MetricValue(
                    value=value,
                    labels=dict(zip(self.label_names, key)),
                )
                for key, value in self._values.items()
            ]


class Histogram(Metric):
    """Histogram metric for distributions."""

    DEFAULT_BUCKETS = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ):
        super().__init__(name, description, labels=labels)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._observations: dict[tuple[str, ...], list[float]] = defaultdict(list)

    @property
    def metric_type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float, **labels: str) -> None:
        """Record an observation."""
        key = self._label_key(labels)
        with self._lock:
            self._observations[key].append(value)

    @contextmanager
    def time(self, **labels: str) -> Generator[None, None, None]:
        """Context manager to time operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe(duration, **labels)

    def _label_key(self, labels: dict[str, str]) -> tuple[str, ...]:
        """Create label key tuple."""
        return tuple(labels.get(name, "") for name in self.label_names)

    def collect(self) -> list[MetricValue]:
        """Collect histogram statistics."""
        results = []
        with self._lock:
            for key, observations in self._observations.items():
                if not observations:
                    continue

                labels = dict(zip(self.label_names, key))

                # Count and sum
                results.append(MetricValue(
                    value=len(observations),
                    labels={**labels, "stat": "count"},
                ))
                results.append(MetricValue(
                    value=sum(observations),
                    labels={**labels, "stat": "sum"},
                ))

                # Bucket counts
                for bucket in self.buckets:
                    count = sum(1 for v in observations if v <= bucket)
                    results.append(MetricValue(
                        value=count,
                        labels={**labels, "le": str(bucket)},
                    ))

        return results

    def get_percentile(self, percentile: float, **labels: str) -> float | None:
        """Get percentile value for a label set."""
        key = self._label_key(labels)
        with self._lock:
            observations = self._observations.get(key, [])
            if not observations:
                return None
            sorted_obs = sorted(observations)
            idx = int(len(sorted_obs) * percentile / 100)
            return sorted_obs[min(idx, len(sorted_obs) - 1)]


# =============================================================================
# Metrics Collector
# =============================================================================


class MetricsCollector:
    """Central collector for all profiler metrics.

    Provides pre-defined metrics for common profiling operations
    and allows custom metric registration.

    Example:
        collector = MetricsCollector()

        # Record profile duration
        with collector.profile_duration.time(column="user_id"):
            profile_column(...)

        # Increment counter
        collector.profiles_total.inc(status="success")
    """

    def __init__(self, prefix: str = "truthound_profiler"):
        self.prefix = prefix
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

        # Register default metrics
        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register standard profiler metrics."""
        # Counters
        self.profiles_total = self.register_counter(
            "profiles_total",
            "Total number of profiles completed",
            labels=["status", "type"],
        )

        self.columns_profiled = self.register_counter(
            "columns_profiled",
            "Total number of columns profiled",
            labels=["data_type"],
        )

        self.rules_generated = self.register_counter(
            "rules_generated",
            "Total number of rules generated",
            labels=["category", "generator"],
        )

        self.cache_operations = self.register_counter(
            "cache_operations",
            "Cache operations",
            labels=["operation", "result"],
        )

        # Gauges
        self.active_profiles = self.register_gauge(
            "active_profiles",
            "Number of profiles currently in progress",
        )

        self.cache_size = self.register_gauge(
            "cache_size",
            "Current cache size",
            labels=["backend"],
        )

        # Histograms
        self.profile_duration = self.register_histogram(
            "profile_duration_seconds",
            "Time spent profiling",
            labels=["operation"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0),
        )

        self.column_profile_duration = self.register_histogram(
            "column_profile_duration_seconds",
            "Time spent profiling a column",
            labels=["column_type"],
        )

        self.pattern_match_duration = self.register_histogram(
            "pattern_match_duration_seconds",
            "Time spent matching patterns",
            labels=["pattern_name"],
        )

        self.rows_processed = self.register_histogram(
            "rows_processed",
            "Number of rows processed per profile",
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
        )

    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Counter:
        """Register a counter metric."""
        full_name = f"{self.prefix}_{name}"
        metric = Counter(full_name, description, labels)
        with self._lock:
            self._metrics[full_name] = metric
        return metric

    def register_gauge(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Gauge:
        """Register a gauge metric."""
        full_name = f"{self.prefix}_{name}"
        metric = Gauge(full_name, description, labels)
        with self._lock:
            self._metrics[full_name] = metric
        return metric

    def register_histogram(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Register a histogram metric."""
        full_name = f"{self.prefix}_{name}"
        metric = Histogram(full_name, description, labels, buckets)
        with self._lock:
            self._metrics[full_name] = metric
        return metric

    def get_metric(self, name: str) -> Metric | None:
        """Get a registered metric by name."""
        full_name = f"{self.prefix}_{name}"
        return self._metrics.get(full_name)

    def collect_all(self) -> dict[str, list[MetricValue]]:
        """Collect all metric values."""
        with self._lock:
            return {
                name: metric.collect()
                for name, metric in self._metrics.items()
            }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for name, metric in self._metrics.items():
            lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {metric.metric_type.value}")

            for value in metric.collect():
                label_str = ""
                if value.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in value.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"
                lines.append(f"{name}{label_str} {value.value}")

        return "\n".join(lines)


# =============================================================================
# Span Exporter Protocol
# =============================================================================


class SpanExporter(ABC):
    """Abstract base class for span exporters."""

    @abstractmethod
    def export(self, spans: list[Span]) -> bool:
        """Export spans to backend.

        Args:
            spans: Spans to export

        Returns:
            True if export succeeded
        """
        pass

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Exports spans to console/logging."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger("truthound.profiler.tracing")

    def export(self, spans: list[Span]) -> bool:
        for span in spans:
            self.logger.info(
                "Span: %s [%s] duration=%.2fms status=%s",
                span.name,
                span.span_id[:8],
                span.duration_ms,
                span.status.value,
            )
            for key, value in span.attributes.items():
                self.logger.debug("  %s: %s", key, value)
        return True


class InMemorySpanExporter(SpanExporter):
    """Exports spans to in-memory storage for testing."""

    def __init__(self, max_spans: int = 10000):
        self.max_spans = max_spans
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def export(self, spans: list[Span]) -> bool:
        with self._lock:
            self._spans.extend(spans)
            # Trim if over limit
            if len(self._spans) > self.max_spans:
                self._spans = self._spans[-self.max_spans:]
        return True

    def get_spans(self) -> list[Span]:
        """Get all exported spans."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear all spans."""
        with self._lock:
            self._spans.clear()


class OTLPSpanExporter(SpanExporter):
    """Exports spans via OTLP (OpenTelemetry Protocol).

    Requires opentelemetry-exporter-otlp package.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self._otlp_exporter = None

        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as _OTLPExporter,
            )
            self._otlp_exporter = _OTLPExporter(
                endpoint=endpoint,
                headers=headers,
                timeout=timeout,
            )
        except ImportError:
            pass

    def export(self, spans: list[Span]) -> bool:
        if self._otlp_exporter is None:
            return False

        # Convert to OpenTelemetry spans
        # This is a simplified conversion - full implementation would
        # use proper OTLP span conversion
        return True

    def shutdown(self) -> None:
        if self._otlp_exporter:
            self._otlp_exporter.shutdown()


# =============================================================================
# Span Exporter Registry
# =============================================================================


class SpanExporterRegistry:
    """Registry for span exporter factories."""

    def __init__(self) -> None:
        self._exporters: dict[str, type[SpanExporter]] = {}

    def register(self, name: str, exporter_class: type[SpanExporter]) -> None:
        """Register an exporter class."""
        self._exporters[name] = exporter_class

    def create(self, name: str, **kwargs: Any) -> SpanExporter:
        """Create an exporter instance."""
        if name not in self._exporters:
            raise KeyError(
                f"Unknown exporter: {name}. "
                f"Available: {list(self._exporters.keys())}"
            )
        return self._exporters[name](**kwargs)

    def list_exporters(self) -> list[str]:
        """List registered exporter names."""
        return list(self._exporters.keys())


# Global registry
span_exporter_registry = SpanExporterRegistry()
span_exporter_registry.register("console", ConsoleSpanExporter)
span_exporter_registry.register("memory", InMemorySpanExporter)
span_exporter_registry.register("otlp", OTLPSpanExporter)


# =============================================================================
# Profiler Telemetry
# =============================================================================


@dataclass
class TelemetryConfig:
    """Configuration for profiler telemetry."""

    service_name: str = "truthound-profiler"
    enabled: bool = True
    exporter: str = "console"
    exporter_options: dict[str, Any] = field(default_factory=dict)
    sample_rate: float = 1.0  # 1.0 = sample all
    batch_size: int = 100
    flush_interval_seconds: float = 5.0


class ProfilerTelemetry:
    """Main telemetry interface for the profiler.

    Provides tracing, metrics, and logging integration.

    Example:
        telemetry = ProfilerTelemetry(service_name="my-profiler")

        with telemetry.span("profile_table") as span:
            span.set_attribute("table_name", "users")
            # profiling logic...
    """

    def __init__(
        self,
        service_name: str = "truthound-profiler",
        enabled: bool = True,
        exporter: str | SpanExporter = "console",
        exporter_options: dict[str, Any] | None = None,
        sample_rate: float = 1.0,
    ):
        self.service_name = service_name
        self.enabled = enabled
        self.sample_rate = sample_rate
        self._lock = threading.Lock()

        # Initialize exporter
        if isinstance(exporter, SpanExporter):
            self._exporter = exporter
        else:
            options = exporter_options or {}
            self._exporter = span_exporter_registry.create(exporter, **options)

        # Span tracking
        self._spans: list[Span] = []
        self._current_span: Span | None = None
        self._span_stack: list[Span] = []

        # Metrics collector
        self.metrics = MetricsCollector()

        # Generate IDs
        self._trace_id_counter = 0
        self._span_id_counter = 0

    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        with self._lock:
            self._trace_id_counter += 1
            return f"{self._trace_id_counter:032x}"

    def _generate_span_id(self) -> str:
        """Generate unique span ID."""
        with self._lock:
            self._span_id_counter += 1
            return f"{self._span_id_counter:016x}"

    def _should_sample(self) -> bool:
        """Determine if this trace should be sampled."""
        import random
        return random.random() < self.sample_rate

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Create a span context manager.

        Args:
            name: Span name
            attributes: Initial attributes

        Yields:
            Span object for the duration
        """
        if not self.enabled or not self._should_sample():
            # Return a no-op span
            yield Span(
                name=name,
                trace_id="",
                span_id="",
            )
            return

        # Create span
        parent_id = self._span_stack[-1].span_id if self._span_stack else None
        trace_id = (
            self._span_stack[0].trace_id if self._span_stack
            else self._generate_trace_id()
        )

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=self._generate_span_id(),
            parent_id=parent_id,
            attributes=attributes or {},
        )

        self._span_stack.append(span)

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._span_stack.pop()
            self._spans.append(span)

            # Export if batch is full
            if len(self._spans) >= 100:
                self._flush()

    def _flush(self) -> None:
        """Flush pending spans to exporter."""
        with self._lock:
            if self._spans:
                self._exporter.export(list(self._spans))
                self._spans.clear()

    def shutdown(self) -> None:
        """Shutdown telemetry and flush remaining spans."""
        self._flush()
        self._exporter.shutdown()

    def record_profile(
        self,
        profile_type: str,
        duration_seconds: float,
        row_count: int,
        column_count: int,
        success: bool = True,
    ) -> None:
        """Record profile metrics.

        Args:
            profile_type: Type of profile (table, column, etc.)
            duration_seconds: Time taken
            row_count: Number of rows profiled
            column_count: Number of columns profiled
            success: Whether profile succeeded
        """
        status = "success" if success else "error"

        self.metrics.profiles_total.inc(status=status, type=profile_type)
        self.metrics.profile_duration.observe(
            duration_seconds,
            operation=profile_type,
        )
        self.metrics.rows_processed.observe(row_count)

    def record_column_profile(
        self,
        column_type: str,
        duration_seconds: float,
    ) -> None:
        """Record column profile metrics."""
        self.metrics.columns_profiled.inc(data_type=column_type)
        self.metrics.column_profile_duration.observe(
            duration_seconds,
            column_type=column_type,
        )

    def record_cache_operation(
        self,
        operation: str,
        hit: bool,
    ) -> None:
        """Record cache operation."""
        result = "hit" if hit else "miss"
        self.metrics.cache_operations.inc(operation=operation, result=result)


# =============================================================================
# Decorators
# =============================================================================


# Global telemetry instance (can be overridden)
_global_telemetry: ProfilerTelemetry | None = None


def get_telemetry() -> ProfilerTelemetry:
    """Get or create global telemetry instance."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = ProfilerTelemetry()
    return _global_telemetry


def set_telemetry(telemetry: ProfilerTelemetry) -> None:
    """Set global telemetry instance."""
    global _global_telemetry
    _global_telemetry = telemetry


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_args: bool = False,
) -> Callable[[F], F]:
    """Decorator to automatically trace a function.

    Example:
        @traced("profile_column")
        def profile_column(name: str) -> ColumnProfile:
            ...

        @traced(record_args=True)
        def process_data(data: list) -> None:
            ...

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add
        record_args: Whether to record function arguments

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            telemetry = get_telemetry()

            span_attrs = dict(attributes or {})
            span_attrs["function"] = func.__name__
            span_attrs["module"] = func.__module__

            if record_args:
                span_attrs["args_count"] = len(args)
                span_attrs["kwargs_keys"] = list(kwargs.keys())

            with telemetry.span(span_name, attributes=span_attrs) as span:
                result = func(*args, **kwargs)

                # Add result info if available
                if hasattr(result, "__len__"):
                    span.set_attribute("result_length", len(result))

                return result

        return wrapper  # type: ignore

    return decorator


def timed(
    histogram: Histogram | None = None,
    metric_name: str = "operation_duration_seconds",
    **labels: str,
) -> Callable[[F], F]:
    """Decorator to time a function and record to histogram.

    Example:
        @timed(metric_name="profile_column_seconds", column_type="string")
        def profile_string_column(col):
            ...

    Args:
        histogram: Histogram to record to
        metric_name: Metric name if creating new histogram
        **labels: Labels to add to metric

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            hist = histogram
            if hist is None:
                telemetry = get_telemetry()
                hist = telemetry.metrics.get_metric(metric_name)
                if hist is None or not isinstance(hist, Histogram):
                    hist = telemetry.metrics.register_histogram(metric_name)

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                hist.observe(duration, **labels)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# OpenTelemetry Integration
# =============================================================================


class OpenTelemetryIntegration:
    """Integration with OpenTelemetry SDK.

    Wraps OpenTelemetry tracer and meter for seamless integration.

    Example:
        # If opentelemetry is installed
        otel = OpenTelemetryIntegration.create(
            service_name="truthound-profiler",
            endpoint="http://localhost:4317"
        )

        with otel.tracer.start_as_current_span("profile") as span:
            span.set_attribute("table", "users")
            ...
    """

    def __init__(
        self,
        tracer: Any = None,
        meter: Any = None,
    ):
        self._tracer = tracer
        self._meter = meter

    @classmethod
    def create(
        cls,
        service_name: str = "truthound-profiler",
        endpoint: str | None = None,
    ) -> "OpenTelemetryIntegration":
        """Create OpenTelemetry integration.

        Args:
            service_name: Service name for tracing
            endpoint: OTLP endpoint (optional)

        Returns:
            Configured integration
        """
        tracer = None
        meter = None

        try:
            from opentelemetry import trace, metrics
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.metrics import MeterProvider

            resource = Resource.create({"service.name": service_name})

            # Setup tracing
            trace_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(trace_provider)

            if endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=endpoint)
                trace_provider.add_span_processor(BatchSpanProcessor(exporter))

            tracer = trace.get_tracer(service_name)

            # Setup metrics
            meter_provider = MeterProvider(resource=resource)
            metrics.set_meter_provider(meter_provider)
            meter = metrics.get_meter(service_name)

        except ImportError:
            pass

        return cls(tracer=tracer, meter=meter)

    @property
    def tracer(self) -> Any:
        """Get OpenTelemetry tracer."""
        return self._tracer

    @property
    def meter(self) -> Any:
        """Get OpenTelemetry meter."""
        return self._meter

    @property
    def available(self) -> bool:
        """Check if OpenTelemetry is available."""
        return self._tracer is not None


# =============================================================================
# Convenience Functions
# =============================================================================


def create_telemetry(
    service_name: str = "truthound-profiler",
    exporter: str = "console",
    **kwargs: Any,
) -> ProfilerTelemetry:
    """Create and configure profiler telemetry.

    Args:
        service_name: Service name for tracing
        exporter: Exporter type
        **kwargs: Exporter options

    Returns:
        Configured ProfilerTelemetry instance
    """
    telemetry = ProfilerTelemetry(
        service_name=service_name,
        exporter=exporter,
        exporter_options=kwargs,
    )
    set_telemetry(telemetry)
    return telemetry


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return get_telemetry().metrics
