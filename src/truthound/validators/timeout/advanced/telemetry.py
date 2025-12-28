"""OpenTelemetry integration for distributed tracing and metrics.

This module provides comprehensive observability for timeout operations:
- Distributed tracing across service boundaries
- Metrics collection for timeout events
- Span context propagation
- Multiple exporter backends (Console, OTLP, etc.)

The design follows OpenTelemetry conventions while providing a clean
abstraction layer for the Truthound timeout system.

Example:
    from truthound.validators.timeout.advanced.telemetry import (
        TelemetryProvider,
        TracingConfig,
        trace_operation,
    )

    # Configure telemetry
    config = TracingConfig(
        service_name="truthound-validator",
        exporter="otlp",
        endpoint="http://jaeger:4317",
    )

    provider = TelemetryProvider(config)

    # Trace an operation
    with provider.trace("validate_batch") as span:
        span.set_attribute("batch_size", 1000)
        result = validate(data)
        span.set_attribute("result", result.status)
"""

from __future__ import annotations

import contextlib
import functools
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SpanStatus(str, Enum):
    """Status codes for spans."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanKind(str, Enum):
    """Kind of span."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class SpanContext:
    """Context for span propagation across service boundaries.

    Attributes:
        trace_id: Unique identifier for the trace
        span_id: Unique identifier for this span
        parent_span_id: Parent span ID (None if root)
        trace_flags: Trace flags (e.g., sampling)
        trace_state: Vendor-specific trace state
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    trace_flags: int = 0
    trace_state: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create_root(cls) -> "SpanContext":
        """Create a root span context."""
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
        )

    @classmethod
    def create_child(cls, parent: "SpanContext") -> "SpanContext":
        """Create a child span context."""
        return cls(
            trace_id=parent.trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.span_id,
            trace_flags=parent.trace_flags,
            trace_state=dict(parent.trace_state),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpanContext":
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            trace_flags=data.get("trace_flags", 0),
            trace_state=data.get("trace_state", {}),
        )

    def to_w3c_traceparent(self) -> str:
        """Convert to W3C traceparent header format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_w3c_traceparent(cls, header: str) -> "SpanContext":
        """Parse W3C traceparent header."""
        parts = header.split("-")
        if len(parts) != 4:
            raise ValueError(f"Invalid traceparent format: {header}")
        return cls(
            trace_id=parts[1],
            span_id=parts[2],
            trace_flags=int(parts[3], 16),
        )


@dataclass
class TracingSpan:
    """Represents a tracing span.

    Attributes:
        name: Span name
        context: Span context
        kind: Span kind
        start_time: Start timestamp
        end_time: End timestamp (None if still active)
        attributes: Span attributes
        events: Span events
        status: Span status
        status_message: Status message for errors
    """

    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple attributes."""
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """Set span status."""
        self.status = status
        self.status_message = message

    def set_ok(self) -> None:
        """Mark span as OK."""
        self.set_status(SpanStatus.OK)

    def set_error(self, message: str = "") -> None:
        """Mark span as error."""
        self.set_status(SpanStatus.ERROR, message)

    def record_exception(self, exception: Exception) -> None:
        """Record an exception."""
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )
        self.set_error(str(exception))

    def end(self, end_time: datetime | None = None) -> None:
        """End the span."""
        self.end_time = end_time or datetime.now(timezone.utc)

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status.value,
            "status_message": self.status_message,
        }


class TelemetryExporter(ABC):
    """Base class for telemetry exporters."""

    @abstractmethod
    def export_span(self, span: TracingSpan) -> bool:
        """Export a completed span.

        Args:
            span: Span to export

        Returns:
            True if export succeeded
        """
        pass

    @abstractmethod
    def export_metrics(self, metrics: dict[str, Any]) -> bool:
        """Export metrics.

        Args:
            metrics: Metrics to export

        Returns:
            True if export succeeded
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleExporter(TelemetryExporter):
    """Exporter that prints to console (for development/debugging)."""

    def __init__(self, verbose: bool = True):
        """Initialize console exporter.

        Args:
            verbose: Whether to print detailed output
        """
        self.verbose = verbose

    def export_span(self, span: TracingSpan) -> bool:
        """Export span to console."""
        if self.verbose:
            print(f"[SPAN] {span.name}")
            print(f"  Trace ID: {span.context.trace_id}")
            print(f"  Span ID: {span.context.span_id}")
            print(f"  Duration: {span.duration_ms:.2f}ms")
            print(f"  Status: {span.status.value}")
            if span.attributes:
                print(f"  Attributes: {span.attributes}")
        else:
            print(
                f"[SPAN] {span.name} - {span.duration_ms:.2f}ms - {span.status.value}"
            )
        return True

    def export_metrics(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to console."""
        print(f"[METRICS] {metrics}")
        return True

    def shutdown(self) -> None:
        """No-op for console exporter."""
        pass


class OTLPExporter(TelemetryExporter):
    """OpenTelemetry Protocol (OTLP) exporter.

    Note: Actual OTLP export requires opentelemetry-exporter-otlp package.
    This implementation provides the interface and buffers spans locally.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
    ):
        """Initialize OTLP exporter.

        Args:
            endpoint: OTLP endpoint URL
            headers: Optional headers for authentication
            timeout: Export timeout in seconds
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self._buffer: list[TracingSpan] = []
        self._metrics_buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def export_span(self, span: TracingSpan) -> bool:
        """Buffer span for export."""
        if self._shutdown:
            return False

        with self._lock:
            self._buffer.append(span)

        # In production, this would send to OTLP endpoint
        # For now, we buffer and log
        logger.debug(f"OTLP: Buffered span {span.name} ({span.context.span_id})")
        return True

    def export_metrics(self, metrics: dict[str, Any]) -> bool:
        """Buffer metrics for export."""
        if self._shutdown:
            return False

        with self._lock:
            self._metrics_buffer.append(metrics)

        logger.debug(f"OTLP: Buffered metrics {metrics.get('name', 'unknown')}")
        return True

    def flush(self) -> int:
        """Flush buffered data.

        Returns:
            Number of items flushed
        """
        with self._lock:
            count = len(self._buffer) + len(self._metrics_buffer)
            # In production, this would send to OTLP endpoint
            self._buffer.clear()
            self._metrics_buffer.clear()
        return count

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._shutdown = True
        self.flush()


@dataclass
class TracingConfig:
    """Configuration for tracing.

    Attributes:
        service_name: Name of the service
        service_version: Version of the service
        environment: Deployment environment
        exporter_type: Type of exporter (console, otlp)
        endpoint: OTLP endpoint URL
        headers: Authentication headers
        sample_rate: Sampling rate (0.0-1.0)
        enabled: Whether tracing is enabled
    """

    service_name: str = "truthound"
    service_version: str = "0.2.0"
    environment: str = "development"
    exporter_type: str = "console"
    endpoint: str = "http://localhost:4317"
    headers: dict[str, str] = field(default_factory=dict)
    sample_rate: float = 1.0
    enabled: bool = True


@dataclass
class MetricsConfig:
    """Configuration for metrics collection.

    Attributes:
        enabled: Whether metrics are enabled
        export_interval: Seconds between exports
        histogram_buckets: Buckets for histogram metrics
    """

    enabled: bool = True
    export_interval: float = 60.0
    histogram_buckets: list[float] = field(
        default_factory=lambda: [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    )


class MetricsCollector:
    """Collector for timeout-related metrics.

    Collects and aggregates metrics for:
    - Execution times
    - Timeout counts
    - Retry counts
    - Circuit breaker state changes
    """

    def __init__(self, config: MetricsConfig | None = None):
        """Initialize metrics collector.

        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    def increment(self, name: str, value: int = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter.

        Args:
            name: Counter name
            value: Increment value
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram value.

        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge value.

        Args:
            name: Gauge name
            value: Value to set
            labels: Optional labels
        """
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a key for metrics storage."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics.

        Returns:
            Dictionary of all metrics
        """
        with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {k: self._compute_histogram_stats(v) for k, v in self._histograms.items()},
                "gauges": dict(self._gauges),
            }

    def _compute_histogram_stats(self, values: list[float]) -> dict[str, Any]:
        """Compute histogram statistics."""
        if not values:
            return {"count": 0}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "sum": sum(values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p90": sorted_values[int(n * 0.9)],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        }

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


class TelemetryProvider:
    """Main provider for telemetry (tracing and metrics).

    This class manages the lifecycle of tracers and metrics collectors,
    providing a unified interface for observability.

    Example:
        provider = TelemetryProvider(TracingConfig(
            service_name="my-validator",
            exporter_type="otlp",
        ))

        with provider.trace("validate") as span:
            span.set_attribute("rows", 10000)
            result = validate(data)
    """

    def __init__(
        self,
        tracing_config: TracingConfig | None = None,
        metrics_config: MetricsConfig | None = None,
    ):
        """Initialize telemetry provider.

        Args:
            tracing_config: Tracing configuration
            metrics_config: Metrics configuration
        """
        self.tracing_config = tracing_config or TracingConfig()
        self.metrics_config = metrics_config or MetricsConfig()
        self._exporter = self._create_exporter()
        self._metrics = MetricsCollector(self.metrics_config)
        self._context_stack: list[SpanContext] = []
        self._lock = threading.Lock()

    def _create_exporter(self) -> TelemetryExporter:
        """Create the appropriate exporter."""
        if self.tracing_config.exporter_type == "otlp":
            return OTLPExporter(
                endpoint=self.tracing_config.endpoint,
                headers=self.tracing_config.headers,
            )
        return ConsoleExporter()

    @contextmanager
    def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[TracingSpan, None, None]:
        """Create a tracing span.

        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes

        Yields:
            TracingSpan
        """
        if not self.tracing_config.enabled:
            # No-op span
            span = TracingSpan(
                name=name,
                context=SpanContext.create_root(),
                kind=kind,
            )
            yield span
            return

        # Check sampling
        import random
        if random.random() > self.tracing_config.sample_rate:
            span = TracingSpan(
                name=name,
                context=SpanContext.create_root(),
                kind=kind,
            )
            yield span
            return

        # Create span context
        with self._lock:
            if self._context_stack:
                context = SpanContext.create_child(self._context_stack[-1])
            else:
                context = SpanContext.create_root()
            self._context_stack.append(context)

        # Create span
        span = TracingSpan(
            name=name,
            context=context,
            kind=kind,
        )
        if attributes:
            span.set_attributes(attributes)

        # Add service attributes
        span.set_attributes({
            "service.name": self.tracing_config.service_name,
            "service.version": self.tracing_config.service_version,
            "deployment.environment": self.tracing_config.environment,
        })

        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_ok()
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._exporter.export_span(span)

            with self._lock:
                if self._context_stack:
                    self._context_stack.pop()

            # Record metrics
            if span.duration_ms is not None:
                self._metrics.record_histogram(
                    "span.duration_ms",
                    span.duration_ms,
                    {"name": name, "status": span.status.value},
                )

    def get_current_context(self) -> SpanContext | None:
        """Get current span context.

        Returns:
            Current SpanContext or None
        """
        with self._lock:
            return self._context_stack[-1] if self._context_stack else None

    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject current context into carrier for propagation.

        Args:
            carrier: Dictionary to inject headers into
        """
        ctx = self.get_current_context()
        if ctx:
            carrier["traceparent"] = ctx.to_w3c_traceparent()

    def extract_context(self, carrier: dict[str, str]) -> SpanContext | None:
        """Extract context from carrier.

        Args:
            carrier: Dictionary containing headers

        Returns:
            Extracted SpanContext or None
        """
        traceparent = carrier.get("traceparent")
        if traceparent:
            try:
                return SpanContext.from_w3c_traceparent(traceparent)
            except ValueError:
                logger.warning(f"Invalid traceparent: {traceparent}")
        return None

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics.

        Returns:
            Dictionary of metrics
        """
        return self._metrics.get_metrics()

    def record_timeout(self, operation: str, timeout_seconds: float) -> None:
        """Record a timeout event.

        Args:
            operation: Operation that timed out
            timeout_seconds: Timeout value
        """
        self._metrics.increment("timeouts.total", 1, {"operation": operation})
        self._metrics.record_histogram(
            "timeout.configured_seconds",
            timeout_seconds,
            {"operation": operation},
        )

    def record_execution(self, operation: str, duration_ms: float, success: bool) -> None:
        """Record an execution.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            success: Whether execution succeeded
        """
        labels = {"operation": operation, "success": str(success).lower()}
        self._metrics.increment("executions.total", 1, labels)
        self._metrics.record_histogram("execution.duration_ms", duration_ms, labels)

    def shutdown(self) -> None:
        """Shutdown the telemetry provider."""
        self._exporter.shutdown()


# Module-level default provider
_default_provider: TelemetryProvider | None = None


def create_tracer(config: TracingConfig | None = None) -> TelemetryProvider:
    """Create or get the default telemetry provider.

    Args:
        config: Optional tracing configuration

    Returns:
        TelemetryProvider
    """
    global _default_provider
    if _default_provider is None or config is not None:
        _default_provider = TelemetryProvider(config)
    return _default_provider


def create_metrics_collector(config: MetricsConfig | None = None) -> MetricsCollector:
    """Create a metrics collector.

    Args:
        config: Optional metrics configuration

    Returns:
        MetricsCollector
    """
    return MetricsCollector(config)


def trace_operation(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to trace a function.

    Args:
        name: Span name
        attributes: Initial attributes

    Returns:
        Decorated function

    Example:
        @trace_operation("validate_column")
        def validate_column(data, column):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            provider = create_tracer()
            with provider.trace(name, attributes=attributes) as span:
                span.set_attribute("function", func.__name__)
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator
