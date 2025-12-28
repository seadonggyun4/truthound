"""Distributed tracing for store operations.

This module provides OpenTelemetry-compatible distributed tracing with
support for multiple backends (OTLP, Jaeger, Zipkin, etc.).

Features:
- OpenTelemetry-compatible API
- Context propagation (W3C TraceContext, B3)
- Multiple exporter support
- Automatic span attributes
- Error recording
- Sampling strategies
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

from truthound.stores.observability.config import TracingConfig, TracingSampler
from truthound.stores.observability.protocols import (
    ObservabilityContext,
    TracingBackend,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SpanKind(str, Enum):
    """Types of spans."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span status codes."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for a span, used for propagation.

    Follows W3C TraceContext specification.
    """

    trace_id: str
    span_id: str
    trace_flags: int = 1  # 1 = sampled
    trace_state: str = ""
    is_remote: bool = False

    @classmethod
    def generate(cls) -> "SpanContext":
        """Generate a new span context."""
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
        )

    @classmethod
    def from_parent(cls, parent: "SpanContext") -> "SpanContext":
        """Create child span context from parent."""
        return cls(
            trace_id=parent.trace_id,
            span_id=uuid.uuid4().hex[:16],
            trace_flags=parent.trace_flags,
            trace_state=parent.trace_state,
        )

    def to_traceparent(self) -> str:
        """Convert to W3C traceparent header format."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_traceparent(cls, value: str) -> "SpanContext | None":
        """Parse W3C traceparent header."""
        try:
            parts = value.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None
            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=int(parts[3], 16),
                is_remote=True,
            )
        except (ValueError, IndexError):
            return None


@dataclass
class SpanEvent:
    """An event within a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanLink:
    """A link to another span."""

    context: SpanContext
    attributes: dict[str, Any] = field(default_factory=dict)


class Span:
    """A span representing a unit of work.

    Spans can be used as context managers for automatic timing and status.
    """

    def __init__(
        self,
        name: str,
        context: SpanContext,
        parent_context: SpanContext | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
        start_time: float | None = None,
    ) -> None:
        self.name = name
        self.context = context
        self.parent_context = parent_context
        self.kind = kind
        self.attributes: dict[str, Any] = attributes or {}
        self.events: list[SpanEvent] = []
        self.links: list[SpanLink] = []
        self.status = SpanStatus.UNSET
        self.status_message: str = ""
        self.start_time = start_time or time.time()
        self.end_time: float | None = None
        self._ended = False

    @property
    def trace_id(self) -> str:
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        return self.context.span_id

    @property
    def duration_seconds(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a span attribute."""
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: dict[str, Any]) -> "Span":
        """Set multiple span attributes."""
        self.attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> "Span":
        """Add an event to the span."""
        self.events.append(
            SpanEvent(
                name=name,
                timestamp=timestamp or time.time(),
                attributes=attributes or {},
            )
        )
        return self

    def add_link(
        self,
        context: SpanContext,
        attributes: dict[str, Any] | None = None,
    ) -> "Span":
        """Add a link to another span."""
        self.links.append(SpanLink(context=context, attributes=attributes or {}))
        return self

    def set_status(self, status: SpanStatus, message: str = "") -> "Span":
        """Set the span status."""
        self.status = status
        self.status_message = message
        return self

    def record_exception(
        self,
        exception: Exception,
        attributes: dict[str, Any] | None = None,
    ) -> "Span":
        """Record an exception in the span."""
        exc_attrs = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            **(attributes or {}),
        }
        self.add_event("exception", exc_attrs)
        self.set_status(SpanStatus.ERROR, str(exception))
        return self

    def end(self, end_time: float | None = None) -> None:
        """End the span."""
        if not self._ended:
            self.end_time = end_time or time.time()
            self._ended = True

    def __enter__(self) -> "Span":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if exc_val is not None and isinstance(exc_val, Exception):
            self.record_exception(exc_val)
        elif self.status == SpanStatus.UNSET:
            self.set_status(SpanStatus.OK)
        self.end()

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_context.span_id if self.parent_context else None,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "links": [
                {
                    "trace_id": l.context.trace_id,
                    "span_id": l.context.span_id,
                    "attributes": l.attributes,
                }
                for l in self.links
            ],
        }


class BaseTracer(ABC):
    """Base class for tracers."""

    def __init__(self, config: TracingConfig) -> None:
        self.config = config
        self._current_span: threading.local = threading.local()

    def _should_sample(self, parent_context: SpanContext | None = None) -> bool:
        """Determine if a span should be sampled."""
        sampler = self.config.sampler

        if sampler == TracingSampler.ALWAYS_ON:
            return True
        elif sampler == TracingSampler.ALWAYS_OFF:
            return False
        elif sampler == TracingSampler.RATIO:
            import random
            return random.random() < self.config.sample_ratio
        elif sampler == TracingSampler.PARENT_BASED:
            if parent_context:
                return parent_context.trace_flags & 1 == 1
            # No parent, use ratio sampling
            import random
            return random.random() < self.config.sample_ratio

        return True

    def get_current_span(self) -> Span | None:
        """Get the currently active span."""
        return getattr(self._current_span, "span", None)

    def _set_current_span(self, span: Span | None) -> None:
        """Set the current span."""
        self._current_span.span = span

    @abstractmethod
    def start_span(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span."""
        ...

    @abstractmethod
    @contextmanager
    def trace(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager for tracing."""
        ...

    @abstractmethod
    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject tracing context into carrier."""
        ...

    @abstractmethod
    def extract_context(self, carrier: dict[str, str]) -> ObservabilityContext | None:
        """Extract tracing context from carrier."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered spans."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the tracer."""
        ...


class NoopTracer(BaseTracer):
    """No-operation tracer for when tracing is disabled."""

    def __init__(self, config: TracingConfig | None = None) -> None:
        super().__init__(config or TracingConfig(enabled=False))

    def start_span(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        span_context = SpanContext.generate()
        return Span(name, span_context, kind=SpanKind(kind), attributes=attributes)

    @contextmanager
    def trace(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        span = self.start_span(name, context, kind, attributes)
        try:
            yield span
        finally:
            span.end()

    def inject_context(self, carrier: dict[str, str]) -> None:
        pass

    def extract_context(self, carrier: dict[str, str]) -> ObservabilityContext | None:
        return None

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class InMemoryTracer(BaseTracer):
    """In-memory tracer for testing."""

    def __init__(self, config: TracingConfig | None = None) -> None:
        super().__init__(config or TracingConfig())
        self._spans: list[Span] = []
        self._lock = threading.Lock()

    def start_span(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        parent = self.get_current_span()
        parent_context = parent.context if parent else None

        if context and context.trace_id:
            span_context = SpanContext(
                trace_id=context.trace_id,
                span_id=uuid.uuid4().hex[:16],
            )
        elif parent_context:
            span_context = SpanContext.from_parent(parent_context)
        else:
            span_context = SpanContext.generate()

        if not self._should_sample(parent_context):
            span_context.trace_flags = 0

        span = Span(
            name=name,
            context=span_context,
            parent_context=parent_context,
            kind=SpanKind(kind),
            attributes=attributes,
        )

        # Add service info
        span.set_attribute("service.name", self.config.service_name)

        self._set_current_span(span)
        return span

    @contextmanager
    def trace(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        span = self.start_span(name, context, kind, attributes)
        previous_span = self.get_current_span()
        self._set_current_span(span)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            with self._lock:
                self._spans.append(span)
            self._set_current_span(previous_span)

    def inject_context(self, carrier: dict[str, str]) -> None:
        span = self.get_current_span()
        if span:
            carrier["traceparent"] = span.context.to_traceparent()
            if span.context.trace_state:
                carrier["tracestate"] = span.context.trace_state

    def extract_context(self, carrier: dict[str, str]) -> ObservabilityContext | None:
        traceparent = carrier.get("traceparent")
        if traceparent:
            span_context = SpanContext.from_traceparent(traceparent)
            if span_context:
                return ObservabilityContext(
                    correlation_id=span_context.trace_id,
                    trace_id=span_context.trace_id,
                    span_id=span_context.span_id,
                )
        return None

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @property
    def spans(self) -> list[Span]:
        """Get all recorded spans (for testing)."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear all spans (for testing)."""
        with self._lock:
            self._spans.clear()


class OpenTelemetryTracer(BaseTracer):
    """OpenTelemetry-based tracer with OTLP export.

    This tracer uses the OpenTelemetry SDK if available, falling back to
    the in-memory tracer if not.
    """

    def __init__(self, config: TracingConfig | None = None) -> None:
        super().__init__(config or TracingConfig())
        self._otel_tracer: Any = None
        self._provider: Any = None
        self._fallback = InMemoryTracer(config)
        self._initialized = False

        if self.config.enabled:
            self._try_initialize_otel()

    def _try_initialize_otel(self) -> None:
        """Try to initialize OpenTelemetry SDK."""
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME

            # Create resource
            resource = Resource.create({SERVICE_NAME: self.config.service_name})

            # Create provider
            self._provider = TracerProvider(resource=resource)

            # Add exporter based on config
            exporter = self._create_exporter()
            if exporter:
                self._provider.add_span_processor(BatchSpanProcessor(exporter))

            # Set as global provider
            otel_trace.set_tracer_provider(self._provider)

            # Get tracer
            self._otel_tracer = otel_trace.get_tracer(
                self.config.service_name,
                schema_url="https://opentelemetry.io/schemas/1.11.0",
            )

            self._initialized = True
            logger.info("OpenTelemetry tracer initialized")

        except ImportError:
            logger.warning(
                "OpenTelemetry SDK not installed. Using in-memory tracer. "
                "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")

    def _create_exporter(self) -> Any:
        """Create the appropriate exporter based on config."""
        exporter_type = self.config.exporter.lower()

        try:
            if exporter_type == "otlp":
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                return OTLPSpanExporter(
                    endpoint=self.config.endpoint or "http://localhost:4317",
                    headers=self.config.headers or None,
                )

            elif exporter_type == "jaeger":
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                return JaegerExporter(
                    agent_host_name=self.config.endpoint or "localhost",
                    agent_port=6831,
                )

            elif exporter_type == "zipkin":
                from opentelemetry.exporter.zipkin.json import ZipkinExporter

                return ZipkinExporter(
                    endpoint=self.config.endpoint or "http://localhost:9411/api/v2/spans",
                )

            elif exporter_type == "console":
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter

                return ConsoleSpanExporter()

            elif exporter_type == "noop":
                return None

        except ImportError as e:
            logger.warning(f"Exporter {exporter_type} not available: {e}")
            return None

        return None

    def start_span(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        if self._initialized and self._otel_tracer:
            return self._start_otel_span(name, context, kind, attributes)
        return self._fallback.start_span(name, context, kind, attributes)

    def _start_otel_span(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a span using OpenTelemetry."""
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import SpanKind as OtelSpanKind

        kind_map = {
            "internal": OtelSpanKind.INTERNAL,
            "server": OtelSpanKind.SERVER,
            "client": OtelSpanKind.CLIENT,
            "producer": OtelSpanKind.PRODUCER,
            "consumer": OtelSpanKind.CONSUMER,
        }

        otel_span = self._otel_tracer.start_span(
            name,
            kind=kind_map.get(kind, OtelSpanKind.INTERNAL),
            attributes=attributes,
        )

        # Wrap in our Span class
        otel_context = otel_span.get_span_context()
        span_context = SpanContext(
            trace_id=format(otel_context.trace_id, "032x"),
            span_id=format(otel_context.span_id, "016x"),
            trace_flags=otel_context.trace_flags,
        )

        span = Span(
            name=name,
            context=span_context,
            kind=SpanKind(kind),
            attributes=attributes,
        )
        span._otel_span = otel_span  # Store reference for proper ending
        return span

    @contextmanager
    def trace(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        if self._initialized and self._otel_tracer:
            with self._trace_otel(name, context, kind, attributes) as span:
                yield span
        else:
            with self._fallback.trace(name, context, kind, attributes) as span:
                yield span

    def _trace_otel(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Trace using OpenTelemetry."""
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import SpanKind as OtelSpanKind

        kind_map = {
            "internal": OtelSpanKind.INTERNAL,
            "server": OtelSpanKind.SERVER,
            "client": OtelSpanKind.CLIENT,
            "producer": OtelSpanKind.PRODUCER,
            "consumer": OtelSpanKind.CONSUMER,
        }

        with self._otel_tracer.start_as_current_span(
            name,
            kind=kind_map.get(kind, OtelSpanKind.INTERNAL),
            attributes=attributes,
        ) as otel_span:
            otel_context = otel_span.get_span_context()
            span_context = SpanContext(
                trace_id=format(otel_context.trace_id, "032x"),
                span_id=format(otel_context.span_id, "016x"),
                trace_flags=otel_context.trace_flags,
            )

            span = Span(
                name=name,
                context=span_context,
                kind=SpanKind(kind),
                attributes=attributes,
            )
            span._otel_span = otel_span

            try:
                yield span
            except Exception as e:
                otel_span.record_exception(e)
                otel_span.set_status(
                    otel_trace.Status(otel_trace.StatusCode.ERROR, str(e))
                )
                raise

    def inject_context(self, carrier: dict[str, str]) -> None:
        if self._initialized:
            try:
                from opentelemetry import propagate

                propagate.inject(carrier)
                return
            except Exception:
                pass
        self._fallback.inject_context(carrier)

    def extract_context(self, carrier: dict[str, str]) -> ObservabilityContext | None:
        if self._initialized:
            try:
                from opentelemetry import propagate, trace as otel_trace

                ctx = propagate.extract(carrier)
                span = otel_trace.get_current_span(ctx)
                if span:
                    span_context = span.get_span_context()
                    return ObservabilityContext(
                        correlation_id=format(span_context.trace_id, "032x"),
                        trace_id=format(span_context.trace_id, "032x"),
                        span_id=format(span_context.span_id, "016x"),
                    )
            except Exception:
                pass
        return self._fallback.extract_context(carrier)

    def flush(self) -> None:
        if self._provider:
            try:
                self._provider.force_flush()
            except Exception as e:
                logger.error(f"Failed to flush traces: {e}")

    def shutdown(self) -> None:
        if self._provider:
            try:
                self._provider.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown tracer: {e}")


class Tracer:
    """Factory for creating tracers based on configuration."""

    _instance: BaseTracer | None = None
    _lock = threading.Lock()

    @classmethod
    def get_tracer(cls, config: TracingConfig | None = None) -> BaseTracer:
        """Get or create a tracer instance."""
        with cls._lock:
            if cls._instance is None:
                config = config or TracingConfig()
                if not config.enabled:
                    cls._instance = NoopTracer(config)
                else:
                    cls._instance = OpenTelemetryTracer(config)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the tracer instance (for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.shutdown()
            cls._instance = None
