"""Protocol definitions for OpenTelemetry compatibility.

This module defines protocols (interfaces) that both Truthound native
and OpenTelemetry SDK implementations must follow. These protocols
enable duck-typing based interoperability.

The protocols are designed to be compatible with:
- OpenTelemetry Python API 1.x
- Truthound native tracing implementation
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)


T = TypeVar("T")


# =============================================================================
# Span Context Protocol
# =============================================================================


@runtime_checkable
class SpanContextProtocol(Protocol):
    """Protocol for span context.

    Compatible with both:
    - opentelemetry.trace.SpanContext
    - truthound.observability.tracing.span.SpanContextData
    """

    @property
    def trace_id(self) -> str | int:
        """Get the trace ID."""
        ...

    @property
    def span_id(self) -> str | int:
        """Get the span ID."""
        ...

    @property
    def trace_flags(self) -> int:
        """Get the trace flags."""
        ...

    @property
    def is_valid(self) -> bool:
        """Check if context is valid."""
        ...

    @property
    def is_remote(self) -> bool:
        """Check if context is from a remote parent."""
        ...


# =============================================================================
# Span Protocol
# =============================================================================


@runtime_checkable
class SpanProtocol(Protocol):
    """Protocol for spans.

    Compatible with both:
    - opentelemetry.trace.Span
    - truthound.observability.tracing.span.Span
    """

    def get_span_context(self) -> SpanContextProtocol:
        """Get the span context."""
        ...

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        ...

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        """Set multiple attributes."""
        ...

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | float | None = None,
    ) -> None:
        """Add an event to the span."""
        ...

    def set_status(self, status: Any, description: str | None = None) -> None:
        """Set span status."""
        ...

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | float | None = None,
        escaped: bool = False,
    ) -> None:
        """Record an exception."""
        ...

    def update_name(self, name: str) -> None:
        """Update span name."""
        ...

    def end(self, end_time: int | float | None = None) -> None:
        """End the span."""
        ...

    def is_recording(self) -> bool:
        """Check if span is recording."""
        ...


# =============================================================================
# Tracer Protocol
# =============================================================================


@runtime_checkable
class TracerProtocol(Protocol):
    """Protocol for tracers.

    Compatible with both:
    - opentelemetry.trace.Tracer
    - truthound.observability.tracing.provider.Tracer
    """

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Any] | None = None,
        start_time: int | float | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> SpanProtocol:
        """Start a new span."""
        ...

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = None,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Any] | None = None,
        start_time: int | float | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[SpanProtocol]:
        """Start a span and set it as current."""
        ...


# =============================================================================
# TracerProvider Protocol
# =============================================================================


@runtime_checkable
class TracerProviderProtocol(Protocol):
    """Protocol for tracer providers.

    Compatible with both:
    - opentelemetry.trace.TracerProvider
    - truthound.observability.tracing.provider.TracerProvider
    """

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str = "",
        schema_url: str = "",
    ) -> TracerProtocol:
        """Get a tracer instance."""
        ...


# =============================================================================
# Span Processor Protocol
# =============================================================================


@runtime_checkable
class SpanProcessorProtocol(Protocol):
    """Protocol for span processors.

    Compatible with both:
    - opentelemetry.sdk.trace.SpanProcessor
    - truthound.observability.tracing.processor.SpanProcessor
    """

    def on_start(
        self,
        span: SpanProtocol,
        parent_context: SpanContextProtocol | None = None,
    ) -> None:
        """Called when a span starts."""
        ...

    def on_end(self, span: SpanProtocol) -> None:
        """Called when a span ends."""
        ...

    def shutdown(self) -> bool:
        """Shutdown the processor."""
        ...

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans."""
        ...


# =============================================================================
# Span Exporter Protocol
# =============================================================================


@runtime_checkable
class SpanExporterProtocol(Protocol):
    """Protocol for span exporters.

    Compatible with both:
    - opentelemetry.sdk.trace.export.SpanExporter
    - truthound.observability.tracing.exporter.SpanExporter
    """

    def export(self, spans: Sequence[SpanProtocol]) -> Any:
        """Export spans."""
        ...

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        ...

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending exports."""
        ...


# =============================================================================
# Sampler Protocol
# =============================================================================


@runtime_checkable
class SamplerProtocol(Protocol):
    """Protocol for samplers.

    Compatible with both:
    - opentelemetry.sdk.trace.Sampler
    - truthound.observability.tracing.sampler.Sampler
    """

    def should_sample(
        self,
        parent_context: SpanContextProtocol | None,
        trace_id: str | int,
        name: str,
        kind: Any = None,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Any] | None = None,
    ) -> Any:
        """Make a sampling decision."""
        ...

    def get_description(self) -> str:
        """Get sampler description."""
        ...


# =============================================================================
# Propagator Protocol
# =============================================================================


@runtime_checkable
class PropagatorProtocol(Protocol):
    """Protocol for context propagators.

    Compatible with both:
    - opentelemetry.propagators.textmap.TextMapPropagator
    - truthound.observability.tracing.propagator.Propagator
    """

    def extract(
        self,
        carrier: Mapping[str, str],
        context: Any = None,
        getter: Any = None,
    ) -> Any:
        """Extract context from carrier."""
        ...

    def inject(
        self,
        carrier: dict[str, str],
        context: Any = None,
        setter: Any = None,
    ) -> None:
        """Inject context into carrier."""
        ...

    @property
    def fields(self) -> set[str]:
        """Get the fields used by this propagator."""
        ...


# =============================================================================
# Resource Protocol
# =============================================================================


@runtime_checkable
class ResourceProtocol(Protocol):
    """Protocol for resources.

    Compatible with both:
    - opentelemetry.sdk.resources.Resource
    - truthound.observability.tracing.resource.Resource
    """

    @property
    def attributes(self) -> Mapping[str, Any]:
        """Get resource attributes."""
        ...

    def merge(self, other: "ResourceProtocol") -> "ResourceProtocol":
        """Merge with another resource."""
        ...


# =============================================================================
# Context Protocol
# =============================================================================


@runtime_checkable
class ContextProtocol(Protocol):
    """Protocol for context management.

    Compatible with both:
    - opentelemetry.context.Context
    - truthound context management
    """

    def get(self, key: str, default: T = None) -> T:
        """Get a value from context."""
        ...

    def set(self, key: str, value: Any) -> "ContextProtocol":
        """Set a value in context."""
        ...


# =============================================================================
# Factory Protocols
# =============================================================================


class TracerProviderFactory(Protocol):
    """Factory for creating TracerProviders."""

    def create(
        self,
        resource: ResourceProtocol | None = None,
        sampler: SamplerProtocol | None = None,
    ) -> TracerProviderProtocol:
        """Create a new TracerProvider."""
        ...


class SpanExporterFactory(Protocol):
    """Factory for creating SpanExporters."""

    def create_otlp(
        self,
        endpoint: str,
        headers: Mapping[str, str] | None = None,
    ) -> SpanExporterProtocol:
        """Create an OTLP exporter."""
        ...

    def create_jaeger(
        self,
        agent_host: str = "localhost",
        agent_port: int = 6831,
    ) -> SpanExporterProtocol:
        """Create a Jaeger exporter."""
        ...

    def create_zipkin(
        self,
        endpoint: str,
    ) -> SpanExporterProtocol:
        """Create a Zipkin exporter."""
        ...

    def create_console(self) -> SpanExporterProtocol:
        """Create a console exporter."""
        ...
