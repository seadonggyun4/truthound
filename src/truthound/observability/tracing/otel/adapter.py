"""OpenTelemetry adapter implementation.

This module provides adapter classes that wrap both Truthound native
and OpenTelemetry SDK implementations behind a unified interface.

The adapter pattern allows:
- Seamless switching between backends
- Gradual migration from Truthound to OTEL SDK
- Using OTEL exporters with Truthound spans
- Using Truthound spans with OTEL processors
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Iterator, Mapping, Sequence

from truthound.observability.tracing.otel.detection import (
    detect_otel_availability,
    is_otel_sdk_available,
)
from truthound.observability.tracing.otel.protocols import (
    SpanContextProtocol,
    SpanProtocol,
    TracerProtocol,
    TracerProviderProtocol,
    SpanProcessorProtocol,
    SpanExporterProtocol,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Selection
# =============================================================================


class TracingBackend(str, Enum):
    """Available tracing backends."""

    AUTO = "auto"  # Auto-detect best available
    TRUTHOUND = "truthound"  # Force Truthound native
    OPENTELEMETRY = "opentelemetry"  # Force OpenTelemetry SDK


@dataclass
class AdapterConfig:
    """Configuration for the tracing adapter.

    Attributes:
        backend: Which backend to use.
        service_name: Service name for traces.
        service_version: Service version.
        environment: Deployment environment.
        exporter_type: Type of exporter (console, otlp, jaeger, zipkin).
        exporter_endpoint: Exporter endpoint URL.
        exporter_headers: Additional headers for exporter.
        sampling_ratio: Sampling ratio (0.0 to 1.0).
        batch_export: Use batch span processor.
        propagators: List of propagator types.
        auto_instrument: Enable auto-instrumentation (if OTEL SDK).
    """

    backend: TracingBackend = TracingBackend.AUTO
    service_name: str = "truthound"
    service_version: str = ""
    environment: str = ""
    exporter_type: str = "console"
    exporter_endpoint: str = ""
    exporter_headers: dict[str, str] = field(default_factory=dict)
    sampling_ratio: float = 1.0
    batch_export: bool = True
    propagators: list[str] = field(default_factory=lambda: ["w3c"])
    auto_instrument: bool = False

    @classmethod
    def from_env(cls) -> "AdapterConfig":
        """Create configuration from environment variables."""
        import os

        backend_str = os.environ.get("TRUTHOUND_TRACING_BACKEND", "auto").lower()
        backend = TracingBackend(backend_str) if backend_str in [b.value for b in TracingBackend] else TracingBackend.AUTO

        # Parse headers
        headers = {}
        headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers_str:
            for pair in headers_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers[key.strip()] = value.strip()

        # Determine exporter type
        exporter_type = "console"
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if endpoint:
            exporter_type = "otlp"

        # Parse sampling ratio
        try:
            sampling_ratio = float(os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0"))
        except ValueError:
            sampling_ratio = 1.0

        return cls(
            backend=backend,
            service_name=os.environ.get("OTEL_SERVICE_NAME", "truthound"),
            service_version=os.environ.get("OTEL_SERVICE_VERSION", ""),
            environment=os.environ.get("DEPLOYMENT_ENVIRONMENT", ""),
            exporter_type=exporter_type,
            exporter_endpoint=endpoint,
            exporter_headers=headers,
            sampling_ratio=sampling_ratio,
        )


# =============================================================================
# Span Context Adapter
# =============================================================================


class SpanContextAdapter:
    """Adapter for span context that works with both backends.

    Normalizes span context data from either Truthound or OTEL format.
    """

    def __init__(
        self,
        trace_id: str | int,
        span_id: str | int,
        trace_flags: int = 1,
        trace_state: str = "",
        is_remote: bool = False,
        _native: Any = None,
    ) -> None:
        """Initialize span context adapter.

        Args:
            trace_id: Trace ID (hex string or int).
            span_id: Span ID (hex string or int).
            trace_flags: Trace flags (0=not sampled, 1=sampled).
            trace_state: W3C trace state string.
            is_remote: Whether context is from remote parent.
            _native: Native span context object (for unwrapping).
        """
        # Normalize to hex strings
        if isinstance(trace_id, int):
            self._trace_id = format(trace_id, "032x")
        else:
            self._trace_id = str(trace_id)

        if isinstance(span_id, int):
            self._span_id = format(span_id, "016x")
        else:
            self._span_id = str(span_id)

        self._trace_flags = trace_flags
        self._trace_state = trace_state
        self._is_remote = is_remote
        self._native = _native

    @property
    def trace_id(self) -> str:
        """Get trace ID as hex string."""
        return self._trace_id

    @property
    def trace_id_int(self) -> int:
        """Get trace ID as integer."""
        return int(self._trace_id, 16)

    @property
    def span_id(self) -> str:
        """Get span ID as hex string."""
        return self._span_id

    @property
    def span_id_int(self) -> int:
        """Get span ID as integer."""
        return int(self._span_id, 16)

    @property
    def trace_flags(self) -> int:
        """Get trace flags."""
        return self._trace_flags

    @property
    def trace_state(self) -> str:
        """Get trace state."""
        return self._trace_state

    @property
    def is_valid(self) -> bool:
        """Check if context is valid."""
        return bool(self._trace_id and self._span_id and
                    self._trace_id != "0" * 32 and self._span_id != "0" * 16)

    @property
    def is_remote(self) -> bool:
        """Check if context is from remote parent."""
        return self._is_remote

    @property
    def is_sampled(self) -> bool:
        """Check if span is sampled."""
        return bool(self._trace_flags & 0x01)

    def to_w3c_traceparent(self) -> str:
        """Convert to W3C traceparent header."""
        return f"00-{self._trace_id}-{self._span_id}-{self._trace_flags:02x}"

    @classmethod
    def from_w3c_traceparent(cls, header: str) -> "SpanContextAdapter | None":
        """Parse from W3C traceparent header."""
        try:
            parts = header.split("-")
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

    @classmethod
    def from_truthound(cls, ctx: Any) -> "SpanContextAdapter":
        """Create from Truthound SpanContextData."""
        return cls(
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            trace_flags=ctx.trace_flags,
            trace_state=getattr(ctx, "trace_state", ""),
            is_remote=getattr(ctx, "is_remote", False),
            _native=ctx,
        )

    @classmethod
    def from_otel(cls, ctx: Any) -> "SpanContextAdapter":
        """Create from OpenTelemetry SpanContext."""
        return cls(
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            trace_flags=ctx.trace_flags,
            trace_state=str(ctx.trace_state) if ctx.trace_state else "",
            is_remote=ctx.is_remote,
            _native=ctx,
        )

    def to_truthound(self) -> Any:
        """Convert to Truthound SpanContextData."""
        from truthound.observability.tracing.span import SpanContextData

        return SpanContextData(
            trace_id=self._trace_id,
            span_id=self._span_id,
            trace_flags=self._trace_flags,
            trace_state=self._trace_state,
            is_remote=self._is_remote,
        )

    def to_otel(self) -> Any:
        """Convert to OpenTelemetry SpanContext."""
        if not is_otel_sdk_available():
            raise ImportError("OpenTelemetry SDK not available")

        from opentelemetry.trace import SpanContext, TraceState

        trace_state = TraceState()
        if self._trace_state:
            for pair in self._trace_state.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    trace_state = trace_state.add(key.strip(), value.strip())

        return SpanContext(
            trace_id=self.trace_id_int,
            span_id=self.span_id_int,
            is_remote=self._is_remote,
            trace_flags=self._trace_flags,
            trace_state=trace_state,
        )

    def unwrap(self) -> Any:
        """Get the native span context object."""
        return self._native


# =============================================================================
# Span Adapter
# =============================================================================


class SpanAdapter:
    """Adapter for spans that works with both backends.

    Provides a unified interface regardless of whether the underlying
    span is from Truthound or OpenTelemetry SDK.
    """

    def __init__(self, span: Any, backend: TracingBackend) -> None:
        """Initialize span adapter.

        Args:
            span: Native span object.
            backend: Backend type.
        """
        self._span = span
        self._backend = backend

    @property
    def context(self) -> SpanContextAdapter:
        """Get span context."""
        if self._backend == TracingBackend.OPENTELEMETRY:
            return SpanContextAdapter.from_otel(self._span.get_span_context())
        else:
            return SpanContextAdapter.from_truthound(self._span.context)

    @property
    def name(self) -> str:
        """Get span name."""
        return self._span.name

    def get_span_context(self) -> SpanContextAdapter:
        """Get span context (OTEL API compatible)."""
        return self.context

    def set_attribute(self, key: str, value: Any) -> "SpanAdapter":
        """Set a span attribute."""
        self._span.set_attribute(key, value)
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> "SpanAdapter":
        """Set multiple attributes."""
        if self._backend == TracingBackend.OPENTELEMETRY:
            self._span.set_attributes(dict(attributes))
        else:
            self._span.set_attributes(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | float | None = None,
    ) -> "SpanAdapter":
        """Add an event to the span."""
        self._span.add_event(name, attributes=attributes, timestamp=timestamp)
        return self

    def set_status(self, status: Any, description: str | None = None) -> "SpanAdapter":
        """Set span status."""
        if self._backend == TracingBackend.OPENTELEMETRY:
            from opentelemetry.trace import StatusCode

            if isinstance(status, str):
                status_map = {"ok": StatusCode.OK, "error": StatusCode.ERROR}
                status = status_map.get(status.lower(), StatusCode.UNSET)
            self._span.set_status(status, description)
        else:
            from truthound.observability.tracing.span import StatusCode

            if isinstance(status, str):
                status_map = {"ok": StatusCode.OK, "error": StatusCode.ERROR}
                status = status_map.get(status.lower(), StatusCode.UNSET)
            self._span.set_status(status, description or "")
        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | float | None = None,
        escaped: bool = False,
    ) -> "SpanAdapter":
        """Record an exception."""
        if self._backend == TracingBackend.OPENTELEMETRY:
            self._span.record_exception(exception, attributes=attributes)
        else:
            self._span.record_exception(exception, attributes=attributes, escaped=escaped)
        return self

    def update_name(self, name: str) -> "SpanAdapter":
        """Update span name."""
        self._span.update_name(name)
        return self

    def end(self, end_time: int | float | None = None) -> None:
        """End the span."""
        self._span.end(end_time)

    def is_recording(self) -> bool:
        """Check if span is recording."""
        return self._span.is_recording()

    def unwrap(self) -> Any:
        """Get the native span object."""
        return self._span

    def __enter__(self) -> "SpanAdapter":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if exc_val is not None:
            self.record_exception(exc_val, escaped=True)
            self.set_status("error", str(exc_val))
        self.end()


# =============================================================================
# Tracer Adapter
# =============================================================================


class TracerAdapter:
    """Adapter for tracers that works with both backends.

    Provides a unified interface for creating spans regardless of
    the underlying implementation.
    """

    def __init__(self, tracer: Any, backend: TracingBackend) -> None:
        """Initialize tracer adapter.

        Args:
            tracer: Native tracer object.
            backend: Backend type.
        """
        self._tracer = tracer
        self._backend = backend

    @property
    def name(self) -> str:
        """Get tracer name."""
        if hasattr(self._tracer, "name"):
            return self._tracer.name
        if hasattr(self._tracer, "_name"):
            return self._tracer._name
        return "unknown"

    def start_span(
        self,
        name: str,
        context: SpanContextAdapter | None = None,
        kind: Any = None,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Any] | None = None,
        start_time: int | float | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> SpanAdapter:
        """Start a new span.

        Args:
            name: Span name.
            context: Parent context (optional).
            kind: Span kind.
            attributes: Initial attributes.
            links: Links to other spans.
            start_time: Start time.
            record_exception: Whether to record exceptions.
            set_status_on_exception: Whether to set error status on exception.

        Returns:
            SpanAdapter wrapping the created span.
        """
        if self._backend == TracingBackend.OPENTELEMETRY:
            return self._start_span_otel(
                name, context, kind, attributes, links, start_time
            )
        else:
            return self._start_span_truthound(
                name, context, kind, attributes, links, start_time
            )

    def _start_span_truthound(
        self,
        name: str,
        context: SpanContextAdapter | None,
        kind: Any,
        attributes: Mapping[str, Any] | None,
        links: Sequence[Any] | None,
        start_time: int | float | None,
    ) -> SpanAdapter:
        """Start span using Truthound backend."""
        from truthound.observability.tracing.span import SpanKind

        # Convert kind
        span_kind = SpanKind.INTERNAL
        if kind is not None:
            if isinstance(kind, SpanKind):
                span_kind = kind
            elif hasattr(kind, "name"):
                kind_map = {
                    "INTERNAL": SpanKind.INTERNAL,
                    "SERVER": SpanKind.SERVER,
                    "CLIENT": SpanKind.CLIENT,
                    "PRODUCER": SpanKind.PRODUCER,
                    "CONSUMER": SpanKind.CONSUMER,
                }
                span_kind = kind_map.get(kind.name, SpanKind.INTERNAL)

        # Convert parent context
        parent = context.to_truthound() if context else None

        span = self._tracer.start_span(
            name=name,
            kind=span_kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            parent=parent,
        )

        return SpanAdapter(span, self._backend)

    def _start_span_otel(
        self,
        name: str,
        context: SpanContextAdapter | None,
        kind: Any,
        attributes: Mapping[str, Any] | None,
        links: Sequence[Any] | None,
        start_time: int | float | None,
    ) -> SpanAdapter:
        """Start span using OpenTelemetry backend."""
        from opentelemetry.trace import SpanKind

        # Convert kind
        span_kind = SpanKind.INTERNAL
        if kind is not None:
            if isinstance(kind, SpanKind):
                span_kind = kind
            elif hasattr(kind, "name"):
                kind_map = {
                    "INTERNAL": SpanKind.INTERNAL,
                    "SERVER": SpanKind.SERVER,
                    "CLIENT": SpanKind.CLIENT,
                    "PRODUCER": SpanKind.PRODUCER,
                    "CONSUMER": SpanKind.CONSUMER,
                }
                span_kind = kind_map.get(kind.name, SpanKind.INTERNAL)

        # Start span (context handling differs in OTEL)
        span = self._tracer.start_span(
            name=name,
            kind=span_kind,
            attributes=dict(attributes) if attributes else None,
            start_time=start_time,
        )

        return SpanAdapter(span, self._backend)

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: SpanContextAdapter | None = None,
        kind: Any = None,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Any] | None = None,
        start_time: int | float | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[SpanAdapter]:
        """Start a span and set it as current.

        Context manager that creates a span, makes it the current span,
        and ends it when the context exits.

        Yields:
            SpanAdapter for the created span.
        """
        if self._backend == TracingBackend.OPENTELEMETRY:
            yield from self._start_as_current_otel(
                name, kind, attributes, links, start_time,
                record_exception, set_status_on_exception, end_on_exit
            )
        else:
            yield from self._start_as_current_truthound(
                name, context, kind, attributes, links, start_time,
                record_exception, set_status_on_exception, end_on_exit
            )

    def _start_as_current_truthound(
        self,
        name: str,
        context: SpanContextAdapter | None,
        kind: Any,
        attributes: Mapping[str, Any] | None,
        links: Sequence[Any] | None,
        start_time: int | float | None,
        record_exception: bool,
        set_status_on_exception: bool,
        end_on_exit: bool,
    ) -> Iterator[SpanAdapter]:
        """Start as current using Truthound backend."""
        with self._tracer.start_as_current_span(
            name=name,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            end_on_exit=end_on_exit,
        ) as span:
            yield SpanAdapter(span, self._backend)

    def _start_as_current_otel(
        self,
        name: str,
        kind: Any,
        attributes: Mapping[str, Any] | None,
        links: Sequence[Any] | None,
        start_time: int | float | None,
        record_exception: bool,
        set_status_on_exception: bool,
        end_on_exit: bool,
    ) -> Iterator[SpanAdapter]:
        """Start as current using OpenTelemetry backend."""
        from opentelemetry.trace import SpanKind

        span_kind = SpanKind.INTERNAL
        if kind is not None:
            if hasattr(kind, "name"):
                kind_map = {
                    "INTERNAL": SpanKind.INTERNAL,
                    "SERVER": SpanKind.SERVER,
                    "CLIENT": SpanKind.CLIENT,
                    "PRODUCER": SpanKind.PRODUCER,
                    "CONSUMER": SpanKind.CONSUMER,
                }
                span_kind = kind_map.get(kind.name, SpanKind.INTERNAL)

        with self._tracer.start_as_current_span(
            name=name,
            kind=span_kind,
            attributes=dict(attributes) if attributes else None,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            end_on_exit=end_on_exit,
        ) as span:
            yield SpanAdapter(span, self._backend)

    def unwrap(self) -> Any:
        """Get the native tracer object."""
        return self._tracer


# =============================================================================
# TracerProvider Adapter
# =============================================================================


class TracerProviderAdapter:
    """Adapter for tracer providers that works with both backends.

    Provides a unified interface for obtaining tracers regardless of
    the underlying implementation.
    """

    def __init__(self, provider: Any, backend: TracingBackend) -> None:
        """Initialize tracer provider adapter.

        Args:
            provider: Native tracer provider object.
            backend: Backend type.
        """
        self._provider = provider
        self._backend = backend
        self._tracers: dict[tuple[str, str], TracerAdapter] = {}
        self._lock = threading.Lock()

    @property
    def backend(self) -> TracingBackend:
        """Get the backend type."""
        return self._backend

    def get_tracer(
        self,
        name: str,
        version: str = "",
        schema_url: str = "",
    ) -> TracerAdapter:
        """Get a tracer.

        Args:
            name: Instrumentation library name.
            version: Instrumentation library version.
            schema_url: Schema URL.

        Returns:
            TracerAdapter wrapping the tracer.
        """
        key = (name, version)

        with self._lock:
            if key not in self._tracers:
                if self._backend == TracingBackend.OPENTELEMETRY:
                    tracer = self._provider.get_tracer(name, version, schema_url)
                else:
                    tracer = self._provider.get_tracer(name, version, schema_url)
                self._tracers[key] = TracerAdapter(tracer, self._backend)
            return self._tracers[key]

    def add_span_processor(self, processor: Any) -> None:
        """Add a span processor.

        Args:
            processor: Span processor to add.
        """
        if hasattr(self._provider, "add_span_processor"):
            self._provider.add_span_processor(processor)
        elif hasattr(self._provider, "add_processor"):
            self._provider.add_processor(processor)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all processors.

        Args:
            timeout_millis: Timeout in milliseconds.

        Returns:
            True if successful.
        """
        return self._provider.force_flush(timeout_millis)

    def shutdown(self) -> bool:
        """Shutdown the provider.

        Returns:
            True if successful.
        """
        return self._provider.shutdown()

    def unwrap(self) -> Any:
        """Get the native provider object."""
        return self._provider


# =============================================================================
# Global State
# =============================================================================


_global_adapter: TracerProviderAdapter | None = None
_adapter_lock = threading.Lock()
_current_config: AdapterConfig | None = None


def _create_truthound_provider(config: AdapterConfig) -> TracerProviderAdapter:
    """Create a Truthound-based provider."""
    from truthound.observability.tracing.config import configure_tracing, TracingConfig

    tracing_config = TracingConfig(
        service_name=config.service_name,
        service_version=config.service_version,
        environment=config.environment,
        exporter=config.exporter_type,
        endpoint=config.exporter_endpoint,
        headers=config.exporter_headers,
        sampling_ratio=config.sampling_ratio,
        batch_export=config.batch_export,
    )

    provider = configure_tracing(tracing_config, set_global=False)
    return TracerProviderAdapter(provider, TracingBackend.TRUTHOUND)


def _create_otel_provider(config: AdapterConfig) -> TracerProviderAdapter:
    """Create an OpenTelemetry SDK-based provider."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased

    # Create resource
    resource_attrs = {SERVICE_NAME: config.service_name}
    if config.service_version:
        resource_attrs[SERVICE_VERSION] = config.service_version
    if config.environment:
        resource_attrs["deployment.environment"] = config.environment

    resource = Resource.create(resource_attrs)

    # Create sampler
    sampler = ParentBased(root=TraceIdRatioBased(config.sampling_ratio))

    # Create provider
    provider = TracerProvider(resource=resource, sampler=sampler)

    # Add exporter
    if config.exporter_type == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = OTLPSpanExporter(
                endpoint=config.exporter_endpoint or "http://localhost:4317",
            )
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        except ImportError:
            logger.warning("OTLP exporter not available, using console")
            _add_console_processor(provider)

    elif config.exporter_type == "jaeger":
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = JaegerExporter()
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        except ImportError:
            logger.warning("Jaeger exporter not available, using console")
            _add_console_processor(provider)

    elif config.exporter_type == "zipkin":
        try:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = ZipkinExporter(endpoint=config.exporter_endpoint)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        except ImportError:
            logger.warning("Zipkin exporter not available, using console")
            _add_console_processor(provider)

    else:
        _add_console_processor(provider)

    return TracerProviderAdapter(provider, TracingBackend.OPENTELEMETRY)


def _add_console_processor(provider: Any) -> None:
    """Add console span processor to provider."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


def configure(
    config: AdapterConfig | None = None,
    *,
    backend: TracingBackend | str | None = None,
    service_name: str | None = None,
    exporter_type: str | None = None,
    exporter_endpoint: str | None = None,
    sampling_ratio: float | None = None,
) -> TracerProviderAdapter:
    """Configure the global tracing adapter.

    This is the main entry point for setting up tracing with automatic
    backend selection.

    Args:
        config: Full configuration object.
        backend: Backend to use (overrides config).
        service_name: Service name (overrides config).
        exporter_type: Exporter type (overrides config).
        exporter_endpoint: Exporter endpoint (overrides config).
        sampling_ratio: Sampling ratio (overrides config).

    Returns:
        Configured TracerProviderAdapter.

    Example:
        >>> # Auto-detect backend
        >>> provider = configure(service_name="my-service")
        >>> tracer = provider.get_tracer("my-component")

        >>> # Force specific backend
        >>> provider = configure(backend="truthound")
    """
    global _global_adapter, _current_config

    with _adapter_lock:
        # Start with config or defaults
        if config is None:
            config = AdapterConfig()

        # Apply overrides
        if backend is not None:
            if isinstance(backend, str):
                backend = TracingBackend(backend)
            config.backend = backend
        if service_name:
            config.service_name = service_name
        if exporter_type:
            config.exporter_type = exporter_type
        if exporter_endpoint:
            config.exporter_endpoint = exporter_endpoint
        if sampling_ratio is not None:
            config.sampling_ratio = sampling_ratio

        # Determine actual backend
        actual_backend = config.backend
        if actual_backend == TracingBackend.AUTO:
            if is_otel_sdk_available():
                logger.info("OpenTelemetry SDK detected, using OTEL backend")
                actual_backend = TracingBackend.OPENTELEMETRY
            else:
                logger.info("OpenTelemetry SDK not found, using Truthound backend")
                actual_backend = TracingBackend.TRUTHOUND

        # Create provider
        if actual_backend == TracingBackend.OPENTELEMETRY:
            _global_adapter = _create_otel_provider(config)
        else:
            _global_adapter = _create_truthound_provider(config)

        _current_config = config
        return _global_adapter


def get_tracer_provider() -> TracerProviderAdapter:
    """Get the global tracer provider adapter.

    Automatically configures with defaults if not already configured.

    Returns:
        Global TracerProviderAdapter.
    """
    global _global_adapter

    with _adapter_lock:
        if _global_adapter is None:
            return configure()
        return _global_adapter


def get_tracer(name: str, version: str = "") -> TracerAdapter:
    """Get a tracer from the global provider.

    Convenience function for quick access.

    Args:
        name: Instrumentation name.
        version: Instrumentation version.

    Returns:
        TracerAdapter.
    """
    return get_tracer_provider().get_tracer(name, version)


def get_current_backend() -> TracingBackend:
    """Get the currently active backend.

    Returns:
        Current TracingBackend.
    """
    provider = get_tracer_provider()
    return provider.backend


def reset_global_adapter() -> None:
    """Reset the global adapter (mainly for testing)."""
    global _global_adapter, _current_config

    with _adapter_lock:
        if _global_adapter is not None:
            try:
                _global_adapter.shutdown()
            except Exception:
                pass
        _global_adapter = None
        _current_config = None
