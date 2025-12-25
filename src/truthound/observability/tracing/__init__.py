"""OpenTelemetry-compatible distributed tracing module.

This module provides a comprehensive distributed tracing implementation
that is compatible with OpenTelemetry standards while remaining backend-agnostic.

Architecture:
    TracerProvider -> Tracer -> Span
         |
    SpanProcessor -> SpanExporter
         |
    Propagator (W3C/B3/Jaeger)

Key Features:
    - Multiple backend support (OTLP, Jaeger, Zipkin, Console)
    - Pluggable span processors (Simple, Batch, Multi)
    - W3C Trace Context, B3, and Jaeger propagation
    - Resource detection for service metadata
    - Sampling strategies (Always, Never, Rate-based, Parent-based)
    - Full OpenTelemetry semantic conventions support

Usage:
    >>> from truthound.observability.tracing import (
    ...     TracerProvider,
    ...     OTLPSpanExporter,
    ...     BatchSpanProcessor,
    ...     configure_tracing,
    ... )
    >>>
    >>> # Quick setup
    >>> provider = configure_tracing(
    ...     service_name="truthound",
    ...     exporter="otlp",
    ...     endpoint="http://localhost:4317",
    ... )
    >>>
    >>> # Get tracer and create spans
    >>> tracer = provider.get_tracer("my.component")
    >>> with tracer.start_span("validation") as span:
    ...     span.set_attribute("checkpoint", "daily_check")
    ...     run_validation()
"""

from truthound.observability.tracing.provider import (
    TracerProvider,
    Tracer,
    get_tracer_provider,
    set_tracer_provider,
)

from truthound.observability.tracing.span import (
    Span,
    SpanKind,
    StatusCode,
    Link,
    Event,
    SpanLimits,
)

from truthound.observability.tracing.processor import (
    SpanProcessor,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    MultiSpanProcessor,
)

from truthound.observability.tracing.exporter import (
    SpanExporter,
    ExportResult,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
    JaegerExporter,
    ZipkinExporter,
)

from truthound.observability.tracing.sampler import (
    Sampler,
    SamplingResult,
    SamplingDecision,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
)

from truthound.observability.tracing.propagator import (
    Propagator,
    CompositePropagator,
    W3CTraceContextPropagator,
    W3CBaggagePropagator,
    B3Propagator,
    JaegerPropagator,
    get_global_propagator,
    set_global_propagator,
)

from truthound.observability.tracing.resource import (
    Resource,
    ResourceDetector,
    ProcessResourceDetector,
    HostResourceDetector,
    ServiceResourceDetector,
    get_aggregated_resources,
)

from truthound.observability.tracing.baggage import (
    Baggage,
    get_baggage,
    set_baggage,
    remove_baggage,
    clear_baggage,
)

from truthound.observability.tracing.config import (
    TracingConfig,
    configure_tracing,
)

__all__ = [
    # Provider
    "TracerProvider",
    "Tracer",
    "get_tracer_provider",
    "set_tracer_provider",
    # Span
    "Span",
    "SpanKind",
    "StatusCode",
    "Link",
    "Event",
    "SpanLimits",
    # Processor
    "SpanProcessor",
    "SimpleSpanProcessor",
    "BatchSpanProcessor",
    "MultiSpanProcessor",
    # Exporter
    "SpanExporter",
    "ExportResult",
    "ConsoleSpanExporter",
    "InMemorySpanExporter",
    "OTLPSpanExporter",
    "JaegerExporter",
    "ZipkinExporter",
    # Sampler
    "Sampler",
    "SamplingResult",
    "SamplingDecision",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
    # Propagator
    "Propagator",
    "CompositePropagator",
    "W3CTraceContextPropagator",
    "W3CBaggagePropagator",
    "B3Propagator",
    "JaegerPropagator",
    "get_global_propagator",
    "set_global_propagator",
    # Resource
    "Resource",
    "ResourceDetector",
    "ProcessResourceDetector",
    "HostResourceDetector",
    "ServiceResourceDetector",
    "get_aggregated_resources",
    # Baggage
    "Baggage",
    "get_baggage",
    "set_baggage",
    "remove_baggage",
    "clear_baggage",
    # Config
    "TracingConfig",
    "configure_tracing",
]
