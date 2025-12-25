"""Enterprise observability module for Truthound.

This module provides a unified interface for structured logging, metrics
collection, and distributed tracing, enabling comprehensive monitoring
of data quality operations.

Features:
    - Structured logging with JSON, logfmt, and human-readable formats
    - Metrics collection with labels/dimensions support
    - Multiple backend support (Prometheus, StatsD, OpenTelemetry)
    - Distributed tracing with OpenTelemetry compatibility
    - Context propagation (W3C Trace Context, B3, Jaeger)
    - Automatic checkpoint instrumentation
    - Log correlation with trace/span IDs

Architecture:
    The observability system follows a provider/exporter pattern:

    Logging:
        Logger -> Handler -> Formatter -> Output
                            (JSON/logfmt/console)

    Metrics:
        Collector -> Metric -> Exporter
                    (Counter/Gauge/Histogram)

    Tracing:
        TracerProvider -> Tracer -> Span
              |
        SpanProcessor -> SpanExporter
              |
        Propagator (W3C/B3/Jaeger)

Usage:
    >>> from truthound.observability import (
    ...     get_logger, get_metrics,
    ...     Counter, Gauge, Histogram,
    ... )
    >>>
    >>> # Structured logging
    >>> logger = get_logger(__name__)
    >>> logger.info("Validation completed",
    ...     checkpoint="daily_check",
    ...     total_issues=5,
    ...     pass_rate=0.95,
    ... )
    >>>
    >>> # Metrics collection
    >>> metrics = get_metrics()
    >>> validation_counter = metrics.counter(
    ...     "validations_total",
    ...     "Total number of validations",
    ...     labels=["checkpoint", "status"],
    ... )
    >>> validation_counter.inc(checkpoint="daily", status="success")
    >>>
    >>> # Histograms for latency
    >>> latency = metrics.histogram(
    ...     "validation_duration_seconds",
    ...     "Validation execution time",
    ...     buckets=[0.1, 0.5, 1.0, 5.0, 10.0],
    ... )
    >>> with latency.time():
    ...     run_validation()
    >>>
    >>> # Distributed Tracing (new)
    >>> from truthound.observability.tracing import (
    ...     configure_tracing, get_tracer,
    ... )
    >>> configure_tracing(service_name="truthound", exporter="otlp")
    >>> tracer = get_tracer("my.component")
    >>> with tracer.start_as_current_span("validation") as span:
    ...     span.set_attribute("checkpoint", "daily")
    ...     run_validation()
"""

from truthound.observability.logging import (
    # Core Logger
    StructuredLogger,
    LogLevel,
    LogRecord,
    # Formatters
    LogFormatter,
    JSONFormatter,
    LogfmtFormatter,
    ConsoleFormatter,
    # Handlers
    LogHandler,
    ConsoleHandler,
    FileHandler,
    RotatingFileHandler,
    # Context
    LogContext,
    log_context,
    # Global access
    get_logger,
    set_default_logger,
    configure_logging,
)

from truthound.observability.metrics import (
    # Metric Types
    Metric,
    Counter,
    Gauge,
    Histogram,
    Summary,
    # Collector
    MetricsCollector,
    MetricsRegistry,
    # Exporters
    MetricsExporter,
    PrometheusExporter,
    StatsDExporter,
    InMemoryExporter,
    # Global access
    get_metrics,
    set_metrics,
    configure_metrics,
)

from truthound.observability.context import (
    # Trace Context
    TraceContext,
    SpanContext,
    SpanStatus,
    # Context management
    current_context,
    with_context,
    create_trace,
    create_span,
)

from truthound.observability.instrumentation import (
    # Decorators
    traced,
    timed,
    counted,
    # Checkpoint integration
    CheckpointInstrumentation,
    instrument_checkpoint,
)

# Distributed Tracing (OpenTelemetry-compatible)
from truthound.observability.tracing import (
    # Provider
    TracerProvider,
    Tracer,
    get_tracer_provider,
    set_tracer_provider,
    # Span
    Span,
    SpanKind,
    StatusCode,
    Link,
    Event,
    SpanLimits,
    # Processor
    SpanProcessor,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    MultiSpanProcessor,
    # Exporter
    SpanExporter,
    ExportResult,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
    JaegerExporter,
    ZipkinExporter,
    # Sampler
    Sampler,
    SamplingResult,
    SamplingDecision,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    # Propagator
    Propagator,
    CompositePropagator,
    W3CTraceContextPropagator,
    W3CBaggagePropagator,
    B3Propagator,
    JaegerPropagator,
    get_global_propagator,
    set_global_propagator,
    # Resource
    Resource,
    ResourceDetector,
    get_aggregated_resources,
    # Baggage
    Baggage,
    get_baggage,
    set_baggage,
    remove_baggage,
    clear_baggage,
    # Config
    TracingConfig,
    configure_tracing,
)

# Convenience imports for tracing
from truthound.observability.tracing.provider import (
    get_tracer,
    get_current_span,
    get_current_context as get_current_trace_context,
)

__all__ = [
    # Logging
    "StructuredLogger",
    "LogLevel",
    "LogRecord",
    "LogFormatter",
    "JSONFormatter",
    "LogfmtFormatter",
    "ConsoleFormatter",
    "LogHandler",
    "ConsoleHandler",
    "FileHandler",
    "RotatingFileHandler",
    "LogContext",
    "log_context",
    "get_logger",
    "set_default_logger",
    "configure_logging",
    # Metrics
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "MetricsCollector",
    "MetricsRegistry",
    "MetricsExporter",
    "PrometheusExporter",
    "StatsDExporter",
    "InMemoryExporter",
    "get_metrics",
    "set_metrics",
    "configure_metrics",
    # Context
    "TraceContext",
    "SpanContext",
    "SpanStatus",
    "current_context",
    "with_context",
    "create_trace",
    "create_span",
    # Instrumentation
    "traced",
    "timed",
    "counted",
    "CheckpointInstrumentation",
    "instrument_checkpoint",
    # Distributed Tracing
    "TracerProvider",
    "Tracer",
    "get_tracer_provider",
    "set_tracer_provider",
    "get_tracer",
    "get_current_span",
    "get_current_trace_context",
    "Span",
    "SpanKind",
    "StatusCode",
    "Link",
    "Event",
    "SpanLimits",
    "SpanProcessor",
    "SimpleSpanProcessor",
    "BatchSpanProcessor",
    "MultiSpanProcessor",
    "SpanExporter",
    "ExportResult",
    "ConsoleSpanExporter",
    "InMemorySpanExporter",
    "OTLPSpanExporter",
    "JaegerExporter",
    "ZipkinExporter",
    "Sampler",
    "SamplingResult",
    "SamplingDecision",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
    "Propagator",
    "CompositePropagator",
    "W3CTraceContextPropagator",
    "W3CBaggagePropagator",
    "B3Propagator",
    "JaegerPropagator",
    "get_global_propagator",
    "set_global_propagator",
    "Resource",
    "ResourceDetector",
    "get_aggregated_resources",
    "Baggage",
    "get_baggage",
    "set_baggage",
    "remove_baggage",
    "clear_baggage",
    "TracingConfig",
    "configure_tracing",
]
