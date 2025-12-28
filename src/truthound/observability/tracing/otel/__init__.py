"""OpenTelemetry Compatibility Layer for Truthound.

This module provides a bridge between Truthound's native tracing implementation
and the official OpenTelemetry API, enabling seamless interoperability.

Design Principles:
    1. **Transparent Fallback**: Use official OpenTelemetry SDK when available,
       fall back to Truthound's native implementation when not.
    2. **Zero Configuration**: Works out of the box with sensible defaults.
    3. **Full API Compatibility**: Truthound spans/traces work with OTEL exporters
       and vice versa.
    4. **Minimal Overhead**: Adapter pattern adds negligible performance cost.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Application Code                         │
    │         (uses OpenTelemetry API or Truthound API)           │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   Adapter Layer (this module)               │
    │   ┌─────────────────────┐  ┌─────────────────────────────┐  │
    │   │  API Adapter        │  │  SDK Bridge                 │  │
    │   │  - TracerProvider   │  │  - SpanProcessor            │  │
    │   │  - Tracer           │  │  - SpanExporter             │  │
    │   │  - Span             │  │  - Sampler                  │  │
    │   └─────────────────────┘  └─────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
    ┌───────────────────┐                   ┌───────────────────┐
    │  OpenTelemetry    │                   │  Truthound Native │
    │  SDK (if present) │                   │  Implementation   │
    └───────────────────┘                   └───────────────────┘

Usage:
    # Automatic backend selection
    >>> from truthound.observability.tracing.otel import get_tracer_provider
    >>> provider = get_tracer_provider()  # Uses OTEL SDK if available
    >>> tracer = provider.get_tracer("my-service")

    # Force Truthound backend
    >>> from truthound.observability.tracing.otel import configure
    >>> configure(backend="truthound")

    # Force OpenTelemetry SDK backend
    >>> configure(backend="opentelemetry")

    # Bridge Truthound spans to OTEL exporters
    >>> from truthound.observability.tracing.otel import OTELSpanBridge
    >>> bridge = OTELSpanBridge()
    >>> otel_span = bridge.to_otel(truthound_span)

See Also:
    - truthound.observability.tracing: Native Truthound tracing
    - opentelemetry.trace: Official OpenTelemetry API
"""

from truthound.observability.tracing.otel.detection import (
    OTELAvailability,
    detect_otel_availability,
    is_otel_sdk_available,
    is_otel_api_available,
    get_otel_version,
)

from truthound.observability.tracing.otel.adapter import (
    TracingBackend,
    AdapterConfig,
    TracerProviderAdapter,
    TracerAdapter,
    SpanAdapter,
    get_tracer_provider,
    get_tracer,
    configure,
    get_current_backend,
)

from truthound.observability.tracing.otel.bridge import (
    SpanBridge,
    SpanContextBridge,
    SpanProcessorBridge,
    SpanExporterBridge,
    SamplerBridge,
    PropagatorBridge,
)

from truthound.observability.tracing.otel.compat import (
    OTELSpanWrapper,
    TruthoundSpanWrapper,
    create_compatible_span,
    to_otel_span_context,
    from_otel_span_context,
)

from truthound.observability.tracing.otel.config import (
    UnifiedTracingConfig,
    setup_tracing,
    setup_tracing_from_env,
    configure_otel_sdk,
    configure_truthound_tracing,
    get_tracing_status,
    diagnose_tracing,
)

__all__ = [
    # Detection
    "OTELAvailability",
    "detect_otel_availability",
    "is_otel_sdk_available",
    "is_otel_api_available",
    "get_otel_version",
    # Adapter
    "TracingBackend",
    "AdapterConfig",
    "TracerProviderAdapter",
    "TracerAdapter",
    "SpanAdapter",
    "get_tracer_provider",
    "get_tracer",
    "configure",
    "get_current_backend",
    # Bridge
    "SpanBridge",
    "SpanContextBridge",
    "SpanProcessorBridge",
    "SpanExporterBridge",
    "SamplerBridge",
    "PropagatorBridge",
    # Compatibility
    "OTELSpanWrapper",
    "TruthoundSpanWrapper",
    "create_compatible_span",
    "to_otel_span_context",
    "from_otel_span_context",
    # Configuration
    "UnifiedTracingConfig",
    "setup_tracing",
    "setup_tracing_from_env",
    "configure_otel_sdk",
    "configure_truthound_tracing",
    "get_tracing_status",
    "diagnose_tracing",
]
