"""Configuration integration for OpenTelemetry compatibility layer.

This module provides unified configuration that works with both
Truthound native and OpenTelemetry SDK backends, and bridges between
the two configuration systems.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from truthound.observability.tracing.otel.detection import (
    detect_otel_availability,
    is_otel_sdk_available,
)
from truthound.observability.tracing.otel.adapter import (
    TracingBackend,
    AdapterConfig,
    TracerProviderAdapter,
    configure,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Configuration
# =============================================================================


ExporterType = Literal["console", "otlp", "jaeger", "zipkin", "none"]
SamplerType = Literal["always_on", "always_off", "trace_id_ratio", "parent_based"]
PropagatorType = Literal["w3c", "b3", "b3_single", "jaeger", "all"]


@dataclass
class UnifiedTracingConfig:
    """Unified tracing configuration for both backends.

    This configuration class can be used to configure either Truthound
    native tracing or OpenTelemetry SDK tracing, with automatic
    backend selection.

    Attributes:
        service_name: Service name for traces.
        service_version: Service version.
        environment: Deployment environment (development, staging, production).
        backend: Backend selection (auto, truthound, opentelemetry).
        exporter: Exporter type.
        endpoint: Exporter endpoint URL.
        headers: Additional headers for exporter (e.g., auth).
        sampler: Sampler type.
        sampling_ratio: Sampling ratio (0.0 to 1.0).
        propagators: List of propagator types.
        batch_export: Whether to use batch span processor.
        max_queue_size: Maximum queue size for batch processor.
        max_export_batch_size: Maximum batch size for export.
        scheduled_delay_millis: Delay between batch exports.
        max_attributes: Maximum attributes per span.
        max_events: Maximum events per span.
        max_links: Maximum links per span.
        console_debug: Enable additional console output.
        auto_instrument: Enable auto-instrumentation (OTEL SDK only).

    Example:
        >>> config = UnifiedTracingConfig(
        ...     service_name="my-service",
        ...     exporter="otlp",
        ...     endpoint="http://localhost:4317",
        ... )
        >>> provider = setup_tracing(config)
    """

    # Service identification
    service_name: str = "truthound"
    service_version: str = ""
    environment: str = ""

    # Backend selection
    backend: TracingBackend = TracingBackend.AUTO

    # Exporter settings
    exporter: ExporterType = "console"
    endpoint: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    # Sampling settings
    sampler: SamplerType = "always_on"
    sampling_ratio: float = 1.0

    # Propagator settings
    propagators: list[PropagatorType] = field(default_factory=lambda: ["w3c"])

    # Batch processor settings
    batch_export: bool = True
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    scheduled_delay_millis: int = 5000

    # Span limits
    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128

    # Debug settings
    console_debug: bool = False

    # OTEL-specific settings
    auto_instrument: bool = False

    @classmethod
    def from_env(cls) -> "UnifiedTracingConfig":
        """Create configuration from environment variables.

        Supports both OTEL standard and Truthound-specific variables.

        Environment Variables:
            OTEL_SERVICE_NAME / TRUTHOUND_SERVICE_NAME: Service name
            OTEL_SERVICE_VERSION / TRUTHOUND_SERVICE_VERSION: Version
            TRUTHOUND_TRACING_BACKEND: Backend (auto, truthound, opentelemetry)
            OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint
            OTEL_EXPORTER_OTLP_HEADERS: Exporter headers
            OTEL_TRACES_SAMPLER: Sampler type
            OTEL_TRACES_SAMPLER_ARG: Sampling ratio
            OTEL_PROPAGATORS: Propagators (comma-separated)

        Returns:
            UnifiedTracingConfig from environment.
        """
        # Backend selection
        backend_str = os.environ.get("TRUTHOUND_TRACING_BACKEND", "auto").lower()
        try:
            backend = TracingBackend(backend_str)
        except ValueError:
            backend = TracingBackend.AUTO

        # Service identification (OTEL vars take precedence)
        service_name = os.environ.get(
            "OTEL_SERVICE_NAME",
            os.environ.get("TRUTHOUND_SERVICE_NAME", "truthound")
        )
        service_version = os.environ.get(
            "OTEL_SERVICE_VERSION",
            os.environ.get("TRUTHOUND_SERVICE_VERSION", "")
        )

        # Parse environment from resource attributes
        environment = ""
        resource_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        if "deployment.environment=" in resource_attrs:
            for attr in resource_attrs.split(","):
                if attr.startswith("deployment.environment="):
                    environment = attr.split("=", 1)[1]
                    break
        if not environment:
            environment = os.environ.get("DEPLOYMENT_ENVIRONMENT", "")

        # Exporter configuration
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        exporter: ExporterType = "console"
        if endpoint:
            exporter = "otlp"
        elif os.environ.get("OTEL_EXPORTER_JAEGER_ENDPOINT"):
            exporter = "jaeger"
            endpoint = os.environ.get("OTEL_EXPORTER_JAEGER_ENDPOINT", "")
        elif os.environ.get("OTEL_EXPORTER_ZIPKIN_ENDPOINT"):
            exporter = "zipkin"
            endpoint = os.environ.get("OTEL_EXPORTER_ZIPKIN_ENDPOINT", "")

        # Parse headers
        headers = {}
        headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers_str:
            for pair in headers_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers[key.strip()] = value.strip()

        # Sampler configuration
        sampler_map = {
            "always_on": "always_on",
            "always_off": "always_off",
            "traceidratio": "trace_id_ratio",
            "parentbased_always_on": "parent_based",
            "parentbased_always_off": "parent_based",
            "parentbased_traceidratio": "parent_based",
        }
        sampler_env = os.environ.get("OTEL_TRACES_SAMPLER", "always_on").lower()
        sampler: SamplerType = sampler_map.get(sampler_env, "always_on")

        # Sampling ratio
        try:
            sampling_ratio = float(os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0"))
        except ValueError:
            sampling_ratio = 1.0

        # Propagators
        propagators_str = os.environ.get("OTEL_PROPAGATORS", "tracecontext,baggage")
        propagator_map = {
            "tracecontext": "w3c",
            "baggage": "w3c",
            "b3": "b3",
            "b3multi": "b3",
            "jaeger": "jaeger",
        }
        propagators: list[PropagatorType] = []
        for p in propagators_str.split(","):
            mapped = propagator_map.get(p.strip().lower())
            if mapped and mapped not in propagators:
                propagators.append(mapped)
        if not propagators:
            propagators = ["w3c"]

        # Batch processor settings
        try:
            max_queue_size = int(os.environ.get("OTEL_BSP_MAX_QUEUE_SIZE", "2048"))
        except ValueError:
            max_queue_size = 2048

        try:
            max_export_batch_size = int(os.environ.get("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "512"))
        except ValueError:
            max_export_batch_size = 512

        try:
            scheduled_delay = int(os.environ.get("OTEL_BSP_SCHEDULE_DELAY", "5000"))
        except ValueError:
            scheduled_delay = 5000

        # Span limits
        try:
            max_attributes = int(os.environ.get("OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT", "128"))
        except ValueError:
            max_attributes = 128

        try:
            max_events = int(os.environ.get("OTEL_SPAN_EVENT_COUNT_LIMIT", "128"))
        except ValueError:
            max_events = 128

        try:
            max_links = int(os.environ.get("OTEL_SPAN_LINK_COUNT_LIMIT", "128"))
        except ValueError:
            max_links = 128

        return cls(
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            backend=backend,
            exporter=exporter,
            endpoint=endpoint,
            headers=headers,
            sampler=sampler,
            sampling_ratio=sampling_ratio,
            propagators=propagators,
            batch_export=True,
            max_queue_size=max_queue_size,
            max_export_batch_size=max_export_batch_size,
            scheduled_delay_millis=scheduled_delay,
            max_attributes=max_attributes,
            max_events=max_events,
            max_links=max_links,
        )

    @classmethod
    def development(cls, service_name: str = "dev-service") -> "UnifiedTracingConfig":
        """Create development configuration.

        Uses console exporter with always-on sampling and debug output.
        """
        return cls(
            service_name=service_name,
            environment="development",
            exporter="console",
            sampler="always_on",
            batch_export=False,
            console_debug=True,
        )

    @classmethod
    def production(
        cls,
        service_name: str,
        endpoint: str,
        sampling_ratio: float = 0.1,
    ) -> "UnifiedTracingConfig":
        """Create production configuration.

        Uses OTLP exporter with parent-based sampling.
        """
        return cls(
            service_name=service_name,
            environment="production",
            exporter="otlp",
            endpoint=endpoint,
            sampler="parent_based",
            sampling_ratio=sampling_ratio,
            batch_export=True,
        )

    @classmethod
    def testing(cls, service_name: str = "test-service") -> "UnifiedTracingConfig":
        """Create testing configuration.

        Uses in-memory or console exporter with always-on sampling.
        """
        return cls(
            service_name=service_name,
            environment="testing",
            exporter="console",
            sampler="always_on",
            batch_export=False,
        )

    def to_truthound_config(self) -> Any:
        """Convert to Truthound TracingConfig.

        Returns:
            truthound.observability.tracing.config.TracingConfig
        """
        from truthound.observability.tracing.config import TracingConfig

        return TracingConfig(
            service_name=self.service_name,
            service_version=self.service_version,
            environment=self.environment,
            exporter=self.exporter,
            endpoint=self.endpoint,
            headers=self.headers,
            sampler=self.sampler,
            sampling_ratio=self.sampling_ratio,
            propagator=self.propagators[0] if self.propagators else "w3c",
            batch_export=self.batch_export,
            max_queue_size=self.max_queue_size,
            max_export_batch_size=self.max_export_batch_size,
            scheduled_delay_millis=self.scheduled_delay_millis,
            max_attributes=self.max_attributes,
            max_events=self.max_events,
            max_links=self.max_links,
            console_debug=self.console_debug,
        )

    def to_adapter_config(self) -> AdapterConfig:
        """Convert to AdapterConfig for the OTEL adapter layer.

        Returns:
            AdapterConfig
        """
        return AdapterConfig(
            backend=self.backend,
            service_name=self.service_name,
            service_version=self.service_version,
            environment=self.environment,
            exporter_type=self.exporter,
            exporter_endpoint=self.endpoint,
            exporter_headers=self.headers,
            sampling_ratio=self.sampling_ratio,
            batch_export=self.batch_export,
            propagators=list(self.propagators),
            auto_instrument=self.auto_instrument,
        )


# =============================================================================
# Setup Functions
# =============================================================================


def setup_tracing(
    config: UnifiedTracingConfig | None = None,
    *,
    service_name: str | None = None,
    backend: TracingBackend | str | None = None,
    exporter: ExporterType | None = None,
    endpoint: str | None = None,
    set_global: bool = True,
) -> TracerProviderAdapter:
    """Set up tracing with unified configuration.

    This is the main entry point for configuring tracing with the
    OpenTelemetry compatibility layer.

    Args:
        config: Full configuration object.
        service_name: Service name (overrides config).
        backend: Backend selection (overrides config).
        exporter: Exporter type (overrides config).
        endpoint: Exporter endpoint (overrides config).
        set_global: Set as global provider.

    Returns:
        Configured TracerProviderAdapter.

    Example:
        >>> # Quick setup with defaults
        >>> provider = setup_tracing(service_name="my-service")

        >>> # Production setup
        >>> provider = setup_tracing(
        ...     config=UnifiedTracingConfig.production(
        ...         service_name="my-service",
        ...         endpoint="http://collector:4317",
        ...     )
        ... )

        >>> # Force specific backend
        >>> provider = setup_tracing(
        ...     service_name="my-service",
        ...     backend="truthound",
        ... )
    """
    # Start with provided config or create default
    if config is None:
        config = UnifiedTracingConfig()

    # Apply overrides
    if service_name:
        config.service_name = service_name
    if backend is not None:
        if isinstance(backend, str):
            config.backend = TracingBackend(backend)
        else:
            config.backend = backend
    if exporter:
        config.exporter = exporter
    if endpoint:
        config.endpoint = endpoint

    # Log configuration
    logger.info(
        "Setting up tracing: service=%s backend=%s exporter=%s",
        config.service_name,
        config.backend.value,
        config.exporter,
    )

    # Convert to adapter config and configure
    adapter_config = config.to_adapter_config()
    return configure(adapter_config)


def setup_tracing_from_env(set_global: bool = True) -> TracerProviderAdapter:
    """Set up tracing from environment variables.

    Convenience function that reads configuration from environment.

    Args:
        set_global: Set as global provider.

    Returns:
        Configured TracerProviderAdapter.
    """
    config = UnifiedTracingConfig.from_env()
    return setup_tracing(config)


# =============================================================================
# OTEL SDK Configuration Bridge
# =============================================================================


def configure_otel_sdk(config: UnifiedTracingConfig) -> Any:
    """Configure OpenTelemetry SDK directly from unified config.

    This bypasses the adapter layer and configures the OTEL SDK
    directly. Useful when you want to use the SDK directly but
    with Truthound's configuration system.

    Args:
        config: Unified tracing configuration.

    Returns:
        OpenTelemetry TracerProvider.

    Raises:
        ImportError: If OTEL SDK is not available.
    """
    if not is_otel_sdk_available():
        raise ImportError("OpenTelemetry SDK is not installed")

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.sdk.trace.sampling import (
        TraceIdRatioBased,
        ParentBased,
        ALWAYS_ON,
        ALWAYS_OFF,
    )
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import set_tracer_provider

    # Create resource
    resource_attrs = {SERVICE_NAME: config.service_name}
    if config.service_version:
        resource_attrs[SERVICE_VERSION] = config.service_version
    if config.environment:
        resource_attrs["deployment.environment"] = config.environment

    resource = Resource.create(resource_attrs)

    # Create sampler
    if config.sampler == "always_off":
        sampler = ALWAYS_OFF
    elif config.sampler == "trace_id_ratio":
        sampler = TraceIdRatioBased(config.sampling_ratio)
    elif config.sampler == "parent_based":
        sampler = ParentBased(root=TraceIdRatioBased(config.sampling_ratio))
    else:
        sampler = ALWAYS_ON

    # Create provider
    provider = TracerProvider(resource=resource, sampler=sampler)

    # Create exporter and processor
    exporter = None

    if config.exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(
                endpoint=config.endpoint or "http://localhost:4317",
            )
        except ImportError:
            logger.warning("OTLP exporter not available")

    elif config.exporter == "jaeger":
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            exporter = JaegerExporter()
        except ImportError:
            logger.warning("Jaeger exporter not available")

    elif config.exporter == "zipkin":
        try:
            from opentelemetry.exporter.zipkin.json import ZipkinExporter

            exporter = ZipkinExporter(
                endpoint=config.endpoint or "http://localhost:9411/api/v2/spans"
            )
        except ImportError:
            logger.warning("Zipkin exporter not available")

    elif config.exporter == "console":
        exporter = ConsoleSpanExporter()

    # Add processor
    if exporter:
        if config.batch_export:
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=config.max_queue_size,
                max_export_batch_size=config.max_export_batch_size,
                schedule_delay_millis=config.scheduled_delay_millis,
            )
        else:
            processor = SimpleSpanProcessor(exporter)

        provider.add_span_processor(processor)

    # Add console debug if enabled
    if config.console_debug and config.exporter != "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Set as global
    set_tracer_provider(provider)

    return provider


def configure_truthound_tracing(config: UnifiedTracingConfig) -> Any:
    """Configure Truthound tracing directly from unified config.

    This bypasses the adapter layer and configures Truthound's
    native tracing directly.

    Args:
        config: Unified tracing configuration.

    Returns:
        Truthound TracerProvider.
    """
    from truthound.observability.tracing.config import configure_tracing

    truthound_config = config.to_truthound_config()
    return configure_tracing(truthound_config, set_global=True)


# =============================================================================
# Status and Diagnostics
# =============================================================================


def get_tracing_status() -> dict[str, Any]:
    """Get current tracing configuration status.

    Returns:
        Dictionary with tracing status information.
    """
    from truthound.observability.tracing.otel.adapter import (
        get_tracer_provider,
        get_current_backend,
        _current_config,
    )

    availability = detect_otel_availability()

    status = {
        "otel_api_available": availability.api_available,
        "otel_sdk_available": availability.sdk_available,
        "otel_version": availability.api_version,
        "current_backend": get_current_backend().value,
        "available_features": list(availability.features),
        "installed_packages": {
            k: v.version for k, v in availability.packages.items()
        },
    }

    if _current_config:
        status["config"] = {
            "service_name": _current_config.service_name,
            "exporter_type": _current_config.exporter_type,
            "exporter_endpoint": _current_config.exporter_endpoint,
            "sampling_ratio": _current_config.sampling_ratio,
            "batch_export": _current_config.batch_export,
        }

    return status


def diagnose_tracing() -> str:
    """Run tracing diagnostics and return human-readable report.

    Returns:
        Diagnostic report string.
    """
    status = get_tracing_status()

    lines = [
        "=" * 60,
        "Truthound Tracing Diagnostics",
        "=" * 60,
        "",
        "OpenTelemetry Availability:",
        f"  API Available: {status['otel_api_available']}",
        f"  SDK Available: {status['otel_sdk_available']}",
        f"  Version: {status.get('otel_version', 'N/A')}",
        "",
        f"Current Backend: {status['current_backend']}",
        "",
        "Available Features:",
    ]

    for feature in status.get("available_features", []):
        lines.append(f"  - {feature}")

    if not status.get("available_features"):
        lines.append("  (none)")

    lines.append("")
    lines.append("Installed OTEL Packages:")

    for pkg, version in status.get("installed_packages", {}).items():
        lines.append(f"  - {pkg}: {version}")

    if not status.get("installed_packages"):
        lines.append("  (none)")

    if "config" in status:
        lines.extend([
            "",
            "Current Configuration:",
            f"  Service: {status['config']['service_name']}",
            f"  Exporter: {status['config']['exporter_type']}",
            f"  Endpoint: {status['config']['exporter_endpoint'] or '(none)'}",
            f"  Sampling: {status['config']['sampling_ratio']}",
            f"  Batch Export: {status['config']['batch_export']}",
        ])

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
