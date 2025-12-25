"""Configuration and setup utilities for tracing.

This module provides convenient configuration functions for setting up
distributed tracing with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from truthound.observability.tracing.provider import TracerProvider, set_tracer_provider
from truthound.observability.tracing.sampler import (
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
)
from truthound.observability.tracing.processor import (
    SpanProcessor,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    BatchConfig,
)
from truthound.observability.tracing.exporter import (
    SpanExporter,
    ConsoleSpanExporter,
    OTLPSpanExporter,
    OTLPConfig,
    JaegerExporter,
    JaegerConfig,
    ZipkinExporter,
    ZipkinConfig,
)
from truthound.observability.tracing.propagator import (
    Propagator,
    CompositePropagator,
    W3CTraceContextPropagator,
    W3CBaggagePropagator,
    B3Propagator,
    JaegerPropagator,
    set_global_propagator,
)
from truthound.observability.tracing.resource import (
    Resource,
    get_aggregated_resources,
)
from truthound.observability.tracing.span import SpanLimits


# =============================================================================
# Tracing Configuration
# =============================================================================


ExporterType = Literal["console", "otlp", "jaeger", "zipkin", "none"]
SamplerType = Literal["always_on", "always_off", "trace_id_ratio", "parent_based"]
PropagatorType = Literal["w3c", "b3", "b3_single", "jaeger", "all"]


@dataclass
class TracingConfig:
    """Configuration for distributed tracing.

    Provides a declarative way to configure tracing components.

    Example:
        >>> config = TracingConfig(
        ...     service_name="my-service",
        ...     service_version="1.0.0",
        ...     exporter="otlp",
        ...     endpoint="http://localhost:4317",
        ...     sampling_ratio=0.1,
        ... )
        >>> provider = configure_tracing(config)
    """

    # Service identification
    service_name: str = "unknown_service"
    service_version: str = ""
    environment: str = ""

    # Exporter settings
    exporter: ExporterType = "console"
    endpoint: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    # Sampling settings
    sampler: SamplerType = "always_on"
    sampling_ratio: float = 1.0

    # Propagator settings
    propagator: PropagatorType = "w3c"

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

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Create configuration from environment variables.

        Environment variables:
            - OTEL_SERVICE_NAME: Service name
            - OTEL_SERVICE_VERSION: Service version
            - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint
            - OTEL_EXPORTER_OTLP_HEADERS: OTLP headers (key=value,...)
            - OTEL_TRACES_SAMPLER: Sampler type
            - OTEL_TRACES_SAMPLER_ARG: Sampling ratio
            - OTEL_PROPAGATORS: Propagators (comma-separated)

        Returns:
            TracingConfig from environment.
        """
        # Parse headers
        headers = {}
        headers_str = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers_str:
            for pair in headers_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    headers[key.strip()] = value.strip()

        # Parse sampler
        sampler_map = {
            "always_on": "always_on",
            "always_off": "always_off",
            "traceidratio": "trace_id_ratio",
            "parentbased_always_on": "parent_based",
            "parentbased_always_off": "parent_based",
            "parentbased_traceidratio": "parent_based",
        }
        sampler_env = os.environ.get("OTEL_TRACES_SAMPLER", "always_on").lower()
        sampler = sampler_map.get(sampler_env, "always_on")

        # Parse sampling ratio
        try:
            sampling_ratio = float(os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0"))
        except ValueError:
            sampling_ratio = 1.0

        # Determine exporter
        exporter = "console"
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        if endpoint:
            exporter = "otlp"

        return cls(
            service_name=os.environ.get("OTEL_SERVICE_NAME", "unknown_service"),
            service_version=os.environ.get("OTEL_SERVICE_VERSION", ""),
            environment=os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").split("deployment.environment=")[-1].split(",")[0] if "deployment.environment=" in os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "") else "",
            exporter=exporter,
            endpoint=endpoint,
            headers=headers,
            sampler=sampler,
            sampling_ratio=sampling_ratio,
        )

    @classmethod
    def development(cls, service_name: str = "dev-service") -> "TracingConfig":
        """Create development configuration.

        Uses console exporter with always-on sampling.
        """
        return cls(
            service_name=service_name,
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
    ) -> "TracingConfig":
        """Create production configuration.

        Uses OTLP exporter with rate-based sampling.
        """
        return cls(
            service_name=service_name,
            exporter="otlp",
            endpoint=endpoint,
            sampler="parent_based",
            sampling_ratio=sampling_ratio,
            batch_export=True,
        )


# =============================================================================
# Configuration Functions
# =============================================================================


def _create_exporter(config: TracingConfig) -> SpanExporter | None:
    """Create exporter from configuration."""
    if config.exporter == "none":
        return None

    if config.exporter == "console":
        return ConsoleSpanExporter(pretty=True)

    if config.exporter == "otlp":
        endpoint = config.endpoint or "http://localhost:4317"
        return OTLPSpanExporter(
            config=OTLPConfig(
                endpoint=endpoint,
                headers=config.headers,
            )
        )

    if config.exporter == "jaeger":
        endpoint = config.endpoint
        if endpoint:
            return JaegerExporter(
                config=JaegerConfig(collector_endpoint=endpoint)
            )
        return JaegerExporter()

    if config.exporter == "zipkin":
        endpoint = config.endpoint or "http://localhost:9411/api/v2/spans"
        return ZipkinExporter(
            config=ZipkinConfig(
                endpoint=endpoint,
                local_node_service_name=config.service_name,
            )
        )

    return ConsoleSpanExporter()


def _create_sampler(config: TracingConfig) -> Sampler:
    """Create sampler from configuration."""
    if config.sampler == "always_off":
        return AlwaysOffSampler()

    if config.sampler == "trace_id_ratio":
        return TraceIdRatioSampler(config.sampling_ratio)

    if config.sampler == "parent_based":
        root = TraceIdRatioSampler(config.sampling_ratio)
        return ParentBasedSampler(root=root)

    return AlwaysOnSampler()


def _create_propagator(config: TracingConfig) -> Propagator:
    """Create propagator from configuration."""
    if config.propagator == "b3":
        return B3Propagator()

    if config.propagator == "b3_single":
        return B3Propagator(single_header=True)

    if config.propagator == "jaeger":
        return JaegerPropagator()

    if config.propagator == "all":
        return CompositePropagator([
            W3CTraceContextPropagator(),
            W3CBaggagePropagator(),
            B3Propagator(),
            JaegerPropagator(),
        ])

    # Default: W3C
    return CompositePropagator([
        W3CTraceContextPropagator(),
        W3CBaggagePropagator(),
    ])


def _create_resource(config: TracingConfig) -> Resource:
    """Create resource from configuration."""
    # Get base resources from detectors first
    base = get_aggregated_resources()

    # Then override with config values (config takes precedence)
    override_attributes = {
        "service.name": config.service_name,
    }

    if config.service_version:
        override_attributes["service.version"] = config.service_version

    if config.environment:
        override_attributes["deployment.environment"] = config.environment

    override = Resource(attributes=override_attributes)
    return base.merge(override)


def configure_tracing(
    config: TracingConfig | None = None,
    *,
    service_name: str | None = None,
    exporter: ExporterType | None = None,
    endpoint: str | None = None,
    sampling_ratio: float | None = None,
    set_global: bool = True,
) -> TracerProvider:
    """Configure and create a TracerProvider.

    This is the main entry point for setting up tracing.
    Can use either a TracingConfig object or keyword arguments.

    Args:
        config: Full configuration object.
        service_name: Service name (overrides config).
        exporter: Exporter type (overrides config).
        endpoint: Exporter endpoint (overrides config).
        sampling_ratio: Sampling ratio (overrides config).
        set_global: Set as global provider.

    Returns:
        Configured TracerProvider.

    Example:
        >>> # Using config object
        >>> config = TracingConfig.production(
        ...     service_name="my-service",
        ...     endpoint="http://collector:4317",
        ... )
        >>> provider = configure_tracing(config)
        >>>
        >>> # Using keyword arguments
        >>> provider = configure_tracing(
        ...     service_name="my-service",
        ...     exporter="otlp",
        ...     endpoint="http://collector:4317",
        ...     sampling_ratio=0.1,
        ... )
    """
    # Start with config or defaults
    if config is None:
        config = TracingConfig()

    # Apply overrides
    if service_name:
        config.service_name = service_name
    if exporter:
        config.exporter = exporter
    if endpoint:
        config.endpoint = endpoint
    if sampling_ratio is not None:
        config.sampling_ratio = sampling_ratio

    # Create components
    resource = _create_resource(config)
    sampler = _create_sampler(config)
    propagator = _create_propagator(config)

    span_limits = SpanLimits(
        max_attributes=config.max_attributes,
        max_events=config.max_events,
        max_links=config.max_links,
    )

    # Create exporter and processor
    processors: list[SpanProcessor] = []

    exporter_instance = _create_exporter(config)
    if exporter_instance:
        if config.batch_export:
            processor = BatchSpanProcessor(
                exporter_instance,
                config=BatchConfig(
                    max_queue_size=config.max_queue_size,
                    max_export_batch_size=config.max_export_batch_size,
                    scheduled_delay_millis=config.scheduled_delay_millis,
                ),
            )
        else:
            processor = SimpleSpanProcessor(exporter_instance)

        processors.append(processor)

    # Add console debug if enabled
    if config.console_debug and config.exporter != "console":
        processors.append(SimpleSpanProcessor(ConsoleSpanExporter(pretty=True)))

    # Create provider
    provider = TracerProvider(
        resource=resource,
        sampler=sampler,
        span_limits=span_limits,
        processors=processors,
    )

    # Set globals
    if set_global:
        set_tracer_provider(provider)
        set_global_propagator(propagator)

    return provider


def configure_tracing_from_env(set_global: bool = True) -> TracerProvider:
    """Configure tracing from environment variables.

    Convenience function that reads configuration from environment.

    Args:
        set_global: Set as global provider.

    Returns:
        Configured TracerProvider.
    """
    config = TracingConfig.from_env()
    return configure_tracing(config, set_global=set_global)
