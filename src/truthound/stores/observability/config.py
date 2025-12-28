"""Configuration for store observability.

This module defines configuration dataclasses for all observability components.
Each component can be independently configured and enabled/disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AuditLogLevel(str, Enum):
    """Audit logging verbosity levels."""

    MINIMAL = "minimal"  # Only write/delete operations
    STANDARD = "standard"  # All CRUD operations
    VERBOSE = "verbose"  # All operations including queries
    DEBUG = "debug"  # Everything including internal operations


class MetricsExportFormat(str, Enum):
    """Metrics export format."""

    PROMETHEUS = "prometheus"
    OPENMETRICS = "openmetrics"
    JSON = "json"
    STATSD = "statsd"


class TracingSampler(str, Enum):
    """Tracing sampling strategy."""

    ALWAYS_ON = "always_on"
    ALWAYS_OFF = "always_off"
    RATIO = "ratio"
    PARENT_BASED = "parent_based"


@dataclass
class AuditConfig:
    """Configuration for audit logging.

    Attributes:
        enabled: Whether audit logging is enabled.
        level: Audit logging verbosity level.
        backend: Backend type ("memory", "file", "json", "elasticsearch", "kafka").
        file_path: Path for file-based backends.
        include_data_preview: Include preview of data in audit logs.
        max_data_preview_size: Maximum size of data preview in bytes.
        redact_sensitive: Redact sensitive fields in audit logs.
        sensitive_fields: List of field names to redact.
        retention_days: Days to retain audit logs (0 = forever).
        batch_size: Batch size for async backends.
        flush_interval_seconds: Flush interval for async backends.
    """

    enabled: bool = True
    level: AuditLogLevel = AuditLogLevel.STANDARD
    backend: str = "json"
    file_path: Path | str | None = None
    include_data_preview: bool = False
    max_data_preview_size: int = 1024
    redact_sensitive: bool = True
    sensitive_fields: list[str] = field(
        default_factory=lambda: [
            "password",
            "secret",
            "token",
            "api_key",
            "credential",
            "ssn",
            "credit_card",
        ]
    )
    retention_days: int = 90
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    extra_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)
        if isinstance(self.level, str):
            self.level = AuditLogLevel(self.level)


@dataclass
class MetricsConfig:
    """Configuration for Prometheus metrics.

    Attributes:
        enabled: Whether metrics collection is enabled.
        export_format: Export format for metrics.
        prefix: Prefix for all metric names.
        labels: Default labels to add to all metrics.
        histogram_buckets: Custom bucket boundaries for histograms.
        enable_http_server: Start HTTP server for metrics endpoint.
        http_port: Port for HTTP metrics endpoint.
        http_path: Path for HTTP metrics endpoint.
        push_gateway_url: URL for Prometheus Push Gateway.
        push_interval_seconds: Interval for pushing metrics.
        include_timestamps: Include timestamps in metrics.
    """

    enabled: bool = True
    export_format: MetricsExportFormat = MetricsExportFormat.PROMETHEUS
    prefix: str = "truthound_store"
    labels: dict[str, str] = field(default_factory=dict)
    histogram_buckets: list[float] = field(
        default_factory=lambda: [
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
    )
    enable_http_server: bool = False
    http_port: int = 9090
    http_path: str = "/metrics"
    push_gateway_url: str | None = None
    push_interval_seconds: float = 10.0
    include_timestamps: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.export_format, str):
            self.export_format = MetricsExportFormat(self.export_format)


@dataclass
class TracingConfig:
    """Configuration for distributed tracing.

    Attributes:
        enabled: Whether tracing is enabled.
        service_name: Name of the service for traces.
        sampler: Sampling strategy.
        sample_ratio: Sampling ratio (0.0-1.0) when using RATIO sampler.
        exporter: Exporter type ("otlp", "jaeger", "zipkin", "console", "noop").
        endpoint: Endpoint URL for trace exporter.
        headers: Additional headers for exporter.
        propagators: Context propagators to use.
        record_exceptions: Record exceptions in spans.
        record_attributes: Additional attributes to record.
        max_attributes: Maximum number of attributes per span.
        max_events: Maximum number of events per span.
    """

    enabled: bool = True
    service_name: str = "truthound-store"
    sampler: TracingSampler = TracingSampler.PARENT_BASED
    sample_ratio: float = 1.0
    exporter: str = "otlp"
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    propagators: list[str] = field(
        default_factory=lambda: ["tracecontext", "baggage"]
    )
    record_exceptions: bool = True
    record_attributes: dict[str, str] = field(default_factory=dict)
    max_attributes: int = 128
    max_events: int = 128

    def __post_init__(self) -> None:
        if isinstance(self.sampler, str):
            self.sampler = TracingSampler(self.sampler)


@dataclass
class ObservabilityConfig:
    """Unified observability configuration.

    This combines audit, metrics, and tracing configurations into a single
    configuration object for convenience.

    Attributes:
        audit: Audit logging configuration.
        metrics: Prometheus metrics configuration.
        tracing: Distributed tracing configuration.
        correlation_id_header: Header name for correlation ID propagation.
        environment: Environment name (dev, staging, prod).
        version: Application version for tagging.
    """

    audit: AuditConfig = field(default_factory=AuditConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    correlation_id_header: str = "X-Correlation-ID"
    environment: str = "development"
    version: str = "unknown"

    # Convenience properties for quick enable/disable
    @property
    def audit_enabled(self) -> bool:
        return self.audit.enabled

    @audit_enabled.setter
    def audit_enabled(self, value: bool) -> None:
        self.audit.enabled = value

    @property
    def metrics_enabled(self) -> bool:
        return self.metrics.enabled

    @metrics_enabled.setter
    def metrics_enabled(self, value: bool) -> None:
        self.metrics.enabled = value

    @property
    def tracing_enabled(self) -> bool:
        return self.tracing.enabled

    @tracing_enabled.setter
    def tracing_enabled(self, value: bool) -> None:
        self.tracing.enabled = value

    @classmethod
    def disabled(cls) -> "ObservabilityConfig":
        """Create a config with all observability disabled."""
        return cls(
            audit=AuditConfig(enabled=False),
            metrics=MetricsConfig(enabled=False),
            tracing=TracingConfig(enabled=False),
        )

    @classmethod
    def minimal(cls) -> "ObservabilityConfig":
        """Create a minimal config for development."""
        return cls(
            audit=AuditConfig(enabled=True, level=AuditLogLevel.MINIMAL),
            metrics=MetricsConfig(enabled=True),
            tracing=TracingConfig(enabled=False),
        )

    @classmethod
    def production(cls, service_name: str = "truthound") -> "ObservabilityConfig":
        """Create a production-ready config."""
        return cls(
            audit=AuditConfig(
                enabled=True,
                level=AuditLogLevel.STANDARD,
                backend="json",
                retention_days=365,
            ),
            metrics=MetricsConfig(
                enabled=True,
                enable_http_server=True,
                labels={"service": service_name},
            ),
            tracing=TracingConfig(
                enabled=True,
                service_name=service_name,
                sampler=TracingSampler.PARENT_BASED,
                sample_ratio=0.1,
            ),
            environment="production",
        )
