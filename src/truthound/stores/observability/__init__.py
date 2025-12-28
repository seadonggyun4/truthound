"""Observability layer for stores.

This module provides a unified observability abstraction layer that includes:
- Audit logging for compliance (SOC2, GDPR, HIPAA)
- Prometheus metrics for operational visibility
- Distributed tracing with OpenTelemetry

The design follows these principles:
1. **Pluggable backends**: Each observability aspect has swappable backends
2. **Zero-cost when disabled**: No overhead when observability is turned off
3. **Composable**: Components can be used independently or together
4. **Non-invasive**: Wrapper pattern doesn't modify original stores

Example:
    >>> from truthound.stores import get_store
    >>> from truthound.stores.observability import ObservableStore, ObservabilityConfig
    >>>
    >>> base_store = get_store("filesystem", path="./results")
    >>> observable = ObservableStore(
    ...     base_store,
    ...     ObservabilityConfig(
    ...         audit_enabled=True,
    ...         metrics_enabled=True,
    ...         tracing_enabled=True,
    ...     )
    ... )
"""

from truthound.stores.observability.config import (
    ObservabilityConfig,
    AuditConfig,
    MetricsConfig,
    TracingConfig,
)
from truthound.stores.observability.protocols import (
    AuditBackend,
    MetricsBackend,
    TracingBackend,
    ObservabilityContext,
)
from truthound.stores.observability.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    InMemoryAuditBackend,
    FileAuditBackend,
    JsonAuditBackend,
    CompositeAuditBackend,
)
from truthound.stores.observability.metrics import (
    StoreMetrics,
    MetricsRegistry,
    InMemoryMetricsBackend,
    PrometheusMetricsBackend,
)
from truthound.stores.observability.tracing import (
    SpanContext,
    Span,
    Tracer,
    NoopTracer,
    OpenTelemetryTracer,
)
from truthound.stores.observability.store import (
    ObservableStore,
)
from truthound.stores.observability.factory import (
    create_observable_store,
    get_default_observability_config,
)

__all__ = [
    # Config
    "ObservabilityConfig",
    "AuditConfig",
    "MetricsConfig",
    "TracingConfig",
    # Protocols
    "AuditBackend",
    "MetricsBackend",
    "TracingBackend",
    "ObservabilityContext",
    # Audit
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "InMemoryAuditBackend",
    "FileAuditBackend",
    "JsonAuditBackend",
    "CompositeAuditBackend",
    # Metrics
    "StoreMetrics",
    "MetricsRegistry",
    "InMemoryMetricsBackend",
    "PrometheusMetricsBackend",
    # Tracing
    "SpanContext",
    "Span",
    "Tracer",
    "NoopTracer",
    "OpenTelemetryTracer",
    # Store
    "ObservableStore",
    # Factory
    "create_observable_store",
    "get_default_observability_config",
]
