"""Factory functions for creating observable stores.

This module provides convenient factory functions for creating stores
with observability features enabled.
"""

from __future__ import annotations

import os
from typing import Any, TypeVar

from truthound.stores.base import BaseStore, StoreConfig, Serializable
from truthound.stores.observability.config import (
    AuditConfig,
    AuditLogLevel,
    MetricsConfig,
    ObservabilityConfig,
    TracingConfig,
    TracingSampler,
)
from truthound.stores.observability.store import ObservableStore

T = TypeVar("T", bound=Serializable)
ConfigT = TypeVar("ConfigT", bound=StoreConfig)


def get_default_observability_config() -> ObservabilityConfig:
    """Get observability config from environment variables.

    Environment variables:
        TRUTHOUND_AUDIT_ENABLED: Enable audit logging (default: true)
        TRUTHOUND_AUDIT_LEVEL: Audit level (minimal, standard, verbose, debug)
        TRUTHOUND_AUDIT_BACKEND: Audit backend (memory, file, json)
        TRUTHOUND_AUDIT_PATH: Path for file-based audit logs

        TRUTHOUND_METRICS_ENABLED: Enable metrics (default: true)
        TRUTHOUND_METRICS_PREFIX: Metrics prefix (default: truthound_store)
        TRUTHOUND_METRICS_PORT: HTTP port for metrics (default: 9090)
        TRUTHOUND_METRICS_HTTP_ENABLED: Enable HTTP metrics endpoint

        TRUTHOUND_TRACING_ENABLED: Enable tracing (default: false)
        TRUTHOUND_TRACING_SERVICE_NAME: Service name for traces
        TRUTHOUND_TRACING_ENDPOINT: OTLP endpoint URL
        TRUTHOUND_TRACING_SAMPLER: Sampler (always_on, always_off, ratio, parent_based)
        TRUTHOUND_TRACING_SAMPLE_RATIO: Sample ratio (0.0-1.0)

        TRUTHOUND_ENVIRONMENT: Environment name (dev, staging, prod)
    """

    def get_bool(key: str, default: bool = False) -> bool:
        value = os.environ.get(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
        return default

    def get_float(key: str, default: float) -> float:
        try:
            return float(os.environ.get(key, default))
        except ValueError:
            return default

    def get_int(key: str, default: int) -> int:
        try:
            return int(os.environ.get(key, default))
        except ValueError:
            return default

    environment = os.environ.get("TRUTHOUND_ENVIRONMENT", "development")

    # Audit config
    audit_enabled = get_bool("TRUTHOUND_AUDIT_ENABLED", True)
    audit_level_str = os.environ.get("TRUTHOUND_AUDIT_LEVEL", "standard")
    try:
        audit_level = AuditLogLevel(audit_level_str.lower())
    except ValueError:
        audit_level = AuditLogLevel.STANDARD

    audit_config = AuditConfig(
        enabled=audit_enabled,
        level=audit_level,
        backend=os.environ.get("TRUTHOUND_AUDIT_BACKEND", "json"),
        file_path=os.environ.get("TRUTHOUND_AUDIT_PATH"),
        retention_days=get_int("TRUTHOUND_AUDIT_RETENTION_DAYS", 90),
    )

    # Metrics config
    metrics_enabled = get_bool("TRUTHOUND_METRICS_ENABLED", True)
    metrics_config = MetricsConfig(
        enabled=metrics_enabled,
        prefix=os.environ.get("TRUTHOUND_METRICS_PREFIX", "truthound_store"),
        http_port=get_int("TRUTHOUND_METRICS_PORT", 9090),
        enable_http_server=get_bool("TRUTHOUND_METRICS_HTTP_ENABLED", False),
        push_gateway_url=os.environ.get("TRUTHOUND_METRICS_PUSH_GATEWAY"),
    )

    # Tracing config
    tracing_enabled = get_bool("TRUTHOUND_TRACING_ENABLED", False)
    sampler_str = os.environ.get("TRUTHOUND_TRACING_SAMPLER", "parent_based")
    try:
        sampler = TracingSampler(sampler_str.lower())
    except ValueError:
        sampler = TracingSampler.PARENT_BASED

    tracing_config = TracingConfig(
        enabled=tracing_enabled,
        service_name=os.environ.get("TRUTHOUND_TRACING_SERVICE_NAME", "truthound-store"),
        endpoint=os.environ.get("TRUTHOUND_TRACING_ENDPOINT"),
        sampler=sampler,
        sample_ratio=get_float("TRUTHOUND_TRACING_SAMPLE_RATIO", 1.0),
        exporter=os.environ.get("TRUTHOUND_TRACING_EXPORTER", "otlp"),
    )

    return ObservabilityConfig(
        audit=audit_config,
        metrics=metrics_config,
        tracing=tracing_config,
        environment=environment,
    )


def create_observable_store(
    store: BaseStore[T, ConfigT],
    config: ObservabilityConfig | None = None,
    use_env_config: bool = True,
    **overrides: Any,
) -> ObservableStore[T, ConfigT]:
    """Create an observable store wrapper.

    Args:
        store: The base store to wrap.
        config: Observability configuration. If None and use_env_config is True,
            configuration is read from environment variables.
        use_env_config: Whether to use environment variables for configuration.
        **overrides: Override specific config values.
            Supported overrides:
            - audit_enabled: bool
            - metrics_enabled: bool
            - tracing_enabled: bool
            - service_name: str
            - environment: str

    Returns:
        Observable store wrapper.

    Example:
        >>> from truthound.stores import get_store
        >>> from truthound.stores.observability import create_observable_store
        >>>
        >>> base_store = get_store("filesystem", path="./results")
        >>> store = create_observable_store(base_store, service_name="my-app")
    """
    if config is None:
        if use_env_config:
            config = get_default_observability_config()
        else:
            config = ObservabilityConfig()

    # Apply overrides
    if "audit_enabled" in overrides:
        config.audit.enabled = overrides["audit_enabled"]
    if "metrics_enabled" in overrides:
        config.metrics.enabled = overrides["metrics_enabled"]
    if "tracing_enabled" in overrides:
        config.tracing.enabled = overrides["tracing_enabled"]
    if "service_name" in overrides:
        config.tracing.service_name = overrides["service_name"]
    if "environment" in overrides:
        config.environment = overrides["environment"]

    return ObservableStore(store, config)


def wrap_store(
    store: BaseStore[T, ConfigT],
    audit: bool = True,
    metrics: bool = True,
    tracing: bool = False,
    **kwargs: Any,
) -> ObservableStore[T, ConfigT]:
    """Simple wrapper function with boolean flags.

    This is a convenience function for quick store wrapping.

    Args:
        store: The base store to wrap.
        audit: Enable audit logging.
        metrics: Enable metrics collection.
        tracing: Enable distributed tracing.
        **kwargs: Additional configuration options.

    Returns:
        Observable store wrapper.

    Example:
        >>> from truthound.stores import get_store
        >>> from truthound.stores.observability import wrap_store
        >>>
        >>> base_store = get_store("filesystem", path="./results")
        >>> store = wrap_store(base_store, audit=True, metrics=True, tracing=False)
    """
    config = ObservabilityConfig(
        audit=AuditConfig(enabled=audit),
        metrics=MetricsConfig(enabled=metrics),
        tracing=TracingConfig(enabled=tracing),
    )
    return create_observable_store(store, config, use_env_config=False, **kwargs)
