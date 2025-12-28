"""Observable store wrapper.

This module provides a wrapper that adds observability (audit, metrics, tracing)
to any store implementation without modifying the original store.

The wrapper follows the Decorator pattern and is transparent to users.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator, Generic, Iterator, TypeVar

from truthound.stores.base import (
    BaseStore,
    StoreConfig,
    StoreQuery,
    Serializable,
)
from truthound.stores.observability.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditStatus,
    InMemoryAuditBackend,
)
from truthound.stores.observability.config import ObservabilityConfig
from truthound.stores.observability.metrics import (
    InMemoryMetricsBackend,
    StoreMetrics,
)
from truthound.stores.observability.protocols import (
    AuditBackend,
    MetricsBackend,
    ObservabilityContext,
    TracingBackend,
)
from truthound.stores.observability.tracing import (
    NoopTracer,
    Span,
    SpanKind,
)

T = TypeVar("T", bound=Serializable)
ConfigT = TypeVar("ConfigT", bound=StoreConfig)


class ObservabilityManager:
    """Manages observability components for a store."""

    def __init__(
        self,
        config: ObservabilityConfig,
        store_type: str,
        store_id: str | None = None,
        audit_backend: AuditBackend | None = None,
        metrics_backend: MetricsBackend | None = None,
        tracer: TracingBackend | None = None,
    ) -> None:
        self.config = config
        self.store_type = store_type
        self.store_id = store_id or str(uuid.uuid4())[:8]

        # Initialize audit
        if config.audit.enabled:
            self._audit_backend = audit_backend or self._create_audit_backend()
            self.audit = AuditLogger(
                self._audit_backend,
                store_type=store_type,
                store_id=self.store_id,
            )
        else:
            self._audit_backend = None
            self.audit = None

        # Initialize metrics
        if config.metrics.enabled:
            self._metrics_backend = metrics_backend or self._create_metrics_backend()
            self.metrics = StoreMetrics(
                self._metrics_backend,
                store_type=store_type,
                store_id=self.store_id,
            )
        else:
            self._metrics_backend = None
            self.metrics = None

        # Initialize tracing
        if config.tracing.enabled:
            self.tracer = tracer or self._create_tracer()
        else:
            self.tracer = NoopTracer()

    def _create_audit_backend(self) -> AuditBackend:
        """Create audit backend based on config."""
        from truthound.stores.observability.audit import (
            FileAuditBackend,
            JsonAuditBackend,
            AsyncAuditBackend,
        )

        backend_type = self.config.audit.backend.lower()

        if backend_type == "memory":
            return InMemoryAuditBackend(self.config.audit)
        elif backend_type == "file":
            return FileAuditBackend(self.config.audit)
        elif backend_type == "json":
            backend = JsonAuditBackend(self.config.audit)
            return AsyncAuditBackend(backend, self.config.audit)
        else:
            return InMemoryAuditBackend(self.config.audit)

    def _create_metrics_backend(self) -> MetricsBackend:
        """Create metrics backend based on config."""
        from truthound.stores.observability.metrics import PrometheusMetricsBackend

        return PrometheusMetricsBackend(
            self.config.metrics,
            auto_start_server=self.config.metrics.enable_http_server,
        )

    def _create_tracer(self) -> TracingBackend:
        """Create tracer based on config."""
        from truthound.stores.observability.tracing import OpenTelemetryTracer

        return OpenTelemetryTracer(self.config.tracing)

    def create_context(
        self,
        correlation_id: str | None = None,
        **kwargs: Any,
    ) -> ObservabilityContext:
        """Create a new observability context."""
        return ObservabilityContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            **kwargs,
        )

    @contextmanager
    def observe(
        self,
        operation: str,
        event_type: AuditEventType,
        resource_type: str = "",
        resource_id: str | None = None,
        context: ObservabilityContext | None = None,
        **attributes: Any,
    ) -> Generator[ObservabilityContext, None, None]:
        """Context manager for observing an operation.

        This creates a span, records metrics, and logs an audit event.
        """
        context = context or self.create_context()
        start_time = time.perf_counter()
        error: Exception | None = None

        # Start tracing span
        with self.tracer.trace(
            f"store.{operation}",
            context=context,
            kind="internal",
            attributes={
                "store.type": self.store_type,
                "store.id": self.store_id,
                "store.operation": operation,
                "resource.type": resource_type,
                "resource.id": resource_id,
                **attributes,
            },
        ) as span:
            # Update context with trace info
            context = context.with_span(span.trace_id, span.span_id)

            try:
                yield context
            except Exception as e:
                error = e
                raise
            finally:
                duration = time.perf_counter() - start_time

                # Record metrics
                if self.metrics:
                    self.metrics.record_operation(
                        operation=operation,
                        duration_seconds=duration,
                        success=error is None,
                    )
                    if error:
                        self.metrics.record_error(type(error).__name__)

                # Log audit event
                if self.audit:
                    self.audit.log_event(
                        event_type=event_type,
                        operation=operation,
                        status=AuditStatus.FAILURE if error else AuditStatus.SUCCESS,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        correlation_id=context.correlation_id,
                        trace_id=context.trace_id,
                        span_id=context.span_id,
                        user_id=context.user_id,
                        tenant_id=context.tenant_id,
                        duration_ms=duration * 1000,
                        error_type=type(error).__name__ if error else None,
                        error_message=str(error) if error else None,
                        metadata=attributes,
                    )

    def flush(self) -> None:
        """Flush all observability backends."""
        if self.audit:
            self.audit.flush()
        if hasattr(self.tracer, "flush"):
            self.tracer.flush()

    def shutdown(self) -> None:
        """Shutdown all observability backends."""
        if self.audit:
            self.audit.close()
        if hasattr(self.tracer, "shutdown"):
            self.tracer.shutdown()


class ObservableStore(BaseStore[T, ConfigT], Generic[T, ConfigT]):
    """Store wrapper that adds observability to any store.

    This wrapper is transparent - it delegates all operations to the
    underlying store while adding audit logging, metrics, and tracing.

    Example:
        >>> from truthound.stores import get_store
        >>> from truthound.stores.observability import ObservableStore, ObservabilityConfig
        >>>
        >>> base_store = get_store("filesystem", path="./results")
        >>> config = ObservabilityConfig.production("my-service")
        >>> store = ObservableStore(base_store, config)
        >>>
        >>> # All operations are now observable
        >>> store.save(result)  # Logged, metriced, traced
    """

    def __init__(
        self,
        store: BaseStore[T, ConfigT],
        config: ObservabilityConfig | None = None,
        context: ObservabilityContext | None = None,
    ) -> None:
        self._store = store
        self._obs_config = config or ObservabilityConfig()
        self._default_context = context
        self._obs = ObservabilityManager(
            config=self._obs_config,
            store_type=type(store).__name__,
        )

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Not used - delegates to wrapped store."""
        raise NotImplementedError("ObservableStore wraps another store")

    @property
    def config(self) -> ConfigT:
        """Get the underlying store's configuration."""
        return self._store.config

    @property
    def observability(self) -> ObservabilityManager:
        """Get the observability manager."""
        return self._obs

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def _do_initialize(self) -> None:
        """Initialize the underlying store with observability."""
        with self._obs.observe(
            "initialize",
            AuditEventType.INITIALIZE,
            context=self._default_context,
        ):
            self._store.initialize()

    def close(self) -> None:
        """Close the store with observability."""
        with self._obs.observe(
            "close",
            AuditEventType.CLOSE,
            context=self._default_context,
        ):
            self._store.close()
        self._obs.shutdown()

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def save(self, item: T, context: ObservabilityContext | None = None) -> str:
        """Save an item with full observability."""
        item_id = getattr(item, "id", None) or "unknown"
        item_type = type(item).__name__

        with self._obs.observe(
            "save",
            AuditEventType.CREATE,
            resource_type=item_type,
            resource_id=str(item_id),
            context=context or self._default_context,
        ):
            result = self._store.save(item)

            # Record save-specific metrics
            if self._obs.metrics:
                # Try to estimate size
                try:
                    import json
                    size = len(json.dumps(item.to_dict()))
                    self._obs.metrics.record_save(0, size)
                except Exception:
                    pass

            return result

    def get(self, item_id: str, context: ObservabilityContext | None = None) -> T:
        """Get an item with full observability."""
        with self._obs.observe(
            "get",
            AuditEventType.READ,
            resource_id=item_id,
            context=context or self._default_context,
        ):
            return self._store.get(item_id)

    def exists(self, item_id: str, context: ObservabilityContext | None = None) -> bool:
        """Check existence with observability."""
        with self._obs.observe(
            "exists",
            AuditEventType.READ,
            resource_id=item_id,
            context=context or self._default_context,
        ):
            return self._store.exists(item_id)

    def delete(self, item_id: str, context: ObservabilityContext | None = None) -> bool:
        """Delete an item with full observability."""
        with self._obs.observe(
            "delete",
            AuditEventType.DELETE,
            resource_id=item_id,
            context=context or self._default_context,
        ):
            return self._store.delete(item_id)

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def list_ids(
        self,
        query: StoreQuery | None = None,
        context: ObservabilityContext | None = None,
    ) -> list[str]:
        """List IDs with observability."""
        with self._obs.observe(
            "list_ids",
            AuditEventType.LIST,
            context=context or self._default_context,
            query=query.data_asset if query else None,
        ):
            return self._store.list_ids(query)

    def query(
        self,
        query: StoreQuery,
        context: ObservabilityContext | None = None,
    ) -> list[T]:
        """Query items with observability."""
        with self._obs.observe(
            "query",
            AuditEventType.QUERY,
            context=context or self._default_context,
            query_asset=query.data_asset,
            query_limit=query.limit,
        ):
            results = self._store.query(query)

            # Record query metrics
            if self._obs.metrics:
                self._obs.metrics.record_query(0, len(results))

            return results

    def iter_query(
        self,
        query: StoreQuery,
        batch_size: int = 100,
        context: ObservabilityContext | None = None,
    ) -> Iterator[T]:
        """Iterate query results with observability."""
        with self._obs.observe(
            "iter_query",
            AuditEventType.QUERY,
            context=context or self._default_context,
            query_asset=query.data_asset,
            batch_size=batch_size,
        ):
            yield from self._store.iter_query(query, batch_size)

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def get_latest(
        self,
        data_asset: str,
        context: ObservabilityContext | None = None,
    ) -> T | None:
        """Get latest item with observability."""
        with self._obs.observe(
            "get_latest",
            AuditEventType.READ,
            context=context or self._default_context,
            data_asset=data_asset,
        ):
            return self._store.get_latest(data_asset)

    def count(
        self,
        query: StoreQuery | None = None,
        context: ObservabilityContext | None = None,
    ) -> int:
        """Count items with observability."""
        with self._obs.observe(
            "count",
            AuditEventType.COUNT,
            context=context or self._default_context,
        ):
            result = self._store.count(query)

            # Update item count gauge
            if self._obs.metrics and query is None:
                self._obs.metrics.set_item_count(result)

            return result

    def clear(
        self,
        query: StoreQuery | None = None,
        context: ObservabilityContext | None = None,
    ) -> int:
        """Clear items with observability."""
        with self._obs.observe(
            "clear",
            AuditEventType.BATCH_DELETE,
            context=context or self._default_context,
        ):
            return self._store.clear(query)

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def save_batch(
        self,
        items: list[T],
        context: ObservabilityContext | None = None,
    ) -> list[str]:
        """Save multiple items with observability."""
        with self._obs.observe(
            "save_batch",
            AuditEventType.BATCH_CREATE,
            context=context or self._default_context,
            batch_size=len(items),
        ) as ctx:
            start = time.perf_counter()
            results = []
            for item in items:
                results.append(self._store.save(item))

            # Record batch metrics
            if self._obs.metrics:
                duration = time.perf_counter() - start
                self._obs.metrics.record_batch("save", len(items), duration)

            return results

    def delete_batch(
        self,
        item_ids: list[str],
        context: ObservabilityContext | None = None,
    ) -> int:
        """Delete multiple items with observability."""
        with self._obs.observe(
            "delete_batch",
            AuditEventType.BATCH_DELETE,
            context=context or self._default_context,
            batch_size=len(item_ids),
        ):
            start = time.perf_counter()
            deleted = 0
            for item_id in item_ids:
                if self._store.delete(item_id):
                    deleted += 1

            # Record batch metrics
            if self._obs.metrics:
                duration = time.perf_counter() - start
                self._obs.metrics.record_batch("delete", len(item_ids), duration)

            return deleted

    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------

    def with_context(self, context: ObservabilityContext) -> "ObservableStore[T, ConfigT]":
        """Create a new store instance with the given default context."""
        return ObservableStore(
            self._store,
            self._obs_config,
            context=context,
        )

    def with_user(self, user_id: str, tenant_id: str | None = None) -> "ObservableStore[T, ConfigT]":
        """Create a new store instance with user context."""
        context = self._obs.create_context(
            user_id=user_id,
            tenant_id=tenant_id,
        )
        return self.with_context(context)

    # -------------------------------------------------------------------------
    # Observability Access
    # -------------------------------------------------------------------------

    def get_metrics(self) -> str:
        """Export current metrics in Prometheus format."""
        if self._obs._metrics_backend:
            return self._obs._metrics_backend.export()
        return ""

    def get_audit_events(
        self,
        start_time=None,
        end_time=None,
        event_type: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events."""
        if self._obs._audit_backend:
            return self._obs._audit_backend.query(
                start_time=start_time,
                end_time=end_time,
                event_type=event_type,
                user_id=user_id,
                resource_id=resource_id,
                limit=limit,
            )
        return []

    def flush_observability(self) -> None:
        """Flush all observability data."""
        self._obs.flush()
