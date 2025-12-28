"""Protocols for observability backends.

This module defines the abstract interfaces that all observability backends
must implement. Using protocols enables duck-typing and loose coupling.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from truthound.stores.observability.audit import AuditEvent
    from truthound.stores.observability.tracing import Span


T = TypeVar("T")


@dataclass
class ObservabilityContext:
    """Context for observability operations.

    This context is passed through all observability operations to provide
    correlation and additional metadata.

    Attributes:
        correlation_id: Unique ID for correlating related operations.
        trace_id: Distributed trace ID.
        span_id: Current span ID.
        user_id: ID of the user performing the operation.
        tenant_id: ID of the tenant (for multi-tenant systems).
        source_ip: Source IP address.
        user_agent: User agent string.
        extra: Additional context-specific data.
        timestamp: When this context was created.
    """

    correlation_id: str
    trace_id: str | None = None
    span_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    source_ip: str | None = None
    user_agent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for logging/serialization."""
        return {
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "extra": self.extra,
            "timestamp": self.timestamp.isoformat(),
        }

    def with_span(self, trace_id: str, span_id: str) -> "ObservabilityContext":
        """Create a new context with updated span information."""
        return ObservabilityContext(
            correlation_id=self.correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            source_ip=self.source_ip,
            user_agent=self.user_agent,
            extra=self.extra,
            timestamp=self.timestamp,
        )


@runtime_checkable
class AuditBackend(Protocol):
    """Protocol for audit logging backends.

    Implementations must be thread-safe and handle failures gracefully
    (audit logging should never break the main operation).
    """

    @abstractmethod
    def log(self, event: "AuditEvent") -> None:
        """Log an audit event.

        Args:
            event: The audit event to log.

        Note:
            Implementations should handle failures gracefully and not
            raise exceptions that would break the main operation.
        """
        ...

    @abstractmethod
    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_type: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
        limit: int = 100,
    ) -> list["AuditEvent"]:
        """Query audit events.

        Args:
            start_time: Filter events after this time.
            end_time: Filter events before this time.
            event_type: Filter by event type.
            user_id: Filter by user ID.
            resource_id: Filter by resource ID.
            limit: Maximum number of events to return.

        Returns:
            List of matching audit events.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered events to persistent storage."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release resources."""
        ...


@runtime_checkable
class MetricsBackend(Protocol):
    """Protocol for metrics collection backends.

    Implementations must be thread-safe and support the standard
    metric types: counter, gauge, histogram, and summary.
    """

    @abstractmethod
    def counter(
        self,
        name: str,
        value: float = 1.0,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name.
            value: Value to increment by (default 1).
            labels: Optional labels for the metric.
        """
        ...

    @abstractmethod
    def gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Set a gauge metric value.

        Args:
            name: Metric name.
            value: Value to set.
            labels: Optional labels for the metric.
        """
        ...

    @abstractmethod
    def histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram observation.

        Args:
            name: Metric name.
            value: Value to observe.
            labels: Optional labels for the metric.
        """
        ...

    @abstractmethod
    def summary(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a summary observation.

        Args:
            name: Metric name.
            value: Value to observe.
            labels: Optional labels for the metric.
        """
        ...

    @abstractmethod
    @contextmanager
    def timer(
        self,
        name: str,
        labels: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to time an operation.

        Args:
            name: Metric name for the timer.
            labels: Optional labels for the metric.

        Yields:
            None - timing is recorded on exit.
        """
        ...

    @abstractmethod
    def export(self) -> str:
        """Export all metrics in the configured format.

        Returns:
            String representation of all metrics.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all metrics to their initial values."""
        ...


@runtime_checkable
class TracingBackend(Protocol):
    """Protocol for distributed tracing backends.

    Implementations must support span creation, context propagation,
    and be compatible with OpenTelemetry semantics.
    """

    @abstractmethod
    def start_span(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> "Span":
        """Start a new span.

        Args:
            name: Span name.
            context: Optional observability context.
            kind: Span kind ("internal", "server", "client", "producer", "consumer").
            attributes: Initial attributes for the span.

        Returns:
            The created span.
        """
        ...

    @abstractmethod
    @contextmanager
    def trace(
        self,
        name: str,
        context: ObservabilityContext | None = None,
        kind: str = "internal",
        attributes: dict[str, Any] | None = None,
    ) -> Generator["Span", None, None]:
        """Context manager for tracing an operation.

        Args:
            name: Span name.
            context: Optional observability context.
            kind: Span kind.
            attributes: Initial attributes for the span.

        Yields:
            The active span.
        """
        ...

    @abstractmethod
    def inject_context(self, carrier: dict[str, str]) -> None:
        """Inject tracing context into a carrier (e.g., HTTP headers).

        Args:
            carrier: Dictionary to inject context into.
        """
        ...

    @abstractmethod
    def extract_context(self, carrier: dict[str, str]) -> ObservabilityContext | None:
        """Extract tracing context from a carrier.

        Args:
            carrier: Dictionary to extract context from.

        Returns:
            Observability context if found, None otherwise.
        """
        ...

    @abstractmethod
    def get_current_span(self) -> "Span | None":
        """Get the currently active span.

        Returns:
            The current span, or None if no span is active.
        """
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered spans to the exporter."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the tracer and release resources."""
        ...


class ObservabilityManager(Protocol):
    """Protocol for unified observability management.

    Combines audit, metrics, and tracing into a single interface.
    """

    @property
    @abstractmethod
    def audit(self) -> AuditBackend:
        """Get the audit backend."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> MetricsBackend:
        """Get the metrics backend."""
        ...

    @property
    @abstractmethod
    def tracer(self) -> TracingBackend:
        """Get the tracing backend."""
        ...

    @abstractmethod
    def create_context(
        self,
        correlation_id: str | None = None,
        **kwargs: Any,
    ) -> ObservabilityContext:
        """Create a new observability context.

        Args:
            correlation_id: Optional correlation ID (generated if not provided).
            **kwargs: Additional context attributes.

        Returns:
            New observability context.
        """
        ...

    @abstractmethod
    @contextmanager
    def observe(
        self,
        operation: str,
        context: ObservabilityContext | None = None,
        **attributes: Any,
    ) -> Generator[ObservabilityContext, None, None]:
        """Context manager for observing an operation.

        This creates a span, records metrics, and logs an audit event
        for the operation.

        Args:
            operation: Operation name.
            context: Optional existing context.
            **attributes: Additional attributes for the operation.

        Yields:
            The observability context for this operation.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown all observability components."""
        ...
