"""Trace context management for distributed tracing.

This module provides context propagation for distributed tracing,
enabling correlation of logs and metrics across service boundaries.

Features:
    - Trace and span context management
    - W3C Trace Context compatible headers
    - Automatic context propagation
    - Integration with logging and metrics

Architecture:
    TraceContext -> SpanContext -> (nested spans)

    Each trace has a unique trace_id and contains multiple spans.
    Spans can be nested to represent call hierarchies.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Trace IDs
# =============================================================================


def generate_trace_id() -> str:
    """Generate a unique trace ID (128-bit hex string)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a unique span ID (64-bit hex string)."""
    return uuid.uuid4().hex[:16]


# =============================================================================
# Span Status
# =============================================================================


class SpanStatus(Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


# =============================================================================
# Span Context
# =============================================================================


@dataclass
class SpanContext:
    """Context for a single span within a trace.

    A span represents a single operation within a trace.
    Spans can be nested to form a tree structure.

    Attributes:
        trace_id: Unique trace identifier.
        span_id: Unique span identifier.
        parent_span_id: Parent span ID (None for root span).
        name: Human-readable span name.
        start_time: When the span started.
        end_time: When the span ended (None if active).
        status: Span status.
        attributes: Key-value attributes.
        events: List of events that occurred during span.
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str = ""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Check if span is still active."""
        return self.end_time is None

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Event attributes.
        """
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: SpanStatus, message: str = "") -> None:
        """Set span status.

        Args:
            status: New status.
            message: Optional status message.
        """
        self.status = status
        if message:
            self.attributes["status_message"] = message

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
        }


# =============================================================================
# Trace Context
# =============================================================================


@dataclass
class TraceContext:
    """Context for a complete trace.

    A trace represents a complete request flow and contains multiple spans.

    Attributes:
        trace_id: Unique trace identifier.
        root_span: The root span of the trace.
        baggage: Key-value pairs propagated across service boundaries.
    """

    trace_id: str
    root_span: SpanContext | None = None
    baggage: dict[str, str] = field(default_factory=dict)

    @property
    def span_id(self) -> str | None:
        """Get current span ID."""
        return self.root_span.span_id if self.root_span else None

    def set_baggage(self, key: str, value: str) -> None:
        """Set a baggage item.

        Baggage is propagated to all downstream services.

        Args:
            key: Baggage key.
            value: Baggage value.
        """
        self.baggage[key] = value

    def get_baggage(self, key: str) -> str | None:
        """Get a baggage item.

        Args:
            key: Baggage key.

        Returns:
            Baggage value or None.
        """
        return self.baggage.get(key)

    def to_headers(self) -> dict[str, str]:
        """Convert to W3C Trace Context headers.

        Returns:
            Dictionary with traceparent and tracestate headers.
        """
        headers = {}

        if self.root_span:
            # W3C traceparent format: version-trace_id-span_id-flags
            traceparent = f"00-{self.trace_id}-{self.root_span.span_id}-01"
            headers["traceparent"] = traceparent

        if self.baggage:
            # W3C baggage format: key=value,key2=value2
            baggage_str = ",".join(f"{k}={v}" for k, v in self.baggage.items())
            headers["baggage"] = baggage_str

        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "TraceContext | None":
        """Parse trace context from W3C headers.

        Args:
            headers: HTTP headers dictionary.

        Returns:
            TraceContext or None if no valid headers.
        """
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None

            version, trace_id, parent_span_id, flags = parts

            # Create context
            ctx = cls(trace_id=trace_id)

            # Parse baggage
            baggage_str = headers.get("baggage", "")
            if baggage_str:
                for item in baggage_str.split(","):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        ctx.baggage[key.strip()] = value.strip()

            # Note: parent_span_id is stored but root_span is created by caller
            ctx._parent_span_id = parent_span_id

            return ctx

        except Exception:
            return None


# =============================================================================
# Context Storage
# =============================================================================


class _ContextStorage:
    """Thread-local storage for trace context."""

    _local = threading.local()

    @classmethod
    def get_context(cls) -> TraceContext | None:
        """Get current trace context."""
        return getattr(cls._local, "context", None)

    @classmethod
    def set_context(cls, context: TraceContext | None) -> None:
        """Set current trace context."""
        cls._local.context = context

    @classmethod
    def get_span_stack(cls) -> list[SpanContext]:
        """Get current span stack."""
        if not hasattr(cls._local, "span_stack"):
            cls._local.span_stack = []
        return cls._local.span_stack

    @classmethod
    def push_span(cls, span: SpanContext) -> None:
        """Push a span onto the stack."""
        stack = cls.get_span_stack()
        stack.append(span)

    @classmethod
    def pop_span(cls) -> SpanContext | None:
        """Pop a span from the stack."""
        stack = cls.get_span_stack()
        return stack.pop() if stack else None

    @classmethod
    def current_span(cls) -> SpanContext | None:
        """Get current span."""
        stack = cls.get_span_stack()
        return stack[-1] if stack else None


# =============================================================================
# Context Management Functions
# =============================================================================


def current_context() -> TraceContext | None:
    """Get the current trace context.

    Returns:
        Current TraceContext or None.
    """
    return _ContextStorage.get_context()


def current_span() -> SpanContext | None:
    """Get the current span.

    Returns:
        Current SpanContext or None.
    """
    return _ContextStorage.current_span()


@contextmanager
def with_context(context: TraceContext) -> Iterator[TraceContext]:
    """Context manager to set trace context.

    Args:
        context: TraceContext to use.

    Yields:
        The trace context.
    """
    previous = _ContextStorage.get_context()
    _ContextStorage.set_context(context)
    try:
        yield context
    finally:
        _ContextStorage.set_context(previous)


def create_trace(
    name: str = "",
    *,
    trace_id: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> TraceContext:
    """Create a new trace with root span.

    Args:
        name: Name for the root span.
        trace_id: Custom trace ID (generates new if None).
        attributes: Initial span attributes.

    Returns:
        New TraceContext.
    """
    tid = trace_id or generate_trace_id()

    root_span = SpanContext(
        trace_id=tid,
        span_id=generate_span_id(),
        name=name,
        attributes=attributes or {},
    )

    return TraceContext(
        trace_id=tid,
        root_span=root_span,
    )


@contextmanager
def create_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Iterator[SpanContext]:
    """Create a new span within current trace.

    Args:
        name: Span name.
        attributes: Span attributes.

    Yields:
        The new SpanContext.
    """
    context = _ContextStorage.get_context()
    parent = _ContextStorage.current_span()

    if context is None:
        # No active trace, create one
        context = create_trace(name)
        _ContextStorage.set_context(context)
        span = context.root_span
    else:
        # Create child span
        span = SpanContext(
            trace_id=context.trace_id,
            span_id=generate_span_id(),
            parent_span_id=parent.span_id if parent else None,
            name=name,
            attributes=attributes or {},
        )

    _ContextStorage.push_span(span)

    try:
        yield span
        if span.status == SpanStatus.UNSET:
            span.set_status(SpanStatus.OK)
    except Exception as e:
        span.set_status(SpanStatus.ERROR, str(e))
        span.set_attribute("exception.type", type(e).__name__)
        span.set_attribute("exception.message", str(e))
        raise
    finally:
        span.end()
        _ContextStorage.pop_span()


# =============================================================================
# Span Collectors
# =============================================================================


class SpanCollector:
    """Collects completed spans for export.

    SpanCollector receives completed spans and can export them
    to various backends (console, file, tracing services).
    """

    def __init__(self) -> None:
        """Initialize collector."""
        self._spans: list[SpanContext] = []
        self._lock = threading.Lock()
        self._handlers: list[Callable[[SpanContext], None]] = []

    def add_handler(self, handler: Callable[[SpanContext], None]) -> None:
        """Add a span handler.

        Args:
            handler: Callable that receives completed spans.
        """
        self._handlers.append(handler)

    def collect(self, span: SpanContext) -> None:
        """Collect a completed span.

        Args:
            span: Completed span.
        """
        with self._lock:
            self._spans.append(span)

        for handler in self._handlers:
            try:
                handler(span)
            except Exception:
                pass

    def get_spans(self) -> list[SpanContext]:
        """Get all collected spans."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear collected spans."""
        with self._lock:
            self._spans.clear()


# Global span collector
_span_collector = SpanCollector()


def get_span_collector() -> SpanCollector:
    """Get the global span collector."""
    return _span_collector


# =============================================================================
# Propagation Helpers
# =============================================================================


def inject_context(
    headers: dict[str, str],
    context: TraceContext | None = None,
) -> dict[str, str]:
    """Inject trace context into headers.

    Args:
        headers: Headers dictionary to modify.
        context: TraceContext (uses current if None).

    Returns:
        Headers with trace context.
    """
    ctx = context or current_context()
    if ctx:
        headers.update(ctx.to_headers())
    return headers


def extract_context(headers: dict[str, str]) -> TraceContext | None:
    """Extract trace context from headers.

    Args:
        headers: HTTP headers.

    Returns:
        TraceContext or None.
    """
    return TraceContext.from_headers(headers)
