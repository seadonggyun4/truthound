"""Compatibility utilities for OpenTelemetry integration.

This module provides wrapper classes and utilities that ensure
Truthound spans and contexts are fully compatible with OpenTelemetry
APIs and vice versa.

The wrappers implement the complete OpenTelemetry Span interface,
allowing Truthound spans to be used anywhere OTEL spans are expected.
"""

from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from truthound.observability.tracing.otel.detection import is_otel_sdk_available

logger = logging.getLogger(__name__)


# =============================================================================
# Span Context Conversion
# =============================================================================


def to_otel_span_context(truthound_ctx: Any) -> Any:
    """Convert Truthound SpanContextData to OTEL SpanContext.

    Args:
        truthound_ctx: Truthound SpanContextData.

    Returns:
        OpenTelemetry SpanContext.

    Raises:
        ImportError: If OTEL API is not available.
    """
    if not is_otel_sdk_available():
        raise ImportError("OpenTelemetry SDK not available")

    from opentelemetry.trace import SpanContext, TraceState

    # Parse trace ID
    trace_id = truthound_ctx.trace_id
    if isinstance(trace_id, str):
        trace_id = int(trace_id, 16)

    # Parse span ID
    span_id = truthound_ctx.span_id
    if isinstance(span_id, str):
        span_id = int(span_id, 16)

    # Parse trace state
    trace_state = TraceState()
    if hasattr(truthound_ctx, "trace_state") and truthound_ctx.trace_state:
        state_str = truthound_ctx.trace_state
        if isinstance(state_str, dict):
            for k, v in state_str.items():
                trace_state = trace_state.add(k, v)
        elif isinstance(state_str, str):
            for pair in state_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    trace_state = trace_state.add(key.strip(), value.strip())

    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=getattr(truthound_ctx, "is_remote", False),
        trace_flags=truthound_ctx.trace_flags,
        trace_state=trace_state,
    )


def from_otel_span_context(otel_ctx: Any) -> Any:
    """Convert OTEL SpanContext to Truthound SpanContextData.

    Args:
        otel_ctx: OpenTelemetry SpanContext.

    Returns:
        Truthound SpanContextData.
    """
    from truthound.observability.tracing.span import SpanContextData

    # Format trace ID as hex string
    trace_id = otel_ctx.trace_id
    if isinstance(trace_id, int):
        trace_id = format(trace_id, "032x")

    # Format span ID as hex string
    span_id = otel_ctx.span_id
    if isinstance(span_id, int):
        span_id = format(span_id, "016x")

    # Convert trace state to string
    trace_state = ""
    if hasattr(otel_ctx, "trace_state") and otel_ctx.trace_state:
        pairs = [f"{k}={v}" for k, v in otel_ctx.trace_state.items()]
        trace_state = ",".join(pairs)

    return SpanContextData(
        trace_id=trace_id,
        span_id=span_id,
        trace_flags=otel_ctx.trace_flags,
        trace_state=trace_state,
        is_remote=otel_ctx.is_remote,
    )


# =============================================================================
# OTEL Span Wrapper
# =============================================================================


class OTELSpanWrapper:
    """Wraps a Truthound span with full OpenTelemetry Span interface.

    This allows Truthound spans to be used anywhere an OTEL Span is expected,
    such as with OTEL context propagation, instrumentation, or exporters.

    Example:
        >>> truthound_span = tracer.start_span("operation")
        >>> otel_span = OTELSpanWrapper(truthound_span)
        >>>
        >>> # Use with OTEL context
        >>> from opentelemetry import context, trace
        >>> ctx = trace.set_span_in_context(otel_span)
        >>> token = context.attach(ctx)
    """

    def __init__(self, truthound_span: Any) -> None:
        """Initialize wrapper.

        Args:
            truthound_span: Truthound Span to wrap.
        """
        self._span = truthound_span
        self._otel_context = None

    def get_span_context(self) -> Any:
        """Get the span context.

        Returns:
            OpenTelemetry SpanContext.
        """
        if self._otel_context is None:
            self._otel_context = to_otel_span_context(self._span.context)
        return self._otel_context

    @property
    def context(self) -> Any:
        """Get the span context (alias for OTEL compatibility)."""
        return self.get_span_context()

    @property
    def name(self) -> str:
        """Get span name."""
        return self._span.name

    def set_attribute(self, key: str, value: Any) -> "OTELSpanWrapper":
        """Set a span attribute.

        Args:
            key: Attribute key.
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        self._span.set_attribute(key, value)
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> "OTELSpanWrapper":
        """Set multiple attributes.

        Args:
            attributes: Dictionary of attributes.

        Returns:
            Self for chaining.
        """
        self._span.set_attributes(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | None = None,
    ) -> "OTELSpanWrapper":
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Event attributes.
            timestamp: Event timestamp in nanoseconds.

        Returns:
            Self for chaining.
        """
        # Convert nanosecond timestamp to seconds
        ts = timestamp / 1_000_000_000 if timestamp else None
        self._span.add_event(name, attributes=attributes, timestamp=ts)
        return self

    def set_status(self, status: Any, description: str | None = None) -> "OTELSpanWrapper":
        """Set span status.

        Args:
            status: OTEL Status or StatusCode.
            description: Optional status description.

        Returns:
            Self for chaining.
        """
        from truthound.observability.tracing.span import StatusCode

        # Handle OTEL Status object
        if hasattr(status, "status_code"):
            status = status.status_code

        # Map OTEL StatusCode to Truthound
        truthound_status = StatusCode.UNSET
        if hasattr(status, "name"):
            status_map = {"OK": StatusCode.OK, "ERROR": StatusCode.ERROR, "UNSET": StatusCode.UNSET}
            truthound_status = status_map.get(status.name, StatusCode.UNSET)

        self._span.set_status(truthound_status, description or "")
        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> "OTELSpanWrapper":
        """Record an exception.

        Args:
            exception: Exception to record.
            attributes: Additional attributes.
            timestamp: Event timestamp in nanoseconds.
            escaped: Whether exception escaped span scope.

        Returns:
            Self for chaining.
        """
        self._span.record_exception(exception, attributes=attributes, escaped=escaped)
        return self

    def update_name(self, name: str) -> "OTELSpanWrapper":
        """Update span name.

        Args:
            name: New span name.

        Returns:
            Self for chaining.
        """
        self._span.update_name(name)
        return self

    def end(self, end_time: int | None = None) -> None:
        """End the span.

        Args:
            end_time: End time in nanoseconds.
        """
        # Convert nanosecond timestamp to seconds
        ts = end_time / 1_000_000_000 if end_time else None
        self._span.end(ts)

    def is_recording(self) -> bool:
        """Check if span is recording."""
        return self._span.is_recording()

    def __enter__(self) -> "OTELSpanWrapper":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if exc_val is not None:
            self.record_exception(exc_val, escaped=True)
            if is_otel_sdk_available():
                from opentelemetry.trace import StatusCode
                self.set_status(StatusCode.ERROR, str(exc_val))
        self.end()

    def unwrap(self) -> Any:
        """Get the underlying Truthound span."""
        return self._span


# =============================================================================
# Truthound Span Wrapper
# =============================================================================


class TruthoundSpanWrapper:
    """Wraps an OTEL span with Truthound Span interface.

    This allows OTEL spans to be used with Truthound APIs
    and processors.

    Example:
        >>> from opentelemetry import trace
        >>> otel_tracer = trace.get_tracer("my-service")
        >>> otel_span = otel_tracer.start_span("operation")
        >>> truthound_span = TruthoundSpanWrapper(otel_span)
        >>>
        >>> # Use with Truthound processor
        >>> processor.on_end(truthound_span)
    """

    def __init__(self, otel_span: Any) -> None:
        """Initialize wrapper.

        Args:
            otel_span: OTEL Span to wrap.
        """
        self._span = otel_span
        self._truthound_context = None

    @property
    def context(self) -> Any:
        """Get the span context as Truthound SpanContextData."""
        if self._truthound_context is None:
            self._truthound_context = from_otel_span_context(
                self._span.get_span_context()
            )
        return self._truthound_context

    @property
    def name(self) -> str:
        """Get span name."""
        if hasattr(self._span, "name"):
            return self._span.name
        return ""

    @property
    def kind(self) -> Any:
        """Get span kind as Truthound SpanKind."""
        from truthound.observability.tracing.span import SpanKind

        if hasattr(self._span, "kind") and self._span.kind:
            kind_map = {
                "INTERNAL": SpanKind.INTERNAL,
                "SERVER": SpanKind.SERVER,
                "CLIENT": SpanKind.CLIENT,
                "PRODUCER": SpanKind.PRODUCER,
                "CONSUMER": SpanKind.CONSUMER,
            }
            return kind_map.get(self._span.kind.name, SpanKind.INTERNAL)
        return SpanKind.INTERNAL

    @property
    def start_time(self) -> float:
        """Get start time in seconds."""
        if hasattr(self._span, "start_time"):
            return self._span.start_time / 1_000_000_000
        return time.time()

    @property
    def end_time(self) -> float | None:
        """Get end time in seconds."""
        if hasattr(self._span, "end_time") and self._span.end_time:
            return self._span.end_time / 1_000_000_000
        return None

    @property
    def duration_ms(self) -> float | None:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def attributes(self) -> dict[str, Any]:
        """Get span attributes."""
        if hasattr(self._span, "attributes") and self._span.attributes:
            return dict(self._span.attributes)
        return {}

    @property
    def events(self) -> list[Any]:
        """Get span events."""
        if hasattr(self._span, "events"):
            return list(self._span.events)
        return []

    @property
    def links(self) -> list[Any]:
        """Get span links."""
        if hasattr(self._span, "links"):
            return list(self._span.links)
        return []

    @property
    def status(self) -> tuple[Any, str]:
        """Get span status as (StatusCode, message) tuple."""
        from truthound.observability.tracing.span import StatusCode

        if hasattr(self._span, "status") and self._span.status:
            status_map = {
                "UNSET": StatusCode.UNSET,
                "OK": StatusCode.OK,
                "ERROR": StatusCode.ERROR,
            }
            code = status_map.get(self._span.status.status_code.name, StatusCode.UNSET)
            message = self._span.status.description or ""
            return (code, message)
        return (StatusCode.UNSET, "")

    @property
    def parent(self) -> Any:
        """Get parent span context."""
        if hasattr(self._span, "parent") and self._span.parent:
            return from_otel_span_context(self._span.parent)
        return None

    def set_attribute(self, key: str, value: Any) -> "TruthoundSpanWrapper":
        """Set a span attribute."""
        self._span.set_attribute(key, value)
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> "TruthoundSpanWrapper":
        """Set multiple attributes."""
        if hasattr(self._span, "set_attributes"):
            self._span.set_attributes(dict(attributes))
        else:
            for k, v in attributes.items():
                self._span.set_attribute(k, v)
        return self

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> "TruthoundSpanWrapper":
        """Add an event to the span."""
        # Convert seconds to nanoseconds
        ts = int(timestamp * 1_000_000_000) if timestamp else None
        self._span.add_event(name, attributes=attributes, timestamp=ts)
        return self

    def set_status(self, code: Any, message: str = "") -> "TruthoundSpanWrapper":
        """Set span status."""
        from truthound.observability.tracing.span import StatusCode

        if is_otel_sdk_available():
            from opentelemetry.trace import StatusCode as OTELStatusCode

            otel_status_map = {
                StatusCode.UNSET: OTELStatusCode.UNSET,
                StatusCode.OK: OTELStatusCode.OK,
                StatusCode.ERROR: OTELStatusCode.ERROR,
            }
            otel_code = otel_status_map.get(code, OTELStatusCode.UNSET)
            self._span.set_status(otel_code, message)

        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        escaped: bool = False,
    ) -> "TruthoundSpanWrapper":
        """Record an exception."""
        self._span.record_exception(exception, attributes=attributes)
        return self

    def update_name(self, name: str) -> "TruthoundSpanWrapper":
        """Update span name."""
        self._span.update_name(name)
        return self

    def end(self, end_time: float | None = None) -> None:
        """End the span."""
        # Convert seconds to nanoseconds
        ts = int(end_time * 1_000_000_000) if end_time else None
        self._span.end(ts)

    def is_recording(self) -> bool:
        """Check if span is recording."""
        return self._span.is_recording()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.parent.span_id if self.parent else None,
            "name": self.name,
            "kind": self.kind.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": {
                "code": self.status[0].name,
                "message": self.status[1],
            },
        }

    def __enter__(self) -> "TruthoundSpanWrapper":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if exc_val is not None:
            self.record_exception(exc_val, escaped=True)
            from truthound.observability.tracing.span import StatusCode
            self.set_status(StatusCode.ERROR, str(exc_val))
        self.end()

    def unwrap(self) -> Any:
        """Get the underlying OTEL span."""
        return self._span


# =============================================================================
# Factory Functions
# =============================================================================


def create_compatible_span(
    span: Any,
    target_api: str = "auto",
) -> Any:
    """Create a span wrapper compatible with the target API.

    Args:
        span: Source span (Truthound or OTEL).
        target_api: Target API ("truthound", "otel", or "auto").

    Returns:
        Wrapped span compatible with target API.

    Example:
        >>> # Make Truthound span OTEL-compatible
        >>> otel_span = create_compatible_span(truthound_span, target_api="otel")
        >>>
        >>> # Make OTEL span Truthound-compatible
        >>> truthound_span = create_compatible_span(otel_span, target_api="truthound")
    """
    # Detect source type
    is_truthound = hasattr(span, "context") and hasattr(span.context, "trace_id")
    is_otel = hasattr(span, "get_span_context")

    # Already wrapped
    if isinstance(span, (OTELSpanWrapper, TruthoundSpanWrapper)):
        if target_api == "otel" and isinstance(span, OTELSpanWrapper):
            return span
        if target_api == "truthound" and isinstance(span, TruthoundSpanWrapper):
            return span
        # Unwrap and re-wrap
        span = span.unwrap()

    # Auto-detect target
    if target_api == "auto":
        if is_otel_sdk_available():
            target_api = "otel"
        else:
            target_api = "truthound"

    # Create wrapper
    if target_api == "otel":
        if is_truthound:
            return OTELSpanWrapper(span)
        return span  # Already OTEL
    else:
        if is_otel:
            return TruthoundSpanWrapper(span)
        return span  # Already Truthound


def wrap_for_otel(span: Any) -> OTELSpanWrapper:
    """Convenience function to wrap a Truthound span for OTEL.

    Args:
        span: Truthound span.

    Returns:
        OTELSpanWrapper.
    """
    if isinstance(span, OTELSpanWrapper):
        return span
    return OTELSpanWrapper(span)


def wrap_for_truthound(span: Any) -> TruthoundSpanWrapper:
    """Convenience function to wrap an OTEL span for Truthound.

    Args:
        span: OTEL span.

    Returns:
        TruthoundSpanWrapper.
    """
    if isinstance(span, TruthoundSpanWrapper):
        return span
    return TruthoundSpanWrapper(span)


# =============================================================================
# Context Utilities
# =============================================================================


@contextmanager
def use_otel_context(truthound_span: Any) -> Iterator[Any]:
    """Set a Truthound span in OTEL context.

    This allows using Truthound spans with OTEL instrumentation
    and context propagation.

    Args:
        truthound_span: Truthound span.

    Yields:
        OTEL context token.

    Example:
        >>> with tracer.start_as_current_span("operation") as span:
        ...     with use_otel_context(span) as ctx:
        ...         # OTEL context now has this span
        ...         # Child operations will see this as parent
        ...         do_work()
    """
    if not is_otel_sdk_available():
        yield None
        return

    from opentelemetry import context, trace

    # Wrap span
    otel_span = OTELSpanWrapper(truthound_span)

    # Set in context
    ctx = trace.set_span_in_context(otel_span)
    token = context.attach(ctx)

    try:
        yield token
    finally:
        context.detach(token)


@contextmanager
def use_truthound_context(otel_span: Any) -> Iterator[Any]:
    """Set an OTEL span in Truthound context.

    This allows using OTEL spans with Truthound context management.

    Args:
        otel_span: OTEL span.

    Yields:
        Truthound span wrapper.
    """
    from truthound.observability.tracing.provider import _ContextStorage

    # Wrap span
    truthound_span = TruthoundSpanWrapper(otel_span)

    # Push to Truthound context stack
    _ContextStorage.push_span(truthound_span.unwrap())

    try:
        yield truthound_span
    finally:
        _ContextStorage.pop_span()
