"""Span implementation for distributed tracing.

This module provides the core Span abstraction following OpenTelemetry
semantic conventions while remaining backend-agnostic.

A Span represents a single operation within a trace. Spans can be nested
to form a tree structure representing the complete request flow.
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Iterator, Mapping, Sequence, TypeVar

T = TypeVar("T")


# =============================================================================
# IDs
# =============================================================================


def generate_trace_id() -> str:
    """Generate a 128-bit trace ID as hex string."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a 64-bit span ID as hex string."""
    return uuid.uuid4().hex[:16]


# =============================================================================
# Enums
# =============================================================================


class SpanKind(Enum):
    """Type of span.

    SpanKind describes the relationship between the span and its parent/children.
    """

    INTERNAL = auto()  # Default, internal operation
    SERVER = auto()  # Server-side of a synchronous RPC
    CLIENT = auto()  # Client-side of a synchronous RPC
    PRODUCER = auto()  # Producer of an async message
    CONSUMER = auto()  # Consumer of an async message


class StatusCode(Enum):
    """Status of a span.

    Follows OpenTelemetry status conventions.
    """

    UNSET = auto()  # Default, status not set
    OK = auto()  # Operation completed successfully
    ERROR = auto()  # Operation failed


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class SpanLimits:
    """Configuration limits for spans.

    Prevents unbounded memory growth from too many attributes/events.
    """

    max_attributes: int = 128
    max_events: int = 128
    max_links: int = 128
    max_attribute_length: int = 4096

    @classmethod
    def default(cls) -> "SpanLimits":
        """Get default limits."""
        return cls()

    @classmethod
    def unlimited(cls) -> "SpanLimits":
        """Get unlimited (for testing)."""
        return cls(
            max_attributes=float("inf"),
            max_events=float("inf"),
            max_links=float("inf"),
            max_attribute_length=float("inf"),
        )


@dataclass
class Event:
    """An event that occurred during a span's lifetime.

    Events are time-stamped annotations with optional attributes.
    """

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class SpanContextData:
    """Immutable span context data for propagation.

    Contains the minimum information needed to identify a span
    and propagate it across service boundaries.
    """

    trace_id: str
    span_id: str
    trace_flags: int = 1  # 0=not sampled, 1=sampled
    trace_state: str = ""
    is_remote: bool = False

    @property
    def is_valid(self) -> bool:
        """Check if context is valid."""
        return bool(self.trace_id and self.span_id)

    @property
    def is_sampled(self) -> bool:
        """Check if span is sampled."""
        return bool(self.trace_flags & 0x01)

    def to_w3c_traceparent(self) -> str:
        """Convert to W3C traceparent header value."""
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_w3c_traceparent(cls, header: str) -> "SpanContextData | None":
        """Parse from W3C traceparent header."""
        try:
            parts = header.split("-")
            if len(parts) != 4:
                return None
            version, trace_id, span_id, flags = parts
            if version != "00":
                return None
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=int(flags, 16),
                is_remote=True,
            )
        except (ValueError, IndexError):
            return None


@dataclass
class Link:
    """A link to another span.

    Links are used to connect spans that are causally related but
    don't have a direct parent-child relationship.
    """

    context: SpanContextData
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "attributes": self.attributes,
        }


# =============================================================================
# Span Interface
# =============================================================================


class SpanBase(ABC):
    """Abstract base class for spans.

    Defines the interface that all span implementations must follow.
    """

    @property
    @abstractmethod
    def context(self) -> SpanContextData:
        """Get span context."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get span name."""
        pass

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> "SpanBase":
        """Set a span attribute.

        Args:
            key: Attribute key (should follow semantic conventions).
            value: Attribute value.

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def set_attributes(self, attributes: Mapping[str, Any]) -> "SpanBase":
        """Set multiple attributes at once.

        Args:
            attributes: Dictionary of attributes.

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> "SpanBase":
        """Add an event to the span.

        Args:
            name: Event name.
            attributes: Event attributes.
            timestamp: Event timestamp (defaults to now).

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def set_status(self, code: StatusCode, message: str = "") -> "SpanBase":
        """Set span status.

        Args:
            code: Status code.
            message: Optional status message (for errors).

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        escaped: bool = False,
    ) -> "SpanBase":
        """Record an exception on the span.

        Args:
            exception: The exception to record.
            attributes: Additional attributes.
            escaped: Whether the exception escaped the span's scope.

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def update_name(self, name: str) -> "SpanBase":
        """Update the span name.

        Args:
            name: New span name.

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def end(self, end_time: float | None = None) -> None:
        """End the span.

        Args:
            end_time: Optional end time (defaults to now).
        """
        pass

    @abstractmethod
    def is_recording(self) -> bool:
        """Check if span is recording.

        Returns:
            True if span is still recording events.
        """
        pass


# =============================================================================
# Span Implementation
# =============================================================================


class Span(SpanBase):
    """Concrete span implementation.

    A Span represents a single operation within a trace. It records
    timing, attributes, events, and status information.
    """

    def __init__(
        self,
        name: str,
        context: SpanContextData,
        *,
        parent: SpanContextData | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        links: Sequence[Link] | None = None,
        attributes: Mapping[str, Any] | None = None,
        start_time: float | None = None,
        limits: SpanLimits | None = None,
    ) -> None:
        """Initialize span.

        Args:
            name: Span name.
            context: Span context.
            parent: Parent span context.
            kind: Span kind.
            links: Links to other spans.
            attributes: Initial attributes.
            start_time: Start time (defaults to now).
            limits: Span limits configuration.
        """
        self._name = name
        self._context = context
        self._parent = parent
        self._kind = kind
        self._limits = limits or SpanLimits.default()

        self._start_time = start_time or time.time()
        self._end_time: float | None = None

        self._attributes: dict[str, Any] = {}
        self._events: list[Event] = []
        self._links: list[Link] = list(links or [])

        self._status_code = StatusCode.UNSET
        self._status_message = ""

        self._lock = threading.Lock()
        self._ended = False

        # Add initial attributes
        if attributes:
            self.set_attributes(attributes)

    @property
    def context(self) -> SpanContextData:
        """Get span context."""
        return self._context

    @property
    def name(self) -> str:
        """Get span name."""
        return self._name

    @property
    def parent(self) -> SpanContextData | None:
        """Get parent span context."""
        return self._parent

    @property
    def kind(self) -> SpanKind:
        """Get span kind."""
        return self._kind

    @property
    def start_time(self) -> float:
        """Get start time."""
        return self._start_time

    @property
    def end_time(self) -> float | None:
        """Get end time."""
        return self._end_time

    @property
    def duration_ns(self) -> int | None:
        """Get duration in nanoseconds."""
        if self._end_time is None:
            return None
        return int((self._end_time - self._start_time) * 1_000_000_000)

    @property
    def duration_ms(self) -> float | None:
        """Get duration in milliseconds."""
        if self._end_time is None:
            return None
        return (self._end_time - self._start_time) * 1000

    @property
    def attributes(self) -> dict[str, Any]:
        """Get span attributes."""
        with self._lock:
            return dict(self._attributes)

    @property
    def events(self) -> list[Event]:
        """Get span events."""
        with self._lock:
            return list(self._events)

    @property
    def links(self) -> list[Link]:
        """Get span links."""
        return list(self._links)

    @property
    def status(self) -> tuple[StatusCode, str]:
        """Get span status."""
        return (self._status_code, self._status_message)

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a span attribute."""
        if not self.is_recording():
            return self

        with self._lock:
            if len(self._attributes) >= self._limits.max_attributes:
                return self

            # Truncate string values if needed
            if isinstance(value, str) and len(value) > self._limits.max_attribute_length:
                value = value[: self._limits.max_attribute_length]

            self._attributes[key] = value

        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> "Span":
        """Set multiple attributes."""
        for key, value in attributes.items():
            self.set_attribute(key, value)
        return self

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> "Span":
        """Add an event."""
        if not self.is_recording():
            return self

        with self._lock:
            if len(self._events) >= self._limits.max_events:
                return self

            event = Event(
                name=name,
                timestamp=timestamp or time.time(),
                attributes=dict(attributes or {}),
            )
            self._events.append(event)

        return self

    def set_status(self, code: StatusCode, message: str = "") -> "Span":
        """Set span status."""
        if not self.is_recording():
            return self

        # Once ERROR is set, it cannot be changed
        if self._status_code == StatusCode.ERROR:
            return self

        with self._lock:
            self._status_code = code
            self._status_message = message

        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        escaped: bool = False,
    ) -> "Span":
        """Record an exception."""
        import traceback

        exc_attributes = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.stacktrace": "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            ),
            "exception.escaped": escaped,
        }

        if attributes:
            exc_attributes.update(attributes)

        self.add_event("exception", exc_attributes)

        # Set error status if not already set
        if self._status_code != StatusCode.ERROR:
            self.set_status(StatusCode.ERROR, str(exception))

        return self

    def update_name(self, name: str) -> "Span":
        """Update span name."""
        if not self.is_recording():
            return self

        with self._lock:
            self._name = name

        return self

    def add_link(
        self,
        context: SpanContextData,
        attributes: Mapping[str, Any] | None = None,
    ) -> "Span":
        """Add a link to another span."""
        if not self.is_recording():
            return self

        with self._lock:
            if len(self._links) >= self._limits.max_links:
                return self

            self._links.append(Link(context, dict(attributes or {})))

        return self

    def end(self, end_time: float | None = None) -> None:
        """End the span."""
        with self._lock:
            if self._ended:
                return
            self._ended = True
            self._end_time = end_time or time.time()

    def is_recording(self) -> bool:
        """Check if span is recording."""
        return not self._ended

    def __enter__(self) -> "Span":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if exc_val is not None:
            self.record_exception(exc_val, escaped=True)
        elif self._status_code == StatusCode.UNSET:
            self.set_status(StatusCode.OK)
        self.end()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "trace_id": self._context.trace_id,
            "span_id": self._context.span_id,
            "parent_span_id": self._parent.span_id if self._parent else None,
            "name": self._name,
            "kind": self._kind.name,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "duration_ms": self.duration_ms,
            "status": {
                "code": self._status_code.name,
                "message": self._status_message,
            },
            "attributes": self._attributes,
            "events": [e.to_dict() for e in self._events],
            "links": [l.to_dict() for l in self._links],
        }


# =============================================================================
# No-op Span
# =============================================================================


class NoOpSpan(SpanBase):
    """No-operation span for when tracing is disabled.

    Used when sampling decides not to record a span.
    """

    _CONTEXT = SpanContextData(
        trace_id="0" * 32,
        span_id="0" * 16,
        trace_flags=0,
    )

    def __init__(self, context: SpanContextData | None = None) -> None:
        """Initialize no-op span."""
        self._context = context or self._CONTEXT

    @property
    def context(self) -> SpanContextData:
        return self._context

    @property
    def name(self) -> str:
        return ""

    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        return self

    def set_attributes(self, attributes: Mapping[str, Any]) -> "NoOpSpan":
        return self

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> "NoOpSpan":
        return self

    def set_status(self, code: StatusCode, message: str = "") -> "NoOpSpan":
        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        escaped: bool = False,
    ) -> "NoOpSpan":
        return self

    def update_name(self, name: str) -> "NoOpSpan":
        return self

    def end(self, end_time: float | None = None) -> None:
        pass

    def is_recording(self) -> bool:
        return False

    def __enter__(self) -> "NoOpSpan":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
