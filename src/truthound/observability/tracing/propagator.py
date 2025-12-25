"""Context propagators for distributed tracing.

Propagators handle injecting and extracting trace context from
carrier formats (HTTP headers, message metadata, etc.).

Supported Formats:
    - W3C Trace Context (traceparent, tracestate)
    - W3C Baggage
    - B3 (Zipkin format)
    - Jaeger
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, TypeVar

from truthound.observability.tracing.span import SpanContextData

T = TypeVar("T")


# =============================================================================
# Carrier Getter/Setter
# =============================================================================


class CarrierGetter(ABC):
    """Abstract interface for getting values from a carrier."""

    @abstractmethod
    def get(self, carrier: Any, key: str) -> str | None:
        """Get a value from the carrier.

        Args:
            carrier: The carrier object.
            key: Key to get.

        Returns:
            Value or None if not found.
        """
        pass

    @abstractmethod
    def keys(self, carrier: Any) -> list[str]:
        """Get all keys from the carrier.

        Args:
            carrier: The carrier object.

        Returns:
            List of keys.
        """
        pass


class CarrierSetter(ABC):
    """Abstract interface for setting values in a carrier."""

    @abstractmethod
    def set(self, carrier: Any, key: str, value: str) -> None:
        """Set a value in the carrier.

        Args:
            carrier: The carrier object.
            key: Key to set.
            value: Value to set.
        """
        pass


class DictCarrierGetter(CarrierGetter):
    """Getter for dictionary-like carriers."""

    def get(self, carrier: Mapping[str, str], key: str) -> str | None:
        return carrier.get(key)

    def keys(self, carrier: Mapping[str, str]) -> list[str]:
        return list(carrier.keys())


class DictCarrierSetter(CarrierSetter):
    """Setter for dictionary-like carriers."""

    def set(self, carrier: MutableMapping[str, str], key: str, value: str) -> None:
        carrier[key] = value


# Default getter/setter for dict carriers
_default_getter = DictCarrierGetter()
_default_setter = DictCarrierSetter()


# =============================================================================
# Propagator Interface
# =============================================================================


class Propagator(ABC):
    """Abstract base class for context propagators.

    Propagators inject and extract trace context from carriers.
    """

    @abstractmethod
    def inject(
        self,
        context: SpanContextData,
        carrier: MutableMapping[str, str],
        setter: CarrierSetter | None = None,
    ) -> None:
        """Inject context into carrier.

        Args:
            context: Span context to inject.
            carrier: Carrier to inject into.
            setter: Carrier setter (default: dict setter).
        """
        pass

    @abstractmethod
    def extract(
        self,
        carrier: Mapping[str, str],
        getter: CarrierGetter | None = None,
    ) -> SpanContextData | None:
        """Extract context from carrier.

        Args:
            carrier: Carrier to extract from.
            getter: Carrier getter (default: dict getter).

        Returns:
            Extracted SpanContextData or None.
        """
        pass

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """Get header fields used by this propagator.

        Returns:
            List of header field names.
        """
        pass


# =============================================================================
# W3C Trace Context Propagator
# =============================================================================


class W3CTraceContextPropagator(Propagator):
    """W3C Trace Context propagator.

    Implements the W3C Trace Context specification for propagating
    trace context in HTTP headers.

    Headers:
        - traceparent: version-trace_id-span_id-flags
        - tracestate: vendor-specific trace context

    Example:
        >>> propagator = W3CTraceContextPropagator()
        >>> headers = {}
        >>> propagator.inject(span.context, headers)
        >>> # headers = {"traceparent": "00-abc...-def...-01"}
    """

    TRACEPARENT_HEADER = "traceparent"
    TRACESTATE_HEADER = "tracestate"
    VERSION = "00"

    @property
    def fields(self) -> list[str]:
        return [self.TRACEPARENT_HEADER, self.TRACESTATE_HEADER]

    def inject(
        self,
        context: SpanContextData,
        carrier: MutableMapping[str, str],
        setter: CarrierSetter | None = None,
    ) -> None:
        """Inject W3C traceparent and tracestate headers."""
        if not context or not context.is_valid:
            return

        setter = setter or _default_setter

        # traceparent: version-trace_id-span_id-flags
        traceparent = f"{self.VERSION}-{context.trace_id}-{context.span_id}-{context.trace_flags:02x}"
        setter.set(carrier, self.TRACEPARENT_HEADER, traceparent)

        # tracestate (if present)
        if context.trace_state:
            setter.set(carrier, self.TRACESTATE_HEADER, context.trace_state)

    def extract(
        self,
        carrier: Mapping[str, str],
        getter: CarrierGetter | None = None,
    ) -> SpanContextData | None:
        """Extract W3C traceparent and tracestate from headers."""
        getter = getter or _default_getter

        traceparent = getter.get(carrier, self.TRACEPARENT_HEADER)
        if not traceparent:
            # Try lowercase
            traceparent = getter.get(carrier, self.TRACEPARENT_HEADER.lower())

        if not traceparent:
            return None

        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None

            version, trace_id, span_id, flags = parts

            # Validate version
            if version != self.VERSION:
                return None

            # Validate trace_id and span_id lengths
            if len(trace_id) != 32 or len(span_id) != 16:
                return None

            # Parse flags
            trace_flags = int(flags, 16)

            # Get tracestate
            trace_state = getter.get(carrier, self.TRACESTATE_HEADER) or ""

            return SpanContextData(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=trace_flags,
                trace_state=trace_state,
                is_remote=True,
            )

        except (ValueError, IndexError):
            return None


# =============================================================================
# W3C Baggage Propagator
# =============================================================================


@dataclass
class BaggageEntry:
    """A single baggage entry."""

    key: str
    value: str
    metadata: str = ""


class W3CBaggagePropagator(Propagator):
    """W3C Baggage propagator.

    Propagates baggage (key-value pairs) across service boundaries.
    Baggage is useful for passing contextual data like user IDs,
    tenant IDs, etc.

    Header format: key1=value1;metadata,key2=value2

    Example:
        >>> propagator = W3CBaggagePropagator()
        >>> baggage = {"user_id": "123", "tenant": "acme"}
        >>> propagator.inject_baggage(baggage, headers)
    """

    BAGGAGE_HEADER = "baggage"

    @property
    def fields(self) -> list[str]:
        return [self.BAGGAGE_HEADER]

    def inject(
        self,
        context: SpanContextData,
        carrier: MutableMapping[str, str],
        setter: CarrierSetter | None = None,
    ) -> None:
        """Inject baggage (no-op, use inject_baggage instead)."""
        pass

    def inject_baggage(
        self,
        baggage: Mapping[str, str],
        carrier: MutableMapping[str, str],
        setter: CarrierSetter | None = None,
    ) -> None:
        """Inject baggage into carrier.

        Args:
            baggage: Baggage key-value pairs.
            carrier: Carrier to inject into.
            setter: Carrier setter.
        """
        if not baggage:
            return

        setter = setter or _default_setter

        parts = []
        for key, value in baggage.items():
            # URL-encode key and value
            import urllib.parse
            encoded_key = urllib.parse.quote(key, safe="")
            encoded_value = urllib.parse.quote(value, safe="")
            parts.append(f"{encoded_key}={encoded_value}")

        setter.set(carrier, self.BAGGAGE_HEADER, ",".join(parts))

    def extract(
        self,
        carrier: Mapping[str, str],
        getter: CarrierGetter | None = None,
    ) -> SpanContextData | None:
        """Extract (no-op, use extract_baggage instead)."""
        return None

    def extract_baggage(
        self,
        carrier: Mapping[str, str],
        getter: CarrierGetter | None = None,
    ) -> dict[str, str]:
        """Extract baggage from carrier.

        Args:
            carrier: Carrier to extract from.
            getter: Carrier getter.

        Returns:
            Dictionary of baggage key-value pairs.
        """
        getter = getter or _default_getter

        baggage_header = getter.get(carrier, self.BAGGAGE_HEADER)
        if not baggage_header:
            baggage_header = getter.get(carrier, self.BAGGAGE_HEADER.lower())

        if not baggage_header:
            return {}

        import urllib.parse

        baggage = {}
        for member in baggage_header.split(","):
            member = member.strip()
            if "=" not in member:
                continue

            # Split on first =
            key_value = member.split("=", 1)
            if len(key_value) != 2:
                continue

            key, value = key_value

            # Handle metadata (;property=value)
            if ";" in value:
                value = value.split(";", 1)[0]

            try:
                key = urllib.parse.unquote(key.strip())
                value = urllib.parse.unquote(value.strip())
                baggage[key] = value
            except Exception:
                continue

        return baggage


# =============================================================================
# B3 Propagator (Zipkin)
# =============================================================================


class B3Propagator(Propagator):
    """B3 propagator for Zipkin compatibility.

    Supports both single-header and multi-header formats.

    Multi-header format:
        - X-B3-TraceId
        - X-B3-SpanId
        - X-B3-ParentSpanId
        - X-B3-Sampled
        - X-B3-Flags

    Single-header format:
        - b3: {TraceId}-{SpanId}-{Sampled}-{ParentSpanId}

    Example:
        >>> propagator = B3Propagator(single_header=True)
        >>> propagator.inject(span.context, headers)
    """

    # Multi-header keys
    TRACE_ID_HEADER = "X-B3-TraceId"
    SPAN_ID_HEADER = "X-B3-SpanId"
    PARENT_SPAN_ID_HEADER = "X-B3-ParentSpanId"
    SAMPLED_HEADER = "X-B3-Sampled"
    FLAGS_HEADER = "X-B3-Flags"

    # Single-header key
    SINGLE_HEADER = "b3"

    def __init__(self, single_header: bool = False) -> None:
        """Initialize B3 propagator.

        Args:
            single_header: Use single b3 header format.
        """
        self._single_header = single_header

    @property
    def fields(self) -> list[str]:
        if self._single_header:
            return [self.SINGLE_HEADER]
        return [
            self.TRACE_ID_HEADER,
            self.SPAN_ID_HEADER,
            self.PARENT_SPAN_ID_HEADER,
            self.SAMPLED_HEADER,
            self.FLAGS_HEADER,
        ]

    def inject(
        self,
        context: SpanContextData,
        carrier: MutableMapping[str, str],
        setter: CarrierSetter | None = None,
    ) -> None:
        """Inject B3 headers."""
        if not context or not context.is_valid:
            return

        setter = setter or _default_setter

        if self._single_header:
            # Single header format: {TraceId}-{SpanId}-{Sampled}
            sampled = "1" if context.is_sampled else "0"
            b3 = f"{context.trace_id}-{context.span_id}-{sampled}"
            setter.set(carrier, self.SINGLE_HEADER, b3)
        else:
            # Multi-header format
            setter.set(carrier, self.TRACE_ID_HEADER, context.trace_id)
            setter.set(carrier, self.SPAN_ID_HEADER, context.span_id)
            setter.set(
                carrier, self.SAMPLED_HEADER, "1" if context.is_sampled else "0"
            )

    def extract(
        self,
        carrier: Mapping[str, str],
        getter: CarrierGetter | None = None,
    ) -> SpanContextData | None:
        """Extract B3 context from headers."""
        getter = getter or _default_getter

        # Try single header first
        b3 = getter.get(carrier, self.SINGLE_HEADER)
        if b3:
            return self._parse_single_header(b3)

        # Try multi-header format
        trace_id = getter.get(carrier, self.TRACE_ID_HEADER)
        if not trace_id:
            trace_id = getter.get(carrier, self.TRACE_ID_HEADER.lower())

        span_id = getter.get(carrier, self.SPAN_ID_HEADER)
        if not span_id:
            span_id = getter.get(carrier, self.SPAN_ID_HEADER.lower())

        if not trace_id or not span_id:
            return None

        # Get sampled flag
        sampled = getter.get(carrier, self.SAMPLED_HEADER)
        if not sampled:
            sampled = getter.get(carrier, self.SAMPLED_HEADER.lower())

        # Check debug flag
        flags = getter.get(carrier, self.FLAGS_HEADER)
        if not flags:
            flags = getter.get(carrier, self.FLAGS_HEADER.lower())

        trace_flags = 0
        if flags == "1" or sampled == "1":
            trace_flags = 1

        # Pad trace_id to 32 chars if needed (B3 allows 16 or 32)
        if len(trace_id) == 16:
            trace_id = "0" * 16 + trace_id

        return SpanContextData(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            is_remote=True,
        )

    def _parse_single_header(self, b3: str) -> SpanContextData | None:
        """Parse single b3 header."""
        try:
            parts = b3.split("-")
            if len(parts) < 2:
                return None

            trace_id = parts[0]
            span_id = parts[1]

            # Pad trace_id if needed
            if len(trace_id) == 16:
                trace_id = "0" * 16 + trace_id

            trace_flags = 0
            if len(parts) > 2:
                sampled = parts[2]
                if sampled in ("1", "d"):
                    trace_flags = 1

            return SpanContextData(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=trace_flags,
                is_remote=True,
            )

        except (ValueError, IndexError):
            return None


# =============================================================================
# Jaeger Propagator
# =============================================================================


class JaegerPropagator(Propagator):
    """Jaeger native format propagator.

    Header format:
        - uber-trace-id: {trace-id}:{span-id}:{parent-span-id}:{flags}

    Example:
        >>> propagator = JaegerPropagator()
        >>> propagator.inject(span.context, headers)
    """

    HEADER = "uber-trace-id"

    @property
    def fields(self) -> list[str]:
        return [self.HEADER]

    def inject(
        self,
        context: SpanContextData,
        carrier: MutableMapping[str, str],
        setter: CarrierSetter | None = None,
    ) -> None:
        """Inject Jaeger header."""
        if not context or not context.is_valid:
            return

        setter = setter or _default_setter

        # Format: {trace-id}:{span-id}:{parent-span-id}:{flags}
        header = f"{context.trace_id}:{context.span_id}:0:{context.trace_flags}"
        setter.set(carrier, self.HEADER, header)

    def extract(
        self,
        carrier: Mapping[str, str],
        getter: CarrierGetter | None = None,
    ) -> SpanContextData | None:
        """Extract Jaeger context."""
        getter = getter or _default_getter

        header = getter.get(carrier, self.HEADER)
        if not header:
            header = getter.get(carrier, self.HEADER.lower())

        if not header:
            return None

        try:
            parts = header.split(":")
            if len(parts) != 4:
                return None

            trace_id, span_id, parent_span_id, flags = parts

            # Pad trace_id to 32 chars if needed
            if len(trace_id) == 16:
                trace_id = "0" * 16 + trace_id

            return SpanContextData(
                trace_id=trace_id,
                span_id=span_id,
                trace_flags=int(flags),
                is_remote=True,
            )

        except (ValueError, IndexError):
            return None


# =============================================================================
# Composite Propagator
# =============================================================================


class CompositePropagator(Propagator):
    """Propagator that combines multiple propagators.

    Useful for supporting multiple header formats simultaneously.

    Example:
        >>> propagator = CompositePropagator([
        ...     W3CTraceContextPropagator(),
        ...     B3Propagator(),
        ... ])
    """

    def __init__(self, propagators: list[Propagator] | None = None) -> None:
        """Initialize composite propagator.

        Args:
            propagators: List of propagators.
        """
        self._propagators = list(propagators or [])

    def add_propagator(self, propagator: Propagator) -> None:
        """Add a propagator."""
        self._propagators.append(propagator)

    @property
    def fields(self) -> list[str]:
        """Get all fields from all propagators."""
        all_fields = []
        for prop in self._propagators:
            all_fields.extend(prop.fields)
        return list(set(all_fields))

    def inject(
        self,
        context: SpanContextData,
        carrier: MutableMapping[str, str],
        setter: CarrierSetter | None = None,
    ) -> None:
        """Inject using all propagators."""
        for propagator in self._propagators:
            try:
                propagator.inject(context, carrier, setter)
            except Exception:
                pass

    def extract(
        self,
        carrier: Mapping[str, str],
        getter: CarrierGetter | None = None,
    ) -> SpanContextData | None:
        """Extract using first matching propagator."""
        for propagator in self._propagators:
            try:
                context = propagator.extract(carrier, getter)
                if context:
                    return context
            except Exception:
                pass
        return None


# =============================================================================
# Global Propagator
# =============================================================================


_global_propagator: Propagator | None = None
_propagator_lock = threading.Lock()


def get_global_propagator() -> Propagator:
    """Get the global propagator.

    Returns:
        Global propagator (default: W3C Trace Context).
    """
    global _global_propagator

    with _propagator_lock:
        if _global_propagator is None:
            _global_propagator = CompositePropagator([
                W3CTraceContextPropagator(),
                W3CBaggagePropagator(),
            ])
        return _global_propagator


def set_global_propagator(propagator: Propagator) -> None:
    """Set the global propagator.

    Args:
        propagator: Propagator to use globally.
    """
    global _global_propagator

    with _propagator_lock:
        _global_propagator = propagator


# =============================================================================
# Convenience Functions
# =============================================================================


def inject_context(
    context: SpanContextData,
    carrier: MutableMapping[str, str],
    propagator: Propagator | None = None,
) -> None:
    """Inject context into carrier.

    Args:
        context: Span context to inject.
        carrier: Carrier to inject into.
        propagator: Propagator to use (default: global).
    """
    prop = propagator or get_global_propagator()
    prop.inject(context, carrier)


def extract_context(
    carrier: Mapping[str, str],
    propagator: Propagator | None = None,
) -> SpanContextData | None:
    """Extract context from carrier.

    Args:
        carrier: Carrier to extract from.
        propagator: Propagator to use (default: global).

    Returns:
        Extracted SpanContextData or None.
    """
    prop = propagator or get_global_propagator()
    return prop.extract(carrier)
