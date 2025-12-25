"""Tracer provider and tracer implementations.

The TracerProvider is the entry point for creating tracers.
Tracers are used to create spans for tracing operations.

Architecture:
    TracerProvider
        -> Tracer (scoped by instrumentation name/version)
            -> Span (represents a single operation)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence, TYPE_CHECKING

from truthound.observability.tracing.span import (
    Span,
    SpanBase,
    NoOpSpan,
    SpanContextData,
    SpanKind,
    SpanLimits,
    Link,
    generate_trace_id,
    generate_span_id,
)
from truthound.observability.tracing.sampler import (
    Sampler,
    AlwaysOnSampler,
    SamplingDecision,
)
from truthound.observability.tracing.resource import Resource, get_default_resource

if TYPE_CHECKING:
    from truthound.observability.tracing.processor import SpanProcessor


# =============================================================================
# Context Storage
# =============================================================================


class _ContextStorage:
    """Thread-local storage for trace context."""

    _local = threading.local()

    @classmethod
    def get_span_stack(cls) -> list[Span]:
        """Get current span stack."""
        if not hasattr(cls._local, "span_stack"):
            cls._local.span_stack = []
        return cls._local.span_stack

    @classmethod
    def push_span(cls, span: Span) -> None:
        """Push a span onto the stack."""
        stack = cls.get_span_stack()
        stack.append(span)

    @classmethod
    def pop_span(cls) -> Span | None:
        """Pop a span from the stack."""
        stack = cls.get_span_stack()
        return stack.pop() if stack else None

    @classmethod
    def current_span(cls) -> Span | None:
        """Get current span."""
        stack = cls.get_span_stack()
        return stack[-1] if stack else None

    @classmethod
    def current_context(cls) -> SpanContextData | None:
        """Get current span context."""
        span = cls.current_span()
        return span.context if span else None


# =============================================================================
# Tracer
# =============================================================================


class Tracer:
    """Creates spans for tracing operations.

    Tracers are obtained from TracerProvider and are scoped by
    instrumentation library name and version.

    Example:
        >>> tracer = provider.get_tracer("my.component", "1.0.0")
        >>> with tracer.start_span("operation") as span:
        ...     span.set_attribute("key", "value")
        ...     do_work()
    """

    def __init__(
        self,
        provider: "TracerProvider",
        name: str,
        version: str = "",
        schema_url: str = "",
    ) -> None:
        """Initialize tracer.

        Args:
            provider: Parent TracerProvider.
            name: Instrumentation library name.
            version: Instrumentation library version.
            schema_url: Schema URL for semantic conventions.
        """
        self._provider = provider
        self._name = name
        self._version = version
        self._schema_url = schema_url

    @property
    def name(self) -> str:
        """Get instrumentation name."""
        return self._name

    @property
    def version(self) -> str:
        """Get instrumentation version."""
        return self._version

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
        start_time: float | None = None,
        parent: SpanContextData | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name.
            kind: Span kind.
            attributes: Initial attributes.
            links: Links to other spans.
            start_time: Start time (defaults to now).
            parent: Parent context (defaults to current).
            record_exception: Record exceptions automatically.
            set_status_on_exception: Set error status on exception.

        Returns:
            New Span.
        """
        return self._provider._start_span(
            name=name,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            parent=parent,
            tracer=self,
        )

    @contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
        start_time: float | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        """Start a span and set it as the current span.

        Context manager that starts a span, makes it the current span,
        and ends it when the context exits.

        Args:
            name: Span name.
            kind: Span kind.
            attributes: Initial attributes.
            links: Links to other spans.
            start_time: Start time.
            record_exception: Record exceptions.
            set_status_on_exception: Set error status.
            end_on_exit: End span on context exit.

        Yields:
            The started Span.

        Example:
            >>> with tracer.start_as_current_span("operation") as span:
            ...     span.set_attribute("key", "value")
            ...     do_work()
        """
        span = self.start_span(
            name=name,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
        )

        _ContextStorage.push_span(span)

        # Notify processor
        parent = _ContextStorage.current_context()
        for processor in self._provider._processors:
            try:
                processor.on_start(span, parent)
            except Exception:
                pass

        try:
            yield span
        except Exception as e:
            if record_exception:
                span.record_exception(e, escaped=True)
            if set_status_on_exception:
                from truthound.observability.tracing.span import StatusCode
                span.set_status(StatusCode.ERROR, str(e))
            raise
        finally:
            _ContextStorage.pop_span()

            if end_on_exit:
                span.end()

            # Notify processors
            for processor in self._provider._processors:
                try:
                    processor.on_end(span)
                except Exception:
                    pass


# =============================================================================
# TracerProvider
# =============================================================================


class TracerProvider:
    """Provides Tracers for creating spans.

    TracerProvider is the main entry point for the tracing API.
    It manages tracers, processors, and sampling.

    Example:
        >>> from truthound.observability.tracing import (
        ...     TracerProvider,
        ...     BatchSpanProcessor,
        ...     OTLPSpanExporter,
        ... )
        >>>
        >>> provider = TracerProvider()
        >>> provider.add_processor(
        ...     BatchSpanProcessor(OTLPSpanExporter("http://localhost:4317"))
        ... )
        >>>
        >>> tracer = provider.get_tracer("my.service")
        >>> with tracer.start_as_current_span("operation"):
        ...     do_work()
    """

    def __init__(
        self,
        *,
        resource: Resource | None = None,
        sampler: Sampler | None = None,
        span_limits: SpanLimits | None = None,
        processors: list["SpanProcessor"] | None = None,
    ) -> None:
        """Initialize tracer provider.

        Args:
            resource: Resource describing this service.
            sampler: Sampler for sampling decisions.
            span_limits: Limits for spans.
            processors: Initial span processors.
        """
        self._resource = resource or get_default_resource()
        self._sampler = sampler or AlwaysOnSampler()
        self._span_limits = span_limits or SpanLimits.default()
        self._processors: list["SpanProcessor"] = list(processors or [])
        self._tracers: dict[tuple[str, str], Tracer] = {}
        self._shutdown = False
        self._lock = threading.Lock()

    @property
    def resource(self) -> Resource:
        """Get the resource."""
        return self._resource

    @property
    def sampler(self) -> Sampler:
        """Get the sampler."""
        return self._sampler

    def get_tracer(
        self,
        name: str,
        version: str = "",
        schema_url: str = "",
    ) -> Tracer:
        """Get or create a tracer.

        Args:
            name: Instrumentation library name.
            version: Instrumentation library version.
            schema_url: Schema URL.

        Returns:
            Tracer for the given instrumentation.
        """
        key = (name, version)

        with self._lock:
            if key not in self._tracers:
                self._tracers[key] = Tracer(
                    provider=self,
                    name=name,
                    version=version,
                    schema_url=schema_url,
                )
            return self._tracers[key]

    def add_processor(self, processor: "SpanProcessor") -> None:
        """Add a span processor.

        Args:
            processor: Processor to add.
        """
        with self._lock:
            self._processors.append(processor)

    def _start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[Link] | None = None,
        start_time: float | None = None,
        parent: SpanContextData | None = None,
        tracer: Tracer | None = None,
    ) -> Span:
        """Internal method to start a span.

        Args:
            name: Span name.
            kind: Span kind.
            attributes: Initial attributes.
            links: Links to other spans.
            start_time: Start time.
            parent: Parent context.
            tracer: Creating tracer.

        Returns:
            New Span (or NoOpSpan if not sampled).
        """
        if self._shutdown:
            return NoOpSpan()

        # Get parent context
        if parent is None:
            parent = _ContextStorage.current_context()

        # Generate IDs
        if parent:
            trace_id = parent.trace_id
        else:
            trace_id = generate_trace_id()

        span_id = generate_span_id()

        # Make sampling decision
        sampling_result = self._sampler.should_sample(
            parent_context=parent,
            trace_id=trace_id,
            name=name,
            kind=kind,
            attributes=attributes,
            links=links,
        )

        if sampling_result.decision == SamplingDecision.DROP:
            # Return no-op span
            return NoOpSpan(
                context=SpanContextData(
                    trace_id=trace_id,
                    span_id=span_id,
                    trace_flags=0,
                )
            )

        # Create span context
        context = SpanContextData(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=1 if sampling_result.is_sampled else 0,
        )

        # Merge sampler attributes with provided attributes
        merged_attrs = dict(sampling_result.attributes)
        if attributes:
            merged_attrs.update(attributes)

        # Add instrumentation info
        if tracer:
            merged_attrs["otel.library.name"] = tracer.name
            if tracer.version:
                merged_attrs["otel.library.version"] = tracer.version

        # Create span
        span = Span(
            name=name,
            context=context,
            parent=parent,
            kind=kind,
            links=links,
            attributes=merged_attrs,
            start_time=start_time,
            limits=self._span_limits,
        )

        return span

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all processors.

        Args:
            timeout_millis: Maximum time to wait.

        Returns:
            True if all processors flushed successfully.
        """
        success = True
        per_processor_timeout = timeout_millis // max(len(self._processors), 1)

        for processor in self._processors:
            try:
                if not processor.force_flush(per_processor_timeout):
                    success = False
            except Exception:
                success = False

        return success

    def shutdown(self) -> bool:
        """Shutdown the provider and all processors.

        Returns:
            True if shutdown completed successfully.
        """
        with self._lock:
            if self._shutdown:
                return True
            self._shutdown = True

        success = True
        for processor in self._processors:
            try:
                if not processor.shutdown():
                    success = False
            except Exception:
                success = False

        return success


# =============================================================================
# Global Provider
# =============================================================================


_global_provider: TracerProvider | None = None
_provider_lock = threading.Lock()


def get_tracer_provider() -> TracerProvider:
    """Get the global tracer provider.

    Returns:
        Global TracerProvider.
    """
    global _global_provider

    with _provider_lock:
        if _global_provider is None:
            _global_provider = TracerProvider()
        return _global_provider


def set_tracer_provider(provider: TracerProvider) -> None:
    """Set the global tracer provider.

    Args:
        provider: TracerProvider to use globally.
    """
    global _global_provider

    with _provider_lock:
        _global_provider = provider


def get_tracer(name: str, version: str = "") -> Tracer:
    """Get a tracer from the global provider.

    Convenience function for quick access.

    Args:
        name: Instrumentation name.
        version: Instrumentation version.

    Returns:
        Tracer from global provider.
    """
    return get_tracer_provider().get_tracer(name, version)


# =============================================================================
# Context Helpers
# =============================================================================


def get_current_span() -> Span | None:
    """Get the current active span.

    Returns:
        Current Span or None.
    """
    return _ContextStorage.current_span()


def get_current_context() -> SpanContextData | None:
    """Get the current span context.

    Returns:
        Current SpanContextData or None.
    """
    return _ContextStorage.current_context()


@contextmanager
def use_span(span: Span, end_on_exit: bool = False) -> Iterator[Span]:
    """Use an existing span as the current span.

    Args:
        span: Span to use.
        end_on_exit: End span on context exit.

    Yields:
        The span.
    """
    _ContextStorage.push_span(span)
    try:
        yield span
    finally:
        _ContextStorage.pop_span()
        if end_on_exit:
            span.end()
