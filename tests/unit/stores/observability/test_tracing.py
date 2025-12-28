"""Tests for distributed tracing."""

from __future__ import annotations

import time

import pytest

from truthound.stores.observability.config import TracingConfig, TracingSampler
from truthound.stores.observability.protocols import ObservabilityContext
from truthound.stores.observability.tracing import (
    InMemoryTracer,
    NoopTracer,
    OpenTelemetryTracer,
    Span,
    SpanContext,
    SpanEvent,
    SpanKind,
    SpanStatus,
    Tracer,
)


class TestSpanContext:
    """Tests for SpanContext."""

    def test_generate(self) -> None:
        ctx = SpanContext.generate()
        assert ctx.trace_id is not None
        assert ctx.span_id is not None
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 16

    def test_from_parent(self) -> None:
        parent = SpanContext.generate()
        child = SpanContext.from_parent(parent)

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.trace_flags == parent.trace_flags

    def test_to_traceparent(self) -> None:
        ctx = SpanContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )
        traceparent = ctx.to_traceparent()
        assert traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_from_traceparent(self) -> None:
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = SpanContext.from_traceparent(traceparent)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.trace_flags == 1
        assert ctx.is_remote is True

    def test_from_traceparent_invalid(self) -> None:
        assert SpanContext.from_traceparent("invalid") is None
        assert SpanContext.from_traceparent("01-abc-def-01") is None


class TestSpan:
    """Tests for Span."""

    def test_create_span(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test-span", ctx)

        assert span.name == "test-span"
        assert span.trace_id == ctx.trace_id
        assert span.span_id == ctx.span_id
        assert span.status == SpanStatus.UNSET

    def test_set_attribute(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test-span", ctx)

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 42

    def test_set_attributes(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test-span", ctx)

        span.set_attributes({"a": 1, "b": 2})

        assert span.attributes["a"] == 1
        assert span.attributes["b"] == 2

    def test_add_event(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test-span", ctx)

        span.add_event("processing_started", {"item_count": 100})

        assert len(span.events) == 1
        assert span.events[0].name == "processing_started"
        assert span.events[0].attributes["item_count"] == 100

    def test_set_status(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test-span", ctx)

        span.set_status(SpanStatus.OK)
        assert span.status == SpanStatus.OK

        span.set_status(SpanStatus.ERROR, "Something failed")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Something failed"

    def test_record_exception(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test-span", ctx)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.record_exception(e)

        assert span.status == SpanStatus.ERROR
        assert len(span.events) == 1
        assert span.events[0].name == "exception"
        assert span.events[0].attributes["exception.type"] == "ValueError"
        assert "Test error" in span.events[0].attributes["exception.message"]

    def test_end_span(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test-span", ctx)

        assert span.end_time is None
        span.end()

        assert span.end_time is not None
        assert span.duration_seconds is not None
        assert span.duration_seconds >= 0

    def test_context_manager_success(self) -> None:
        ctx = SpanContext.generate()
        with Span("test-span", ctx) as span:
            span.set_attribute("test", True)

        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_context_manager_error(self) -> None:
        ctx = SpanContext.generate()

        with pytest.raises(ValueError):
            with Span("test-span", ctx) as span:
                raise ValueError("Test error")

        assert span.status == SpanStatus.ERROR
        assert len(span.events) == 1

    def test_to_dict(self) -> None:
        ctx = SpanContext.generate()
        span = Span(
            "test-span",
            ctx,
            kind=SpanKind.CLIENT,
            attributes={"key": "value"},
        )
        span.add_event("test-event")
        span.set_status(SpanStatus.OK)
        span.end()

        data = span.to_dict()

        assert data["name"] == "test-span"
        assert data["trace_id"] == ctx.trace_id
        assert data["kind"] == "client"
        assert data["status"] == "ok"
        assert data["attributes"]["key"] == "value"
        assert len(data["events"]) == 1


class TestNoopTracer:
    """Tests for NoopTracer."""

    def test_start_span(self) -> None:
        tracer = NoopTracer()
        span = tracer.start_span("test")

        assert span is not None
        assert span.name == "test"

    def test_trace_context_manager(self) -> None:
        tracer = NoopTracer()

        with tracer.trace("test-operation") as span:
            span.set_attribute("test", True)

        assert span.end_time is not None

    def test_inject_does_nothing(self) -> None:
        tracer = NoopTracer()
        carrier: dict[str, str] = {}

        tracer.inject_context(carrier)

        assert len(carrier) == 0

    def test_extract_returns_none(self) -> None:
        tracer = NoopTracer()
        carrier = {"traceparent": "00-abc-def-01"}

        result = tracer.extract_context(carrier)

        assert result is None


class TestInMemoryTracer:
    """Tests for InMemoryTracer."""

    def test_start_span(self) -> None:
        tracer = InMemoryTracer()
        span = tracer.start_span("test-span")

        assert span is not None
        assert span.name == "test-span"
        assert span.trace_id is not None

    def test_trace_records_span(self) -> None:
        tracer = InMemoryTracer()

        with tracer.trace("test-operation") as span:
            span.set_attribute("key", "value")

        assert len(tracer.spans) == 1
        assert tracer.spans[0].name == "test-operation"
        assert tracer.spans[0].attributes["key"] == "value"

    def test_nested_spans(self) -> None:
        tracer = InMemoryTracer()

        with tracer.trace("parent") as parent:
            with tracer.trace("child") as child:
                pass

        assert len(tracer.spans) == 2
        # Both should share the same trace ID
        assert tracer.spans[0].trace_id == tracer.spans[1].trace_id

    def test_context_propagation(self) -> None:
        tracer = InMemoryTracer()

        with tracer.trace("operation") as span:
            carrier: dict[str, str] = {}
            tracer.inject_context(carrier)

        assert "traceparent" in carrier
        assert span.trace_id in carrier["traceparent"]

    def test_context_extraction(self) -> None:
        tracer = InMemoryTracer()
        carrier = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }

        ctx = tracer.extract_context(carrier)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"

    def test_clear(self) -> None:
        tracer = InMemoryTracer()

        with tracer.trace("test"):
            pass

        assert len(tracer.spans) == 1
        tracer.clear()
        assert len(tracer.spans) == 0

    def test_service_name_attribute(self) -> None:
        config = TracingConfig(service_name="my-service")
        tracer = InMemoryTracer(config)

        span = tracer.start_span("test")

        assert span.attributes.get("service.name") == "my-service"

    def test_span_with_context(self) -> None:
        tracer = InMemoryTracer()
        context = ObservabilityContext(
            correlation_id="corr-123",
            trace_id="existing-trace-id",
        )

        span = tracer.start_span("test", context=context)

        assert span.trace_id == "existing-trace-id"

    def test_sampling_always_off(self) -> None:
        config = TracingConfig(sampler=TracingSampler.ALWAYS_OFF)
        tracer = InMemoryTracer(config)

        span = tracer.start_span("test")

        # Span should be created but not sampled
        assert span.context.trace_flags == 0

    def test_sampling_always_on(self) -> None:
        config = TracingConfig(sampler=TracingSampler.ALWAYS_ON)
        tracer = InMemoryTracer(config)

        span = tracer.start_span("test")

        assert span.context.trace_flags == 1

    def test_error_recording_in_trace(self) -> None:
        tracer = InMemoryTracer()

        with pytest.raises(RuntimeError):
            with tracer.trace("failing-operation") as span:
                raise RuntimeError("Operation failed")

        assert len(tracer.spans) == 1
        assert tracer.spans[0].status == SpanStatus.ERROR


class TestOpenTelemetryTracer:
    """Tests for OpenTelemetryTracer."""

    def test_fallback_to_inmemory_when_otel_not_installed(self) -> None:
        # This test checks that the tracer falls back gracefully
        config = TracingConfig(enabled=True)
        tracer = OpenTelemetryTracer(config)

        # Should still work even if OpenTelemetry is not installed
        with tracer.trace("test") as span:
            span.set_attribute("test", True)

        assert span is not None

    def test_disabled_config(self) -> None:
        config = TracingConfig(enabled=False)
        tracer = OpenTelemetryTracer(config)

        assert not tracer._initialized


class TestTracerFactory:
    """Tests for Tracer factory."""

    def teardown_method(self) -> None:
        Tracer.reset()

    def test_get_tracer_returns_same_instance(self) -> None:
        tracer1 = Tracer.get_tracer()
        tracer2 = Tracer.get_tracer()

        assert tracer1 is tracer2

    def test_get_tracer_disabled(self) -> None:
        config = TracingConfig(enabled=False)
        tracer = Tracer.get_tracer(config)

        assert isinstance(tracer, NoopTracer)

    def test_reset(self) -> None:
        tracer1 = Tracer.get_tracer()
        Tracer.reset()
        tracer2 = Tracer.get_tracer()

        assert tracer1 is not tracer2


class TestSpanKind:
    """Tests for SpanKind enum."""

    def test_span_kinds(self) -> None:
        assert SpanKind.INTERNAL.value == "internal"
        assert SpanKind.SERVER.value == "server"
        assert SpanKind.CLIENT.value == "client"
        assert SpanKind.PRODUCER.value == "producer"
        assert SpanKind.CONSUMER.value == "consumer"

    def test_span_with_kind(self) -> None:
        ctx = SpanContext.generate()
        span = Span("test", ctx, kind=SpanKind.CLIENT)

        assert span.kind == SpanKind.CLIENT
        assert span.to_dict()["kind"] == "client"
