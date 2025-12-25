"""Comprehensive tests for distributed tracing module.

Tests cover:
- Span creation and lifecycle
- TracerProvider and Tracer
- Span processors (Simple, Batch, Multi)
- Span exporters (Console, InMemory, OTLP, Jaeger, Zipkin)
- Samplers (AlwaysOn, AlwaysOff, TraceIdRatio, ParentBased)
- Propagators (W3C, B3, Jaeger)
- Resource detection
- Baggage management
- Configuration utilities
"""

from __future__ import annotations

import json
import os
import threading
import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from truthound.observability.tracing import (
    # Provider
    TracerProvider,
    Tracer,
    get_tracer_provider,
    set_tracer_provider,
    # Span
    Span,
    SpanKind,
    StatusCode,
    Link,
    Event,
    SpanLimits,
    # Processor
    SpanProcessor,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    MultiSpanProcessor,
    # Exporter
    SpanExporter,
    ExportResult,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
    JaegerExporter,
    ZipkinExporter,
    # Sampler
    Sampler,
    SamplingResult,
    SamplingDecision,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    # Propagator
    Propagator,
    CompositePropagator,
    W3CTraceContextPropagator,
    W3CBaggagePropagator,
    B3Propagator,
    JaegerPropagator,
    get_global_propagator,
    set_global_propagator,
    # Resource
    Resource,
    ResourceDetector,
    get_aggregated_resources,
    # Baggage
    Baggage,
    get_baggage,
    set_baggage,
    remove_baggage,
    clear_baggage,
    # Config
    TracingConfig,
    configure_tracing,
)

from truthound.observability.tracing.span import (
    SpanContextData,
    generate_trace_id,
    generate_span_id,
    NoOpSpan,
)

from truthound.observability.tracing.provider import (
    get_tracer,
    get_current_span,
)

from truthound.observability.tracing.resource import (
    ProcessResourceDetector,
    HostResourceDetector,
    ServiceResourceDetector,
    SDKResourceDetector,
)

from truthound.observability.tracing.baggage import baggage_context

from truthound.observability.tracing.propagator import inject_context, extract_context


# =============================================================================
# Span Tests
# =============================================================================


class TestSpanContextData:
    """Tests for SpanContextData."""

    def test_creation(self):
        """Test span context creation."""
        ctx = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
        )
        assert ctx.is_valid
        assert ctx.is_sampled  # Default trace_flags=1

    def test_not_sampled(self):
        """Test unsampled span context."""
        ctx = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            trace_flags=0,
        )
        assert ctx.is_valid
        assert not ctx.is_sampled

    def test_to_w3c_traceparent(self):
        """Test W3C traceparent generation."""
        ctx = SpanContextData(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )
        traceparent = ctx.to_w3c_traceparent()
        assert traceparent == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_from_w3c_traceparent(self):
        """Test W3C traceparent parsing."""
        header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = SpanContextData.from_w3c_traceparent(header)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.is_sampled
        assert ctx.is_remote


class TestSpan:
    """Tests for Span."""

    def test_creation(self):
        """Test span creation."""
        ctx = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
        )
        span = Span(name="test-operation", context=ctx)

        assert span.name == "test-operation"
        assert span.context == ctx
        assert span.is_recording()
        assert span.kind == SpanKind.INTERNAL

    def test_attributes(self):
        """Test setting span attributes."""
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)
        span.set_attributes({"key3": True, "key4": 3.14})

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 42
        assert span.attributes["key3"] is True
        assert span.attributes["key4"] == 3.14

    def test_events(self):
        """Test adding span events."""
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )

        span.add_event("event1", {"attr": "value"})
        span.add_event("event2")

        assert len(span.events) == 2
        assert span.events[0].name == "event1"
        assert span.events[0].attributes["attr"] == "value"
        assert span.events[1].name == "event2"

    def test_status(self):
        """Test setting span status."""
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )

        span.set_status(StatusCode.OK)
        assert span.status == (StatusCode.OK, "")

        # Once ERROR is set, it cannot be changed
        span.set_status(StatusCode.ERROR, "Something went wrong")
        assert span.status == (StatusCode.ERROR, "Something went wrong")

        span.set_status(StatusCode.OK)  # Should be ignored
        assert span.status[0] == StatusCode.ERROR

    def test_end(self):
        """Test ending a span."""
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )

        assert span.is_recording()
        assert span.end_time is None

        time.sleep(0.01)
        span.end()

        assert not span.is_recording()
        assert span.end_time is not None
        assert span.duration_ms >= 10

    def test_context_manager(self):
        """Test span as context manager."""
        ctx = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
        )
        span = Span(name="test", context=ctx)

        with span as s:
            s.set_attribute("in_context", True)

        assert not span.is_recording()
        assert span.status[0] == StatusCode.OK

    def test_context_manager_exception(self):
        """Test span context manager with exception."""
        ctx = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
        )
        span = Span(name="test", context=ctx)

        with pytest.raises(ValueError):
            with span:
                raise ValueError("Test error")

        assert not span.is_recording()
        assert span.status[0] == StatusCode.ERROR
        assert "Test error" in span.status[1]

    def test_record_exception(self):
        """Test recording exception."""
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )

        try:
            raise ValueError("Test exception")
        except ValueError as e:
            span.record_exception(e)

        assert len(span.events) == 1
        assert span.events[0].name == "exception"
        assert span.events[0].attributes["exception.type"] == "ValueError"

    def test_span_limits(self):
        """Test span limits."""
        limits = SpanLimits(max_attributes=2, max_events=2)
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
            limits=limits,
        )

        span.set_attribute("a", 1)
        span.set_attribute("b", 2)
        span.set_attribute("c", 3)  # Should be ignored

        assert len(span.attributes) == 2
        assert "c" not in span.attributes

    def test_to_dict(self):
        """Test span serialization."""
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id="abcd1234" * 4,
                span_id="efgh5678" * 2,
            ),
            kind=SpanKind.SERVER,
        )
        span.set_attribute("key", "value")
        span.add_event("event")
        span.end()

        data = span.to_dict()

        assert data["name"] == "test"
        assert data["kind"] == "SERVER"
        assert data["attributes"]["key"] == "value"
        assert len(data["events"]) == 1


class TestNoOpSpan:
    """Tests for NoOpSpan."""

    def test_is_no_op(self):
        """Test NoOpSpan doesn't record."""
        span = NoOpSpan()

        assert not span.is_recording()
        span.set_attribute("key", "value")  # Should be no-op
        span.add_event("event")  # Should be no-op

    def test_context_manager(self):
        """Test NoOpSpan as context manager."""
        span = NoOpSpan()

        with span as s:
            pass

        # Should not raise


# =============================================================================
# Provider Tests
# =============================================================================


class TestTracerProvider:
    """Tests for TracerProvider."""

    def test_get_tracer(self):
        """Test getting a tracer."""
        provider = TracerProvider()
        tracer = provider.get_tracer("my.component", "1.0.0")

        assert isinstance(tracer, Tracer)
        assert tracer.name == "my.component"
        assert tracer.version == "1.0.0"

    def test_tracer_caching(self):
        """Test tracer caching."""
        provider = TracerProvider()

        tracer1 = provider.get_tracer("component", "1.0")
        tracer2 = provider.get_tracer("component", "1.0")

        assert tracer1 is tracer2

    def test_start_span(self):
        """Test starting spans."""
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider = TracerProvider(processors=[processor])

        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("operation") as span:
            span.set_attribute("key", "value")

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "operation"

    def test_nested_spans(self):
        """Test nested spans."""
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider = TracerProvider(processors=[processor])

        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("parent") as parent:
            with tracer.start_as_current_span("child") as child:
                assert child.parent is not None
                assert child.parent.span_id == parent.context.span_id

        spans = exporter.get_finished_spans()
        assert len(spans) == 2


class TestGlobalProvider:
    """Tests for global provider functions."""

    def test_set_get_provider(self):
        """Test setting and getting global provider."""
        provider = TracerProvider()
        set_tracer_provider(provider)

        assert get_tracer_provider() is provider

    def test_get_tracer_convenience(self):
        """Test get_tracer convenience function."""
        provider = TracerProvider()
        set_tracer_provider(provider)

        tracer = get_tracer("test.component")
        assert isinstance(tracer, Tracer)


# =============================================================================
# Sampler Tests
# =============================================================================


class TestSamplers:
    """Tests for samplers."""

    def test_always_on(self):
        """Test AlwaysOnSampler."""
        sampler = AlwaysOnSampler()
        result = sampler.should_sample(
            parent_context=None,
            trace_id=generate_trace_id(),
            name="test",
            kind=SpanKind.INTERNAL,
        )

        assert result.is_sampled
        assert result.decision == SamplingDecision.RECORD_AND_SAMPLE

    def test_always_off(self):
        """Test AlwaysOffSampler."""
        sampler = AlwaysOffSampler()
        result = sampler.should_sample(
            parent_context=None,
            trace_id=generate_trace_id(),
            name="test",
            kind=SpanKind.INTERNAL,
        )

        assert not result.is_sampled
        assert result.decision == SamplingDecision.DROP

    def test_trace_id_ratio(self):
        """Test TraceIdRatioSampler."""
        sampler = TraceIdRatioSampler(0.5)

        sampled = 0
        total = 1000

        for _ in range(total):
            result = sampler.should_sample(
                parent_context=None,
                trace_id=generate_trace_id(),
                name="test",
                kind=SpanKind.INTERNAL,
            )
            if result.is_sampled:
                sampled += 1

        # Should be roughly 50%
        ratio = sampled / total
        assert 0.4 < ratio < 0.6

    def test_trace_id_ratio_deterministic(self):
        """Test TraceIdRatioSampler is deterministic."""
        sampler = TraceIdRatioSampler(0.5)
        trace_id = generate_trace_id()

        result1 = sampler.should_sample(
            None, trace_id, "test", SpanKind.INTERNAL
        )
        result2 = sampler.should_sample(
            None, trace_id, "test", SpanKind.INTERNAL
        )

        assert result1.is_sampled == result2.is_sampled

    def test_parent_based(self):
        """Test ParentBasedSampler."""
        sampler = ParentBasedSampler(root=TraceIdRatioSampler(1.0))

        # Root span
        result = sampler.should_sample(
            parent_context=None,
            trace_id=generate_trace_id(),
            name="test",
            kind=SpanKind.INTERNAL,
        )
        assert result.is_sampled

        # With sampled parent
        parent = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            trace_flags=1,
        )
        result = sampler.should_sample(
            parent_context=parent,
            trace_id=parent.trace_id,
            name="test",
            kind=SpanKind.INTERNAL,
        )
        assert result.is_sampled


# =============================================================================
# Processor Tests
# =============================================================================


class TestSimpleSpanProcessor:
    """Tests for SimpleSpanProcessor."""

    def test_export_on_end(self):
        """Test spans are exported on end."""
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)

        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )

        processor.on_start(span)
        span.end()
        processor.on_end(span)

        assert len(exporter.get_finished_spans()) == 1

    def test_shutdown(self):
        """Test processor shutdown."""
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)

        assert processor.shutdown()


class TestBatchSpanProcessor:
    """Tests for BatchSpanProcessor."""

    def test_batch_export(self):
        """Test batch export."""
        from truthound.observability.tracing.processor import BatchConfig

        exporter = InMemorySpanExporter()
        processor = BatchSpanProcessor(
            exporter,
            config=BatchConfig(
                scheduled_delay_millis=100,
                max_export_batch_size=10,
            ),
        )

        # Add spans
        for i in range(5):
            span = Span(
                name=f"test-{i}",
                context=SpanContextData(
                    trace_id=generate_trace_id(),
                    span_id=generate_span_id(),
                ),
            )
            span.end()
            processor.on_end(span)

        # Force flush
        processor.force_flush()

        # Check exported
        assert len(exporter.get_finished_spans()) == 5

        processor.shutdown()

    def test_shutdown_exports_pending(self):
        """Test shutdown exports pending spans."""
        exporter = InMemorySpanExporter()
        processor = BatchSpanProcessor(exporter)

        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )
        span.end()
        processor.on_end(span)

        processor.shutdown()

        assert len(exporter.get_finished_spans()) >= 1


class TestMultiSpanProcessor:
    """Tests for MultiSpanProcessor."""

    def test_fan_out(self):
        """Test fan-out to multiple processors."""
        exporter1 = InMemorySpanExporter()
        exporter2 = InMemorySpanExporter()

        multi = MultiSpanProcessor([
            SimpleSpanProcessor(exporter1),
            SimpleSpanProcessor(exporter2),
        ])

        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )
        span.end()

        multi.on_end(span)

        assert len(exporter1.get_finished_spans()) == 1
        assert len(exporter2.get_finished_spans()) == 1


# =============================================================================
# Exporter Tests
# =============================================================================


class TestConsoleSpanExporter:
    """Tests for ConsoleSpanExporter."""

    def test_export_pretty(self):
        """Test pretty print export."""
        output = StringIO()
        exporter = ConsoleSpanExporter(output=output, pretty=True)

        span = Span(
            name="test-operation",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )
        span.set_attribute("key", "value")
        span.end()

        result = exporter.export([span])

        assert result == ExportResult.SUCCESS
        output_str = output.getvalue()
        assert "test-operation" in output_str
        assert "key" in output_str

    def test_export_json(self):
        """Test JSON export."""
        output = StringIO()
        exporter = ConsoleSpanExporter(output=output, json_output=True)

        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )
        span.end()

        exporter.export([span])

        output_str = output.getvalue()
        data = json.loads(output_str.strip())
        assert data["name"] == "test"


class TestInMemorySpanExporter:
    """Tests for InMemorySpanExporter."""

    def test_export_and_get(self):
        """Test export and retrieval."""
        exporter = InMemorySpanExporter()

        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )
        span.end()

        exporter.export([span])

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test"

    def test_clear(self):
        """Test clearing spans."""
        exporter = InMemorySpanExporter()

        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )
        span.end()

        exporter.export([span])
        exporter.clear()

        assert len(exporter.get_finished_spans()) == 0


# =============================================================================
# Propagator Tests
# =============================================================================


class TestW3CTraceContextPropagator:
    """Tests for W3C Trace Context propagator."""

    def test_inject(self):
        """Test context injection."""
        propagator = W3CTraceContextPropagator()
        ctx = SpanContextData(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )

        headers = {}
        propagator.inject(ctx, headers)

        assert "traceparent" in headers
        assert headers["traceparent"] == "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    def test_extract(self):
        """Test context extraction."""
        propagator = W3CTraceContextPropagator()
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        }

        ctx = propagator.extract(headers)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.span_id == "b7ad6b7169203331"
        assert ctx.is_sampled

    def test_roundtrip(self):
        """Test inject/extract roundtrip."""
        propagator = W3CTraceContextPropagator()
        original = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            trace_flags=1,
        )

        headers = {}
        propagator.inject(original, headers)
        extracted = propagator.extract(headers)

        assert extracted is not None
        assert extracted.trace_id == original.trace_id
        assert extracted.span_id == original.span_id


class TestB3Propagator:
    """Tests for B3 propagator."""

    def test_inject_multi_header(self):
        """Test multi-header injection."""
        propagator = B3Propagator(single_header=False)
        ctx = SpanContextData(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )

        headers = {}
        propagator.inject(ctx, headers)

        assert headers["X-B3-TraceId"] == "0af7651916cd43dd8448eb211c80319c"
        assert headers["X-B3-SpanId"] == "b7ad6b7169203331"
        assert headers["X-B3-Sampled"] == "1"

    def test_inject_single_header(self):
        """Test single-header injection."""
        propagator = B3Propagator(single_header=True)
        ctx = SpanContextData(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )

        headers = {}
        propagator.inject(ctx, headers)

        assert "b3" in headers
        assert headers["b3"] == "0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-1"

    def test_extract_multi_header(self):
        """Test multi-header extraction."""
        propagator = B3Propagator()
        headers = {
            "X-B3-TraceId": "0af7651916cd43dd8448eb211c80319c",
            "X-B3-SpanId": "b7ad6b7169203331",
            "X-B3-Sampled": "1",
        }

        ctx = propagator.extract(headers)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.is_sampled


class TestJaegerPropagator:
    """Tests for Jaeger propagator."""

    def test_inject(self):
        """Test Jaeger header injection."""
        propagator = JaegerPropagator()
        ctx = SpanContextData(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )

        headers = {}
        propagator.inject(ctx, headers)

        assert "uber-trace-id" in headers
        assert "0af7651916cd43dd8448eb211c80319c" in headers["uber-trace-id"]

    def test_extract(self):
        """Test Jaeger header extraction."""
        propagator = JaegerPropagator()
        headers = {
            "uber-trace-id": "0af7651916cd43dd8448eb211c80319c:b7ad6b7169203331:0:1",
        }

        ctx = propagator.extract(headers)

        assert ctx is not None
        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"


class TestCompositePropagator:
    """Tests for CompositePropagator."""

    def test_inject_all(self):
        """Test injection to all propagators."""
        composite = CompositePropagator([
            W3CTraceContextPropagator(),
            B3Propagator(),
        ])

        ctx = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
        )

        headers = {}
        composite.inject(ctx, headers)

        assert "traceparent" in headers
        assert "X-B3-TraceId" in headers

    def test_extract_first_match(self):
        """Test extraction from first matching propagator."""
        composite = CompositePropagator([
            W3CTraceContextPropagator(),
            B3Propagator(),
        ])

        headers = {
            "traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01",
        }

        ctx = composite.extract(headers)
        assert ctx is not None


# =============================================================================
# Resource Tests
# =============================================================================


class TestResource:
    """Tests for Resource."""

    def test_creation(self):
        """Test resource creation."""
        resource = Resource(attributes={
            "service.name": "test-service",
            "service.version": "1.0.0",
        })

        assert resource.get("service.name") == "test-service"
        assert resource.get("service.version") == "1.0.0"

    def test_merge(self):
        """Test resource merging."""
        r1 = Resource(attributes={"a": 1, "b": 2})
        r2 = Resource(attributes={"b": 3, "c": 4})

        merged = r1.merge(r2)

        assert merged.get("a") == 1
        assert merged.get("b") == 3  # Overwritten
        assert merged.get("c") == 4


class TestResourceDetectors:
    """Tests for resource detectors."""

    def test_process_detector(self):
        """Test ProcessResourceDetector."""
        detector = ProcessResourceDetector()
        resource = detector.detect()

        assert resource.get("process.pid") is not None
        assert resource.get("process.runtime.name") is not None

    def test_host_detector(self):
        """Test HostResourceDetector."""
        detector = HostResourceDetector()
        resource = detector.detect()

        assert resource.get("host.name") is not None

    def test_service_detector(self):
        """Test ServiceResourceDetector."""
        detector = ServiceResourceDetector(service_name="test-service")
        resource = detector.detect()

        assert resource.get("service.name") == "test-service"

    def test_sdk_detector(self):
        """Test SDKResourceDetector."""
        detector = SDKResourceDetector()
        resource = detector.detect()

        assert resource.get("telemetry.sdk.name") == "truthound"

    def test_aggregated_resources(self):
        """Test get_aggregated_resources."""
        resource = get_aggregated_resources()

        # Should have attributes from multiple detectors
        assert resource.get("telemetry.sdk.name") is not None
        assert resource.get("process.pid") is not None


# =============================================================================
# Baggage Tests
# =============================================================================


class TestBaggage:
    """Tests for Baggage."""

    def test_set_get(self):
        """Test setting and getting baggage."""
        baggage = Baggage()
        baggage = baggage.set("key", "value")

        assert baggage.get("key") == "value"

    def test_remove(self):
        """Test removing baggage."""
        baggage = Baggage().set("key", "value")
        baggage = baggage.remove("key")

        assert baggage.get("key") is None

    def test_immutability(self):
        """Test baggage immutability."""
        b1 = Baggage()
        b2 = b1.set("key", "value")

        assert b1.get("key") is None
        assert b2.get("key") == "value"


class TestBaggageContext:
    """Tests for baggage context management."""

    def test_context_functions(self):
        """Test baggage context functions."""
        clear_baggage()

        set_baggage("user_id", "123")
        assert get_baggage().get("user_id") == "123"

        remove_baggage("user_id")
        assert get_baggage().get("user_id") is None

    def test_context_manager(self):
        """Test baggage_context context manager."""
        clear_baggage()

        with baggage_context(user_id="123", tenant="acme"):
            assert get_baggage().get("user_id") == "123"
            assert get_baggage().get("tenant") == "acme"

        # Should be restored
        assert get_baggage().get("user_id") is None


# =============================================================================
# Configuration Tests
# =============================================================================


class TestTracingConfig:
    """Tests for TracingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TracingConfig()

        assert config.service_name == "unknown_service"
        assert config.exporter == "console"
        assert config.sampler == "always_on"

    def test_development_config(self):
        """Test development configuration."""
        config = TracingConfig.development("dev-service")

        assert config.service_name == "dev-service"
        assert config.exporter == "console"
        assert config.batch_export is False

    def test_production_config(self):
        """Test production configuration."""
        config = TracingConfig.production(
            service_name="prod-service",
            endpoint="http://collector:4317",
            sampling_ratio=0.1,
        )

        assert config.service_name == "prod-service"
        assert config.exporter == "otlp"
        assert config.sampling_ratio == 0.1

    def test_from_env(self):
        """Test configuration from environment."""
        with patch.dict(os.environ, {
            "OTEL_SERVICE_NAME": "env-service",
            "OTEL_SERVICE_VERSION": "1.2.3",
        }):
            config = TracingConfig.from_env()

        assert config.service_name == "env-service"
        assert config.service_version == "1.2.3"


class TestConfigureTracing:
    """Tests for configure_tracing."""

    def test_basic_configuration(self):
        """Test basic tracing configuration."""
        provider = configure_tracing(
            service_name="test-service",
            exporter="console",
            set_global=False,
        )

        assert isinstance(provider, TracerProvider)
        assert provider.resource.get("service.name") == "test-service"

    def test_with_config_object(self):
        """Test configuration with TracingConfig."""
        config = TracingConfig(
            service_name="config-service",
            exporter="console",
            sampler="always_on",
        )

        provider = configure_tracing(config, set_global=False)

        assert provider.resource.get("service.name") == "config-service"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_span_thread_safety(self):
        """Test span operations are thread-safe."""
        span = Span(
            name="test",
            context=SpanContextData(
                trace_id=generate_trace_id(),
                span_id=generate_span_id(),
            ),
        )

        def add_attributes():
            for i in range(100):
                span.set_attribute(f"key_{threading.current_thread().name}_{i}", i)

        threads = [threading.Thread(target=add_attributes) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have attributes from all threads
        assert len(span.attributes) > 0

    def test_exporter_thread_safety(self):
        """Test exporter is thread-safe."""
        exporter = InMemorySpanExporter()

        def export_span():
            for _ in range(100):
                span = Span(
                    name="test",
                    context=SpanContextData(
                        trace_id=generate_trace_id(),
                        span_id=generate_span_id(),
                    ),
                )
                span.end()
                exporter.export([span])

        threads = [threading.Thread(target=export_span) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(exporter.get_finished_spans()) == 500


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete tracing workflows."""

    def test_end_to_end_tracing(self):
        """Test complete tracing workflow."""
        exporter = InMemorySpanExporter()
        processor = SimpleSpanProcessor(exporter)
        provider = TracerProvider(
            resource=Resource(attributes={"service.name": "integration-test"}),
            processors=[processor],
        )

        tracer = provider.get_tracer("test.integration", "1.0.0")

        with tracer.start_as_current_span("parent-operation") as parent:
            parent.set_attribute("parent.attr", "value")

            with tracer.start_as_current_span("child-operation") as child:
                child.add_event("processing", {"step": 1})

        spans = exporter.get_finished_spans()
        assert len(spans) == 2

        # Check parent-child relationship
        child_span = [s for s in spans if s.name == "child-operation"][0]
        parent_span = [s for s in spans if s.name == "parent-operation"][0]

        assert child_span.parent.span_id == parent_span.context.span_id

    def test_context_propagation(self):
        """Test context propagation across boundaries."""
        # Inject context
        ctx = SpanContextData(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
            trace_flags=1,
        )

        headers = {}
        inject_context(ctx, headers)

        # Extract and continue trace
        extracted = extract_context(headers)

        assert extracted is not None
        assert extracted.trace_id == ctx.trace_id
        assert extracted.is_remote


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
