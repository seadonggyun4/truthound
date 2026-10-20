"""Real SDK span lifetime/error evidence without exporter network or global state."""

import pytest

from truthound.stores.observability.config import TracingConfig
from truthound.stores.observability.tracing import OpenTelemetryTracer


@pytest.mark.integration
def test_real_otel_context_closes_spans_and_preserves_exceptions():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OpenTelemetryTracer(TracingConfig(enabled=False))
    tracer._otel_tracer = provider.get_tracer("synthetic-test")
    tracer._initialized = True
    try:
        with tracer.trace("success") as span:
            span.set_attribute("synthetic", True)
        assert span.end_time is not None
        with (
            pytest.raises(RuntimeError, match="synthetic failure"),
            tracer.trace("failure") as failed,
        ):
            raise RuntimeError("synthetic failure")
        assert failed.end_time is not None and failed.status.value == "error"
        spans = exporter.get_finished_spans()
        assert [s.name for s in spans] == ["success", "failure"]
        assert all(s.end_time is not None for s in spans)
        assert spans[1].status.status_code == StatusCode.ERROR
        assert any(event.name == "exception" for event in spans[1].events)
    finally:
        provider.shutdown()
