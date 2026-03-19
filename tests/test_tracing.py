from __future__ import annotations

import threading

import pytest

from truthound.observability.tracing.baggage import (
    baggage_context,
    clear_baggage,
    get_baggage_item,
)
from truthound.observability.tracing.processor import SimpleSpanProcessor
from truthound.observability.tracing.propagator import (
    W3CTraceContextPropagator,
    extract_context,
    inject_context,
)
from truthound.observability.tracing.span import Span, SpanContextData


pytestmark = pytest.mark.fault


def _context() -> SpanContextData:
    return SpanContextData(
        trace_id="0af7651916cd43dd8448eb211c80319c",
        span_id="b7ad6b7169203331",
        trace_flags=1,
    )


class _ExplodingExporter:
    def export(self, spans: list[Span]) -> None:
        raise RuntimeError("exporter offline")

    def shutdown(self) -> bool:
        return True


@pytest.mark.contract
def test_trace_context_round_trips_through_w3c_headers():
    propagator = W3CTraceContextPropagator()
    carrier: dict[str, str] = {}

    inject_context(_context(), carrier, propagator=propagator)
    restored = extract_context(carrier, propagator=propagator)

    assert restored is not None
    assert restored.trace_id == "0af7651916cd43dd8448eb211c80319c"
    assert restored.span_id == "b7ad6b7169203331"


def test_malformed_trace_headers_are_ignored_without_throwing():
    propagator = W3CTraceContextPropagator()

    assert propagator.extract({"traceparent": "00-short"}) is None
    assert propagator.extract({"traceparent": "ff-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}) is None
    assert SpanContextData.from_w3c_traceparent("not-a-valid-header") is None


def test_simple_span_processor_isolates_exporter_crashes():
    span = Span(name="validation.run", context=_context())
    span.end()
    processor = SimpleSpanProcessor(_ExplodingExporter())

    processor.on_end(span)

    assert processor.shutdown() is True


def test_baggage_context_is_thread_local_and_restored():
    clear_baggage()
    observed: dict[str, str | None] = {}

    with baggage_context(tenant="alpha", run_id="main"):
        def worker() -> None:
            observed["tenant"] = get_baggage_item("tenant")
            observed["run_id"] = get_baggage_item("run_id")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert get_baggage_item("tenant") == "alpha"
        assert get_baggage_item("run_id") == "main"

    assert observed == {"tenant": None, "run_id": None}
    assert get_baggage_item("tenant") is None
    assert get_baggage_item("run_id") is None
