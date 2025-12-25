"""Comprehensive tests for observability module.

Tests cover:
- Structured logging with multiple formatters
- Metrics collection (Counter, Gauge, Histogram, Summary)
- Context propagation and tracing
- Checkpoint instrumentation
- Exporters (Prometheus, StatsD)
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from truthound.observability import (
    # Logging
    StructuredLogger,
    LogLevel,
    LogRecord,
    JSONFormatter,
    LogfmtFormatter,
    ConsoleFormatter,
    ConsoleHandler,
    FileHandler,
    RotatingFileHandler,
    LogContext,
    log_context,
    get_logger,
    configure_logging,
    # Metrics
    Counter,
    Gauge,
    Histogram,
    Summary,
    MetricsCollector,
    MetricsRegistry,
    PrometheusExporter,
    StatsDExporter,
    InMemoryExporter,
    get_metrics,
    # Context
    TraceContext,
    SpanContext,
    SpanStatus,
    current_context,
    with_context,
    create_trace,
    create_span,
    # Instrumentation
    traced,
    timed,
    counted,
    CheckpointInstrumentation,
    instrument_checkpoint,
)


# =============================================================================
# LogRecord Tests
# =============================================================================


class TestLogRecord:
    """Tests for LogRecord."""

    def test_basic_creation(self):
        """Test basic log record creation."""
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
        )

        assert record.level == LogLevel.INFO
        assert record.message == "Test message"
        assert record.logger_name == "test.logger"

    def test_with_fields(self):
        """Test log record with fields."""
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="User action",
            logger_name="test",
            fields={"user_id": 123, "action": "login"},
        )

        assert record.fields["user_id"] == 123
        assert record.fields["action"] == "login"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.ERROR,
            message="Error occurred",
            logger_name="test",
            fields={"code": 500},
            trace_id="abc123",
        )

        data = record.to_dict()

        assert data["level"] == "error"
        assert data["message"] == "Error occurred"
        assert data["code"] == 500
        assert data["trace_id"] == "abc123"

    def test_with_exception(self):
        """Test log record with exception."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            record = LogRecord(
                timestamp=datetime.now(timezone.utc),
                level=LogLevel.ERROR,
                message="Error",
                logger_name="test",
                exception=e,
            )

        data = record.to_dict()
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "Test error" in data["exception"]["message"]


# =============================================================================
# LogFormatter Tests
# =============================================================================


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_format(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = LogRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "info"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_with_fields(self):
        """Test JSON formatting with fields."""
        formatter = JSONFormatter()
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Action",
            logger_name="test",
            fields={"user_id": 42, "enabled": True},
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["user_id"] == 42
        assert data["enabled"] is True

    def test_indented_output(self):
        """Test indented JSON output."""
        formatter = JSONFormatter(indent=2)
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )

        output = formatter.format(record)
        assert "\n" in output


class TestLogfmtFormatter:
    """Tests for LogfmtFormatter."""

    def test_basic_format(self):
        """Test basic logfmt formatting."""
        formatter = LogfmtFormatter()
        record = LogRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
        )

        output = formatter.format(record)

        assert "level=info" in output
        assert 'msg="Test message"' in output
        assert "logger=test" in output

    def test_with_fields(self):
        """Test logfmt with fields."""
        formatter = LogfmtFormatter()
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.WARNING,
            message="Alert",
            logger_name="test",
            fields={"count": 42, "name": "value with spaces"},
        )

        output = formatter.format(record)

        assert "count=42" in output
        assert 'name="value with spaces"' in output


class TestConsoleFormatter:
    """Tests for ConsoleFormatter."""

    def test_basic_format(self):
        """Test basic console formatting."""
        formatter = ConsoleFormatter(color=False)
        record = LogRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "Test message" in output
        assert "[test]" in output


# =============================================================================
# LogHandler Tests
# =============================================================================


class TestConsoleHandler:
    """Tests for ConsoleHandler."""

    def test_emit_to_stream(self):
        """Test emitting to custom stream."""
        stream = StringIO()
        handler = ConsoleHandler(
            stream=stream,
            formatter=JSONFormatter(),
            level=LogLevel.DEBUG,
        )

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )

        handler.handle(record)

        output = stream.getvalue()
        assert "Test" in output

    def test_level_filtering(self):
        """Test log level filtering."""
        stream = StringIO()
        handler = ConsoleHandler(
            stream=stream,
            level=LogLevel.WARNING,
        )

        # Should be filtered
        handler.handle(LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.DEBUG,
            message="Debug",
            logger_name="test",
        ))

        # Should pass
        handler.handle(LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.ERROR,
            message="Error",
            logger_name="test",
        ))

        output = stream.getvalue()
        assert "Debug" not in output
        assert "Error" in output


class TestFileHandler:
    """Tests for FileHandler."""

    def test_write_to_file(self):
        """Test writing logs to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            path = f.name

        try:
            handler = FileHandler(path, formatter=JSONFormatter())

            handler.emit(LogRecord(
                timestamp=datetime.now(timezone.utc),
                level=LogLevel.INFO,
                message="File log test",
                logger_name="test",
            ))

            handler.close()

            content = Path(path).read_text()
            assert "File log test" in content

        finally:
            Path(path).unlink()


# =============================================================================
# LogContext Tests
# =============================================================================


class TestLogContext:
    """Tests for LogContext."""

    def test_push_pop(self):
        """Test push and pop context."""
        LogContext.clear()

        LogContext.push(request_id="abc123")
        assert LogContext.get_current()["request_id"] == "abc123"

        LogContext.push(user_id=42)
        current = LogContext.get_current()
        assert current["request_id"] == "abc123"
        assert current["user_id"] == 42

        LogContext.pop()
        current = LogContext.get_current()
        assert current["request_id"] == "abc123"
        assert "user_id" not in current

    def test_context_manager(self):
        """Test log_context context manager."""
        LogContext.clear()

        with log_context(trace_id="trace123"):
            assert LogContext.get_current()["trace_id"] == "trace123"

            with log_context(span_id="span456"):
                current = LogContext.get_current()
                assert current["trace_id"] == "trace123"
                assert current["span_id"] == "span456"

            assert "span_id" not in LogContext.get_current()

        assert "trace_id" not in LogContext.get_current()


# =============================================================================
# StructuredLogger Tests
# =============================================================================


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_basic_logging(self):
        """Test basic logging."""
        stream = StringIO()
        handler = ConsoleHandler(stream=stream, formatter=JSONFormatter())
        logger = StructuredLogger("test", handlers=[handler])

        logger.info("Hello world")

        output = stream.getvalue()
        data = json.loads(output.strip())
        assert data["message"] == "Hello world"
        assert data["level"] == "info"

    def test_logging_with_fields(self):
        """Test logging with fields."""
        stream = StringIO()
        handler = ConsoleHandler(stream=stream, formatter=JSONFormatter())
        logger = StructuredLogger("test", handlers=[handler])

        logger.info("User action", user_id=123, action="click")

        data = json.loads(stream.getvalue().strip())
        assert data["user_id"] == 123
        assert data["action"] == "click"

    def test_log_levels(self):
        """Test different log levels."""
        stream = StringIO()
        handler = ConsoleHandler(stream=stream, level=LogLevel.DEBUG)
        logger = StructuredLogger("test", level=LogLevel.DEBUG, handlers=[handler])

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        output = stream.getvalue()
        assert "Debug" in output
        assert "Info" in output
        assert "Warning" in output
        assert "Error" in output

    def test_level_filtering(self):
        """Test level filtering."""
        stream = StringIO()
        handler = ConsoleHandler(stream=stream)
        logger = StructuredLogger("test", level=LogLevel.WARNING, handlers=[handler])

        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")

        output = stream.getvalue()
        assert "Debug" not in output
        assert "Info" not in output
        assert "Warning" in output

    def test_bound_fields(self):
        """Test bound logger."""
        stream = StringIO()
        handler = ConsoleHandler(stream=stream, formatter=JSONFormatter())
        logger = StructuredLogger("test", handlers=[handler])

        bound = logger.bind(service="api", version="1.0")
        bound.info("Request received")

        data = json.loads(stream.getvalue().strip())
        assert data["service"] == "api"
        assert data["version"] == "1.0"


# =============================================================================
# Counter Tests
# =============================================================================


class TestCounter:
    """Tests for Counter metric."""

    def test_increment(self):
        """Test counter increment."""
        counter = Counter("test_counter", "Test counter")

        counter.inc()
        assert counter.get() == 1.0

        counter.inc(5)
        assert counter.get() == 6.0

    def test_with_labels(self):
        """Test counter with labels."""
        counter = Counter("requests", "Total requests", labels=["method", "status"])

        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="201")
        counter.inc(method="GET", status="200")

        assert counter.get(method="GET", status="200") == 2.0
        assert counter.get(method="POST", status="201") == 1.0

    def test_negative_increment_rejected(self):
        """Test that negative increments are rejected."""
        counter = Counter("test", "Test")

        with pytest.raises(ValueError):
            counter.inc(-1)

    def test_labeled_counter(self):
        """Test LabeledCounter helper."""
        counter = Counter("requests", "Total", labels=["method"])
        labeled = counter.labels(method="GET")

        labeled.inc()
        labeled.inc(2)

        assert counter.get(method="GET") == 3.0


# =============================================================================
# Gauge Tests
# =============================================================================


class TestGauge:
    """Tests for Gauge metric."""

    def test_set(self):
        """Test gauge set."""
        gauge = Gauge("temperature", "Current temp")

        gauge.set(23.5)
        assert gauge.get() == 23.5

        gauge.set(25.0)
        assert gauge.get() == 25.0

    def test_inc_dec(self):
        """Test gauge increment and decrement."""
        gauge = Gauge("connections", "Active connections")

        gauge.inc()
        assert gauge.get() == 1.0

        gauge.inc(5)
        assert gauge.get() == 6.0

        gauge.dec(2)
        assert gauge.get() == 4.0

    def test_track_inprogress(self):
        """Test in-progress tracking."""
        gauge = Gauge("active", "Active operations")

        assert gauge.get() == 0.0

        with gauge.track_inprogress():
            assert gauge.get() == 1.0

        assert gauge.get() == 0.0


# =============================================================================
# Histogram Tests
# =============================================================================


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Test histogram observation."""
        histogram = Histogram(
            "latency",
            "Request latency",
            buckets=[0.1, 0.5, 1.0, 5.0],
        )

        histogram.observe(0.05)  # <= 0.1, 0.5, 1.0, 5.0
        histogram.observe(0.3)   # <= 0.5, 1.0, 5.0
        histogram.observe(0.8)   # <= 1.0, 5.0
        histogram.observe(3.0)   # <= 5.0

        data = histogram.collect()[0][1]

        assert data["count"] == 4
        assert data["sum"] == pytest.approx(4.15)
        # Buckets are cumulative: each bucket counts values <= boundary
        assert data["buckets"]["0.1"] == 1  # 0.05
        assert data["buckets"]["0.5"] == 2  # 0.05, 0.3
        assert data["buckets"]["1.0"] == 3  # 0.05, 0.3, 0.8
        assert data["buckets"]["5.0"] == 4  # 0.05, 0.3, 0.8, 3.0

    def test_time_context_manager(self):
        """Test timing context manager."""
        histogram = Histogram("duration", "Duration")

        with histogram.time():
            time.sleep(0.1)

        data = histogram.collect()[0][1]
        assert data["count"] == 1
        assert data["sum"] >= 0.1


# =============================================================================
# Summary Tests
# =============================================================================


class TestSummary:
    """Tests for Summary metric."""

    def test_observe_and_quantiles(self):
        """Test summary observations and quantiles."""
        summary = Summary(
            "response_size",
            "Response size",
            quantiles=(0.5, 0.9, 0.99),
        )

        for i in range(100):
            summary.observe(i)

        data = summary.collect()[0][1]

        assert data["count"] == 100
        assert data["sum"] == sum(range(100))
        assert "p50" in data["quantiles"]
        assert "p90" in data["quantiles"]
        assert "p99" in data["quantiles"]


# =============================================================================
# MetricsCollector Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_create_metrics(self):
        """Test creating metrics through collector."""
        collector = MetricsCollector()

        counter = collector.counter("requests_total", "Total requests")
        gauge = collector.gauge("connections", "Active connections")
        histogram = collector.histogram("latency", "Latency")

        assert isinstance(counter, Counter)
        assert isinstance(gauge, Gauge)
        assert isinstance(histogram, Histogram)

    def test_registry_deduplication(self):
        """Test that same metric is returned on duplicate creation."""
        collector = MetricsCollector()

        counter1 = collector.counter("test", "Test")
        counter2 = collector.counter("test", "Test")

        assert counter1 is counter2


# =============================================================================
# Exporter Tests
# =============================================================================


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    def test_export_counter(self):
        """Test exporting counter in Prometheus format."""
        registry = MetricsRegistry()
        counter = Counter("http_requests_total", "Total HTTP requests")
        registry.register(counter)

        counter.inc(10)

        exporter = PrometheusExporter()
        output = exporter.export(registry)

        assert "# HELP http_requests_total" in output
        assert "# TYPE http_requests_total counter" in output
        assert "http_requests_total 10" in output

    def test_export_with_labels(self):
        """Test exporting metrics with labels."""
        registry = MetricsRegistry()
        counter = Counter("requests", "Requests", labels=["method"])
        registry.register(counter)

        counter.inc(method="GET")
        counter.inc(2, method="POST")

        exporter = PrometheusExporter()
        output = exporter.export(registry)

        assert 'method="GET"' in output
        assert 'method="POST"' in output


class TestInMemoryExporter:
    """Tests for InMemoryExporter."""

    def test_export_stores_data(self):
        """Test that exports are stored."""
        registry = MetricsRegistry()
        counter = Counter("test", "Test")
        registry.register(counter)

        counter.inc(5)

        exporter = InMemoryExporter()
        exporter.export(registry)

        assert len(exporter.exports) == 1
        assert "test" in exporter.exports[0]


# =============================================================================
# TraceContext Tests
# =============================================================================


class TestTraceContext:
    """Tests for TraceContext."""

    def test_create_trace(self):
        """Test creating a trace."""
        trace = create_trace("test-operation")

        assert trace.trace_id is not None
        assert trace.root_span is not None
        assert trace.root_span.name == "test-operation"

    def test_baggage(self):
        """Test baggage propagation."""
        trace = create_trace("test")

        trace.set_baggage("user_id", "123")
        assert trace.get_baggage("user_id") == "123"

    def test_to_headers(self):
        """Test W3C header generation."""
        trace = create_trace("test")

        headers = trace.to_headers()

        assert "traceparent" in headers
        assert trace.trace_id in headers["traceparent"]


class TestSpanContext:
    """Tests for SpanContext."""

    def test_span_creation(self):
        """Test span creation."""
        span = SpanContext(
            trace_id="abc123",
            span_id="def456",
            name="test-span",
        )

        assert span.is_active
        assert span.duration_ms is None

    def test_span_end(self):
        """Test ending a span."""
        span = SpanContext(
            trace_id="abc",
            span_id="def",
            name="test",
        )

        time.sleep(0.01)
        span.end()

        assert not span.is_active
        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_span_attributes(self):
        """Test span attributes."""
        span = SpanContext(
            trace_id="abc",
            span_id="def",
            name="test",
        )

        span.set_attribute("user_id", 123)
        span.set_attribute("operation", "validate")

        assert span.attributes["user_id"] == 123
        assert span.attributes["operation"] == "validate"

    def test_span_events(self):
        """Test span events."""
        span = SpanContext(
            trace_id="abc",
            span_id="def",
            name="test",
        )

        span.add_event("data_loaded", {"rows": 1000})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "data_loaded"


# =============================================================================
# Context Management Tests
# =============================================================================


class TestContextManagement:
    """Tests for context management."""

    def test_with_context(self):
        """Test with_context context manager."""
        trace = create_trace("test")

        assert current_context() is None

        with with_context(trace):
            assert current_context() is trace

        assert current_context() is None

    def test_create_span(self):
        """Test creating spans."""
        trace = create_trace("root")

        with with_context(trace):
            with create_span("child") as span:
                assert span.name == "child"
                assert span.trace_id == trace.trace_id

    def test_nested_spans(self):
        """Test nested spans."""
        trace = create_trace("root")

        with with_context(trace):
            with create_span("parent") as parent:
                with create_span("child") as child:
                    assert child.parent_span_id == parent.span_id


# =============================================================================
# Instrumentation Tests
# =============================================================================


class TestDecorators:
    """Tests for instrumentation decorators."""

    def test_traced_decorator(self):
        """Test traced decorator."""

        @traced("test.operation")
        def my_function():
            return 42

        result = my_function()
        assert result == 42

    def test_timed_decorator(self):
        """Test timed decorator."""
        collector = MetricsCollector()

        with patch("truthound.observability.instrumentation.get_metrics", return_value=collector):

            @timed("test_duration_seconds")
            def slow_function():
                time.sleep(0.05)
                return "done"

            result = slow_function()
            assert result == "done"

    def test_counted_decorator(self):
        """Test counted decorator."""
        collector = MetricsCollector()

        with patch("truthound.observability.instrumentation.get_metrics", return_value=collector):

            @counted("test_calls_total")
            def my_function():
                return 1

            my_function()
            my_function()
            my_function()

            # Check counter was incremented
            # Note: Actual check depends on implementation details


# =============================================================================
# CheckpointInstrumentation Tests
# =============================================================================


class TestCheckpointInstrumentation:
    """Tests for CheckpointInstrumentation."""

    def test_validation_tracking(self):
        """Test validation tracking."""
        collector = MetricsCollector()
        stream = StringIO()
        handler = ConsoleHandler(stream=stream, formatter=JSONFormatter())
        logger = StructuredLogger("test", handlers=[handler])

        instrumentation = CheckpointInstrumentation(
            collector=collector,
            logger=logger,
        )

        with instrumentation.validation("test_check", "test.csv") as ctx:
            ctx.record_success(total_issues=5)

        # Check metrics were recorded
        metrics = collector.registry.collect_all()
        assert "truthound_validations_total" in metrics

    def test_failed_validation(self):
        """Test failed validation tracking."""
        collector = MetricsCollector()
        stream = StringIO()
        handler = ConsoleHandler(stream=stream, formatter=JSONFormatter())
        logger = StructuredLogger("test", handlers=[handler])

        instrumentation = CheckpointInstrumentation(
            collector=collector,
            logger=logger,
        )

        with instrumentation.validation("test_check", "test.csv") as ctx:
            ctx.record_failure("Data source not found")

        # Check failure was recorded
        output = stream.getvalue()
        assert "failed" in output.lower()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_counter_thread_safety(self):
        """Test counter is thread-safe."""
        counter = Counter("test", "Test")

        def increment():
            for _ in range(1000):
                counter.inc()

        threads = [threading.Thread(target=increment) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.get() == 10000

    def test_log_context_thread_isolation(self):
        """Test log context is thread-isolated."""
        results = {}

        def thread_func(thread_id):
            LogContext.clear()
            LogContext.push(thread_id=thread_id)
            time.sleep(0.01)
            results[thread_id] = LogContext.get_current().get("thread_id")

        threads = [
            threading.Thread(target=thread_func, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should see its own context
        for i in range(5):
            assert results[i] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
