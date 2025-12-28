"""Tests for enterprise logging system."""

import io
import json
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from truthound.infrastructure.logging import (
    # Core
    LogLevel,
    LogRecord,
    EnterpriseLogger,
    LogConfig,
    # Correlation
    CorrelationContext,
    correlation_context,
    get_correlation_id,
    set_correlation_id,
    generate_correlation_id,
    # Sinks
    LogSink,
    ConsoleSink,
    FileSink,
    JsonFileSink,
    ElasticsearchSink,
    LokiSink,
    FluentdSink,
    # Factory
    get_logger,
    configure_logging,
    reset_logging,
)


class TestLogLevel:
    """Tests for LogLevel."""

    def test_log_level_values(self):
        """Test log level numeric values."""
        assert LogLevel.TRACE < LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.INFO < LogLevel.WARNING < LogLevel.ERROR
        assert LogLevel.ERROR < LogLevel.CRITICAL < LogLevel.AUDIT

    def test_from_string(self):
        """Test log level from string."""
        assert LogLevel.from_string("debug") == LogLevel.DEBUG
        assert LogLevel.from_string("DEBUG") == LogLevel.DEBUG
        assert LogLevel.from_string("info") == LogLevel.INFO
        assert LogLevel.from_string("warning") == LogLevel.WARNING
        assert LogLevel.from_string("warn") == LogLevel.WARNING
        assert LogLevel.from_string("error") == LogLevel.ERROR
        assert LogLevel.from_string("critical") == LogLevel.CRITICAL
        assert LogLevel.from_string("fatal") == LogLevel.CRITICAL
        assert LogLevel.from_string("audit") == LogLevel.AUDIT
        assert LogLevel.from_string("unknown") == LogLevel.INFO  # default


class TestCorrelationContext:
    """Tests for CorrelationContext."""

    def setup_method(self):
        """Clear context before each test."""
        CorrelationContext.clear()

    def teardown_method(self):
        """Clear context after each test."""
        CorrelationContext.clear()

    def test_empty_context(self):
        """Test empty correlation context."""
        ctx = CorrelationContext.get_current()
        assert ctx == {}

    def test_push_pop(self):
        """Test push and pop context."""
        CorrelationContext.push(request_id="req-123")
        ctx = CorrelationContext.get_current()
        assert ctx["request_id"] == "req-123"

        CorrelationContext.pop()
        ctx = CorrelationContext.get_current()
        assert "request_id" not in ctx

    def test_nested_context(self):
        """Test nested context merging."""
        CorrelationContext.push(request_id="req-123")
        CorrelationContext.push(user_id="user-456")

        ctx = CorrelationContext.get_current()
        assert ctx["request_id"] == "req-123"
        assert ctx["user_id"] == "user-456"

        CorrelationContext.pop()
        ctx = CorrelationContext.get_current()
        assert ctx["request_id"] == "req-123"
        assert "user_id" not in ctx

    def test_context_manager(self):
        """Test correlation_context context manager."""
        assert get_correlation_id() is None

        with correlation_context(request_id="req-123", user_id="user-456"):
            assert get_correlation_id() == "req-123"
            ctx = CorrelationContext.get_current()
            assert ctx["user_id"] == "user-456"

            with correlation_context(trace_id="trace-789"):
                assert CorrelationContext.get_trace_id() == "trace-789"
                assert get_correlation_id() == "req-123"

        assert get_correlation_id() is None

    def test_auto_generate_correlation_id(self):
        """Test auto-generation of correlation ID."""
        with correlation_context():
            cid = get_correlation_id()
            assert cid is not None
            assert len(cid) > 0

    def test_generate_correlation_id_format(self):
        """Test correlation ID format."""
        cid = generate_correlation_id()
        assert "-" in cid
        parts = cid.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 8  # hex timestamp
        assert len(parts[1]) == 12  # random part

    def test_to_headers(self):
        """Test context to HTTP headers."""
        CorrelationContext.push(request_id="req-123", trace_id="trace-456")
        headers = CorrelationContext.to_headers()

        assert "X-Correlation-Request-Id" in headers
        assert headers["X-Correlation-Request-Id"] == "req-123"
        assert "X-Correlation-Trace-Id" in headers

    def test_from_headers(self):
        """Test context from HTTP headers."""
        headers = {
            "X-Correlation-Request-Id": "req-123",
            "X-Correlation-Trace-Id": "trace-456",
            "Content-Type": "application/json",
        }
        ctx = CorrelationContext.from_headers(headers)

        assert ctx["request_id"] == "req-123"
        assert ctx["trace_id"] == "trace-456"
        assert "content_type" not in ctx

    def test_thread_isolation(self):
        """Test context is thread-local."""
        results = {}

        def thread_func(name, request_id):
            with correlation_context(request_id=request_id):
                time.sleep(0.01)  # Allow other thread to run
                results[name] = get_correlation_id()

        t1 = threading.Thread(target=thread_func, args=("t1", "req-1"))
        t2 = threading.Thread(target=thread_func, args=("t2", "req-2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == "req-1"
        assert results["t2"] == "req-2"


class TestLogRecord:
    """Tests for LogRecord."""

    def test_create_record(self):
        """Test creating a log record."""
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test.logger",
            fields={"key": "value"},
        )

        assert record.level == LogLevel.INFO
        assert record.message == "Test message"
        assert record.logger_name == "test.logger"
        assert record.fields["key"] == "value"

    def test_record_auto_metadata(self):
        """Test automatic metadata population."""
        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )

        assert record.thread_id > 0
        assert record.process_id > 0
        assert len(record.hostname) > 0

    def test_to_dict(self):
        """Test record to dictionary conversion."""
        record = LogRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
            fields={"key": "value"},
            correlation_id="corr-123",
            trace_id="trace-456",
        )

        data = record.to_dict()

        assert data["level"] == "info"
        assert data["message"] == "Test"
        assert data["logger"] == "test"
        assert data["key"] == "value"
        assert data["correlation_id"] == "corr-123"
        assert data["trace_id"] == "trace-456"
        assert "@timestamp" in data  # ELK compatibility

    def test_to_json(self):
        """Test record to JSON."""
        record = LogRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )

        json_str = record.to_json()
        data = json.loads(json_str)

        assert data["level"] == "info"
        assert data["message"] == "Test"

    def test_to_logfmt(self):
        """Test record to logfmt."""
        record = LogRecord(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
            fields={"count": 42},
            correlation_id="corr-123",
        )

        logfmt = record.to_logfmt()

        assert "level=info" in logfmt
        assert 'msg="Test message"' in logfmt
        assert "count=42" in logfmt
        assert "correlation_id=corr-123" in logfmt

    def test_record_with_exception(self):
        """Test record with exception."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            record = LogRecord(
                timestamp=datetime.now(timezone.utc),
                level=LogLevel.ERROR,
                message="Error occurred",
                logger_name="test",
                exception=e,
            )

        data = record.to_dict()
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error"
        assert len(data["exception"]["traceback"]) > 0


class TestConsoleSink:
    """Tests for ConsoleSink."""

    def test_emit_to_stream(self):
        """Test emitting to stream."""
        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, format="console", color=False)

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
        )

        sink.emit(record)
        output = stream.getvalue()

        assert "INFO" in output
        assert "Test message" in output
        assert "[test]" in output

    def test_json_format(self):
        """Test JSON format output."""
        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, format="json")

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )

        sink.emit(record)
        output = stream.getvalue().strip()
        data = json.loads(output)

        assert data["level"] == "info"
        assert data["message"] == "Test"

    def test_logfmt_format(self):
        """Test logfmt format output."""
        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, format="logfmt")

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )

        sink.emit(record)
        output = stream.getvalue()

        assert "level=info" in output
        assert 'msg="Test"' in output

    def test_level_filtering(self):
        """Test level filtering."""
        stream = io.StringIO()
        sink = ConsoleSink(stream=stream, level=LogLevel.WARNING)

        info_record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Info",
            logger_name="test",
        )
        warning_record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.WARNING,
            message="Warning",
            logger_name="test",
        )

        assert not sink.should_emit(info_record)
        assert sink.should_emit(warning_record)


class TestFileSink:
    """Tests for FileSink."""

    def test_write_to_file(self, tmp_path):
        """Test writing to file."""
        log_file = tmp_path / "test.log"
        sink = FileSink(log_file, format="json")

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=LogLevel.INFO,
            message="Test",
            logger_name="test",
        )

        sink.emit(record)
        sink.close()

        content = log_file.read_text()
        data = json.loads(content.strip())

        assert data["level"] == "info"
        assert data["message"] == "Test"

    def test_file_rotation(self, tmp_path):
        """Test file rotation."""
        log_file = tmp_path / "test.log"
        sink = FileSink(log_file, format="text", max_bytes=100, backup_count=2)

        # Write enough to trigger rotation
        for i in range(10):
            record = LogRecord(
                timestamp=datetime.now(timezone.utc),
                level=LogLevel.INFO,
                message=f"Message {i} " + "x" * 20,
                logger_name="test",
            )
            sink.emit(record)

        sink.close()

        # Check backup files exist
        assert log_file.exists()
        assert (tmp_path / "test.log.1").exists()


class TestEnterpriseLogger:
    """Tests for EnterpriseLogger."""

    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()
        CorrelationContext.clear()

    def teardown_method(self):
        """Reset after each test."""
        reset_logging()
        CorrelationContext.clear()

    def test_create_logger(self):
        """Test creating a logger."""
        logger = EnterpriseLogger("test.logger")

        assert logger.name == "test.logger"
        assert logger.level == LogLevel.INFO

    def test_log_levels(self):
        """Test different log levels."""
        stream = io.StringIO()
        config = LogConfig(level=LogLevel.DEBUG, async_logging=False)
        sink = ConsoleSink(stream=stream, format="json", level=LogLevel.DEBUG)

        logger = EnterpriseLogger("test", config=config, sinks=[sink])

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        output = stream.getvalue()
        lines = [json.loads(line) for line in output.strip().split("\n")]

        levels = [line["level"] for line in lines]
        assert "debug" in levels
        assert "info" in levels
        assert "warning" in levels
        assert "error" in levels

    def test_correlation_propagation(self):
        """Test correlation ID propagation."""
        stream = io.StringIO()
        config = LogConfig(async_logging=False)
        sink = ConsoleSink(stream=stream, format="json")
        logger = EnterpriseLogger("test", config=config, sinks=[sink])

        with correlation_context(request_id="req-123"):
            logger.info("Test message")

        output = stream.getvalue()
        data = json.loads(output.strip())

        # request_id is propagated as-is from context
        assert data["request_id"] == "req-123"

    def test_field_binding(self):
        """Test field binding."""
        stream = io.StringIO()
        config = LogConfig(async_logging=False)
        sink = ConsoleSink(stream=stream, format="json")
        logger = EnterpriseLogger("test", config=config, sinks=[sink])

        bound_logger = logger.bind(user_id="user-123", request_id="req-456")
        bound_logger.info("Test message", extra_field="value")

        output = stream.getvalue()
        data = json.loads(output.strip())

        assert data["user_id"] == "user-123"
        assert data["request_id"] == "req-456"
        assert data["extra_field"] == "value"

    def test_exception_logging(self):
        """Test exception logging."""
        stream = io.StringIO()
        config = LogConfig(async_logging=False)
        sink = ConsoleSink(stream=stream, format="json")
        logger = EnterpriseLogger("test", config=config, sinks=[sink])

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("Error occurred")

        output = stream.getvalue()
        data = json.loads(output.strip())

        assert data["level"] == "error"
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"

    def test_service_and_environment(self):
        """Test service and environment in logs."""
        stream = io.StringIO()
        config = LogConfig(
            service="my-service",
            environment="production",
            async_logging=False,
        )
        sink = ConsoleSink(stream=stream, format="json")
        logger = EnterpriseLogger("test", config=config, sinks=[sink])

        logger.info("Test message")

        output = stream.getvalue()
        data = json.loads(output.strip())

        assert data["service"] == "my-service"
        assert data["environment"] == "production"


class TestGlobalLogging:
    """Tests for global logging functions."""

    def setup_method(self):
        """Reset logging before each test."""
        reset_logging()

    def teardown_method(self):
        """Reset after each test."""
        reset_logging()

    def test_get_logger(self):
        """Test getting a logger."""
        logger1 = get_logger("test.module")
        logger2 = get_logger("test.module")

        assert logger1 is logger2  # Same instance
        assert logger1.name == "test.module"

    def test_configure_logging(self):
        """Test configuring global logging."""
        configure_logging(
            level=LogLevel.DEBUG,
            format="json",
            service="test-service",
        )

        logger = get_logger("test")
        assert logger._config.service == "test-service"
        assert logger._config.format == "json"


class TestLogConfig:
    """Tests for LogConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = LogConfig()

        assert config.level == LogLevel.INFO
        assert config.format == "console"

    def test_development_config(self):
        """Test development configuration."""
        config = LogConfig.development()

        assert config.level == LogLevel.DEBUG
        assert config.include_caller is True
        assert config.async_logging is False

    def test_production_config(self):
        """Test production configuration."""
        config = LogConfig.production("my-service")

        assert config.level == LogLevel.INFO
        assert config.format == "json"
        assert config.service == "my-service"
        assert config.async_logging is True

    def test_from_environment(self):
        """Test loading from environment."""
        with patch.dict("os.environ", {
            "LOG_LEVEL": "DEBUG",
            "LOG_FORMAT": "json",
            "SERVICE_NAME": "env-service",
            "ENVIRONMENT": "staging",
        }):
            config = LogConfig.from_environment()

        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.service == "env-service"
        assert config.environment == "staging"
