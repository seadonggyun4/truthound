"""Enterprise structured logging system for Truthound.

This module extends the base observability logging with enterprise features:
- Correlation ID propagation across distributed systems
- Multiple log sinks (Elasticsearch, Loki, Fluentd)
- JSON structured logging for log aggregation
- Environment-aware configuration
- Async buffered logging for high throughput

Architecture:
    CorrelationContext (thread-local)
           |
           v
    EnterpriseLogger
           |
           +---> LogSink[] (parallel dispatch)
                   |
                   +---> ConsoleSink
                   +---> FileSink
                   +---> JsonFileSink
                   +---> ElasticsearchSink
                   +---> LokiSink
                   +---> FluentdSink

Usage:
    >>> from truthound.infrastructure.logging import (
    ...     get_logger, configure_logging,
    ...     correlation_context, get_correlation_id,
    ... )
    >>>
    >>> # Configure for production
    >>> configure_logging(
    ...     environment="production",
    ...     format="json",
    ...     sinks=[
    ...         {"type": "console"},
    ...         {"type": "elasticsearch", "url": "http://elk:9200"},
    ...     ],
    ... )
    >>>
    >>> # Use correlation context
    >>> with correlation_context(request_id="req-123", user_id="user-456"):
    ...     logger = get_logger(__name__)
    ...     logger.info("Processing request", action="validate")
    ...     # All logs include request_id and user_id automatically
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO, TypeVar

# Re-export base logging components for compatibility
from truthound.observability.logging import (
    LogLevel as BaseLogLevel,
    LogRecord as BaseLogRecord,
    LogContext as BaseLogContext,
)


# =============================================================================
# Log Levels (Extended)
# =============================================================================


class LogLevel(IntEnum):
    """Extended log severity levels."""

    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AUDIT = 60  # Special level for audit events

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Convert string to LogLevel."""
        mapping = {
            "trace": cls.TRACE,
            "debug": cls.DEBUG,
            "info": cls.INFO,
            "warning": cls.WARNING,
            "warn": cls.WARNING,
            "error": cls.ERROR,
            "critical": cls.CRITICAL,
            "fatal": cls.CRITICAL,
            "audit": cls.AUDIT,
        }
        return mapping.get(level.lower(), cls.INFO)


# =============================================================================
# Correlation Context
# =============================================================================


class CorrelationContext:
    """Thread-local correlation context for distributed tracing.

    Maintains correlation IDs and contextual fields that are automatically
    included in all log messages within the context.

    This enables tracking requests across service boundaries and correlating
    logs from different components of a distributed system.

    Example:
        >>> with correlation_context(
        ...     request_id="req-123",
        ...     user_id="user-456",
        ...     trace_id="trace-789",
        ... ):
        ...     logger.info("Processing")  # Includes all context fields
        ...     call_downstream_service()  # Context propagates
    """

    _local = threading.local()
    _CONTEXT_HEADER_PREFIX = "X-Correlation-"

    @classmethod
    def get_current(cls) -> dict[str, Any]:
        """Get current context fields (merged from all levels)."""
        if not hasattr(cls._local, "stack"):
            cls._local.stack = [{}]
        result: dict[str, Any] = {}
        for ctx in cls._local.stack:
            result.update(ctx)
        return result

    @classmethod
    def get_correlation_id(cls) -> str | None:
        """Get the current correlation/request ID."""
        ctx = cls.get_current()
        return ctx.get("correlation_id") or ctx.get("request_id")

    @classmethod
    def get_trace_id(cls) -> str | None:
        """Get the current trace ID."""
        return cls.get_current().get("trace_id")

    @classmethod
    def get_span_id(cls) -> str | None:
        """Get the current span ID."""
        return cls.get_current().get("span_id")

    @classmethod
    def push(cls, **fields: Any) -> None:
        """Push new context level."""
        if not hasattr(cls._local, "stack"):
            cls._local.stack = [{}]
        cls._local.stack.append(fields)

    @classmethod
    def pop(cls) -> dict[str, Any]:
        """Pop context level."""
        if hasattr(cls._local, "stack") and len(cls._local.stack) > 1:
            return cls._local.stack.pop()
        return {}

    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._local.stack = [{}]

    @classmethod
    def to_headers(cls) -> dict[str, str]:
        """Convert context to HTTP headers for propagation.

        Returns:
            Dictionary of header name to value.
        """
        ctx = cls.get_current()
        headers = {}
        for key, value in ctx.items():
            header_name = f"{cls._CONTEXT_HEADER_PREFIX}{key.replace('_', '-').title()}"
            headers[header_name] = str(value)
        return headers

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> dict[str, Any]:
        """Extract context from HTTP headers.

        Args:
            headers: HTTP headers mapping.

        Returns:
            Extracted context fields.
        """
        prefix = cls._CONTEXT_HEADER_PREFIX.lower()
        context = {}
        for key, value in headers.items():
            if key.lower().startswith(prefix):
                field_name = key[len(prefix) :].lower().replace("-", "_")
                context[field_name] = value
        return context


@contextmanager
def correlation_context(**fields: Any) -> Iterator[None]:
    """Context manager for adding correlation fields.

    Args:
        **fields: Key-value pairs to add to context.
                 Common fields: request_id, correlation_id, trace_id,
                 span_id, user_id, session_id, tenant_id.

    Example:
        >>> with correlation_context(request_id="abc", user_id="123"):
        ...     logger.info("User action")  # Includes request_id and user_id
    """
    # Auto-generate correlation_id if not provided and not in current context
    if "correlation_id" not in fields and "request_id" not in fields:
        # Check if correlation already exists in parent context
        existing_cid = CorrelationContext.get_correlation_id()
        if existing_cid is None:
            fields["correlation_id"] = generate_correlation_id()

    CorrelationContext.push(**fields)
    try:
        yield
    finally:
        CorrelationContext.pop()


def get_correlation_id() -> str | None:
    """Get the current correlation ID."""
    return CorrelationContext.get_correlation_id()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in current context.

    Note: This modifies the current context level. Use correlation_context()
    for proper scoping.
    """
    ctx = CorrelationContext.get_current()
    if hasattr(CorrelationContext._local, "stack") and CorrelationContext._local.stack:
        CorrelationContext._local.stack[-1]["correlation_id"] = correlation_id


def generate_correlation_id() -> str:
    """Generate a unique correlation ID.

    Format: {timestamp_hex}-{random_hex}
    Example: 65a1b2c3-4d5e6f7a8b9c
    """
    timestamp = int(time.time() * 1000) & 0xFFFFFFFF
    random_part = uuid.uuid4().hex[:12]
    return f"{timestamp:08x}-{random_part}"


# =============================================================================
# Log Record (Extended)
# =============================================================================


@dataclass
class LogRecord:
    """Extended log record with correlation and enterprise features.

    Attributes:
        timestamp: When the log was created (UTC).
        level: Log severity level.
        message: Human-readable log message.
        logger_name: Name of the logger.
        fields: Structured key-value data.
        exception: Exception info if present.
        correlation_id: Distributed correlation ID.
        trace_id: Distributed trace ID.
        span_id: Current span ID.
        caller: Source location (file:line:function).
        thread_id: Thread identifier.
        process_id: Process identifier.
        hostname: Machine hostname.
        service: Service name.
        environment: Environment name.
    """

    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    fields: dict[str, Any] = field(default_factory=dict)
    exception: BaseException | None = None
    correlation_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    caller: str | None = None
    thread_id: int = 0
    thread_name: str = ""
    process_id: int = 0
    hostname: str = ""
    service: str = ""
    environment: str = ""

    def __post_init__(self) -> None:
        """Set process/thread info."""
        if not self.thread_id:
            self.thread_id = threading.current_thread().ident or 0
        if not self.thread_name:
            self.thread_name = threading.current_thread().name
        if not self.process_id:
            self.process_id = os.getpid()
        if not self.hostname:
            self.hostname = socket.gethostname()

    def to_dict(self, include_meta: bool = True) -> dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_meta: Include metadata fields (thread, process, etc).

        Returns:
            Dictionary representation.
        """
        data = {
            "timestamp": self.timestamp.isoformat(),
            "@timestamp": self.timestamp.isoformat(),  # ELK compatibility
            "level": self.level.name.lower(),
            "message": self.message,
            "logger": self.logger_name,
            **self.fields,
        }

        # Correlation fields
        if self.correlation_id:
            data["correlation_id"] = self.correlation_id
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id

        # Location
        if self.caller:
            data["caller"] = self.caller

        # Service info
        if self.service:
            data["service"] = self.service
        if self.environment:
            data["environment"] = self.environment

        # Metadata
        if include_meta:
            data["thread_id"] = self.thread_id
            data["thread_name"] = self.thread_name
            data["process_id"] = self.process_id
            data["hostname"] = self.hostname

        # Exception
        if self.exception:
            data["exception"] = {
                "type": type(self.exception).__name__,
                "message": str(self.exception),
                "traceback": traceback.format_exception(
                    type(self.exception),
                    self.exception,
                    self.exception.__traceback__,
                ),
            }

        return data

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=indent)

    def to_logfmt(self) -> str:
        """Convert to logfmt format."""
        parts = [
            f'ts={self.timestamp.isoformat()}',
            f'level={self.level.name.lower()}',
            f'msg="{self._escape(self.message)}"',
            f'logger={self.logger_name}',
        ]

        if self.correlation_id:
            parts.append(f"correlation_id={self.correlation_id}")
        if self.trace_id:
            parts.append(f"trace_id={self.trace_id}")

        for key, value in self.fields.items():
            parts.append(f"{key}={self._format_value(value)}")

        return " ".join(parts)

    def _escape(self, value: str) -> str:
        """Escape special characters."""
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    def _format_value(self, value: Any) -> str:
        """Format a value for logfmt."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            if " " in value or '"' in value or "=" in value:
                return f'"{self._escape(value)}"'
            return value
        else:
            return f'"{self._escape(str(value))}"'


# =============================================================================
# Log Sinks
# =============================================================================


class LogSink(ABC):
    """Abstract base class for log output sinks.

    Sinks are responsible for delivering log records to their destinations.
    Multiple sinks can be configured to send logs to different systems.
    """

    def __init__(
        self,
        level: LogLevel = LogLevel.DEBUG,
        filters: list[Callable[[LogRecord], bool]] | None = None,
    ) -> None:
        """Initialize sink.

        Args:
            level: Minimum log level to accept.
            filters: Optional filter functions.
        """
        self._level = level
        self._filters = filters or []
        self._lock = threading.Lock()

    @property
    def level(self) -> LogLevel:
        """Get minimum log level."""
        return self._level

    def should_emit(self, record: LogRecord) -> bool:
        """Check if record should be emitted.

        Args:
            record: Log record to check.

        Returns:
            True if should emit.
        """
        if record.level < self._level:
            return False
        for f in self._filters:
            if not f(record):
                return False
        return True

    @abstractmethod
    def emit(self, record: LogRecord) -> None:
        """Emit a log record.

        Args:
            record: Record to emit.
        """
        pass

    def emit_batch(self, records: list[LogRecord]) -> None:
        """Emit multiple records (default: emit one by one).

        Args:
            records: Records to emit.
        """
        for record in records:
            if self.should_emit(record):
                self.emit(record)

    def close(self) -> None:
        """Clean up sink resources."""
        pass

    def flush(self) -> None:
        """Flush any buffered records."""
        pass


class ConsoleSink(LogSink):
    """Console output sink with optional coloring.

    Outputs logs to stdout/stderr with human-readable formatting.
    """

    COLORS = {
        LogLevel.TRACE: "\033[90m",     # Gray
        LogLevel.DEBUG: "\033[36m",     # Cyan
        LogLevel.INFO: "\033[32m",      # Green
        LogLevel.WARNING: "\033[33m",   # Yellow
        LogLevel.ERROR: "\033[31m",     # Red
        LogLevel.CRITICAL: "\033[35m",  # Magenta
        LogLevel.AUDIT: "\033[34m",     # Blue
    }
    RESET = "\033[0m"

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        color: bool = True,
        format: str = "console",  # console, json, logfmt
        split_stderr: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
        **kwargs: Any,
    ) -> None:
        """Initialize console sink.

        Args:
            stream: Output stream (None for auto).
            color: Enable ANSI colors.
            format: Output format (console, json, logfmt).
            split_stderr: Send warnings+ to stderr.
            timestamp_format: strftime format.
            **kwargs: Arguments for LogSink.
        """
        super().__init__(**kwargs)
        self._stream = stream
        self._color = color and (stream is None or stream.isatty())
        self._format = format
        self._split_stderr = split_stderr
        self._timestamp_format = timestamp_format

    def emit(self, record: LogRecord) -> None:
        """Write log to console."""
        if self._format == "json":
            message = record.to_json()
        elif self._format == "logfmt":
            message = record.to_logfmt()
        else:
            message = self._format_console(record)

        stream = self._get_stream(record)
        with self._lock:
            try:
                stream.write(message + "\n")
                stream.flush()
            except Exception:
                pass

    def _get_stream(self, record: LogRecord) -> TextIO:
        """Get appropriate output stream."""
        if self._stream:
            return self._stream
        if self._split_stderr and record.level >= LogLevel.WARNING:
            return sys.stderr
        return sys.stdout

    def _format_console(self, record: LogRecord) -> str:
        """Format record for console output."""
        parts = []

        # Timestamp
        ts = record.timestamp.strftime(self._timestamp_format)
        parts.append(ts)

        # Level
        level = record.level.name.ljust(8)
        if self._color:
            color = self.COLORS.get(record.level, "")
            level = f"{color}{level}{self.RESET}"
        parts.append(level)

        # Correlation ID (short form)
        if record.correlation_id:
            cid = record.correlation_id[:8]
            parts.append(f"[{cid}]")

        # Logger name
        parts.append(f"[{record.logger_name}]")

        # Message
        parts.append(record.message)

        # Fields
        if record.fields:
            field_strs = [f"{k}={v}" for k, v in record.fields.items()]
            parts.append(" ".join(field_strs))

        result = " ".join(parts)

        # Exception
        if record.exception:
            tb = "".join(
                traceback.format_exception(
                    type(record.exception),
                    record.exception,
                    record.exception.__traceback__,
                )
            )
            result = f"{result}\n{tb}"

        return result


class FileSink(LogSink):
    """File output sink with rotation support."""

    def __init__(
        self,
        path: str | Path,
        *,
        format: str = "json",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        """Initialize file sink.

        Args:
            path: Path to log file.
            format: Output format (json, logfmt, text).
            max_bytes: Max file size before rotation.
            backup_count: Number of backup files.
            encoding: File encoding.
            **kwargs: Arguments for LogSink.
        """
        super().__init__(**kwargs)
        self._path = Path(path)
        self._format = format
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._encoding = encoding
        self._file: TextIO | None = None

    def emit(self, record: LogRecord) -> None:
        """Write log to file."""
        if self._format == "json":
            message = record.to_json()
        elif self._format == "logfmt":
            message = record.to_logfmt()
        else:
            message = f"{record.timestamp.isoformat()} {record.level.name} [{record.logger_name}] {record.message}"

        with self._lock:
            try:
                if self._should_rotate():
                    self._rotate()
                f = self._ensure_file()
                f.write(message + "\n")
                f.flush()
            except Exception:
                pass

    def _ensure_file(self) -> TextIO:
        """Ensure file is open."""
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a", encoding=self._encoding)
        return self._file

    def _should_rotate(self) -> bool:
        """Check if rotation is needed."""
        try:
            return self._path.exists() and self._path.stat().st_size >= self._max_bytes
        except Exception:
            return False

    def _rotate(self) -> None:
        """Rotate log files."""
        if self._file:
            self._file.close()
            self._file = None

        # Rotate existing backups
        for i in range(self._backup_count - 1, 0, -1):
            src = self._path.with_suffix(f"{self._path.suffix}.{i}")
            dst = self._path.with_suffix(f"{self._path.suffix}.{i + 1}")
            if src.exists():
                src.rename(dst)

        # Move current to .1
        if self._path.exists():
            self._path.rename(self._path.with_suffix(f"{self._path.suffix}.1"))

    def close(self) -> None:
        """Close file."""
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


class JsonFileSink(FileSink):
    """JSON file sink (convenience wrapper)."""

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        super().__init__(path, format="json", **kwargs)


class ElasticsearchSink(LogSink):
    """Elasticsearch log sink for centralized logging.

    Sends logs to Elasticsearch/OpenSearch for aggregation and search.
    Supports bulk indexing for high throughput.
    """

    def __init__(
        self,
        url: str,
        *,
        index_prefix: str = "truthound-logs",
        index_pattern: str = "daily",  # daily, weekly, monthly
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Elasticsearch sink.

        Args:
            url: Elasticsearch URL.
            index_prefix: Index name prefix.
            index_pattern: Index rotation pattern.
            username: Basic auth username.
            password: Basic auth password.
            api_key: API key for auth.
            batch_size: Batch size for bulk indexing.
            flush_interval: Flush interval in seconds.
            **kwargs: Arguments for LogSink.
        """
        super().__init__(**kwargs)
        self._url = url.rstrip("/")
        self._index_prefix = index_prefix
        self._index_pattern = index_pattern
        self._username = username
        self._password = password
        self._api_key = api_key
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._buffer: list[LogRecord] = []
        self._last_flush = time.time()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="es-sink")
        self._running = True

        # Start background flusher
        self._flush_thread = threading.Thread(
            target=self._background_flush,
            daemon=True,
            name="es-sink-flusher",
        )
        self._flush_thread.start()

    def emit(self, record: LogRecord) -> None:
        """Buffer record for bulk indexing."""
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._batch_size:
                self._flush_buffer()

    def _background_flush(self) -> None:
        """Background thread for periodic flushing."""
        while self._running:
            time.sleep(1)
            with self._lock:
                if (
                    self._buffer
                    and time.time() - self._last_flush >= self._flush_interval
                ):
                    self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered records to Elasticsearch."""
        if not self._buffer:
            return

        records = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()

        # Submit to executor
        self._executor.submit(self._bulk_index, records)

    def _get_index_name(self, timestamp: datetime) -> str:
        """Get index name for timestamp."""
        if self._index_pattern == "daily":
            suffix = timestamp.strftime("%Y.%m.%d")
        elif self._index_pattern == "weekly":
            suffix = timestamp.strftime("%Y.%W")
        elif self._index_pattern == "monthly":
            suffix = timestamp.strftime("%Y.%m")
        else:
            suffix = timestamp.strftime("%Y.%m.%d")
        return f"{self._index_prefix}-{suffix}"

    def _bulk_index(self, records: list[LogRecord]) -> None:
        """Bulk index records to Elasticsearch."""
        try:
            import urllib.request
            import urllib.error

            # Build bulk request body
            lines = []
            for record in records:
                index_name = self._get_index_name(record.timestamp)
                action = json.dumps({"index": {"_index": index_name}})
                doc = json.dumps(record.to_dict(), default=str)
                lines.append(action)
                lines.append(doc)
            body = "\n".join(lines) + "\n"

            # Build request
            url = f"{self._url}/_bulk"
            headers = {"Content-Type": "application/x-ndjson"}

            if self._api_key:
                headers["Authorization"] = f"ApiKey {self._api_key}"

            request = urllib.request.Request(
                url,
                data=body.encode("utf-8"),
                headers=headers,
                method="POST",
            )

            if self._username and self._password:
                import base64

                credentials = base64.b64encode(
                    f"{self._username}:{self._password}".encode()
                ).decode()
                request.add_header("Authorization", f"Basic {credentials}")

            with urllib.request.urlopen(request, timeout=30):
                pass

        except Exception:
            pass  # Silently fail - don't break logging

    def flush(self) -> None:
        """Flush buffered records."""
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Close sink."""
        self._running = False
        self.flush()
        self._executor.shutdown(wait=True)


class LokiSink(LogSink):
    """Grafana Loki log sink.

    Sends logs to Loki for aggregation with Prometheus-style labels.
    """

    def __init__(
        self,
        url: str,
        *,
        labels: dict[str, str] | None = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Initialize Loki sink.

        Args:
            url: Loki push URL (e.g., http://loki:3100/loki/api/v1/push).
            labels: Static labels to add to all logs.
            batch_size: Batch size.
            flush_interval: Flush interval in seconds.
            **kwargs: Arguments for LogSink.
        """
        super().__init__(**kwargs)
        self._url = url
        self._labels = labels or {}
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: list[LogRecord] = []
        self._last_flush = time.time()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="loki-sink")

    def emit(self, record: LogRecord) -> None:
        """Buffer record for batch push."""
        with self._lock:
            self._buffer.append(record)
            if len(self._buffer) >= self._batch_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered records to Loki."""
        if not self._buffer:
            return

        records = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()
        self._executor.submit(self._push_to_loki, records)

    def _push_to_loki(self, records: list[LogRecord]) -> None:
        """Push records to Loki."""
        try:
            import urllib.request

            # Group by labels
            streams: dict[str, list[tuple[str, str]]] = {}
            for record in records:
                labels = {
                    **self._labels,
                    "level": record.level.name.lower(),
                    "logger": record.logger_name,
                }
                if record.service:
                    labels["service"] = record.service
                if record.environment:
                    labels["environment"] = record.environment

                label_str = "{" + ",".join(f'{k}="{v}"' for k, v in sorted(labels.items())) + "}"
                if label_str not in streams:
                    streams[label_str] = []

                # Loki expects nanosecond timestamps
                ts_ns = str(int(record.timestamp.timestamp() * 1_000_000_000))
                streams[label_str].append([ts_ns, record.to_json()])

            # Build Loki push format
            payload = {
                "streams": [
                    {"stream": json.loads(label_str.replace("=", ":")), "values": values}
                    for label_str, values in streams.items()
                ]
            }

            request = urllib.request.Request(
                self._url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=30):
                pass

        except Exception:
            pass

    def flush(self) -> None:
        """Flush buffered records."""
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Close sink."""
        self.flush()
        self._executor.shutdown(wait=True)


class FluentdSink(LogSink):
    """Fluentd log sink.

    Sends logs to Fluentd using the Forward protocol.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 24224,
        *,
        tag: str = "truthound",
        **kwargs: Any,
    ) -> None:
        """Initialize Fluentd sink.

        Args:
            host: Fluentd host.
            port: Fluentd port.
            tag: Fluentd tag prefix.
            **kwargs: Arguments for LogSink.
        """
        super().__init__(**kwargs)
        self._host = host
        self._port = port
        self._tag = tag
        self._socket: socket.socket | None = None

    def emit(self, record: LogRecord) -> None:
        """Send record to Fluentd."""
        try:
            import msgpack  # type: ignore
        except ImportError:
            # Fallback to JSON
            self._emit_json(record)
            return

        try:
            sock = self._get_socket()
            tag = f"{self._tag}.{record.level.name.lower()}"
            timestamp = int(record.timestamp.timestamp())
            data = record.to_dict(include_meta=True)

            # Forward protocol: [tag, time, record]
            message = msgpack.packb([tag, timestamp, data])
            sock.sendall(message)

        except Exception:
            self._socket = None  # Reset socket on error

    def _emit_json(self, record: LogRecord) -> None:
        """Emit using JSON (fallback)."""
        try:
            sock = self._get_socket()
            data = {
                "tag": f"{self._tag}.{record.level.name.lower()}",
                "time": int(record.timestamp.timestamp()),
                "record": record.to_dict(include_meta=True),
            }
            message = json.dumps(data).encode("utf-8") + b"\n"
            sock.sendall(message)
        except Exception:
            self._socket = None

    def _get_socket(self) -> socket.socket:
        """Get or create socket."""
        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._host, self._port))
        return self._socket

    def close(self) -> None:
        """Close socket."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None


# =============================================================================
# Log Configuration
# =============================================================================


@dataclass
class LogConfig:
    """Logging configuration.

    Example:
        >>> config = LogConfig(
        ...     level="info",
        ...     format="json",
        ...     service="truthound",
        ...     environment="production",
        ...     sinks=[
        ...         {"type": "console"},
        ...         {"type": "file", "path": "/var/log/truthound.log"},
        ...         {"type": "elasticsearch", "url": "http://elk:9200"},
        ...     ],
        ... )
    """

    level: str | LogLevel = LogLevel.INFO
    format: str = "console"  # console, json, logfmt
    service: str = ""
    environment: str = ""
    include_caller: bool = False
    include_meta: bool = True

    # Sinks configuration
    sinks: list[dict[str, Any]] = field(default_factory=lambda: [{"type": "console"}])

    # Buffering
    async_logging: bool = True
    buffer_size: int = 1000
    flush_interval: float = 1.0

    @classmethod
    def development(cls) -> "LogConfig":
        """Development configuration."""
        return cls(
            level=LogLevel.DEBUG,
            format="console",
            environment="development",
            include_caller=True,
            async_logging=False,
        )

    @classmethod
    def production(cls, service: str) -> "LogConfig":
        """Production configuration."""
        return cls(
            level=LogLevel.INFO,
            format="json",
            service=service,
            environment="production",
            include_caller=False,
            async_logging=True,
        )

    @classmethod
    def from_environment(cls) -> "LogConfig":
        """Load configuration from environment variables."""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "console"),
            service=os.getenv("SERVICE_NAME", ""),
            environment=os.getenv("ENVIRONMENT", "development"),
            include_caller=os.getenv("LOG_INCLUDE_CALLER", "").lower() == "true",
        )


# =============================================================================
# Enterprise Logger
# =============================================================================


class EnterpriseLogger:
    """Enterprise-grade structured logger.

    Features:
    - Automatic correlation ID propagation
    - Multiple output sinks
    - Async buffered logging
    - Environment-aware configuration
    - Field binding for contextual logging

    Example:
        >>> logger = EnterpriseLogger("my.module", config=LogConfig.production("my-service"))
        >>> logger.info("Request received", path="/api/users", method="GET")
        >>>
        >>> # Bind fields for reuse
        >>> req_logger = logger.bind(request_id="abc123", user_id="user-456")
        >>> req_logger.info("Processing")  # Includes request_id and user_id
    """

    def __init__(
        self,
        name: str,
        *,
        config: LogConfig | None = None,
        sinks: list[LogSink] | None = None,
    ) -> None:
        """Initialize logger.

        Args:
            name: Logger name (usually module name).
            config: Logging configuration.
            sinks: Direct sink instances (overrides config.sinks).
        """
        self._name = name
        self._config = config or LogConfig()
        self._level = (
            LogLevel.from_string(self._config.level)
            if isinstance(self._config.level, str)
            else self._config.level
        )
        self._bound_fields: dict[str, Any] = {}

        # Initialize sinks
        if sinks:
            self._sinks = sinks
        else:
            self._sinks = self._create_sinks_from_config()

        # Async buffer
        self._buffer: queue.Queue[LogRecord] = queue.Queue(
            maxsize=self._config.buffer_size
        )
        self._running = True

        if self._config.async_logging:
            self._worker = threading.Thread(
                target=self._process_buffer,
                daemon=True,
                name=f"logger-{name}",
            )
            self._worker.start()

            # Register cleanup
            atexit.register(self.close)

    def _create_sinks_from_config(self) -> list[LogSink]:
        """Create sinks from configuration."""
        sinks = []
        for sink_config in self._config.sinks:
            sink_type = sink_config.get("type", "console")
            sink = self._create_sink(sink_type, sink_config)
            if sink:
                sinks.append(sink)
        return sinks or [ConsoleSink(format=self._config.format)]

    def _create_sink(
        self, sink_type: str, config: dict[str, Any]
    ) -> LogSink | None:
        """Create a sink from configuration."""
        config = config.copy()
        config.pop("type", None)

        if sink_type == "console":
            return ConsoleSink(format=self._config.format, **config)
        elif sink_type == "file":
            return FileSink(**config)
        elif sink_type == "json_file":
            return JsonFileSink(**config)
        elif sink_type == "elasticsearch":
            return ElasticsearchSink(**config)
        elif sink_type == "loki":
            return LokiSink(**config)
        elif sink_type == "fluentd":
            return FluentdSink(**config)
        else:
            return None

    @property
    def name(self) -> str:
        """Get logger name."""
        return self._name

    @property
    def level(self) -> LogLevel:
        """Get log level."""
        return self._level

    @level.setter
    def level(self, value: LogLevel) -> None:
        """Set log level."""
        self._level = value

    def bind(self, **fields: Any) -> "EnterpriseLogger":
        """Create child logger with bound fields.

        Args:
            **fields: Fields to bind.

        Returns:
            New logger with bound fields.
        """
        child = EnterpriseLogger(
            self._name,
            config=self._config,
            sinks=self._sinks,
        )
        child._bound_fields = {**self._bound_fields, **fields}
        child._running = self._running
        child._buffer = self._buffer  # Share buffer
        return child

    def _get_caller(self) -> str | None:
        """Get caller location."""
        if not self._config.include_caller:
            return None

        import inspect

        frame = inspect.currentframe()
        if frame:
            for _ in range(5):  # Skip internal frames
                if frame.f_back:
                    frame = frame.f_back
            filename = os.path.basename(frame.f_code.co_filename)
            return f"{filename}:{frame.f_lineno}:{frame.f_code.co_name}"
        return None

    def _log(
        self,
        level: LogLevel,
        message: str,
        exception: BaseException | None = None,
        **fields: Any,
    ) -> None:
        """Internal log method."""
        if level < self._level:
            return

        # Get correlation context
        ctx = CorrelationContext.get_current()

        # Merge fields: bound -> context -> call-time
        merged_fields = {
            **self._bound_fields,
            **ctx,
            **fields,
        }

        # Extract special fields
        correlation_id = merged_fields.pop("correlation_id", None)
        trace_id = merged_fields.pop("trace_id", None)
        span_id = merged_fields.pop("span_id", None)

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            logger_name=self._name,
            fields=merged_fields,
            exception=exception,
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            caller=self._get_caller(),
            service=self._config.service,
            environment=self._config.environment,
        )

        if self._config.async_logging:
            try:
                self._buffer.put_nowait(record)
            except queue.Full:
                # Buffer full, emit directly
                self._emit_record(record)
        else:
            self._emit_record(record)

    def _emit_record(self, record: LogRecord) -> None:
        """Emit record to all sinks."""
        for sink in self._sinks:
            try:
                if sink.should_emit(record):
                    sink.emit(record)
            except Exception:
                pass

    def _process_buffer(self) -> None:
        """Process buffered records."""
        while self._running:
            try:
                record = self._buffer.get(timeout=0.1)
                self._emit_record(record)
            except queue.Empty:
                pass
            except Exception:
                pass

        # Drain remaining records
        while not self._buffer.empty():
            try:
                record = self._buffer.get_nowait()
                self._emit_record(record)
            except queue.Empty:
                break

    def trace(self, message: str, **fields: Any) -> None:
        """Log at TRACE level."""
        self._log(LogLevel.TRACE, message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        """Alias for warning."""
        self.warning(message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, **fields)

    def critical(self, message: str, **fields: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, **fields)

    def fatal(self, message: str, **fields: Any) -> None:
        """Alias for critical."""
        self.critical(message, **fields)

    def exception(
        self,
        message: str,
        exc: BaseException | None = None,
        **fields: Any,
    ) -> None:
        """Log exception with traceback."""
        if exc is None:
            exc = sys.exc_info()[1]
        self._log(LogLevel.ERROR, message, exception=exc, **fields)

    def audit(self, message: str, **fields: Any) -> None:
        """Log audit event (special level)."""
        self._log(LogLevel.AUDIT, message, **fields)

    def flush(self) -> None:
        """Flush all sinks."""
        # Wait for buffer to drain
        while not self._buffer.empty():
            time.sleep(0.01)

        for sink in self._sinks:
            try:
                sink.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close logger and all sinks."""
        self._running = False
        self.flush()

        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                pass


# =============================================================================
# Global Logger Management
# =============================================================================

_loggers: dict[str, EnterpriseLogger] = {}
_global_config: LogConfig | None = None
_lock = threading.Lock()


def configure_logging(
    *,
    level: str | LogLevel = LogLevel.INFO,
    format: str = "console",
    service: str = "",
    environment: str = "",
    sinks: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> None:
    """Configure global logging.

    Args:
        level: Log level.
        format: Output format (console, json, logfmt).
        service: Service name.
        environment: Environment name.
        sinks: Sink configurations.
        **kwargs: Additional LogConfig parameters.
    """
    global _global_config, _loggers

    with _lock:
        _global_config = LogConfig(
            level=level,
            format=format,
            service=service,
            environment=environment,
            sinks=sinks or [{"type": "console"}],
            **kwargs,
        )
        # Clear existing loggers so they pick up new config
        for logger in _loggers.values():
            logger.close()
        _loggers.clear()


def get_logger(name: str) -> EnterpriseLogger:
    """Get or create a logger.

    Args:
        name: Logger name (usually __name__).

    Returns:
        EnterpriseLogger instance.
    """
    global _loggers, _global_config

    with _lock:
        if name not in _loggers:
            config = _global_config or LogConfig.from_environment()
            _loggers[name] = EnterpriseLogger(name, config=config)
        return _loggers[name]


def reset_logging() -> None:
    """Reset logging to defaults."""
    global _loggers, _global_config

    with _lock:
        for logger in _loggers.values():
            logger.close()
        _loggers.clear()
        _global_config = None
