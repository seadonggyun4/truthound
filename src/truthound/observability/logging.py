"""Structured logging system for Truthound.

This module provides a structured logging implementation that supports:
- Multiple output formats (JSON, logfmt, console)
- Contextual logging with automatic field propagation
- Log levels with filtering
- Multiple handlers with routing
- Integration with standard logging module

Design Principles:
    1. Structured by default: All logs are key-value pairs
    2. Context propagation: Fields automatically inherited
    3. Format agnostic: Same API for all output formats
    4. Zero-cost when disabled: Minimal overhead when not logging
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO


# =============================================================================
# Log Levels
# =============================================================================


class LogLevel(IntEnum):
    """Log severity levels."""

    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

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
        }
        return mapping.get(level.lower(), cls.INFO)


# =============================================================================
# Log Record
# =============================================================================


@dataclass
class LogRecord:
    """Immutable log record with structured data.

    Attributes:
        timestamp: When the log was created (UTC).
        level: Log severity level.
        message: Human-readable log message.
        logger_name: Name of the logger.
        fields: Structured key-value data.
        exception: Exception info if present.
        trace_id: Distributed trace ID.
        span_id: Current span ID.
        caller: Source location (file:line).
    """

    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    fields: dict[str, Any] = field(default_factory=dict)
    exception: BaseException | None = None
    trace_id: str | None = None
    span_id: str | None = None
    caller: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name.lower(),
            "message": self.message,
            "logger": self.logger_name,
            **self.fields,
        }

        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        if self.caller:
            data["caller"] = self.caller
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


# =============================================================================
# Log Context
# =============================================================================


class LogContext:
    """Thread-local context for structured logging.

    LogContext provides automatic field propagation for structured logs.
    Fields added to the context are automatically included in all logs
    within that context.

    Example:
        >>> with log_context(request_id="abc123", user_id=42):
        ...     logger.info("Processing request")  # includes request_id, user_id
        ...     with log_context(action="validate"):
        ...         logger.info("Validating")  # includes all three fields
    """

    _local = threading.local()

    @classmethod
    def get_current(cls) -> dict[str, Any]:
        """Get current context fields."""
        if not hasattr(cls._local, "stack"):
            cls._local.stack = [{}]
        # Merge all context levels
        result: dict[str, Any] = {}
        for ctx in cls._local.stack:
            result.update(ctx)
        return result

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


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Context manager for adding fields to log context.

    Args:
        **fields: Key-value pairs to add to context.

    Example:
        >>> with log_context(checkpoint="daily", run_id="abc"):
        ...     logger.info("Starting validation")
    """
    LogContext.push(**fields)
    try:
        yield
    finally:
        LogContext.pop()


# =============================================================================
# Log Formatters
# =============================================================================


class LogFormatter(ABC):
    """Abstract base class for log formatters.

    Formatters convert LogRecord objects to string output.
    """

    @abstractmethod
    def format(self, record: LogRecord) -> str:
        """Format a log record.

        Args:
            record: The log record to format.

        Returns:
            Formatted string.
        """
        pass


class JSONFormatter(LogFormatter):
    """JSON log formatter.

    Outputs logs as JSON objects, one per line.
    Ideal for log aggregation systems like ELK, Splunk, etc.

    Example output:
        {"timestamp":"2024-01-15T10:30:00Z","level":"info","message":"Starting",...}
    """

    def __init__(
        self,
        *,
        indent: int | None = None,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
    ) -> None:
        """Initialize JSON formatter.

        Args:
            indent: JSON indentation (None for compact).
            sort_keys: Sort keys alphabetically.
            ensure_ascii: Escape non-ASCII characters.
        """
        self._indent = indent
        self._sort_keys = sort_keys
        self._ensure_ascii = ensure_ascii

    def format(self, record: LogRecord) -> str:
        """Format record as JSON."""
        data = record.to_dict()
        return json.dumps(
            data,
            indent=self._indent,
            sort_keys=self._sort_keys,
            ensure_ascii=self._ensure_ascii,
            default=str,
        )


class LogfmtFormatter(LogFormatter):
    """Logfmt formatter.

    Outputs logs in logfmt format (key=value pairs).
    Popular with Prometheus/Grafana ecosystem.

    Example output:
        ts=2024-01-15T10:30:00Z level=info msg="Starting validation" checkpoint=daily
    """

    def __init__(self, *, timestamp_key: str = "ts") -> None:
        """Initialize logfmt formatter.

        Args:
            timestamp_key: Key name for timestamp field.
        """
        self._timestamp_key = timestamp_key

    def format(self, record: LogRecord) -> str:
        """Format record as logfmt."""
        parts = [
            f'{self._timestamp_key}={record.timestamp.isoformat()}',
            f'level={record.level.name.lower()}',
            f'msg="{self._escape(record.message)}"',
            f'logger={record.logger_name}',
        ]

        if record.trace_id:
            parts.append(f"trace_id={record.trace_id}")
        if record.span_id:
            parts.append(f"span_id={record.span_id}")
        if record.caller:
            parts.append(f"caller={record.caller}")

        for key, value in record.fields.items():
            formatted = self._format_value(value)
            parts.append(f"{key}={formatted}")

        if record.exception:
            parts.append(f'error="{self._escape(str(record.exception))}"')

        return " ".join(parts)

    def _escape(self, value: str) -> str:
        """Escape special characters in value."""
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


class ConsoleFormatter(LogFormatter):
    """Human-readable console formatter.

    Outputs colored, readable logs for development.

    Example output:
        2024-01-15 10:30:00 INFO  [my.module] Starting validation checkpoint=daily
    """

    # ANSI color codes
    COLORS = {
        LogLevel.TRACE: "\033[90m",     # Gray
        LogLevel.DEBUG: "\033[36m",     # Cyan
        LogLevel.INFO: "\033[32m",      # Green
        LogLevel.WARNING: "\033[33m",   # Yellow
        LogLevel.ERROR: "\033[31m",     # Red
        LogLevel.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(
        self,
        *,
        color: bool = True,
        show_timestamp: bool = True,
        show_caller: bool = False,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        """Initialize console formatter.

        Args:
            color: Use ANSI colors.
            show_timestamp: Show timestamp.
            show_caller: Show caller location.
            timestamp_format: strftime format for timestamp.
        """
        self._color = color and sys.stdout.isatty()
        self._show_timestamp = show_timestamp
        self._show_caller = show_caller
        self._timestamp_format = timestamp_format

    def format(self, record: LogRecord) -> str:
        """Format record for console."""
        parts = []

        if self._show_timestamp:
            ts = record.timestamp.strftime(self._timestamp_format)
            parts.append(ts)

        level = record.level.name.ljust(5)
        if self._color:
            color = self.COLORS.get(record.level, "")
            level = f"{color}{level}{self.RESET}"
        parts.append(level)

        parts.append(f"[{record.logger_name}]")
        parts.append(record.message)

        # Add fields
        if record.fields:
            field_strs = [f"{k}={v}" for k, v in record.fields.items()]
            parts.append(" ".join(field_strs))

        if self._show_caller and record.caller:
            parts.append(f"({record.caller})")

        result = " ".join(parts)

        # Add exception if present
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


# =============================================================================
# Log Handlers
# =============================================================================


class LogHandler(ABC):
    """Abstract base class for log handlers.

    Handlers are responsible for outputting formatted log records.
    """

    def __init__(
        self,
        formatter: LogFormatter | None = None,
        level: LogLevel = LogLevel.DEBUG,
    ) -> None:
        """Initialize handler.

        Args:
            formatter: Log formatter to use.
            level: Minimum log level to handle.
        """
        self._formatter = formatter or ConsoleFormatter()
        self._level = level
        self._lock = threading.Lock()

    @property
    def formatter(self) -> LogFormatter:
        """Get the formatter."""
        return self._formatter

    @formatter.setter
    def formatter(self, value: LogFormatter) -> None:
        """Set the formatter."""
        self._formatter = value

    def should_handle(self, record: LogRecord) -> bool:
        """Check if this handler should process the record."""
        return record.level >= self._level

    @abstractmethod
    def emit(self, record: LogRecord) -> None:
        """Emit a log record.

        Args:
            record: The log record to emit.
        """
        pass

    def handle(self, record: LogRecord) -> None:
        """Handle a log record (thread-safe).

        Args:
            record: The log record to handle.
        """
        if self.should_handle(record):
            with self._lock:
                self.emit(record)

    def close(self) -> None:
        """Clean up handler resources."""
        pass


class ConsoleHandler(LogHandler):
    """Handler that writes to console (stdout/stderr).

    Writes INFO and below to stdout, WARNING and above to stderr.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        split_stderr: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize console handler.

        Args:
            stream: Output stream (None for auto stdout/stderr).
            split_stderr: Send WARNING+ to stderr.
            **kwargs: Arguments passed to LogHandler.
        """
        super().__init__(**kwargs)
        self._stream = stream
        self._split_stderr = split_stderr

    def emit(self, record: LogRecord) -> None:
        """Write log to console."""
        message = self._formatter.format(record)

        if self._stream:
            stream = self._stream
        elif self._split_stderr and record.level >= LogLevel.WARNING:
            stream = sys.stderr
        else:
            stream = sys.stdout

        try:
            stream.write(message + "\n")
            stream.flush()
        except Exception:
            pass  # Don't fail on logging errors


class FileHandler(LogHandler):
    """Handler that writes to a file."""

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "a",
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> None:
        """Initialize file handler.

        Args:
            path: Path to log file.
            mode: File open mode.
            encoding: File encoding.
            **kwargs: Arguments passed to LogHandler.
        """
        super().__init__(**kwargs)
        self._path = Path(path)
        self._mode = mode
        self._encoding = encoding
        self._file: TextIO | None = None

    def _ensure_file(self) -> TextIO:
        """Ensure file is open."""
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(
                self._path, self._mode, encoding=self._encoding
            )
        return self._file

    def emit(self, record: LogRecord) -> None:
        """Write log to file."""
        message = self._formatter.format(record)
        try:
            f = self._ensure_file()
            f.write(message + "\n")
            f.flush()
        except Exception:
            pass

    def close(self) -> None:
        """Close the file."""
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


class RotatingFileHandler(FileHandler):
    """Handler that rotates log files based on size.

    Creates backup files (log.1, log.2, etc.) when max size reached.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize rotating file handler.

        Args:
            path: Path to log file.
            max_bytes: Maximum file size before rotation.
            backup_count: Number of backup files to keep.
            **kwargs: Arguments passed to FileHandler.
        """
        super().__init__(path, **kwargs)
        self._max_bytes = max_bytes
        self._backup_count = backup_count

    def _should_rotate(self) -> bool:
        """Check if rotation is needed."""
        try:
            return self._path.stat().st_size >= self._max_bytes
        except FileNotFoundError:
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
            self._path.rename(
                self._path.with_suffix(f"{self._path.suffix}.1")
            )

    def emit(self, record: LogRecord) -> None:
        """Write log with rotation check."""
        if self._should_rotate():
            self._rotate()
        super().emit(record)


# =============================================================================
# Structured Logger
# =============================================================================


class StructuredLogger:
    """Structured logger with context and multiple handlers.

    StructuredLogger provides structured logging with automatic field
    propagation, multiple output handlers, and format flexibility.

    Example:
        >>> logger = StructuredLogger("my.module")
        >>> logger.add_handler(ConsoleHandler(formatter=JSONFormatter()))
        >>>
        >>> logger.info("User logged in", user_id=123, ip="192.168.1.1")
        >>> # Output: {"timestamp":"...","level":"info","message":"User logged in","user_id":123,"ip":"192.168.1.1"}
    """

    def __init__(
        self,
        name: str,
        *,
        level: LogLevel = LogLevel.INFO,
        handlers: list[LogHandler] | None = None,
        include_caller: bool = False,
    ) -> None:
        """Initialize structured logger.

        Args:
            name: Logger name (usually module name).
            level: Minimum log level.
            handlers: Initial handlers.
            include_caller: Include caller location in logs.
        """
        self._name = name
        self._level = level
        self._handlers: list[LogHandler] = handlers or []
        self._include_caller = include_caller
        self._bound_fields: dict[str, Any] = {}

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

    def add_handler(self, handler: LogHandler) -> None:
        """Add a log handler."""
        self._handlers.append(handler)

    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a log handler."""
        self._handlers.remove(handler)

    def bind(self, **fields: Any) -> "StructuredLogger":
        """Create a child logger with bound fields.

        Args:
            **fields: Fields to bind.

        Returns:
            New logger with bound fields.
        """
        new_logger = StructuredLogger(
            self._name,
            level=self._level,
            handlers=self._handlers,
            include_caller=self._include_caller,
        )
        new_logger._bound_fields = {**self._bound_fields, **fields}
        return new_logger

    def _get_caller(self) -> str | None:
        """Get caller location."""
        if not self._include_caller:
            return None

        import inspect
        frame = inspect.currentframe()
        if frame:
            # Go up the stack to find the actual caller
            for _ in range(4):  # Skip internal frames
                if frame.f_back:
                    frame = frame.f_back
            return f"{frame.f_code.co_filename}:{frame.f_lineno}"
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

        # Get context from LogContext
        from truthound.observability.context import current_context

        ctx = current_context()

        # Merge fields: bound -> context -> call-time
        merged_fields = {
            **self._bound_fields,
            **LogContext.get_current(),
            **fields,
        }

        record = LogRecord(
            timestamp=datetime.now(timezone.utc),
            level=level,
            message=message,
            logger_name=self._name,
            fields=merged_fields,
            exception=exception,
            trace_id=ctx.trace_id if ctx else None,
            span_id=ctx.span_id if ctx else None,
            caller=self._get_caller(),
        )

        for handler in self._handlers:
            try:
                handler.handle(record)
            except Exception:
                pass  # Don't fail on handler errors

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


# =============================================================================
# Global Logger Management
# =============================================================================

_loggers: dict[str, StructuredLogger] = {}
_default_handlers: list[LogHandler] = []
_default_level: LogLevel = LogLevel.INFO
_lock = threading.Lock()


def configure_logging(
    *,
    level: LogLevel | str = LogLevel.INFO,
    format: str = "console",
    handlers: list[LogHandler] | None = None,
    json_output: bool = False,
) -> None:
    """Configure global logging settings.

    Args:
        level: Default log level.
        format: Output format ("console", "json", "logfmt").
        handlers: Custom handlers (overrides format).
        json_output: Shortcut for JSON format.
    """
    global _default_handlers, _default_level

    if isinstance(level, str):
        level = LogLevel.from_string(level)
    _default_level = level

    if handlers:
        _default_handlers = handlers
    else:
        # Create default handler based on format
        if json_output or format == "json":
            formatter = JSONFormatter()
        elif format == "logfmt":
            formatter = LogfmtFormatter()
        else:
            formatter = ConsoleFormatter()

        _default_handlers = [ConsoleHandler(formatter=formatter, level=level)]


def get_logger(name: str) -> StructuredLogger:
    """Get or create a logger.

    Args:
        name: Logger name (usually __name__).

    Returns:
        StructuredLogger instance.
    """
    global _loggers

    with _lock:
        if name not in _loggers:
            logger = StructuredLogger(
                name,
                level=_default_level,
                handlers=_default_handlers.copy() if _default_handlers else [
                    ConsoleHandler(formatter=ConsoleFormatter())
                ],
            )
            _loggers[name] = logger

        return _loggers[name]


def set_default_logger(logger: StructuredLogger) -> None:
    """Set the default logger for a name.

    Args:
        logger: Logger to register.
    """
    global _loggers

    with _lock:
        _loggers[logger.name] = logger


# Initialize default configuration
configure_logging()
