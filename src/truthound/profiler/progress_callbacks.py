"""Standardized Progress Callback System.

This module provides a comprehensive, extensible progress callback system
with Protocol-based design, adapters for various output targets, and
advanced features like filtering, throttling, and aggregation.

Key features:
- Protocol-based callback abstraction
- Registry for callback type discovery
- Multiple output adapters (console, file, logging, webhook)
- Progress filtering and throttling
- Hierarchical progress aggregation
- Event batching and buffering
- Async callback support

Example:
    from truthound.profiler.progress_callbacks import (
        CallbackRegistry,
        ConsoleAdapter,
        LoggingAdapter,
        create_callback_chain,
    )

    # Create callbacks
    console = ConsoleAdapter()
    logger = LoggingAdapter(logger_name="profiler")

    # Chain callbacks
    chain = create_callback_chain(console, logger)

    # Use with profiler
    profiler.profile(data, progress_callback=chain)
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)


# =============================================================================
# Event Types and Levels
# =============================================================================


class EventLevel(Enum):
    """Severity levels for progress events."""

    DEBUG = auto()      # Detailed debugging info
    INFO = auto()       # Normal progress updates
    NOTICE = auto()     # Notable milestones
    WARNING = auto()    # Non-critical issues
    ERROR = auto()      # Errors that don't stop processing
    CRITICAL = auto()   # Fatal errors


class EventType(str, Enum):
    """Types of progress events."""

    # Lifecycle events
    START = "start"
    COMPLETE = "complete"
    FAIL = "fail"
    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"

    # Progress events
    PROGRESS = "progress"
    COLUMN_START = "column_start"
    COLUMN_COMPLETE = "column_complete"
    COLUMN_PROGRESS = "column_progress"
    ANALYZER_START = "analyzer_start"
    ANALYZER_COMPLETE = "analyzer_complete"

    # Batch/partition events
    BATCH_START = "batch_start"
    BATCH_COMPLETE = "batch_complete"
    PARTITION_START = "partition_start"
    PARTITION_COMPLETE = "partition_complete"

    # Diagnostic events
    CHECKPOINT = "checkpoint"
    HEARTBEAT = "heartbeat"
    METRIC = "metric"
    LOG = "log"


@dataclass(frozen=True)
class ProgressContext:
    """Context information for a progress event.

    Provides structured context that can be nested for hierarchical operations.
    """

    operation_id: str = ""          # Unique operation identifier
    table_name: str = ""            # Current table
    column_name: str = ""           # Current column
    analyzer_name: str = ""         # Current analyzer
    batch_index: int = 0            # Batch number
    partition_index: int = 0        # Partition number
    parent_context: "ProgressContext | None" = None  # Parent for nesting
    tags: tuple[str, ...] = ()      # Custom tags for filtering

    def with_column(self, column: str) -> "ProgressContext":
        """Create child context for a column."""
        return ProgressContext(
            operation_id=self.operation_id,
            table_name=self.table_name,
            column_name=column,
            parent_context=self,
            tags=self.tags,
        )

    def with_analyzer(self, analyzer: str) -> "ProgressContext":
        """Create child context for an analyzer."""
        return ProgressContext(
            operation_id=self.operation_id,
            table_name=self.table_name,
            column_name=self.column_name,
            analyzer_name=analyzer,
            parent_context=self,
            tags=self.tags,
        )

    def get_path(self) -> str:
        """Get hierarchical path string."""
        parts = []
        if self.table_name:
            parts.append(self.table_name)
        if self.column_name:
            parts.append(self.column_name)
        if self.analyzer_name:
            parts.append(self.analyzer_name)
        return "/".join(parts) if parts else ""


@dataclass(frozen=True)
class ProgressMetrics:
    """Timing and throughput metrics."""

    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    rows_processed: int = 0
    rows_per_second: float = 0.0
    columns_completed: int = 0
    columns_total: int = 0
    memory_used_mb: float | None = None

    @property
    def columns_remaining(self) -> int:
        """Get number of columns remaining."""
        return max(0, self.columns_total - self.columns_completed)

    @property
    def throughput_string(self) -> str:
        """Get human-readable throughput."""
        if self.rows_per_second >= 1_000_000:
            return f"{self.rows_per_second / 1_000_000:.1f}M rows/s"
        elif self.rows_per_second >= 1_000:
            return f"{self.rows_per_second / 1_000:.1f}K rows/s"
        else:
            return f"{self.rows_per_second:.0f} rows/s"


@dataclass(frozen=True)
class StandardProgressEvent:
    """Standard progress event with full context.

    This is the primary event type that flows through the callback system.
    All adapters receive this standardized event format.
    """

    event_type: EventType
    level: EventLevel = EventLevel.INFO
    progress: float = 0.0           # 0.0 to 1.0
    message: str = ""
    context: ProgressContext = field(default_factory=ProgressContext)
    metrics: ProgressMetrics = field(default_factory=ProgressMetrics)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def percent(self) -> float:
        """Get progress as percentage."""
        return self.progress * 100

    @property
    def is_complete(self) -> bool:
        """Check if this is a completion event."""
        return self.event_type in {EventType.COMPLETE, EventType.FAIL, EventType.CANCEL}

    @property
    def is_error(self) -> bool:
        """Check if this is an error event."""
        return self.level in {EventLevel.ERROR, EventLevel.CRITICAL}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "level": self.level.name,
            "progress": self.progress,
            "message": self.message,
            "context": {
                "operation_id": self.context.operation_id,
                "table_name": self.context.table_name,
                "column_name": self.context.column_name,
                "analyzer_name": self.context.analyzer_name,
                "path": self.context.get_path(),
                "tags": list(self.context.tags),
            },
            "metrics": {
                "elapsed_seconds": self.metrics.elapsed_seconds,
                "estimated_remaining": self.metrics.estimated_remaining_seconds,
                "rows_processed": self.metrics.rows_processed,
                "rows_per_second": self.metrics.rows_per_second,
                "columns_completed": self.metrics.columns_completed,
                "columns_total": self.metrics.columns_total,
            },
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Callback Protocol
# =============================================================================


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress callbacks.

    Any class implementing this protocol can be used as a progress callback.
    This enables type-safe callbacks while maintaining extensibility.
    """

    def on_progress(self, event: StandardProgressEvent) -> None:
        """Handle a progress event.

        Args:
            event: The progress event to handle
        """
        ...


@runtime_checkable
class AsyncProgressCallback(Protocol):
    """Protocol for async progress callbacks."""

    async def on_progress_async(self, event: StandardProgressEvent) -> None:
        """Handle a progress event asynchronously.

        Args:
            event: The progress event to handle
        """
        ...


@runtime_checkable
class LifecycleCallback(Protocol):
    """Protocol for callbacks with lifecycle management."""

    def start(self) -> None:
        """Called when profiling starts."""
        ...

    def stop(self) -> None:
        """Called when profiling ends."""
        ...


# =============================================================================
# Base Callback Adapter
# =============================================================================


class CallbackAdapter(ABC):
    """Abstract base class for callback adapters.

    Provides common functionality for all callback types including
    lifecycle management and configuration.
    """

    def __init__(
        self,
        *,
        name: str = "",
        enabled: bool = True,
        min_level: EventLevel = EventLevel.INFO,
        event_types: set[EventType] | None = None,
    ):
        """Initialize adapter.

        Args:
            name: Adapter name for identification
            enabled: Whether adapter is active
            min_level: Minimum event level to process
            event_types: Event types to handle (None = all)
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.min_level = min_level
        self.event_types = event_types
        self._started = False

    def should_handle(self, event: StandardProgressEvent) -> bool:
        """Check if event should be handled.

        Args:
            event: Event to check

        Returns:
            True if event should be processed
        """
        if not self.enabled:
            return False

        if event.level.value < self.min_level.value:
            return False

        if self.event_types and event.event_type not in self.event_types:
            return False

        return True

    def on_progress(self, event: StandardProgressEvent) -> None:
        """Handle progress event with filtering.

        Args:
            event: Progress event
        """
        if not self.should_handle(event):
            return

        self._handle_event(event)

    @abstractmethod
    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Handle the event (implemented by subclasses).

        Args:
            event: Progress event to handle
        """
        pass

    def start(self) -> None:
        """Start the adapter."""
        self._started = True

    def stop(self) -> None:
        """Stop the adapter."""
        self._started = False


# =============================================================================
# Console Adapters
# =============================================================================


@dataclass
class ConsoleStyle:
    """Console output styling configuration."""

    bar_width: int = 40
    bar_fill: str = "█"
    bar_empty: str = "░"
    show_eta: bool = True
    show_column: bool = True
    show_throughput: bool = True
    color_enabled: bool = True
    clear_on_complete: bool = True

    # ANSI color codes
    colors: dict[EventLevel, str] = field(default_factory=lambda: {
        EventLevel.DEBUG: "\033[90m",      # Gray
        EventLevel.INFO: "\033[0m",        # Default
        EventLevel.NOTICE: "\033[94m",     # Blue
        EventLevel.WARNING: "\033[93m",    # Yellow
        EventLevel.ERROR: "\033[91m",      # Red
        EventLevel.CRITICAL: "\033[91;1m", # Bold Red
    })
    reset: str = "\033[0m"


class ConsoleAdapter(CallbackAdapter):
    """Console output adapter with progress bar.

    Provides rich console output with progress bars, ETA estimation,
    and colored status messages.

    Example:
        adapter = ConsoleAdapter(style=ConsoleStyle(bar_width=50))
        tracker.add_callback(adapter)
    """

    def __init__(
        self,
        *,
        style: ConsoleStyle | None = None,
        stream: Any = None,  # TextIO
        **kwargs: Any,
    ):
        """Initialize console adapter.

        Args:
            style: Console styling configuration
            stream: Output stream (default: sys.stderr)
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.style = style or ConsoleStyle()
        self._stream = stream
        self._last_line_length = 0

    @property
    def stream(self) -> Any:
        """Get output stream."""
        if self._stream is None:
            import sys
            return sys.stderr
        return self._stream

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Handle event with console output."""
        if event.event_type == EventType.PROGRESS:
            self._render_progress_bar(event)
        elif event.is_complete:
            self._render_completion(event)
        else:
            self._render_message(event)

    def _render_progress_bar(self, event: StandardProgressEvent) -> None:
        """Render progress bar."""
        s = self.style

        # Build bar
        filled = int(event.progress * s.bar_width)
        bar = s.bar_fill * filled + s.bar_empty * (s.bar_width - filled)

        # Build parts
        parts = [f"\r[{bar}] {event.percent:5.1f}%"]

        if s.show_column and event.context.column_name:
            parts.append(f" | {event.context.column_name}")

        if s.show_throughput and event.metrics.rows_per_second > 0:
            parts.append(f" | {event.metrics.throughput_string}")

        if s.show_eta and event.metrics.estimated_remaining_seconds is not None:
            eta = self._format_time(event.metrics.estimated_remaining_seconds)
            parts.append(f" | ETA: {eta}")

        line = "".join(parts)

        # Pad to overwrite previous line
        if len(line) < self._last_line_length:
            line += " " * (self._last_line_length - len(line))
        self._last_line_length = len(line)

        print(line, end="", flush=True, file=self.stream)

    def _render_completion(self, event: StandardProgressEvent) -> None:
        """Render completion message."""
        if self.style.clear_on_complete:
            print("\r" + " " * self._last_line_length + "\r", end="", file=self.stream)

        color = self._get_color(event.level)
        elapsed = self._format_time(event.metrics.elapsed_seconds)

        if event.event_type == EventType.COMPLETE:
            msg = f"✓ Complete in {elapsed}"
        elif event.event_type == EventType.FAIL:
            msg = f"✗ Failed: {event.message}"
        else:
            msg = f"○ Cancelled after {elapsed}"

        print(f"{color}{msg}{self.style.reset}", file=self.stream)

    def _render_message(self, event: StandardProgressEvent) -> None:
        """Render status message."""
        color = self._get_color(event.level)
        print(f"{color}{event.message}{self.style.reset}", file=self.stream)

    def _get_color(self, level: EventLevel) -> str:
        """Get ANSI color for level."""
        if not self.style.color_enabled:
            return ""
        return self.style.colors.get(level, "")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"


class MinimalConsoleAdapter(CallbackAdapter):
    """Minimal console output showing only milestones.

    Useful for environments where minimal output is preferred.
    """

    def __init__(
        self,
        *,
        show_columns: bool = False,
        milestone_interval: int = 10,  # Show every N%
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.show_columns = show_columns
        self.milestone_interval = milestone_interval
        self._last_milestone = -1

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Handle event with minimal output."""
        if event.event_type == EventType.START:
            print("Starting profiling...")
        elif event.event_type == EventType.COMPLETE:
            elapsed = event.metrics.elapsed_seconds
            print(f"Completed in {elapsed:.1f}s")
        elif event.event_type == EventType.FAIL:
            print(f"Failed: {event.message}")
        elif event.event_type == EventType.PROGRESS:
            milestone = int(event.percent // self.milestone_interval) * self.milestone_interval
            if milestone > self._last_milestone:
                self._last_milestone = milestone
                print(f"Progress: {milestone}%")
        elif self.show_columns and event.event_type == EventType.COLUMN_COMPLETE:
            print(f"  Completed: {event.context.column_name}")


# =============================================================================
# Logging Adapter
# =============================================================================


class LoggingAdapter(CallbackAdapter):
    """Logging framework adapter.

    Routes progress events to Python's logging framework.

    Example:
        adapter = LoggingAdapter(
            logger_name="profiler.progress",
            min_level=EventLevel.INFO,
        )
    """

    # Map event levels to logging levels
    LEVEL_MAP = {
        EventLevel.DEBUG: logging.DEBUG,
        EventLevel.INFO: logging.INFO,
        EventLevel.NOTICE: logging.INFO,
        EventLevel.WARNING: logging.WARNING,
        EventLevel.ERROR: logging.ERROR,
        EventLevel.CRITICAL: logging.CRITICAL,
    }

    def __init__(
        self,
        *,
        logger_name: str = "truthound.progress",
        logger: logging.Logger | None = None,
        include_context: bool = True,
        include_metrics: bool = True,
        **kwargs: Any,
    ):
        """Initialize logging adapter.

        Args:
            logger_name: Logger name to use
            logger: Existing logger instance
            include_context: Include context in log extras
            include_metrics: Include metrics in log extras
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self._logger = logger or logging.getLogger(logger_name)
        self.include_context = include_context
        self.include_metrics = include_metrics

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Log the event."""
        level = self.LEVEL_MAP.get(event.level, logging.INFO)

        extra: dict[str, Any] = {
            "event_type": event.event_type.value,
            "progress": event.progress,
        }

        if self.include_context:
            extra["context"] = {
                "operation": event.context.operation_id,
                "table": event.context.table_name,
                "column": event.context.column_name,
                "path": event.context.get_path(),
            }

        if self.include_metrics:
            extra["metrics"] = {
                "elapsed": event.metrics.elapsed_seconds,
                "rows": event.metrics.rows_processed,
                "throughput": event.metrics.rows_per_second,
            }

        message = self._format_message(event)
        self._logger.log(level, message, extra=extra)

    def _format_message(self, event: StandardProgressEvent) -> str:
        """Format log message."""
        if event.message:
            return event.message

        if event.event_type == EventType.PROGRESS:
            return f"Progress: {event.percent:.1f}%"
        elif event.event_type == EventType.COLUMN_START:
            return f"Starting column: {event.context.column_name}"
        elif event.event_type == EventType.COLUMN_COMPLETE:
            return f"Completed column: {event.context.column_name}"
        elif event.event_type == EventType.COMPLETE:
            return f"Profiling complete ({event.metrics.elapsed_seconds:.1f}s)"
        elif event.event_type == EventType.FAIL:
            return f"Profiling failed: {event.message}"
        else:
            return f"{event.event_type.value}: {event.message or 'No message'}"


# =============================================================================
# File Adapter
# =============================================================================


@dataclass
class FileOutputConfig:
    """Configuration for file output."""

    format: str = "jsonl"  # jsonl, json, csv
    include_all_events: bool = True
    rotate_size_mb: int = 100
    compress_rotated: bool = True
    encoding: str = "utf-8"


class FileAdapter(CallbackAdapter):
    """File output adapter.

    Writes progress events to a file in various formats.

    Example:
        adapter = FileAdapter(
            path="profiling_progress.jsonl",
            config=FileOutputConfig(format="jsonl"),
        )
    """

    def __init__(
        self,
        path: str | Path,
        *,
        config: FileOutputConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize file adapter.

        Args:
            path: Output file path
            config: File output configuration
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.path = Path(path)
        self.config = config or FileOutputConfig()
        self._file: Any = None
        self._events: list[dict[str, Any]] = []

    def start(self) -> None:
        """Open file for writing."""
        super().start()
        if self.config.format == "jsonl":
            self._file = open(self.path, "a", encoding=self.config.encoding)
        else:
            self._events = []

    def stop(self) -> None:
        """Close file."""
        super().stop()
        if self._file:
            self._file.close()
            self._file = None

        if self.config.format == "json" and self._events:
            with open(self.path, "w", encoding=self.config.encoding) as f:
                json.dump(self._events, f, indent=2)

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Write event to file."""
        event_dict = event.to_dict()

        if self.config.format == "jsonl" and self._file:
            self._file.write(json.dumps(event_dict) + "\n")
            self._file.flush()
        elif self.config.format == "json":
            self._events.append(event_dict)


# =============================================================================
# Callback Chain
# =============================================================================


class CallbackChain:
    """Chains multiple callbacks together.

    Events are dispatched to all callbacks in the chain.
    Supports adding/removing callbacks dynamically.

    Example:
        chain = CallbackChain()
        chain.add(console_adapter)
        chain.add(logging_adapter)
        chain.add(file_adapter)

        # Use as single callback
        profiler.profile(data, progress_callback=chain)
    """

    def __init__(
        self,
        callbacks: Sequence[CallbackAdapter] | None = None,
        *,
        stop_on_error: bool = False,
    ):
        """Initialize callback chain.

        Args:
            callbacks: Initial callbacks
            stop_on_error: Stop chain on callback error
        """
        self._callbacks: list[CallbackAdapter] = list(callbacks or [])
        self.stop_on_error = stop_on_error
        self._errors: list[tuple[CallbackAdapter, Exception]] = []

    def add(self, callback: CallbackAdapter) -> "CallbackChain":
        """Add callback to chain.

        Args:
            callback: Callback to add

        Returns:
            Self for chaining
        """
        self._callbacks.append(callback)
        return self

    def remove(self, callback: CallbackAdapter) -> bool:
        """Remove callback from chain.

        Args:
            callback: Callback to remove

        Returns:
            True if callback was removed
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Remove all callbacks."""
        self._callbacks.clear()

    def on_progress(self, event: StandardProgressEvent) -> None:
        """Dispatch event to all callbacks.

        Args:
            event: Event to dispatch
        """
        for callback in self._callbacks:
            try:
                callback.on_progress(event)
            except Exception as e:
                self._errors.append((callback, e))
                if self.stop_on_error:
                    raise

    def start(self) -> None:
        """Start all callbacks."""
        for callback in self._callbacks:
            if hasattr(callback, "start"):
                callback.start()

    def stop(self) -> None:
        """Stop all callbacks."""
        for callback in self._callbacks:
            if hasattr(callback, "stop"):
                callback.stop()

    @property
    def errors(self) -> list[tuple[CallbackAdapter, Exception]]:
        """Get errors from callbacks."""
        return self._errors.copy()

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[CallbackAdapter]:
        return iter(self._callbacks)


# =============================================================================
# Filtering and Throttling
# =============================================================================


@dataclass
class FilterConfig:
    """Configuration for event filtering."""

    min_level: EventLevel = EventLevel.INFO
    event_types: set[EventType] | None = None
    include_tags: set[str] | None = None
    exclude_tags: set[str] | None = None
    column_patterns: list[str] | None = None  # Glob patterns
    table_patterns: list[str] | None = None


class FilteringAdapter(CallbackAdapter):
    """Filtering wrapper for callbacks.

    Filters events based on configurable criteria before
    passing to the wrapped callback.

    Example:
        # Only log column events
        filtered = FilteringAdapter(
            wrapped=LoggingAdapter(),
            config=FilterConfig(
                event_types={EventType.COLUMN_START, EventType.COLUMN_COMPLETE},
            ),
        )
    """

    def __init__(
        self,
        wrapped: CallbackAdapter,
        config: FilterConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize filtering adapter.

        Args:
            wrapped: Wrapped callback
            config: Filter configuration
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.wrapped = wrapped
        self.filter_config = config or FilterConfig()

    def should_handle(self, event: StandardProgressEvent) -> bool:
        """Apply filtering rules."""
        if not super().should_handle(event):
            return False

        fc = self.filter_config

        # Level check
        if event.level.value < fc.min_level.value:
            return False

        # Event type check
        if fc.event_types and event.event_type not in fc.event_types:
            return False

        # Tag checks
        if fc.include_tags:
            if not any(tag in event.context.tags for tag in fc.include_tags):
                return False

        if fc.exclude_tags:
            if any(tag in event.context.tags for tag in fc.exclude_tags):
                return False

        return True

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Pass event to wrapped callback."""
        self.wrapped.on_progress(event)

    def start(self) -> None:
        """Start wrapped callback."""
        super().start()
        if hasattr(self.wrapped, "start"):
            self.wrapped.start()

    def stop(self) -> None:
        """Stop wrapped callback."""
        super().stop()
        if hasattr(self.wrapped, "stop"):
            self.wrapped.stop()


@dataclass
class ThrottleConfig:
    """Configuration for event throttling."""

    min_interval_ms: int = 100          # Minimum ms between events
    max_events_per_second: int = 10     # Max events per second
    always_emit_types: set[EventType] = field(default_factory=lambda: {
        EventType.START, EventType.COMPLETE, EventType.FAIL,
        EventType.COLUMN_START, EventType.COLUMN_COMPLETE,
    })


class ThrottlingAdapter(CallbackAdapter):
    """Throttling wrapper for callbacks.

    Limits the rate of events passed to the wrapped callback.
    Always passes lifecycle events regardless of throttle.

    Example:
        # Limit console updates to 5 per second
        throttled = ThrottlingAdapter(
            wrapped=ConsoleAdapter(),
            config=ThrottleConfig(max_events_per_second=5),
        )
    """

    def __init__(
        self,
        wrapped: CallbackAdapter,
        config: ThrottleConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize throttling adapter.

        Args:
            wrapped: Wrapped callback
            config: Throttle configuration
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.wrapped = wrapped
        self.throttle_config = config or ThrottleConfig()
        self._last_emit_time: float = 0
        self._event_times: deque[float] = deque(maxlen=100)

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Handle event with throttling."""
        tc = self.throttle_config
        now = time.time()

        # Always emit certain event types
        if event.event_type in tc.always_emit_types:
            self._emit(event, now)
            return

        # Check minimum interval
        interval_ms = (now - self._last_emit_time) * 1000
        if interval_ms < tc.min_interval_ms:
            return

        # Check rate limit
        cutoff = now - 1.0  # Last second
        while self._event_times and self._event_times[0] < cutoff:
            self._event_times.popleft()

        if len(self._event_times) >= tc.max_events_per_second:
            return

        self._emit(event, now)

    def _emit(self, event: StandardProgressEvent, now: float) -> None:
        """Emit event to wrapped callback."""
        self._last_emit_time = now
        self._event_times.append(now)
        self.wrapped.on_progress(event)

    def start(self) -> None:
        """Start wrapped callback."""
        super().start()
        self._last_emit_time = 0
        self._event_times.clear()
        if hasattr(self.wrapped, "start"):
            self.wrapped.start()

    def stop(self) -> None:
        """Stop wrapped callback."""
        super().stop()
        if hasattr(self.wrapped, "stop"):
            self.wrapped.stop()


# =============================================================================
# Buffering and Batching
# =============================================================================


@dataclass
class BufferConfig:
    """Configuration for event buffering."""

    max_size: int = 100               # Max events in buffer
    flush_interval_seconds: float = 5.0  # Auto-flush interval
    flush_on_complete: bool = True    # Flush on completion events


class BufferingAdapter(CallbackAdapter):
    """Buffering wrapper for callbacks.

    Buffers events and flushes them in batches to reduce
    callback overhead.

    Example:
        # Buffer events and flush every 100 events or 5 seconds
        buffered = BufferingAdapter(
            wrapped=FileAdapter("events.jsonl"),
            config=BufferConfig(max_size=100, flush_interval_seconds=5),
        )
    """

    def __init__(
        self,
        wrapped: CallbackAdapter,
        config: BufferConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize buffering adapter.

        Args:
            wrapped: Wrapped callback
            config: Buffer configuration
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.wrapped = wrapped
        self.buffer_config = config or BufferConfig()
        self._buffer: list[StandardProgressEvent] = []
        self._last_flush: float = 0
        self._lock = threading.Lock()

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Buffer event."""
        bc = self.buffer_config

        with self._lock:
            self._buffer.append(event)

            # Check if we should flush
            should_flush = (
                len(self._buffer) >= bc.max_size or
                (bc.flush_on_complete and event.is_complete)
            )

            # Check interval
            now = time.time()
            if now - self._last_flush >= bc.flush_interval_seconds:
                should_flush = True

            if should_flush:
                self._flush()

    def _flush(self) -> None:
        """Flush buffer to wrapped callback."""
        if not self._buffer:
            return

        events = self._buffer
        self._buffer = []
        self._last_flush = time.time()

        for event in events:
            self.wrapped.on_progress(event)

    def stop(self) -> None:
        """Flush and stop."""
        with self._lock:
            self._flush()
        super().stop()
        if hasattr(self.wrapped, "stop"):
            self.wrapped.stop()


# =============================================================================
# Async Adapter
# =============================================================================


class AsyncAdapter(CallbackAdapter):
    """Async callback adapter.

    Processes events asynchronously using an event loop.

    Example:
        async def handle_event(event):
            await send_to_service(event)

        adapter = AsyncAdapter(async_handler=handle_event)
    """

    def __init__(
        self,
        async_handler: Callable[[StandardProgressEvent], Any],
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        **kwargs: Any,
    ):
        """Initialize async adapter.

        Args:
            async_handler: Async function to handle events
            loop: Event loop to use
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self._handler = async_handler
        self._loop = loop
        self._queue: queue.Queue[StandardProgressEvent | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start async processing thread."""
        super().start()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop async processing."""
        self._running = False
        self._queue.put(None)  # Signal to stop
        if self._thread:
            self._thread.join(timeout=5.0)
        super().stop()

    def _handle_event(self, event: StandardProgressEvent) -> None:
        """Queue event for async processing."""
        if self._running:
            self._queue.put(event)

    def _run_loop(self) -> None:
        """Run event loop in background thread."""
        loop = self._loop or asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
                if event is None:
                    break
                loop.run_until_complete(self._handler(event))
            except queue.Empty:
                continue
            except Exception:
                pass  # Log error in production

        loop.close()


# =============================================================================
# Registry Pattern
# =============================================================================


class CallbackRegistry:
    """Registry for callback adapters.

    Provides discovery and factory pattern for callbacks.

    Example:
        registry = CallbackRegistry()

        # Register custom callback
        @registry.register("custom")
        class CustomAdapter(CallbackAdapter):
            ...

        # Create callback by name
        callback = registry.create("console", bar_width=50)
    """

    _instance: "CallbackRegistry | None" = None

    def __init__(self) -> None:
        self._adapters: dict[str, type[CallbackAdapter]] = {}
        self._factories: dict[str, Callable[..., CallbackAdapter]] = {}

        # Register built-in adapters
        self._register_builtin()

    @classmethod
    def get_instance(cls) -> "CallbackRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_builtin(self) -> None:
        """Register built-in adapters."""
        self._adapters["console"] = ConsoleAdapter
        self._adapters["minimal_console"] = MinimalConsoleAdapter
        self._adapters["logging"] = LoggingAdapter
        self._adapters["file"] = FileAdapter

    def register(
        self,
        name: str,
    ) -> Callable[[type[CallbackAdapter]], type[CallbackAdapter]]:
        """Decorator to register a callback adapter.

        Args:
            name: Registration name

        Returns:
            Decorator function
        """
        def decorator(cls: type[CallbackAdapter]) -> type[CallbackAdapter]:
            self._adapters[name] = cls
            return cls
        return decorator

    def register_factory(
        self,
        name: str,
        factory: Callable[..., CallbackAdapter],
    ) -> None:
        """Register a factory function.

        Args:
            name: Registration name
            factory: Factory function
        """
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> CallbackAdapter:
        """Create callback by name.

        Args:
            name: Registered name
            **kwargs: Callback arguments

        Returns:
            Created callback

        Raises:
            KeyError: If name not registered
        """
        if name in self._factories:
            return self._factories[name](**kwargs)

        if name in self._adapters:
            return self._adapters[name](**kwargs)

        raise KeyError(f"Unknown callback adapter: {name}")

    def list_adapters(self) -> list[str]:
        """List all registered adapter names."""
        return sorted(set(self._adapters.keys()) | set(self._factories.keys()))

    def get_adapter_class(self, name: str) -> type[CallbackAdapter] | None:
        """Get adapter class by name."""
        return self._adapters.get(name)


# =============================================================================
# Event Emitter
# =============================================================================


class ProgressEmitter:
    """Emits standardized progress events.

    This class is used by profilers to emit progress events
    in a standardized format.

    Example:
        emitter = ProgressEmitter(
            callback=chain,
            operation_id="prof_001",
            table_name="users",
            total_columns=10,
        )

        emitter.start()
        for col in columns:
            emitter.column_start(col)
            # ... profile
            emitter.column_complete(col)
        emitter.complete()
    """

    def __init__(
        self,
        callback: ProgressCallback | CallbackChain | None = None,
        *,
        operation_id: str = "",
        table_name: str = "",
        total_columns: int = 0,
        total_rows: int | None = None,
    ):
        """Initialize emitter.

        Args:
            callback: Callback to receive events
            operation_id: Unique operation identifier
            table_name: Name of table being profiled
            total_columns: Total columns to profile
            total_rows: Total rows (if known)
        """
        self._callback = callback
        self._context = ProgressContext(
            operation_id=operation_id or self._generate_id(),
            table_name=table_name,
        )
        self._total_columns = total_columns
        self._total_rows = total_rows
        self._completed_columns = 0
        self._rows_processed = 0
        self._start_time: datetime | None = None
        self._current_column: str | None = None

    def _generate_id(self) -> str:
        """Generate unique operation ID."""
        import uuid
        return f"op_{uuid.uuid4().hex[:8]}"

    def start(self, message: str = "Starting profiling") -> None:
        """Emit start event."""
        self._start_time = datetime.now()
        self._emit(StandardProgressEvent(
            event_type=EventType.START,
            level=EventLevel.NOTICE,
            progress=0.0,
            message=message,
            context=self._context,
            metrics=self._build_metrics(),
        ))

    def column_start(self, column: str) -> None:
        """Emit column start event."""
        self._current_column = column
        context = self._context.with_column(column)

        self._emit(StandardProgressEvent(
            event_type=EventType.COLUMN_START,
            level=EventLevel.INFO,
            progress=self._calculate_progress(),
            message=f"Starting column: {column}",
            context=context,
            metrics=self._build_metrics(),
        ))

    def column_progress(
        self,
        column: str,
        progress: float,
        *,
        rows: int = 0,
        analyzer: str | None = None,
    ) -> None:
        """Emit column progress event."""
        self._rows_processed += rows
        context = self._context.with_column(column)
        if analyzer:
            context = context.with_analyzer(analyzer)

        self._emit(StandardProgressEvent(
            event_type=EventType.COLUMN_PROGRESS,
            level=EventLevel.DEBUG,
            progress=self._calculate_progress(progress),
            message=f"Profiling {column}" + (f" ({analyzer})" if analyzer else ""),
            context=context,
            metrics=self._build_metrics(),
        ))

    def column_complete(self, column: str) -> None:
        """Emit column complete event."""
        self._completed_columns += 1
        self._current_column = None
        context = self._context.with_column(column)

        self._emit(StandardProgressEvent(
            event_type=EventType.COLUMN_COMPLETE,
            level=EventLevel.INFO,
            progress=self._calculate_progress(),
            message=f"Completed column: {column}",
            context=context,
            metrics=self._build_metrics(),
        ))

    def progress(self, progress: float, message: str = "") -> None:
        """Emit generic progress event."""
        self._emit(StandardProgressEvent(
            event_type=EventType.PROGRESS,
            level=EventLevel.INFO,
            progress=progress,
            message=message,
            context=self._context,
            metrics=self._build_metrics(),
        ))

    def complete(self, message: str = "Profiling complete") -> None:
        """Emit completion event."""
        self._emit(StandardProgressEvent(
            event_type=EventType.COMPLETE,
            level=EventLevel.NOTICE,
            progress=1.0,
            message=message,
            context=self._context,
            metrics=self._build_metrics(),
        ))

    def fail(self, message: str, error: Exception | None = None) -> None:
        """Emit failure event."""
        metadata = {"error": str(error)} if error else {}

        self._emit(StandardProgressEvent(
            event_type=EventType.FAIL,
            level=EventLevel.ERROR,
            progress=self._calculate_progress(),
            message=message,
            context=self._context,
            metrics=self._build_metrics(),
            metadata=metadata,
        ))

    def checkpoint(self, name: str, **metadata: Any) -> None:
        """Emit checkpoint event."""
        self._emit(StandardProgressEvent(
            event_type=EventType.CHECKPOINT,
            level=EventLevel.NOTICE,
            progress=self._calculate_progress(),
            message=f"Checkpoint: {name}",
            context=self._context,
            metrics=self._build_metrics(),
            metadata=metadata,
        ))

    def _emit(self, event: StandardProgressEvent) -> None:
        """Emit event to callback."""
        if self._callback:
            try:
                self._callback.on_progress(event)
            except Exception:
                pass  # Don't let callback errors stop profiling

    def _calculate_progress(self, column_progress: float = 0.0) -> float:
        """Calculate overall progress."""
        if self._total_columns == 0:
            return 0.0

        base = self._completed_columns / self._total_columns
        current = column_progress / self._total_columns
        return min(1.0, base + current)

    def _build_metrics(self) -> ProgressMetrics:
        """Build current metrics."""
        elapsed = 0.0
        if self._start_time:
            elapsed = (datetime.now() - self._start_time).total_seconds()

        rows_per_second = self._rows_processed / elapsed if elapsed > 0 else 0.0

        progress = self._calculate_progress()
        estimated_remaining = None
        if progress > 0 and elapsed > 0:
            total_estimated = elapsed / progress
            estimated_remaining = max(0, total_estimated - elapsed)

        return ProgressMetrics(
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=estimated_remaining,
            rows_processed=self._rows_processed,
            rows_per_second=rows_per_second,
            columns_completed=self._completed_columns,
            columns_total=self._total_columns,
        )


# =============================================================================
# Presets
# =============================================================================


class CallbackPresets:
    """Pre-configured callback setups for common use cases."""

    @staticmethod
    def console_only(
        *,
        show_eta: bool = True,
        color: bool = True,
    ) -> CallbackChain:
        """Console output only."""
        style = ConsoleStyle(show_eta=show_eta, color_enabled=color)
        return CallbackChain([ConsoleAdapter(style=style)])

    @staticmethod
    def logging_only(
        *,
        logger_name: str = "truthound.progress",
        min_level: EventLevel = EventLevel.INFO,
    ) -> CallbackChain:
        """Logging output only."""
        return CallbackChain([
            LoggingAdapter(logger_name=logger_name, min_level=min_level)
        ])

    @staticmethod
    def console_and_logging(
        *,
        logger_name: str = "truthound.progress",
    ) -> CallbackChain:
        """Console and logging output."""
        return CallbackChain([
            ConsoleAdapter(),
            LoggingAdapter(logger_name=logger_name),
        ])

    @staticmethod
    def full_observability(
        *,
        log_file: str | Path,
        logger_name: str = "truthound.progress",
    ) -> CallbackChain:
        """Full observability with console, logging, and file output."""
        return CallbackChain([
            ConsoleAdapter(),
            LoggingAdapter(logger_name=logger_name),
            FileAdapter(log_file),
        ])

    @staticmethod
    def production(
        *,
        logger_name: str = "truthound.progress",
        max_events_per_second: int = 10,
    ) -> CallbackChain:
        """Production setup with throttling."""
        logging_adapter = LoggingAdapter(
            logger_name=logger_name,
            min_level=EventLevel.INFO,
        )

        throttled = ThrottlingAdapter(
            wrapped=logging_adapter,
            config=ThrottleConfig(max_events_per_second=max_events_per_second),
        )

        return CallbackChain([throttled])

    @staticmethod
    def silent() -> CallbackChain:
        """No output (useful for testing)."""
        return CallbackChain([])


# =============================================================================
# Convenience Functions
# =============================================================================


def create_callback_chain(
    *adapters: CallbackAdapter,
    stop_on_error: bool = False,
) -> CallbackChain:
    """Create a callback chain from adapters.

    Args:
        *adapters: Adapters to chain
        stop_on_error: Stop on callback errors

    Returns:
        Configured callback chain
    """
    return CallbackChain(list(adapters), stop_on_error=stop_on_error)


def create_console_callback(
    *,
    bar_width: int = 40,
    show_eta: bool = True,
    color: bool = True,
) -> ConsoleAdapter:
    """Create a console callback.

    Args:
        bar_width: Progress bar width
        show_eta: Show ETA
        color: Enable color

    Returns:
        Console adapter
    """
    style = ConsoleStyle(
        bar_width=bar_width,
        show_eta=show_eta,
        color_enabled=color,
    )
    return ConsoleAdapter(style=style)


def create_logging_callback(
    logger_name: str = "truthound.progress",
    *,
    min_level: EventLevel = EventLevel.INFO,
) -> LoggingAdapter:
    """Create a logging callback.

    Args:
        logger_name: Logger name
        min_level: Minimum event level

    Returns:
        Logging adapter
    """
    return LoggingAdapter(logger_name=logger_name, min_level=min_level)


def create_file_callback(
    path: str | Path,
    *,
    format: str = "jsonl",
) -> FileAdapter:
    """Create a file callback.

    Args:
        path: Output file path
        format: Output format (jsonl, json)

    Returns:
        File adapter
    """
    config = FileOutputConfig(format=format)
    return FileAdapter(path, config=config)


def with_throttling(
    callback: CallbackAdapter,
    *,
    max_per_second: int = 10,
    min_interval_ms: int = 100,
) -> ThrottlingAdapter:
    """Wrap callback with throttling.

    Args:
        callback: Callback to wrap
        max_per_second: Max events per second
        min_interval_ms: Min interval between events

    Returns:
        Throttled callback
    """
    config = ThrottleConfig(
        max_events_per_second=max_per_second,
        min_interval_ms=min_interval_ms,
    )
    return ThrottlingAdapter(wrapped=callback, config=config)


def with_filtering(
    callback: CallbackAdapter,
    *,
    min_level: EventLevel = EventLevel.INFO,
    event_types: set[EventType] | None = None,
) -> FilteringAdapter:
    """Wrap callback with filtering.

    Args:
        callback: Callback to wrap
        min_level: Minimum level to pass
        event_types: Event types to pass

    Returns:
        Filtered callback
    """
    config = FilterConfig(min_level=min_level, event_types=event_types)
    return FilteringAdapter(wrapped=callback, config=config)


def with_buffering(
    callback: CallbackAdapter,
    *,
    max_size: int = 100,
    flush_interval: float = 5.0,
) -> BufferingAdapter:
    """Wrap callback with buffering.

    Args:
        callback: Callback to wrap
        max_size: Max buffer size
        flush_interval: Flush interval in seconds

    Returns:
        Buffered callback
    """
    config = BufferConfig(
        max_size=max_size,
        flush_interval_seconds=flush_interval,
    )
    return BufferingAdapter(wrapped=callback, config=config)
