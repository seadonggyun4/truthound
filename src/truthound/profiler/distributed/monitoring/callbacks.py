"""Callback adapters for distributed monitoring.

This module provides callback adapters for receiving and processing
monitoring events from the distributed profiling system.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from truthound.profiler.distributed.monitoring.protocols import (
    EventSeverity,
    IMonitorCallback,
    MonitorEvent,
    MonitorEventType,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Base Callback Adapter
# =============================================================================


class MonitorCallbackAdapter(ABC, IMonitorCallback):
    """Abstract base class for monitor callback adapters.

    Provides common functionality including filtering, throttling,
    and lifecycle management.
    """

    def __init__(
        self,
        *,
        name: str = "",
        enabled: bool = True,
        min_severity: EventSeverity = EventSeverity.INFO,
        event_types: set[MonitorEventType] | None = None,
        throttle_ms: int = 0,
    ) -> None:
        """Initialize adapter.

        Args:
            name: Adapter name for identification
            enabled: Whether adapter is active
            min_severity: Minimum event severity to handle
            event_types: Event types to handle (None = all)
            throttle_ms: Minimum ms between events (0 = no throttle)
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled
        self.min_severity = min_severity
        self.event_types = event_types
        self.throttle_ms = throttle_ms

        self._started = False
        self._last_event_time: float = 0

    def should_handle(self, event: MonitorEvent) -> bool:
        """Check if event should be handled.

        Args:
            event: Event to check

        Returns:
            True if event should be processed
        """
        if not self.enabled:
            return False

        if event.severity.value < self.min_severity.value:
            return False

        if self.event_types and event.event_type not in self.event_types:
            return False

        # Throttle check
        if self.throttle_ms > 0:
            now = time.time() * 1000
            if now - self._last_event_time < self.throttle_ms:
                return False
            self._last_event_time = now

        return True

    def on_event(self, event: MonitorEvent) -> None:
        """Handle monitoring event with filtering.

        Args:
            event: Monitoring event
        """
        if not self.should_handle(event):
            return

        self._handle_event(event)

    @abstractmethod
    def _handle_event(self, event: MonitorEvent) -> None:
        """Handle the event (implemented by subclasses).

        Args:
            event: Monitoring event to handle
        """
        pass

    def start(self) -> None:
        """Start the adapter."""
        self._started = True

    def stop(self) -> None:
        """Stop the adapter."""
        self._started = False


# =============================================================================
# Console Callback
# =============================================================================


@dataclass
class ConsoleStyle:
    """Console output styling configuration."""

    show_timestamp: bool = True
    show_severity: bool = True
    show_event_type: bool = True
    show_progress: bool = True
    color_enabled: bool = True
    timestamp_format: str = "%H:%M:%S"

    # ANSI color codes
    colors: dict[EventSeverity, str] = field(
        default_factory=lambda: {
            EventSeverity.DEBUG: "\033[90m",  # Gray
            EventSeverity.INFO: "\033[0m",  # Default
            EventSeverity.WARNING: "\033[93m",  # Yellow
            EventSeverity.ERROR: "\033[91m",  # Red
            EventSeverity.CRITICAL: "\033[91;1m",  # Bold Red
        }
    )
    reset: str = "\033[0m"


class ConsoleMonitorCallback(MonitorCallbackAdapter):
    """Console output callback for monitoring events.

    Provides formatted console output with colors and severity indicators.

    Example:
        callback = ConsoleMonitorCallback(
            style=ConsoleStyle(show_progress=True),
            min_severity=EventSeverity.INFO,
        )
        monitor.add_callback(callback)
    """

    def __init__(
        self,
        *,
        style: ConsoleStyle | None = None,
        stream: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize console callback.

        Args:
            style: Console styling configuration
            stream: Output stream (default: sys.stderr)
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.style = style or ConsoleStyle()
        self._stream = stream

    @property
    def stream(self) -> Any:
        """Get output stream."""
        if self._stream is None:
            import sys

            return sys.stderr
        return self._stream

    def _handle_event(self, event: MonitorEvent) -> None:
        """Handle event with console output."""
        parts = []
        s = self.style

        # Color prefix
        color = ""
        if s.color_enabled:
            color = s.colors.get(event.severity, "")

        # Timestamp
        if s.show_timestamp:
            ts = event.timestamp.strftime(s.timestamp_format)
            parts.append(f"[{ts}]")

        # Severity
        if s.show_severity:
            severity_char = event.severity.name[0]  # D, I, W, E, C
            parts.append(f"[{severity_char}]")

        # Event type
        if s.show_event_type:
            event_type = event.event_type.value.replace("_", " ").title()
            parts.append(f"[{event_type}]")

        # Message
        parts.append(event.message)

        # Progress
        if s.show_progress and event.progress > 0:
            parts.append(f"({event.progress * 100:.1f}%)")

        # Worker/Task context
        if event.worker_id:
            parts.append(f"[worker:{event.worker_id}]")
        if event.task_id:
            parts.append(f"[task:{event.task_id}]")

        line = " ".join(parts)
        print(f"{color}{line}{s.reset}", file=self.stream)


class ProgressBarCallback(MonitorCallbackAdapter):
    """Progress bar callback for monitoring events.

    Displays a live progress bar for distributed profiling operations.
    """

    def __init__(
        self,
        *,
        width: int = 40,
        show_eta: bool = True,
        show_throughput: bool = True,
        stream: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize progress bar callback.

        Args:
            width: Progress bar width
            show_eta: Show estimated time remaining
            show_throughput: Show throughput
            stream: Output stream
            **kwargs: Base adapter arguments
        """
        # Only handle progress events
        super().__init__(
            event_types={
                MonitorEventType.PROGRESS_UPDATE,
                MonitorEventType.PROGRESS_MILESTONE,
                MonitorEventType.AGGREGATION_COMPLETE,
            },
            **kwargs,
        )
        self.width = width
        self.show_eta = show_eta
        self.show_throughput = show_throughput
        self._stream = stream
        self._last_line_length = 0

    @property
    def stream(self) -> Any:
        """Get output stream."""
        if self._stream is None:
            import sys

            return sys.stderr
        return self._stream

    def _handle_event(self, event: MonitorEvent) -> None:
        """Handle event with progress bar."""
        if event.event_type == MonitorEventType.AGGREGATION_COMPLETE:
            # Clear progress bar and show completion
            print("\r" + " " * self._last_line_length + "\r", end="", file=self.stream)
            elapsed = event.metadata.get("elapsed_seconds", 0)
            print(f"✓ Complete in {elapsed:.1f}s", file=self.stream)
            return

        # Build progress bar
        progress = event.progress
        filled = int(progress * self.width)
        bar = "█" * filled + "░" * (self.width - filled)

        parts = [f"\r[{bar}] {progress * 100:5.1f}%"]

        # Throughput
        if self.show_throughput:
            rows_per_sec = event.metadata.get("rows_per_second", 0)
            if rows_per_sec > 0:
                if rows_per_sec >= 1_000_000:
                    parts.append(f" | {rows_per_sec / 1_000_000:.1f}M rows/s")
                elif rows_per_sec >= 1_000:
                    parts.append(f" | {rows_per_sec / 1_000:.1f}K rows/s")
                else:
                    parts.append(f" | {rows_per_sec:.0f} rows/s")

        # ETA
        if self.show_eta:
            eta = event.metadata.get("estimated_remaining_seconds")
            if eta is not None:
                if eta < 60:
                    parts.append(f" | ETA: {eta:.0f}s")
                elif eta < 3600:
                    mins, secs = divmod(int(eta), 60)
                    parts.append(f" | ETA: {mins}m {secs}s")
                else:
                    hours, remainder = divmod(int(eta), 3600)
                    mins = remainder // 60
                    parts.append(f" | ETA: {hours}h {mins}m")

        line = "".join(parts)

        # Pad to overwrite previous line
        if len(line) < self._last_line_length:
            line += " " * (self._last_line_length - len(line))
        self._last_line_length = len(line)

        print(line, end="", flush=True, file=self.stream)


# =============================================================================
# Logging Callback
# =============================================================================


class LoggingMonitorCallback(MonitorCallbackAdapter):
    """Logging framework callback for monitoring events.

    Routes monitoring events to Python's logging framework.

    Example:
        callback = LoggingMonitorCallback(
            logger_name="truthound.distributed",
            min_severity=EventSeverity.INFO,
        )
    """

    # Map event severity to logging levels
    LEVEL_MAP = {
        EventSeverity.DEBUG: logging.DEBUG,
        EventSeverity.INFO: logging.INFO,
        EventSeverity.WARNING: logging.WARNING,
        EventSeverity.ERROR: logging.ERROR,
        EventSeverity.CRITICAL: logging.CRITICAL,
    }

    def __init__(
        self,
        *,
        logger_name: str = "truthound.distributed",
        log_instance: logging.Logger | None = None,
        include_metadata: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize logging callback.

        Args:
            logger_name: Logger name to use
            log_instance: Existing logger instance
            include_metadata: Include metadata in log extras
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self._logger = log_instance or logging.getLogger(logger_name)
        self.include_metadata = include_metadata

    def _handle_event(self, event: MonitorEvent) -> None:
        """Log the event."""
        level = self.LEVEL_MAP.get(event.severity, logging.INFO)

        extra: dict[str, Any] = {
            "event_type": event.event_type.value,
            "progress": event.progress,
        }

        if event.task_id:
            extra["task_id"] = event.task_id
        if event.worker_id:
            extra["worker_id"] = event.worker_id
        if event.partition_id is not None:
            extra["partition_id"] = event.partition_id

        if self.include_metadata and event.metadata:
            extra["metadata"] = event.metadata

        self._logger.log(level, event.message, extra=extra)


# =============================================================================
# File Callback
# =============================================================================


class FileMonitorCallback(MonitorCallbackAdapter):
    """File output callback for monitoring events.

    Writes events to a file in JSON Lines format for later analysis.

    Example:
        callback = FileMonitorCallback(
            path="monitoring.jsonl",
            rotate_size_mb=100,
        )
    """

    def __init__(
        self,
        path: str | Path,
        *,
        rotate_size_mb: int = 0,
        compress_rotated: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize file callback.

        Args:
            path: Output file path
            rotate_size_mb: Rotate when file reaches this size (0 = no rotation)
            compress_rotated: Compress rotated files
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.path = Path(path)
        self.rotate_size_mb = rotate_size_mb
        self.compress_rotated = compress_rotated
        self._file: Any = None

    def start(self) -> None:
        """Open file for writing."""
        super().start()
        self._file = open(self.path, "a", encoding="utf-8")

    def stop(self) -> None:
        """Close file."""
        super().stop()
        if self._file:
            self._file.close()
            self._file = None

    def _handle_event(self, event: MonitorEvent) -> None:
        """Write event to file."""
        if self._file is None:
            return

        event_dict = event.to_dict()
        self._file.write(json.dumps(event_dict) + "\n")
        self._file.flush()

        # Check for rotation
        if self.rotate_size_mb > 0:
            size_mb = self.path.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotate_size_mb:
                self._rotate()

    def _rotate(self) -> None:
        """Rotate log file."""
        if self._file:
            self._file.close()

        # Rename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = self.path.with_suffix(f".{timestamp}.jsonl")
        self.path.rename(rotated_path)

        # Compress if enabled
        if self.compress_rotated:
            import gzip

            with open(rotated_path, "rb") as f_in:
                with gzip.open(f"{rotated_path}.gz", "wb") as f_out:
                    f_out.writelines(f_in)
            rotated_path.unlink()

        # Reopen main file
        self._file = open(self.path, "a", encoding="utf-8")


# =============================================================================
# Webhook Callback
# =============================================================================


class WebhookMonitorCallback(MonitorCallbackAdapter):
    """Webhook callback for monitoring events.

    Sends events to an HTTP endpoint for external processing.

    Example:
        callback = WebhookMonitorCallback(
            url="https://api.example.com/events",
            headers={"Authorization": "Bearer token"},
            batch_size=10,
        )
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 10.0,
        batch_size: int = 1,
        batch_timeout_seconds: float = 5.0,
        **kwargs: Any,
    ) -> None:
        """Initialize webhook callback.

        Args:
            url: Webhook URL
            headers: HTTP headers
            timeout_seconds: Request timeout
            batch_size: Events per batch (1 = no batching)
            batch_timeout_seconds: Max time to wait for batch
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        self._batch: list[dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._last_flush_time = time.time()

    def _handle_event(self, event: MonitorEvent) -> None:
        """Handle event with batching."""
        with self._batch_lock:
            self._batch.append(event.to_dict())

            should_flush = (
                len(self._batch) >= self.batch_size
                or time.time() - self._last_flush_time >= self.batch_timeout_seconds
            )

            if should_flush:
                self._flush_batch()

    def _flush_batch(self) -> None:
        """Send batched events to webhook."""
        if not self._batch:
            return

        events = self._batch
        self._batch = []
        self._last_flush_time = time.time()

        # Send async to not block
        threading.Thread(target=self._send_events, args=(events,), daemon=True).start()

    def _send_events(self, events: list[dict[str, Any]]) -> None:
        """Send events to webhook.

        Args:
            events: Events to send
        """
        try:
            import urllib.request

            payload = json.dumps({"events": events}).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                **self.headers,
            }

            req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            urllib.request.urlopen(req, timeout=self.timeout_seconds)
        except Exception as e:
            logger.warning(f"Failed to send events to webhook: {e}")

    def stop(self) -> None:
        """Flush remaining events and stop."""
        with self._batch_lock:
            self._flush_batch()
        super().stop()


# =============================================================================
# Callback Chain
# =============================================================================


class CallbackChain(IMonitorCallback):
    """Chains multiple callbacks together.

    Events are dispatched to all callbacks in the chain.

    Example:
        chain = CallbackChain()
        chain.add(console_callback)
        chain.add(logging_callback)
        chain.add(file_callback)

        monitor.set_callback(chain)
    """

    def __init__(
        self,
        callbacks: Sequence[MonitorCallbackAdapter] | None = None,
        *,
        stop_on_error: bool = False,
    ) -> None:
        """Initialize callback chain.

        Args:
            callbacks: Initial callbacks
            stop_on_error: Stop chain on callback error
        """
        self._callbacks: list[MonitorCallbackAdapter] = list(callbacks or [])
        self.stop_on_error = stop_on_error
        self._errors: list[tuple[MonitorCallbackAdapter, Exception]] = []

    def add(self, callback: MonitorCallbackAdapter) -> "CallbackChain":
        """Add callback to chain.

        Args:
            callback: Callback to add

        Returns:
            Self for chaining
        """
        self._callbacks.append(callback)
        return self

    def remove(self, callback: MonitorCallbackAdapter) -> bool:
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

    def on_event(self, event: MonitorEvent) -> None:
        """Dispatch event to all callbacks.

        Args:
            event: Event to dispatch
        """
        for callback in self._callbacks:
            try:
                callback.on_event(event)
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
    def errors(self) -> list[tuple[MonitorCallbackAdapter, Exception]]:
        """Get errors from callbacks."""
        return self._errors.copy()

    def __len__(self) -> int:
        return len(self._callbacks)

    def __iter__(self) -> Iterator[MonitorCallbackAdapter]:
        return iter(self._callbacks)


# =============================================================================
# Async Callback
# =============================================================================


class AsyncMonitorCallback(MonitorCallbackAdapter):
    """Asynchronous callback that processes events in a background thread.

    Useful for callbacks that may block (e.g., network calls).

    Example:
        async_callback = AsyncMonitorCallback(
            wrapped=WebhookMonitorCallback(url="..."),
            buffer_size=1000,
        )
    """

    def __init__(
        self,
        wrapped: MonitorCallbackAdapter,
        *,
        buffer_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize async callback.

        Args:
            wrapped: Callback to wrap
            buffer_size: Event queue size
            **kwargs: Base adapter arguments
        """
        super().__init__(**kwargs)
        self.wrapped = wrapped
        self._queue: queue.Queue[MonitorEvent | None] = queue.Queue(maxsize=buffer_size)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start async processing thread."""
        super().start()
        if hasattr(self.wrapped, "start"):
            self.wrapped.start()

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop async processing."""
        self._running = False
        self._queue.put(None)  # Signal to stop

        if self._thread:
            self._thread.join(timeout=5.0)

        if hasattr(self.wrapped, "stop"):
            self.wrapped.stop()

        super().stop()

    def _handle_event(self, event: MonitorEvent) -> None:
        """Queue event for async processing."""
        if self._running:
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                logger.warning("Async callback queue full, dropping event")

    def _process_loop(self) -> None:
        """Process events from queue."""
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
                if event is None:
                    break
                self.wrapped.on_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Error in async callback: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_console_callback(
    *,
    color: bool = True,
    show_timestamp: bool = True,
    min_severity: EventSeverity = EventSeverity.INFO,
) -> ConsoleMonitorCallback:
    """Create a console callback.

    Args:
        color: Enable color output
        show_timestamp: Show timestamps
        min_severity: Minimum severity

    Returns:
        Console callback
    """
    style = ConsoleStyle(color_enabled=color, show_timestamp=show_timestamp)
    return ConsoleMonitorCallback(style=style, min_severity=min_severity)


def create_logging_callback(
    logger_name: str = "truthound.distributed",
    *,
    min_severity: EventSeverity = EventSeverity.INFO,
) -> LoggingMonitorCallback:
    """Create a logging callback.

    Args:
        logger_name: Logger name
        min_severity: Minimum severity

    Returns:
        Logging callback
    """
    return LoggingMonitorCallback(logger_name=logger_name, min_severity=min_severity)


def create_file_callback(
    path: str | Path,
    *,
    min_severity: EventSeverity = EventSeverity.INFO,
) -> FileMonitorCallback:
    """Create a file callback.

    Args:
        path: Output file path
        min_severity: Minimum severity

    Returns:
        File callback
    """
    return FileMonitorCallback(path, min_severity=min_severity)


def create_callback_chain(*callbacks: MonitorCallbackAdapter) -> CallbackChain:
    """Create a callback chain.

    Args:
        *callbacks: Callbacks to chain

    Returns:
        Callback chain
    """
    return CallbackChain(list(callbacks))
