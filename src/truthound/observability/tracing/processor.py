"""Span processors for trace data handling.

Span processors receive spans as they start and end, allowing for
various processing strategies like batching, filtering, and export.

Processor Types:
    - SimpleSpanProcessor: Synchronous export on span end
    - BatchSpanProcessor: Batch export for performance
    - MultiSpanProcessor: Fan-out to multiple processors
"""

from __future__ import annotations

import atexit
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from truthound.observability.tracing.span import Span
    from truthound.observability.tracing.exporter import SpanExporter


# =============================================================================
# Processor Interface
# =============================================================================


class SpanProcessor(ABC):
    """Abstract base class for span processors.

    Span processors receive notifications when spans start and end,
    and are responsible for forwarding spans to exporters.
    """

    @abstractmethod
    def on_start(self, span: "Span", parent_context: Any = None) -> None:
        """Called when a span starts.

        Args:
            span: The starting span.
            parent_context: Parent context (if any).
        """
        pass

    @abstractmethod
    def on_end(self, span: "Span") -> None:
        """Called when a span ends.

        Args:
            span: The ended span.
        """
        pass

    @abstractmethod
    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown the processor.

        Args:
            timeout_millis: Maximum time to wait for shutdown.

        Returns:
            True if shutdown completed within timeout.
        """
        pass

    @abstractmethod
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all pending spans.

        Args:
            timeout_millis: Maximum time to wait for flush.

        Returns:
            True if flush completed within timeout.
        """
        pass


# =============================================================================
# Simple Span Processor
# =============================================================================


class SimpleSpanProcessor(SpanProcessor):
    """Synchronous span processor.

    Exports spans immediately when they end. This is simple and
    guarantees no data loss, but adds latency to every span.

    Best for:
        - Development and testing
        - Low-volume services
        - When immediate export is required

    Example:
        >>> exporter = ConsoleSpanExporter()
        >>> processor = SimpleSpanProcessor(exporter)
        >>> provider = TracerProvider(processors=[processor])
    """

    def __init__(self, exporter: "SpanExporter") -> None:
        """Initialize with exporter.

        Args:
            exporter: The span exporter to use.
        """
        self._exporter = exporter
        self._shutdown = False
        self._lock = threading.Lock()

    def on_start(self, span: "Span", parent_context: Any = None) -> None:
        """Called when span starts (no-op for simple processor)."""
        pass

    def on_end(self, span: "Span") -> None:
        """Export span immediately on end."""
        if self._shutdown:
            return

        with self._lock:
            if self._shutdown:
                return
            try:
                self._exporter.export([span])
            except Exception:
                pass  # Log error but don't fail

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown processor and exporter."""
        with self._lock:
            self._shutdown = True

        try:
            self._exporter.shutdown()
            return True
        except Exception:
            return False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush (no-op for simple processor)."""
        return True


# =============================================================================
# Batch Span Processor
# =============================================================================


@dataclass
class BatchConfig:
    """Configuration for batch span processor."""

    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    scheduled_delay_millis: int = 5000
    export_timeout_millis: int = 30000

    @classmethod
    def default(cls) -> "BatchConfig":
        """Get default configuration."""
        return cls()

    @classmethod
    def development(cls) -> "BatchConfig":
        """Get development configuration (faster flush)."""
        return cls(
            max_queue_size=256,
            max_export_batch_size=32,
            scheduled_delay_millis=1000,
        )

    @classmethod
    def production(cls) -> "BatchConfig":
        """Get production configuration (optimized)."""
        return cls(
            max_queue_size=4096,
            max_export_batch_size=512,
            scheduled_delay_millis=5000,
        )


class BatchSpanProcessor(SpanProcessor):
    """Batching span processor for high-performance export.

    Collects spans in a queue and exports them in batches, either
    when the batch is full or after a scheduled delay.

    Features:
        - Background export thread
        - Configurable batch size and timing
        - Graceful shutdown with flush
        - Overflow handling

    Best for:
        - Production environments
        - High-volume services
        - When export latency matters

    Example:
        >>> exporter = OTLPSpanExporter("http://collector:4317")
        >>> processor = BatchSpanProcessor(
        ...     exporter,
        ...     config=BatchConfig(
        ...         max_export_batch_size=256,
        ...         scheduled_delay_millis=2000,
        ...     ),
        ... )
        >>> provider = TracerProvider(processors=[processor])
    """

    def __init__(
        self,
        exporter: "SpanExporter",
        config: BatchConfig | None = None,
    ) -> None:
        """Initialize batch processor.

        Args:
            exporter: The span exporter to use.
            config: Batch configuration.
        """
        self._exporter = exporter
        self._config = config or BatchConfig.default()

        self._queue: queue.Queue["Span"] = queue.Queue(
            maxsize=self._config.max_queue_size
        )
        self._shutdown_event = threading.Event()
        self._flush_event = threading.Event()
        self._export_lock = threading.Lock()

        # Stats
        self._dropped_spans = 0
        self._exported_spans = 0

        # Start background thread
        self._worker = threading.Thread(
            target=self._export_loop,
            daemon=True,
            name="BatchSpanProcessor-Worker",
        )
        self._worker.start()

        # Register shutdown handler
        atexit.register(self._atexit_handler)

    def _atexit_handler(self) -> None:
        """Handle process exit."""
        self.shutdown(timeout_millis=5000)

    @property
    def dropped_spans(self) -> int:
        """Get count of dropped spans."""
        return self._dropped_spans

    @property
    def exported_spans(self) -> int:
        """Get count of exported spans."""
        return self._exported_spans

    def on_start(self, span: "Span", parent_context: Any = None) -> None:
        """Called when span starts (no-op for batch processor)."""
        pass

    def on_end(self, span: "Span") -> None:
        """Add span to queue for batch export."""
        if self._shutdown_event.is_set():
            return

        try:
            self._queue.put_nowait(span)
        except queue.Full:
            self._dropped_spans += 1

    def _export_loop(self) -> None:
        """Background export loop."""
        while not self._shutdown_event.is_set():
            # Wait for flush signal or scheduled delay
            self._flush_event.wait(
                timeout=self._config.scheduled_delay_millis / 1000
            )
            self._flush_event.clear()

            self._export_batch()

        # Final export on shutdown
        self._export_batch()

    def _export_batch(self) -> None:
        """Export a batch of spans."""
        batch: list["Span"] = []

        # Drain up to max_export_batch_size spans
        while len(batch) < self._config.max_export_batch_size:
            try:
                span = self._queue.get_nowait()
                batch.append(span)
            except queue.Empty:
                break

        if not batch:
            return

        with self._export_lock:
            try:
                self._exporter.export(batch)
                self._exported_spans += len(batch)
            except Exception:
                pass  # Log error but continue

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown processor."""
        self._shutdown_event.set()
        self._flush_event.set()

        # Wait for worker to finish
        self._worker.join(timeout=timeout_millis / 1000)

        try:
            self._exporter.shutdown()
            return True
        except Exception:
            return False

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans."""
        if self._shutdown_event.is_set():
            return False

        self._flush_event.set()

        # Wait for export to complete
        deadline = time.time() + timeout_millis / 1000

        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.01)

        return self._queue.empty()


# =============================================================================
# Multi Span Processor
# =============================================================================


class MultiSpanProcessor(SpanProcessor):
    """Processor that fans out to multiple processors.

    Useful for sending spans to multiple destinations
    (e.g., console for debugging + OTLP for production).

    Example:
        >>> console_processor = SimpleSpanProcessor(ConsoleSpanExporter())
        >>> otlp_processor = BatchSpanProcessor(OTLPSpanExporter(...))
        >>> multi = MultiSpanProcessor([console_processor, otlp_processor])
        >>> provider = TracerProvider(processors=[multi])
    """

    def __init__(self, processors: list[SpanProcessor] | None = None) -> None:
        """Initialize with list of processors.

        Args:
            processors: List of span processors.
        """
        self._processors = list(processors or [])
        self._lock = threading.Lock()

    def add_processor(self, processor: SpanProcessor) -> None:
        """Add a processor.

        Args:
            processor: Processor to add.
        """
        with self._lock:
            self._processors.append(processor)

    def remove_processor(self, processor: SpanProcessor) -> bool:
        """Remove a processor.

        Args:
            processor: Processor to remove.

        Returns:
            True if removed.
        """
        with self._lock:
            try:
                self._processors.remove(processor)
                return True
            except ValueError:
                return False

    def on_start(self, span: "Span", parent_context: Any = None) -> None:
        """Forward to all processors."""
        with self._lock:
            for processor in self._processors:
                try:
                    processor.on_start(span, parent_context)
                except Exception:
                    pass

    def on_end(self, span: "Span") -> None:
        """Forward to all processors."""
        with self._lock:
            for processor in self._processors:
                try:
                    processor.on_end(span)
                except Exception:
                    pass

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown all processors."""
        success = True
        per_processor_timeout = timeout_millis // max(len(self._processors), 1)

        with self._lock:
            for processor in self._processors:
                try:
                    if not processor.shutdown(per_processor_timeout):
                        success = False
                except Exception:
                    success = False

        return success

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all processors."""
        success = True
        per_processor_timeout = timeout_millis // max(len(self._processors), 1)

        with self._lock:
            for processor in self._processors:
                try:
                    if not processor.force_flush(per_processor_timeout):
                        success = False
                except Exception:
                    success = False

        return success


# =============================================================================
# Filtering Span Processor
# =============================================================================


class FilteringSpanProcessor(SpanProcessor):
    """Processor that filters spans before forwarding.

    Useful for:
        - Dropping noisy spans (health checks, etc.)
        - Sampling based on span properties
        - Routing spans to different processors

    Example:
        >>> def should_export(span):
        ...     # Don't export health check spans
        ...     return span.name != "GET /health"
        >>>
        >>> processor = FilteringSpanProcessor(
        ...     delegate=BatchSpanProcessor(exporter),
        ...     filter_fn=should_export,
        ... )
    """

    def __init__(
        self,
        delegate: SpanProcessor,
        filter_fn: callable,
    ) -> None:
        """Initialize filtering processor.

        Args:
            delegate: Processor to forward matching spans to.
            filter_fn: Function that returns True to export span.
        """
        self._delegate = delegate
        self._filter_fn = filter_fn

    def on_start(self, span: "Span", parent_context: Any = None) -> None:
        """Forward start to delegate."""
        self._delegate.on_start(span, parent_context)

    def on_end(self, span: "Span") -> None:
        """Filter and forward to delegate."""
        try:
            if self._filter_fn(span):
                self._delegate.on_end(span)
        except Exception:
            pass

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown delegate."""
        return self._delegate.shutdown(timeout_millis)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush delegate."""
        return self._delegate.force_flush(timeout_millis)


# =============================================================================
# Span Enriching Processor
# =============================================================================


class EnrichingSpanProcessor(SpanProcessor):
    """Processor that enriches spans with additional attributes.

    Useful for:
        - Adding common attributes (service version, environment)
        - Computing derived attributes
        - Normalizing span data

    Example:
        >>> def enrich(span):
        ...     span.set_attribute("environment", "production")
        ...     span.set_attribute("service.version", "1.2.3")
        >>>
        >>> processor = EnrichingSpanProcessor(
        ...     delegate=BatchSpanProcessor(exporter),
        ...     enricher_fn=enrich,
        ... )
    """

    def __init__(
        self,
        delegate: SpanProcessor,
        enricher_fn: callable,
    ) -> None:
        """Initialize enriching processor.

        Args:
            delegate: Processor to forward enriched spans to.
            enricher_fn: Function to enrich spans.
        """
        self._delegate = delegate
        self._enricher_fn = enricher_fn

    def on_start(self, span: "Span", parent_context: Any = None) -> None:
        """Enrich span on start."""
        try:
            self._enricher_fn(span)
        except Exception:
            pass
        self._delegate.on_start(span, parent_context)

    def on_end(self, span: "Span") -> None:
        """Forward to delegate."""
        self._delegate.on_end(span)

    def shutdown(self, timeout_millis: int = 30000) -> bool:
        """Shutdown delegate."""
        return self._delegate.shutdown(timeout_millis)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush delegate."""
        return self._delegate.force_flush(timeout_millis)
