"""Base classes and configuration for batch writing.

This module defines the data structures and configurations used
for batch write operations.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar


class BatchStatus(str, Enum):
    """Status of a batch operation."""

    PENDING = "pending"  # Batch not yet started
    BUFFERING = "buffering"  # Collecting items
    FLUSHING = "flushing"  # Writing to storage
    COMPLETED = "completed"  # Successfully written
    FAILED = "failed"  # Write failed
    RETRYING = "retrying"  # Retrying after failure


@dataclass
class BatchConfig:
    """Configuration for batch writing.

    Attributes:
        batch_size: Number of items per batch.
        flush_interval_seconds: Auto-flush interval.
        max_buffer_memory_mb: Maximum memory for buffer.
        max_pending_batches: Maximum batches waiting to be written.
        parallelism: Number of parallel write operations.
        compression_enabled: Enable compression for batched data.
        compression_level: Compression level (1-9).
        retry_enabled: Enable automatic retry on failure.
        max_retries: Maximum retry attempts.
        retry_delay_seconds: Initial retry delay.
        retry_backoff_multiplier: Delay multiplier for exponential backoff.
        preserve_order: Preserve insertion order in batches.
        enable_metrics: Collect batch metrics.
    """

    batch_size: int = 1000
    flush_interval_seconds: float = 5.0
    max_buffer_memory_mb: int = 100
    max_pending_batches: int = 10
    parallelism: int = 4
    compression_enabled: bool = True
    compression_level: int = 6
    retry_enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    preserve_order: bool = True
    enable_metrics: bool = True

    def validate(self) -> None:
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.flush_interval_seconds < 0:
            raise ValueError("flush_interval_seconds must be non-negative")
        if self.max_buffer_memory_mb <= 0:
            raise ValueError("max_buffer_memory_mb must be positive")
        if self.max_pending_batches <= 0:
            raise ValueError("max_pending_batches must be positive")
        if self.parallelism <= 0:
            raise ValueError("parallelism must be positive")
        if not 1 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 1 and 9")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if self.retry_backoff_multiplier < 1:
            raise ValueError("retry_backoff_multiplier must be >= 1")


@dataclass
class BatchMetrics:
    """Metrics for batch write operations.

    Attributes:
        total_items: Total items added to batches.
        batches_created: Number of batches created.
        batches_written: Number of batches successfully written.
        batches_failed: Number of batches that failed.
        items_written: Total items successfully written.
        items_failed: Total items that failed to write.
        bytes_written: Total bytes written.
        bytes_compressed: Total bytes after compression.
        flush_count: Number of flush operations.
        retry_count: Number of retry attempts.
        total_write_time_ms: Total time spent writing.
        average_batch_size: Average items per batch.
        average_write_time_ms: Average time per batch write.
        peak_buffer_size: Peak buffer size in items.
        peak_memory_usage_mb: Peak memory usage.
        start_time: When batch operations started.
        last_flush_time: Last flush time.
    """

    total_items: int = 0
    batches_created: int = 0
    batches_written: int = 0
    batches_failed: int = 0
    items_written: int = 0
    items_failed: int = 0
    bytes_written: int = 0
    bytes_compressed: int = 0
    flush_count: int = 0
    retry_count: int = 0
    total_write_time_ms: float = 0.0
    average_batch_size: float = 0.0
    average_write_time_ms: float = 0.0
    peak_buffer_size: int = 0
    peak_memory_usage_mb: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_flush_time: datetime | None = None

    def record_item_added(self, buffer_size: int) -> None:
        """Record an item added to buffer."""
        self.total_items += 1
        if buffer_size > self.peak_buffer_size:
            self.peak_buffer_size = buffer_size

    def record_batch_created(self, size: int) -> None:
        """Record a batch created."""
        self.batches_created += 1

    def record_batch_written(self, size: int, bytes_written: int, time_ms: float) -> None:
        """Record a successful batch write."""
        self.batches_written += 1
        self.items_written += size
        self.bytes_written += bytes_written
        self.total_write_time_ms += time_ms
        self._update_averages()

    def record_batch_failed(self, size: int) -> None:
        """Record a failed batch."""
        self.batches_failed += 1
        self.items_failed += size

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_count += 1

    def record_flush(self) -> None:
        """Record a flush operation."""
        self.flush_count += 1
        self.last_flush_time = datetime.now()

    def record_compression(self, original_size: int, compressed_size: int) -> None:
        """Record compression metrics."""
        self.bytes_compressed += compressed_size

    def record_memory_usage(self, usage_mb: float) -> None:
        """Record memory usage."""
        if usage_mb > self.peak_memory_usage_mb:
            self.peak_memory_usage_mb = usage_mb

    def _update_averages(self) -> None:
        """Update average metrics."""
        if self.batches_written > 0:
            self.average_batch_size = self.items_written / self.batches_written
            self.average_write_time_ms = self.total_write_time_ms / self.batches_written

    def get_compression_ratio(self) -> float:
        """Get compression ratio (0-1, lower is better compression)."""
        if self.bytes_written == 0:
            return 1.0
        return self.bytes_compressed / self.bytes_written

    def get_throughput(self) -> float:
        """Get items per second throughput."""
        if self.total_write_time_ms == 0:
            return 0.0
        return (self.items_written / self.total_write_time_ms) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_items": self.total_items,
            "batches_created": self.batches_created,
            "batches_written": self.batches_written,
            "batches_failed": self.batches_failed,
            "items_written": self.items_written,
            "items_failed": self.items_failed,
            "bytes_written": self.bytes_written,
            "bytes_compressed": self.bytes_compressed,
            "flush_count": self.flush_count,
            "retry_count": self.retry_count,
            "total_write_time_ms": self.total_write_time_ms,
            "average_batch_size": self.average_batch_size,
            "average_write_time_ms": self.average_write_time_ms,
            "peak_buffer_size": self.peak_buffer_size,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "compression_ratio": self.get_compression_ratio(),
            "throughput_items_per_sec": self.get_throughput(),
            "start_time": self.start_time.isoformat(),
            "last_flush_time": (
                self.last_flush_time.isoformat() if self.last_flush_time else None
            ),
        }


T = TypeVar("T")


@dataclass
class Batch(Generic[T]):
    """A batch of items to be written.

    Attributes:
        batch_id: Unique identifier for the batch.
        items: Items in the batch.
        status: Current batch status.
        created_at: When the batch was created.
        size_bytes: Estimated size in bytes.
        retry_count: Number of retry attempts.
        error: Last error if failed.
    """

    batch_id: str
    items: list[T]
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    retry_count: int = 0
    error: str | None = None

    @property
    def size(self) -> int:
        """Get number of items in batch."""
        return len(self.items)

    def mark_flushing(self) -> None:
        """Mark batch as flushing."""
        self.status = BatchStatus.FLUSHING

    def mark_completed(self) -> None:
        """Mark batch as completed."""
        self.status = BatchStatus.COMPLETED

    def mark_failed(self, error: str) -> None:
        """Mark batch as failed."""
        self.status = BatchStatus.FAILED
        self.error = error

    def mark_retrying(self) -> None:
        """Mark batch for retry."""
        self.status = BatchStatus.RETRYING
        self.retry_count += 1


@dataclass
class BatchState:
    """Current state of batch operations.

    Attributes:
        is_active: Whether batch writing is active.
        pending_batches: Number of batches pending write.
        current_buffer_size: Current items in buffer.
        last_flush: Last flush timestamp.
        is_flushing: Whether currently flushing.
    """

    is_active: bool = False
    pending_batches: int = 0
    current_buffer_size: int = 0
    last_flush: datetime | None = None
    is_flushing: bool = False

    def start(self) -> None:
        """Mark as started."""
        self.is_active = True

    def stop(self) -> None:
        """Mark as stopped."""
        self.is_active = False

    def begin_flush(self) -> None:
        """Mark flush as started."""
        self.is_flushing = True

    def end_flush(self) -> None:
        """Mark flush as completed."""
        self.is_flushing = False
        self.last_flush = datetime.now()

    def add_pending(self) -> None:
        """Increment pending batches."""
        self.pending_batches += 1

    def remove_pending(self) -> None:
        """Decrement pending batches."""
        self.pending_batches = max(0, self.pending_batches - 1)

    def update_buffer_size(self, size: int) -> None:
        """Update current buffer size."""
        self.current_buffer_size = size
