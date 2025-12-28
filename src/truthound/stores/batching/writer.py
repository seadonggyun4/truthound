"""Batch writer implementations.

This module provides batch writer classes for efficiently writing
batches of items to storage backends.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generic, TypeVar

from truthound.stores.batching.base import (
    Batch,
    BatchConfig,
    BatchMetrics,
    BatchState,
    BatchStatus,
)
from truthound.stores.batching.buffer import MemoryAwareBuffer


T = TypeVar("T")


class BatchWriterError(Exception):
    """Error during batch writing."""

    def __init__(self, message: str, batch_id: str | None = None) -> None:
        self.batch_id = batch_id
        super().__init__(message)


class BatchWriter(Generic[T]):
    """Synchronous batch writer.

    Collects items into batches and writes them to storage using
    a provided write function.

    Example:
        >>> def write_batch(items: list[dict]) -> None:
        ...     store.save_many(items)
        >>>
        >>> writer = BatchWriter(write_func=write_batch, batch_size=500)
        >>> for item in items:
        ...     writer.add(item)
        >>> writer.flush()
        >>> writer.close()
    """

    def __init__(
        self,
        write_func: Callable[[list[T]], None],
        config: BatchConfig | None = None,
        serializer: Callable[[T], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize batch writer.

        Args:
            write_func: Function to write a batch of items.
            config: Batch configuration.
            serializer: Optional function to serialize items.
        """
        self._write_func = write_func
        self._config = config or BatchConfig()
        self._config.validate()
        self._serializer = serializer

        self._buffer = MemoryAwareBuffer[T](
            max_size=self._config.batch_size * 10,
            max_memory_mb=self._config.max_buffer_memory_mb,
        )
        self._metrics = BatchMetrics()
        self._state = BatchState()
        self._pending_batches: list[Batch[T]] = []

    @property
    def config(self) -> BatchConfig:
        """Get configuration."""
        return self._config

    @property
    def metrics(self) -> BatchMetrics:
        """Get metrics."""
        return self._metrics

    @property
    def state(self) -> BatchState:
        """Get current state."""
        return self._state

    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return self._buffer.size

    def add(self, item: T) -> bool:
        """Add an item to the buffer.

        Args:
            item: Item to add.

        Returns:
            True if added successfully.
        """
        if not self._state.is_active:
            self._state.start()

        result = self._buffer.add(item)
        if result:
            self._metrics.record_item_added(self._buffer.size)
            self._state.update_buffer_size(self._buffer.size)

            # Auto-flush if buffer reaches batch size
            if self._buffer.size >= self._config.batch_size:
                self._create_batch()

        return result

    def add_many(self, items: list[T]) -> int:
        """Add multiple items to the buffer.

        Args:
            items: Items to add.

        Returns:
            Number of items added.
        """
        added = 0
        for item in items:
            if self.add(item):
                added += 1
        return added

    def _create_batch(self) -> Batch[T] | None:
        """Create a batch from buffered items."""
        items = self._buffer.drain(self._config.batch_size)
        if not items:
            return None

        batch = Batch[T](
            batch_id=str(uuid.uuid4()),
            items=items,
            status=BatchStatus.PENDING,
        )
        self._metrics.record_batch_created(len(items))
        self._pending_batches.append(batch)
        self._state.add_pending()
        return batch

    def _write_batch(self, batch: Batch[T]) -> bool:
        """Write a single batch.

        Args:
            batch: Batch to write.

        Returns:
            True if successful.
        """
        batch.mark_flushing()
        start_time = time.monotonic()

        try:
            items_to_write = batch.items
            if self._serializer:
                items_to_write = [self._serializer(item) for item in batch.items]

            self._write_func(items_to_write)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            batch.mark_completed()
            self._metrics.record_batch_written(batch.size, batch.size_bytes, elapsed_ms)
            return True

        except Exception as e:
            batch.mark_failed(str(e))
            return False

    def _retry_batch(self, batch: Batch[T]) -> bool:
        """Retry writing a failed batch.

        Args:
            batch: Batch to retry.

        Returns:
            True if successful.
        """
        if batch.retry_count >= self._config.max_retries:
            self._metrics.record_batch_failed(batch.size)
            return False

        delay = self._config.retry_delay_seconds * (
            self._config.retry_backoff_multiplier ** batch.retry_count
        )
        time.sleep(delay)

        batch.mark_retrying()
        self._metrics.record_retry()

        return self._write_batch(batch)

    def flush(self) -> int:
        """Flush all pending batches.

        Returns:
            Number of batches successfully written.
        """
        self._state.begin_flush()

        # Create batch from remaining buffer
        while self._buffer.size > 0:
            self._create_batch()

        written = 0
        failed_batches = []

        for batch in self._pending_batches:
            success = self._write_batch(batch)

            if not success and self._config.retry_enabled:
                success = self._retry_batch(batch)

            if success:
                written += 1
                self._state.remove_pending()
            else:
                failed_batches.append(batch)

        self._pending_batches = failed_batches
        self._metrics.record_flush()
        self._state.end_flush()

        return written

    def close(self) -> None:
        """Close the writer and flush remaining items."""
        if self._state.is_active:
            self.flush()
            self._state.stop()

    def __enter__(self) -> "BatchWriter[T]":
        """Context manager entry."""
        self._state.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


class AsyncBatchWriter(Generic[T]):
    """Asynchronous batch writer with parallel processing.

    Collects items into batches and writes them asynchronously
    with configurable parallelism.

    Example:
        >>> async def write_batch(items: list[dict]) -> None:
        ...     await store.save_many_async(items)
        >>>
        >>> writer = AsyncBatchWriter(write_func=write_batch)
        >>> async with writer:
        ...     for item in items:
        ...         await writer.add(item)
        ...     await writer.flush()
    """

    def __init__(
        self,
        write_func: Callable[[list[T]], Any],
        config: BatchConfig | None = None,
        serializer: Callable[[T], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize async batch writer.

        Args:
            write_func: Async function to write a batch of items.
            config: Batch configuration.
            serializer: Optional function to serialize items.
        """
        self._write_func = write_func
        self._config = config or BatchConfig()
        self._config.validate()
        self._serializer = serializer

        self._buffer = MemoryAwareBuffer[T](
            max_size=self._config.batch_size * 10,
            max_memory_mb=self._config.max_buffer_memory_mb,
        )
        self._metrics = BatchMetrics()
        self._state = BatchState()
        self._pending_batches: list[Batch[T]] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._config.parallelism)
        self._flush_task: asyncio.Task | None = None
        self._auto_flush_task: asyncio.Task | None = None

    @property
    def config(self) -> BatchConfig:
        """Get configuration."""
        return self._config

    @property
    def metrics(self) -> BatchMetrics:
        """Get metrics."""
        return self._metrics

    @property
    def state(self) -> BatchState:
        """Get current state."""
        return self._state

    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return self._buffer.size

    async def add(self, item: T) -> bool:
        """Add an item to the buffer.

        Args:
            item: Item to add.

        Returns:
            True if added successfully.
        """
        async with self._lock:
            if not self._state.is_active:
                self._state.start()

            result = self._buffer.add(item)
            if result:
                self._metrics.record_item_added(self._buffer.size)
                self._state.update_buffer_size(self._buffer.size)

                # Auto-flush if buffer reaches batch size
                if self._buffer.size >= self._config.batch_size:
                    await self._create_batch()

            return result

    async def add_many(self, items: list[T]) -> int:
        """Add multiple items to the buffer.

        Args:
            items: Items to add.

        Returns:
            Number of items added.
        """
        added = 0
        for item in items:
            if await self.add(item):
                added += 1
        return added

    async def _create_batch(self) -> Batch[T] | None:
        """Create a batch from buffered items."""
        items = self._buffer.drain(self._config.batch_size)
        if not items:
            return None

        batch = Batch[T](
            batch_id=str(uuid.uuid4()),
            items=items,
            status=BatchStatus.PENDING,
        )
        self._metrics.record_batch_created(len(items))
        self._pending_batches.append(batch)
        self._state.add_pending()
        return batch

    async def _write_batch(self, batch: Batch[T]) -> bool:
        """Write a single batch asynchronously.

        Args:
            batch: Batch to write.

        Returns:
            True if successful.
        """
        async with self._semaphore:
            batch.mark_flushing()
            start_time = time.monotonic()

            try:
                items_to_write = batch.items
                if self._serializer:
                    items_to_write = [self._serializer(item) for item in batch.items]

                # Support both sync and async write functions
                if asyncio.iscoroutinefunction(self._write_func):
                    await self._write_func(items_to_write)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._write_func, items_to_write
                    )

                elapsed_ms = (time.monotonic() - start_time) * 1000
                batch.mark_completed()
                self._metrics.record_batch_written(
                    batch.size, batch.size_bytes, elapsed_ms
                )
                return True

            except Exception as e:
                batch.mark_failed(str(e))
                return False

    async def _retry_batch(self, batch: Batch[T]) -> bool:
        """Retry writing a failed batch.

        Args:
            batch: Batch to retry.

        Returns:
            True if successful.
        """
        if batch.retry_count >= self._config.max_retries:
            self._metrics.record_batch_failed(batch.size)
            return False

        delay = self._config.retry_delay_seconds * (
            self._config.retry_backoff_multiplier ** batch.retry_count
        )
        await asyncio.sleep(delay)

        batch.mark_retrying()
        self._metrics.record_retry()

        return await self._write_batch(batch)

    async def flush(self) -> int:
        """Flush all pending batches.

        Returns:
            Number of batches successfully written.
        """
        async with self._lock:
            self._state.begin_flush()

            # Create batches from remaining buffer
            while self._buffer.size > 0:
                await self._create_batch()

            if not self._pending_batches:
                self._state.end_flush()
                return 0

            # Write batches in parallel
            tasks = [self._write_batch(batch) for batch in self._pending_batches]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            written = 0
            failed_batches = []

            for batch, result in zip(self._pending_batches, results):
                if result is True:
                    written += 1
                    self._state.remove_pending()
                elif self._config.retry_enabled:
                    # Retry failed batches
                    success = await self._retry_batch(batch)
                    if success:
                        written += 1
                        self._state.remove_pending()
                    else:
                        failed_batches.append(batch)
                else:
                    failed_batches.append(batch)

            self._pending_batches = failed_batches
            self._metrics.record_flush()
            self._state.end_flush()

            return written

    async def _auto_flush_loop(self) -> None:
        """Background task for auto-flushing."""
        while self._state.is_active:
            await asyncio.sleep(self._config.flush_interval_seconds)
            if self._buffer.size > 0 and not self._state.is_flushing:
                await self.flush()

    async def start_auto_flush(self) -> None:
        """Start background auto-flush task."""
        if self._auto_flush_task is None or self._auto_flush_task.done():
            self._auto_flush_task = asyncio.create_task(self._auto_flush_loop())

    async def stop_auto_flush(self) -> None:
        """Stop background auto-flush task."""
        if self._auto_flush_task:
            self._auto_flush_task.cancel()
            try:
                await self._auto_flush_task
            except asyncio.CancelledError:
                pass
            self._auto_flush_task = None

    async def close(self) -> None:
        """Close the writer and flush remaining items."""
        if self._state.is_active:
            await self.stop_auto_flush()
            await self.flush()
            self._state.stop()

    async def __aenter__(self) -> "AsyncBatchWriter[T]":
        """Async context manager entry."""
        self._state.start()
        await self.start_auto_flush()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
