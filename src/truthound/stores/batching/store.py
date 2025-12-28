"""Batched store wrapper implementation.

This module provides a wrapper around storage backends that
adds batch writing capabilities.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.stores.batching.base import (
    BatchConfig,
    BatchMetrics,
    BatchState,
)
from truthound.stores.batching.writer import AsyncBatchWriter

if TYPE_CHECKING:
    from truthound.stores.base import ValidationStore
    from truthound.stores.results import ValidationResult


T = TypeVar("T")
ConfigT = TypeVar("ConfigT")


class BatchedStore(Generic[T, ConfigT]):
    """Store wrapper with batch writing support.

    Wraps an underlying store to provide batch writing capabilities,
    reducing I/O overhead for high-throughput scenarios.

    Example:
        >>> from truthound.stores import FileSystemStore
        >>> from truthound.stores.batching import BatchedStore, BatchConfig
        >>>
        >>> inner_store = FileSystemStore()
        >>> config = BatchConfig(batch_size=1000, flush_interval_seconds=5.0)
        >>> batched = BatchedStore(inner_store, config)
        >>>
        >>> async with batched:
        ...     for result in results:
        ...         await batched.add(result)
    """

    def __init__(
        self,
        store: "ValidationStore[ConfigT]",
        config: BatchConfig | None = None,
    ) -> None:
        """Initialize batched store.

        Args:
            store: Underlying store to wrap.
            config: Batch configuration.
        """
        self._store = store
        self._config = config or BatchConfig()
        self._config.validate()

        self._metrics = BatchMetrics()
        self._state = BatchState()
        self._writer: AsyncBatchWriter | None = None
        self._lock = asyncio.Lock()

    @property
    def store(self) -> "ValidationStore[ConfigT]":
        """Get underlying store."""
        return self._store

    @property
    def config(self) -> BatchConfig:
        """Get batch configuration."""
        return self._config

    @property
    def metrics(self) -> BatchMetrics:
        """Get batch metrics."""
        if self._writer:
            return self._writer.metrics
        return self._metrics

    @property
    def state(self) -> BatchState:
        """Get current state."""
        if self._writer:
            return self._writer.state
        return self._state

    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        if self._writer:
            return self._writer.buffer_size
        return 0

    def _serialize_result(self, result: "ValidationResult") -> dict[str, Any]:
        """Serialize a validation result."""
        return result.to_dict()

    def _write_batch(self, items: list["ValidationResult"]) -> None:
        """Write a batch of results to the underlying store."""
        for item in items:
            self._store.save(item)

    async def start(self) -> None:
        """Start the batched store."""
        if self._writer is not None:
            return

        self._writer = AsyncBatchWriter(
            write_func=self._write_batch,
            config=self._config,
        )
        self._state.start()
        await self._writer.start_auto_flush()

    async def stop(self) -> None:
        """Stop the batched store and flush remaining items."""
        if self._writer is None:
            return

        await self._writer.close()
        self._writer = None
        self._state.stop()

    async def add(self, item: T) -> bool:
        """Add an item to the batch buffer.

        Args:
            item: Item to add.

        Returns:
            True if added successfully.
        """
        if self._writer is None:
            await self.start()

        return await self._writer.add(item)

    async def add_many(self, items: list[T]) -> int:
        """Add multiple items to the batch buffer.

        Args:
            items: Items to add.

        Returns:
            Number of items added.
        """
        if self._writer is None:
            await self.start()

        return await self._writer.add_many(items)

    async def flush(self) -> int:
        """Flush pending batches to storage.

        Returns:
            Number of batches written.
        """
        if self._writer is None:
            return 0

        return await self._writer.flush()

    # Passthrough methods to underlying store

    def get(self, item_id: str) -> T:
        """Get an item from the underlying store.

        Args:
            item_id: Item identifier.

        Returns:
            The item.
        """
        return self._store.get(item_id)

    def exists(self, item_id: str) -> bool:
        """Check if an item exists in the underlying store.

        Args:
            item_id: Item identifier.

        Returns:
            True if exists.
        """
        return self._store.exists(item_id)

    def delete(self, item_id: str) -> bool:
        """Delete an item from the underlying store.

        Args:
            item_id: Item identifier.

        Returns:
            True if deleted.
        """
        return self._store.delete(item_id)

    def list_ids(self, **kwargs: Any) -> list[str]:
        """List item IDs in the underlying store.

        Returns:
            List of item IDs.
        """
        return self._store.list_ids(**kwargs)

    def save(self, item: T) -> str:
        """Save an item directly (bypassing batch).

        For immediate writes that shouldn't be batched.

        Args:
            item: Item to save.

        Returns:
            Item ID.
        """
        return self._store.save(item)

    def get_stats(self) -> dict[str, Any]:
        """Get batched store statistics."""
        return {
            "config": {
                "batch_size": self._config.batch_size,
                "flush_interval_seconds": self._config.flush_interval_seconds,
                "max_buffer_memory_mb": self._config.max_buffer_memory_mb,
                "parallelism": self._config.parallelism,
            },
            "metrics": self.metrics.to_dict(),
            "state": {
                "is_active": self.state.is_active,
                "pending_batches": self.state.pending_batches,
                "buffer_size": self.buffer_size,
                "is_flushing": self.state.is_flushing,
            },
        }

    async def __aenter__(self) -> "BatchedStore[T, ConfigT]":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
