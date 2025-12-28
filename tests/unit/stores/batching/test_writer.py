"""Tests for batch writer module."""

from __future__ import annotations

import asyncio
import pytest
from typing import Any

from truthound.stores.batching.base import BatchConfig
from truthound.stores.batching.writer import (
    AsyncBatchWriter,
    BatchWriter,
)


class TestBatchWriter:
    """Tests for BatchWriter."""

    def test_creation(self) -> None:
        """Test writer creation."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        writer = BatchWriter(write_func=write_func)
        assert writer.buffer_size == 0

    def test_add_item(self) -> None:
        """Test adding item."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        writer = BatchWriter(write_func=write_func)
        result = writer.add({"key": "value"})
        assert result is True
        assert writer.buffer_size == 1

    def test_auto_batch_creation(self) -> None:
        """Test automatic batch creation at batch_size."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=5)
        writer = BatchWriter(write_func=write_func, config=config)

        for i in range(5):
            writer.add({"i": i})

        # Should have created a batch but not flushed yet
        assert writer.state.pending_batches == 1

    def test_flush(self) -> None:
        """Test manual flush."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=10)
        writer = BatchWriter(write_func=write_func, config=config)

        for i in range(3):
            writer.add({"i": i})

        batches_written = writer.flush()
        assert batches_written == 1
        assert len(written_items) == 3

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=10)

        with BatchWriter(write_func=write_func, config=config) as writer:
            for i in range(5):
                writer.add({"i": i})

        # Should flush on exit
        assert len(written_items) == 5

    def test_serializer(self) -> None:
        """Test custom serializer."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        def serializer(item: Any) -> dict:
            return {"serialized": str(item)}

        config = BatchConfig(batch_size=10)
        writer = BatchWriter(
            write_func=write_func, config=config, serializer=serializer
        )

        writer.add(123)
        writer.flush()

        assert written_items[0] == {"serialized": "123"}

    def test_metrics(self) -> None:
        """Test metrics collection."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=10)
        writer = BatchWriter(write_func=write_func, config=config)

        for i in range(5):
            writer.add({"i": i})
        writer.flush()

        metrics = writer.metrics
        assert metrics.total_items == 5
        assert metrics.batches_created == 1
        assert metrics.batches_written == 1
        assert metrics.items_written == 5
        assert metrics.flush_count == 1

    def test_retry_on_failure(self) -> None:
        """Test retry on failure."""
        call_count = 0
        written_items = []

        def write_func(items: list[dict]) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            written_items.extend(items)

        config = BatchConfig(
            batch_size=10,
            retry_enabled=True,
            max_retries=3,
            retry_delay_seconds=0.01,
        )
        writer = BatchWriter(write_func=write_func, config=config)

        writer.add({"key": "value"})
        batches_written = writer.flush()

        assert batches_written == 1
        assert len(written_items) == 1
        assert writer.metrics.retry_count >= 1


class TestAsyncBatchWriter:
    """Tests for AsyncBatchWriter."""

    @pytest.mark.asyncio
    async def test_creation(self) -> None:
        """Test async writer creation."""
        written_items = []

        async def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        writer = AsyncBatchWriter(write_func=write_func)
        assert writer.buffer_size == 0

    @pytest.mark.asyncio
    async def test_add_item(self) -> None:
        """Test adding item asynchronously."""
        written_items = []

        async def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        writer = AsyncBatchWriter(write_func=write_func)
        result = await writer.add({"key": "value"})
        assert result is True
        assert writer.buffer_size == 1

    @pytest.mark.asyncio
    async def test_flush(self) -> None:
        """Test async flush."""
        written_items = []

        async def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=10)
        writer = AsyncBatchWriter(write_func=write_func, config=config)

        for i in range(3):
            await writer.add({"i": i})

        batches_written = await writer.flush()
        assert batches_written == 1
        assert len(written_items) == 3

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager usage."""
        written_items = []

        async def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=10, flush_interval_seconds=1.0)

        async with AsyncBatchWriter(write_func=write_func, config=config) as writer:
            for i in range(5):
                await writer.add({"i": i})

        # Should flush on exit
        assert len(written_items) == 5

    @pytest.mark.asyncio
    async def test_parallel_writes(self) -> None:
        """Test parallel batch writes."""
        write_times = []

        async def write_func(items: list[dict]) -> None:
            import time

            start = time.monotonic()
            await asyncio.sleep(0.05)  # Simulate I/O
            write_times.append(time.monotonic() - start)

        config = BatchConfig(batch_size=3, parallelism=4)
        writer = AsyncBatchWriter(write_func=write_func, config=config)

        # Create enough items for multiple batches
        for i in range(12):
            await writer.add({"i": i})

        await writer.flush()

        # With parallelism, total time should be less than sequential
        # 4 batches of 3 items each, with parallelism=4, should be ~1 wait time

    @pytest.mark.asyncio
    async def test_sync_write_func(self) -> None:
        """Test with synchronous write function."""
        written_items = []

        def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=10)
        writer = AsyncBatchWriter(write_func=write_func, config=config)

        await writer.add({"key": "value"})
        await writer.flush()

        assert len(written_items) == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test async retry on failure."""
        call_count = 0
        written_items = []

        async def write_func(items: list[dict]) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            written_items.extend(items)

        config = BatchConfig(
            batch_size=10,
            retry_enabled=True,
            max_retries=3,
            retry_delay_seconds=0.01,
        )
        writer = AsyncBatchWriter(write_func=write_func, config=config)

        await writer.add({"key": "value"})
        batches_written = await writer.flush()

        assert batches_written == 1
        assert len(written_items) == 1

    @pytest.mark.asyncio
    async def test_metrics(self) -> None:
        """Test async metrics collection."""
        written_items = []

        async def write_func(items: list[dict]) -> None:
            written_items.extend(items)

        config = BatchConfig(batch_size=10)
        writer = AsyncBatchWriter(write_func=write_func, config=config)

        for i in range(5):
            await writer.add({"i": i})
        await writer.flush()

        metrics = writer.metrics
        assert metrics.total_items == 5
        assert metrics.batches_created == 1
        assert metrics.batches_written == 1
