"""Tests for batching buffer module."""

from __future__ import annotations

import pytest

from truthound.stores.batching.buffer import (
    BatchBuffer,
    MemoryAwareBuffer,
)


class TestBatchBuffer:
    """Tests for BatchBuffer."""

    def test_creation(self) -> None:
        """Test buffer creation."""
        buffer = BatchBuffer[int](max_size=100)
        assert buffer.size == 0
        assert buffer.is_empty is True
        assert buffer.is_full is False

    def test_add_item(self) -> None:
        """Test adding item."""
        buffer = BatchBuffer[int](max_size=100)
        result = buffer.add(1)
        assert result is True
        assert buffer.size == 1
        assert buffer.is_empty is False

    def test_add_when_full(self) -> None:
        """Test adding when buffer is full."""
        buffer = BatchBuffer[int](max_size=3)
        buffer.add(1)
        buffer.add(2)
        buffer.add(3)
        assert buffer.is_full is True

        result = buffer.add(4)
        assert result is False
        assert buffer.size == 3

    def test_add_many(self) -> None:
        """Test adding multiple items."""
        buffer = BatchBuffer[int](max_size=10)
        added = buffer.add_many([1, 2, 3, 4, 5])
        assert added == 5
        assert buffer.size == 5

    def test_add_many_partial(self) -> None:
        """Test adding more items than space."""
        buffer = BatchBuffer[int](max_size=3)
        added = buffer.add_many([1, 2, 3, 4, 5])
        assert added == 3
        assert buffer.size == 3

    def test_drain(self) -> None:
        """Test draining items."""
        buffer = BatchBuffer[int](max_size=10)
        buffer.add_many([1, 2, 3, 4, 5])

        items = buffer.drain(3)
        assert items == [1, 2, 3]
        assert buffer.size == 2

    def test_drain_more_than_available(self) -> None:
        """Test draining more than available."""
        buffer = BatchBuffer[int](max_size=10)
        buffer.add_many([1, 2, 3])

        items = buffer.drain(10)
        assert items == [1, 2, 3]
        assert buffer.size == 0

    def test_drain_all(self) -> None:
        """Test draining all items."""
        buffer = BatchBuffer[int](max_size=10)
        buffer.add_many([1, 2, 3, 4, 5])

        items = buffer.drain_all()
        assert items == [1, 2, 3, 4, 5]
        assert buffer.is_empty is True

    def test_peek(self) -> None:
        """Test peeking items without removing."""
        buffer = BatchBuffer[int](max_size=10)
        buffer.add_many([1, 2, 3, 4, 5])

        items = buffer.peek(3)
        assert items == [1, 2, 3]
        assert buffer.size == 5  # Not removed

    def test_clear(self) -> None:
        """Test clearing buffer."""
        buffer = BatchBuffer[int](max_size=10)
        buffer.add_many([1, 2, 3, 4, 5])

        count = buffer.clear()
        assert count == 5
        assert buffer.is_empty is True

    def test_len(self) -> None:
        """Test len() function."""
        buffer = BatchBuffer[int](max_size=10)
        buffer.add_many([1, 2, 3])
        assert len(buffer) == 3

    def test_iteration(self) -> None:
        """Test iterating over buffer."""
        buffer = BatchBuffer[int](max_size=10)
        buffer.add_many([1, 2, 3, 4, 5])

        items = list(buffer)
        assert items == [1, 2, 3, 4, 5]
        assert buffer.is_empty is True


class TestMemoryAwareBuffer:
    """Tests for MemoryAwareBuffer."""

    def test_creation(self) -> None:
        """Test buffer creation."""
        buffer = MemoryAwareBuffer[dict](max_size=100, max_memory_mb=10.0)
        assert buffer.size == 0
        assert buffer.memory_usage_bytes == 0
        assert buffer.is_empty is True

    def test_add_item(self) -> None:
        """Test adding item with memory tracking."""
        buffer = MemoryAwareBuffer[dict](max_size=100, max_memory_mb=10.0)
        result = buffer.add({"key": "value"})
        assert result is True
        assert buffer.size == 1
        assert buffer.memory_usage_bytes > 0

    def test_memory_limit(self) -> None:
        """Test memory limit enforcement."""
        # Very small memory limit
        buffer = MemoryAwareBuffer[bytes](
            max_size=1000,
            max_memory_mb=0.001,  # ~1KB
        )

        # Add items until memory limit
        large_item = b"x" * 512
        result1 = buffer.add(large_item)
        assert result1 is True

        result2 = buffer.add(large_item)
        assert result2 is False  # Should fail due to memory

    def test_can_add(self) -> None:
        """Test can_add check."""
        buffer = MemoryAwareBuffer[dict](max_size=2, max_memory_mb=10.0)
        buffer.add({"a": 1})
        buffer.add({"b": 2})

        # Buffer full by count
        assert buffer.can_add({"c": 3}) is False

    def test_drain(self) -> None:
        """Test draining with memory tracking."""
        buffer = MemoryAwareBuffer[dict](max_size=100, max_memory_mb=10.0)
        buffer.add_many([{"a": 1}, {"b": 2}, {"c": 3}])

        initial_memory = buffer.memory_usage_bytes
        items = buffer.drain(2)
        assert len(items) == 2
        assert buffer.memory_usage_bytes < initial_memory

    def test_drain_bytes(self) -> None:
        """Test draining by byte limit."""
        buffer = MemoryAwareBuffer[bytes](max_size=100, max_memory_mb=10.0)
        # Add items of known size
        # Note: sys.getsizeof includes Python object overhead (~33 bytes for bytes)
        buffer.add(b"a" * 100)  # ~133 bytes with overhead
        buffer.add(b"b" * 100)
        buffer.add(c := b"c" * 100)

        # Drain up to enough bytes for at least 2 items
        # sys.getsizeof(b"x" * 100) is ~133 bytes, so 300 bytes should get 2
        items = buffer.drain_bytes(300)
        assert len(items) >= 2

    def test_memory_pressure(self) -> None:
        """Test memory pressure calculation."""
        buffer = MemoryAwareBuffer[bytes](
            max_size=1000,
            max_memory_mb=0.01,  # 10KB
        )

        # Add item
        buffer.add(b"x" * 1024)  # 1KB
        pressure = buffer.memory_pressure
        assert 5 <= pressure <= 20  # Approximately 10% with overhead

    def test_get_stats(self) -> None:
        """Test getting buffer statistics."""
        buffer = MemoryAwareBuffer[dict](max_size=100, max_memory_mb=10.0)
        buffer.add_many([{"a": 1}, {"b": 2}])

        stats = buffer.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["memory_bytes"] > 0
        assert "memory_usage_percent" in stats
        assert "is_full" in stats

    def test_custom_size_estimator(self) -> None:
        """Test custom size estimator."""

        def fixed_size(item: dict) -> int:
            return 100  # Fixed 100 bytes per item

        buffer = MemoryAwareBuffer[dict](
            max_size=1000,
            max_memory_mb=0.001,  # ~1KB
            size_estimator=fixed_size,
        )

        # Should fit about 10 items (1KB / 100 bytes)
        for i in range(15):
            buffer.add({"i": i})

        assert buffer.size <= 10
