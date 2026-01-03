"""Buffer implementations for batch writing.

This module provides buffer implementations for collecting items
before batching, with support for memory management.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class BufferConfig:
    """Configuration for buffer.

    Attributes:
        max_size: Maximum number of items.
        max_memory_mb: Maximum memory usage in MB.
        enable_memory_tracking: Track memory usage.
    """

    max_size: int = 10000
    max_memory_mb: float = 100.0
    enable_memory_tracking: bool = True


class BatchBuffer(Generic[T]):
    """Thread-safe buffer for collecting items before batching.

    Provides a simple FIFO buffer with configurable size limits.

    Example:
        >>> buffer = BatchBuffer[dict](max_size=1000)
        >>> buffer.add({"key": "value"})
        >>> batch = buffer.drain(100)
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize buffer.

        Args:
            max_size: Maximum number of items in buffer.
        """
        self._max_size = max_size
        self._items: deque[T] = deque()
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Get current number of items."""
        with self._lock:
            return len(self._items)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.size >= self._max_size

    def add(self, item: T) -> bool:
        """Add an item to the buffer.

        Args:
            item: Item to add.

        Returns:
            True if added, False if buffer full.
        """
        with self._lock:
            if len(self._items) >= self._max_size:
                return False
            self._items.append(item)
            return True

    def add_many(self, items: list[T]) -> int:
        """Add multiple items to the buffer.

        Args:
            items: Items to add.

        Returns:
            Number of items actually added.
        """
        with self._lock:
            space_available = self._max_size - len(self._items)
            items_to_add = items[:space_available]
            self._items.extend(items_to_add)
            return len(items_to_add)

    def drain(self, count: int) -> list[T]:
        """Remove and return items from buffer.

        Args:
            count: Maximum number of items to drain.

        Returns:
            List of drained items.
        """
        with self._lock:
            result = []
            for _ in range(min(count, len(self._items))):
                result.append(self._items.popleft())
            return result

    def drain_all(self) -> list[T]:
        """Remove and return all items from buffer."""
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items

    def peek(self, count: int) -> list[T]:
        """View items without removing.

        Args:
            count: Number of items to peek.

        Returns:
            List of items (copies).
        """
        with self._lock:
            return list(self._items)[:count]

    def clear(self) -> int:
        """Clear all items from buffer.

        Returns:
            Number of items cleared.
        """
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def __len__(self) -> int:
        """Get number of items."""
        return self.size

    def __iter__(self) -> Iterator[T]:
        """Iterate over items (drains buffer)."""
        while not self.is_empty:
            items = self.drain(100)
            yield from items


class MemoryAwareBuffer(Generic[T]):
    """Buffer with memory usage tracking and limits.

    Tracks approximate memory usage of items and enforces
    memory limits in addition to count limits.

    Example:
        >>> buffer = MemoryAwareBuffer[dict](max_size=10000, max_memory_mb=50.0)
        >>> buffer.add({"key": "value"})
        >>> print(f"Memory usage: {buffer.memory_usage_mb:.2f} MB")
    """

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 100.0,
        size_estimator: callable | None = None,
    ) -> None:
        """Initialize memory-aware buffer.

        Args:
            max_size: Maximum number of items.
            max_memory_mb: Maximum memory usage in MB.
            size_estimator: Optional function to estimate item size.
        """
        self._max_size = max_size
        self._max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._size_estimator = size_estimator or self._default_size_estimator
        self._items: deque[tuple[T, int]] = deque()  # (item, size_bytes)
        self._current_memory_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _default_size_estimator(item: Any) -> int:
        """Estimate size of an item in bytes."""
        return sys.getsizeof(item)

    @property
    def size(self) -> int:
        """Get current number of items."""
        with self._lock:
            return len(self._items)

    @property
    def memory_usage_bytes(self) -> int:
        """Get current memory usage in bytes."""
        with self._lock:
            return self._current_memory_bytes

    @property
    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.memory_usage_bytes / (1024 * 1024)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.size == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        with self._lock:
            return (
                len(self._items) >= self._max_size
                or self._current_memory_bytes >= self._max_memory_bytes
            )

    @property
    def memory_pressure(self) -> float:
        """Get memory pressure as percentage (0-100)."""
        with self._lock:
            return (self._current_memory_bytes / self._max_memory_bytes) * 100

    def can_add(self, item: T) -> bool:
        """Check if an item can be added without exceeding limits."""
        if self.is_full:
            return False

        size = self._size_estimator(item)
        with self._lock:
            return (
                self._current_memory_bytes + size <= self._max_memory_bytes
                and len(self._items) < self._max_size
            )

    def add(self, item: T) -> bool:
        """Add an item to the buffer.

        Args:
            item: Item to add.

        Returns:
            True if added, False if would exceed limits.
        """
        size = self._size_estimator(item)

        with self._lock:
            if len(self._items) >= self._max_size:
                return False
            if self._current_memory_bytes + size > self._max_memory_bytes:
                return False

            self._items.append((item, size))
            self._current_memory_bytes += size
            return True

    def add_many(self, items: list[T]) -> int:
        """Add multiple items to the buffer.

        Args:
            items: Items to add.

        Returns:
            Number of items actually added.
        """
        added = 0
        for item in items:
            if self.add(item):
                added += 1
            else:
                break
        return added

    def drain(self, count: int) -> list[T]:
        """Remove and return items from buffer.

        Args:
            count: Maximum number of items to drain.

        Returns:
            List of drained items.
        """
        with self._lock:
            result = []
            for _ in range(min(count, len(self._items))):
                item, size = self._items.popleft()
                self._current_memory_bytes -= size
                result.append(item)
            return result

    def drain_all(self) -> list[T]:
        """Remove and return all items from buffer."""
        with self._lock:
            items = [item for item, _ in self._items]
            self._items.clear()
            self._current_memory_bytes = 0
            return items

    def drain_bytes(self, max_bytes: int) -> list[T]:
        """Drain items up to a byte limit.

        Args:
            max_bytes: Maximum bytes to drain.

        Returns:
            List of drained items.
        """
        with self._lock:
            result = []
            total_bytes = 0

            while self._items and total_bytes < max_bytes:
                item, size = self._items[0]
                if total_bytes + size > max_bytes and result:
                    break
                self._items.popleft()
                self._current_memory_bytes -= size
                total_bytes += size
                result.append(item)

            return result

    def peek(self, count: int) -> list[T]:
        """View items without removing."""
        with self._lock:
            return [item for item, _ in list(self._items)[:count]]

    def clear(self) -> int:
        """Clear all items from buffer."""
        with self._lock:
            count = len(self._items)
            self._items.clear()
            self._current_memory_bytes = 0
            return count

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            is_full = (
                len(self._items) >= self._max_size
                or self._current_memory_bytes >= self._max_memory_bytes
            )
            return {
                "size": len(self._items),
                "max_size": self._max_size,
                "memory_bytes": self._current_memory_bytes,
                "max_memory_bytes": self._max_memory_bytes,
                "memory_usage_percent": (
                    self._current_memory_bytes / self._max_memory_bytes * 100
                    if self._max_memory_bytes > 0
                    else 0
                ),
                "is_full": is_full,
            }

    def __len__(self) -> int:
        """Get number of items."""
        return self.size

    def __iter__(self) -> Iterator[T]:
        """Iterate over items (drains buffer)."""
        while not self.is_empty:
            items = self.drain(100)
            yield from items
