"""Priority-based execution for validators.

This module provides priority-based execution ordering for validators,
ensuring critical validations complete first within time constraints.

Features:
- Priority levels (CRITICAL, HIGH, MEDIUM, LOW)
- Time-aware scheduling
- Preemption support
- Deadline-aware execution

Example:
    from truthound.validators.timeout.advanced.priority import (
        PriorityExecutor,
        ValidationPriority,
        execute_by_priority,
    )

    executor = PriorityExecutor()

    # Add validators with priorities
    executor.add("null_check", ValidationPriority.CRITICAL, null_validator)
    executor.add("format_check", ValidationPriority.LOW, format_validator)

    # Execute in priority order
    results = executor.execute_all(deadline_seconds=30)
"""

from __future__ import annotations

import asyncio
import heapq
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class ValidationPriority(IntEnum):
    """Priority levels for validators.

    Lower value = higher priority.
    """

    CRITICAL = 0    # Must complete, affects data integrity
    HIGH = 10       # Important for business logic
    MEDIUM = 50     # Standard validations
    LOW = 100       # Nice-to-have checks
    BACKGROUND = 200  # Can be skipped under time pressure


@dataclass(order=True)
class PriorityItem(Generic[T]):
    """Item in the priority queue.

    Ordering is based on (priority, enqueue_time) for FIFO within same priority.
    """

    priority: int
    enqueue_time: float = field(compare=True)
    name: str = field(compare=False)
    operation: Callable[[], T] = field(compare=False)
    timeout_ms: float = field(compare=False, default=5000.0)
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        priority: ValidationPriority,
        operation: Callable[[], T],
        timeout_ms: float = 5000.0,
        metadata: dict[str, Any] | None = None,
    ) -> "PriorityItem[T]":
        """Create a priority item.

        Args:
            name: Operation name
            priority: Priority level
            operation: Operation to execute
            timeout_ms: Timeout in milliseconds
            metadata: Additional metadata

        Returns:
            PriorityItem
        """
        return cls(
            priority=int(priority),
            enqueue_time=time.time(),
            name=name,
            operation=operation,
            timeout_ms=timeout_ms,
            metadata=metadata or {},
        )


class PriorityQueue(Generic[T]):
    """Thread-safe priority queue.

    Provides priority-based ordering with FIFO within same priority.
    """

    def __init__(self) -> None:
        """Initialize priority queue."""
        self._heap: list[PriorityItem[T]] = []
        self._lock = threading.Lock()
        self._counter = 0

    def push(self, item: PriorityItem[T]) -> None:
        """Add an item to the queue.

        Args:
            item: Item to add
        """
        with self._lock:
            heapq.heappush(self._heap, item)
            self._counter += 1

    def pop(self) -> PriorityItem[T] | None:
        """Remove and return highest priority item.

        Returns:
            Highest priority item or None if empty
        """
        with self._lock:
            if self._heap:
                return heapq.heappop(self._heap)
            return None

    def peek(self) -> PriorityItem[T] | None:
        """Return highest priority item without removing.

        Returns:
            Highest priority item or None if empty
        """
        with self._lock:
            if self._heap:
                return self._heap[0]
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        with self._lock:
            return len(self._heap) == 0

    def clear(self) -> None:
        """Clear the queue."""
        with self._lock:
            self._heap.clear()
            self._counter = 0

    def get_by_priority(self, priority: ValidationPriority) -> list[PriorityItem[T]]:
        """Get all items with specific priority.

        Args:
            priority: Priority level

        Returns:
            List of matching items
        """
        with self._lock:
            return [item for item in self._heap if item.priority == int(priority)]


@dataclass
class ExecutionResult(Generic[T]):
    """Result of a priority-based execution.

    Attributes:
        name: Operation name
        priority: Priority level used
        value: Result value
        success: Whether execution succeeded
        duration_ms: Execution time in milliseconds
        error: Error message if failed
        skipped: Whether execution was skipped
        skip_reason: Reason for skipping
    """

    name: str
    priority: int
    value: T | None = None
    success: bool = True
    duration_ms: float = 0.0
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @classmethod
    def success_result(
        cls,
        name: str,
        priority: int,
        value: T,
        duration_ms: float,
    ) -> "ExecutionResult[T]":
        """Create success result."""
        return cls(
            name=name,
            priority=priority,
            value=value,
            success=True,
            duration_ms=duration_ms,
            completed_at=datetime.now(timezone.utc),
        )

    @classmethod
    def failure_result(
        cls,
        name: str,
        priority: int,
        error: str,
        duration_ms: float,
    ) -> "ExecutionResult[T]":
        """Create failure result."""
        return cls(
            name=name,
            priority=priority,
            success=False,
            error=error,
            duration_ms=duration_ms,
            completed_at=datetime.now(timezone.utc),
        )

    @classmethod
    def skipped_result(
        cls,
        name: str,
        priority: int,
        reason: str,
    ) -> "ExecutionResult[T]":
        """Create skipped result."""
        return cls(
            name=name,
            priority=priority,
            success=True,
            skipped=True,
            skip_reason=reason,
            completed_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "priority": self.priority,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class PriorityConfig:
    """Configuration for priority executor.

    Attributes:
        min_priority_to_execute: Minimum priority that must execute
        skip_low_priority_under_pressure: Skip low priority under time pressure
        time_pressure_threshold: Remaining time fraction for pressure
        preemption_enabled: Enable preemption of lower priority tasks
        max_concurrent: Maximum concurrent executions
    """

    min_priority_to_execute: ValidationPriority = ValidationPriority.CRITICAL
    skip_low_priority_under_pressure: bool = True
    time_pressure_threshold: float = 0.2  # 20% time remaining
    preemption_enabled: bool = False
    max_concurrent: int = 1


class PriorityExecutor(Generic[T]):
    """Executor for priority-based validation.

    Executes validators in priority order, respecting time constraints
    and optionally skipping lower priority items under time pressure.

    Example:
        executor = PriorityExecutor()

        executor.add("critical_check", ValidationPriority.CRITICAL, check_fn)
        executor.add("optional_check", ValidationPriority.LOW, optional_fn)

        results = executor.execute_all(deadline_seconds=30)
    """

    def __init__(self, config: PriorityConfig | None = None):
        """Initialize executor.

        Args:
            config: Executor configuration
        """
        self.config = config or PriorityConfig()
        self._queue: PriorityQueue[T] = PriorityQueue()
        self._results: list[ExecutionResult[T]] = []
        self._lock = threading.Lock()

    def add(
        self,
        name: str,
        priority: ValidationPriority,
        operation: Callable[[], T],
        timeout_ms: float = 5000.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a validator to the queue.

        Args:
            name: Validator name
            priority: Priority level
            operation: Validation function
            timeout_ms: Timeout in milliseconds
            metadata: Additional metadata
        """
        item = PriorityItem.create(
            name=name,
            priority=priority,
            operation=operation,
            timeout_ms=timeout_ms,
            metadata=metadata,
        )
        self._queue.push(item)

    def execute_all(
        self,
        deadline_seconds: float | None = None,
    ) -> list[ExecutionResult[T]]:
        """Execute all validators in priority order.

        Args:
            deadline_seconds: Overall deadline in seconds

        Returns:
            List of execution results
        """
        results: list[ExecutionResult[T]] = []
        start_time = time.time()
        deadline = start_time + deadline_seconds if deadline_seconds else float("inf")

        while not self._queue.is_empty():
            item = self._queue.pop()
            if item is None:
                break

            current_time = time.time()
            remaining = deadline - current_time

            # Check if deadline exceeded
            if remaining <= 0:
                results.append(ExecutionResult.skipped_result(
                    item.name,
                    item.priority,
                    "deadline_exceeded",
                ))
                continue

            # Check time pressure
            if deadline_seconds:
                remaining_fraction = remaining / deadline_seconds
                if (
                    self.config.skip_low_priority_under_pressure
                    and remaining_fraction < self.config.time_pressure_threshold
                    and item.priority > int(self.config.min_priority_to_execute)
                ):
                    results.append(ExecutionResult.skipped_result(
                        item.name,
                        item.priority,
                        "time_pressure",
                    ))
                    continue

            # Execute
            result = self._execute_item(item, min(remaining * 1000, item.timeout_ms))
            results.append(result)

        with self._lock:
            self._results = results

        return results

    def _execute_item(
        self,
        item: PriorityItem[T],
        timeout_ms: float,
    ) -> ExecutionResult[T]:
        """Execute a single item.

        Args:
            item: Item to execute
            timeout_ms: Timeout in milliseconds

        Returns:
            ExecutionResult
        """
        start = time.time()

        try:
            # Simple execution (no true timeout for sync functions)
            value = item.operation()
            duration = (time.time() - start) * 1000

            return ExecutionResult.success_result(
                item.name,
                item.priority,
                value,
                duration,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return ExecutionResult.failure_result(
                item.name,
                item.priority,
                str(e),
                duration,
            )

    async def execute_all_async(
        self,
        deadline_seconds: float | None = None,
    ) -> list[ExecutionResult[T]]:
        """Execute all validators asynchronously.

        Args:
            deadline_seconds: Overall deadline in seconds

        Returns:
            List of execution results
        """
        results: list[ExecutionResult[T]] = []
        start_time = time.time()
        deadline = start_time + deadline_seconds if deadline_seconds else float("inf")

        while not self._queue.is_empty():
            item = self._queue.pop()
            if item is None:
                break

            current_time = time.time()
            remaining = deadline - current_time

            if remaining <= 0:
                results.append(ExecutionResult.skipped_result(
                    item.name,
                    item.priority,
                    "deadline_exceeded",
                ))
                continue

            # Check time pressure
            if deadline_seconds:
                remaining_fraction = remaining / deadline_seconds
                if (
                    self.config.skip_low_priority_under_pressure
                    and remaining_fraction < self.config.time_pressure_threshold
                    and item.priority > int(self.config.min_priority_to_execute)
                ):
                    results.append(ExecutionResult.skipped_result(
                        item.name,
                        item.priority,
                        "time_pressure",
                    ))
                    continue

            # Execute with timeout
            timeout_sec = min(remaining, item.timeout_ms / 1000)
            result = await self._execute_item_async(item, timeout_sec)
            results.append(result)

        with self._lock:
            self._results = results

        return results

    async def _execute_item_async(
        self,
        item: PriorityItem[T],
        timeout_seconds: float,
    ) -> ExecutionResult[T]:
        """Execute item asynchronously with timeout.

        Args:
            item: Item to execute
            timeout_seconds: Timeout in seconds

        Returns:
            ExecutionResult
        """
        start = time.time()

        try:
            # Run in executor with timeout
            loop = asyncio.get_event_loop()
            value = await asyncio.wait_for(
                loop.run_in_executor(None, item.operation),
                timeout=timeout_seconds,
            )
            duration = (time.time() - start) * 1000

            return ExecutionResult.success_result(
                item.name,
                item.priority,
                value,
                duration,
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start) * 1000
            return ExecutionResult.failure_result(
                item.name,
                item.priority,
                f"Timeout after {timeout_seconds:.1f}s",
                duration,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return ExecutionResult.failure_result(
                item.name,
                item.priority,
                str(e),
                duration,
            )

    def get_results(self) -> list[ExecutionResult[T]]:
        """Get execution results.

        Returns:
            List of execution results
        """
        with self._lock:
            return list(self._results)

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary.

        Returns:
            Summary statistics
        """
        with self._lock:
            results = self._results

        if not results:
            return {"total": 0}

        succeeded = sum(1 for r in results if r.success and not r.skipped)
        failed = sum(1 for r in results if not r.success)
        skipped = sum(1 for r in results if r.skipped)
        total_duration = sum(r.duration_ms for r in results)

        by_priority: dict[int, int] = {}
        for r in results:
            by_priority[r.priority] = by_priority.get(r.priority, 0) + 1

        return {
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total_duration_ms": total_duration,
            "by_priority": by_priority,
        }

    def clear(self) -> None:
        """Clear queue and results."""
        self._queue.clear()
        with self._lock:
            self._results.clear()


def execute_by_priority(
    operations: list[tuple[str, ValidationPriority, Callable[[], T]]],
    deadline_seconds: float | None = None,
    config: PriorityConfig | None = None,
) -> list[ExecutionResult[T]]:
    """Execute operations by priority.

    Args:
        operations: List of (name, priority, operation) tuples
        deadline_seconds: Overall deadline
        config: Executor configuration

    Returns:
        List of execution results
    """
    executor: PriorityExecutor[T] = PriorityExecutor(config)

    for name, priority, operation in operations:
        executor.add(name, priority, operation)

    return executor.execute_all(deadline_seconds)


def create_priority_executor(
    config: PriorityConfig | None = None,
) -> PriorityExecutor[Any]:
    """Create a priority executor.

    Args:
        config: Executor configuration

    Returns:
        PriorityExecutor
    """
    return PriorityExecutor(config)
