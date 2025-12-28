"""Bulkhead pattern implementations.

This module provides resource isolation to prevent one failing
component from exhausting all available resources.
"""

from __future__ import annotations

import functools
import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from truthound.common.resilience.config import BulkheadConfig
from truthound.common.resilience.protocols import BulkheadProtocol

logger = logging.getLogger(__name__)

R = TypeVar("R")


class BulkheadFullError(Exception):
    """Raised when bulkhead is full and cannot accept more requests."""

    def __init__(self, name: str, max_concurrent: int):
        self.bulkhead_name = name
        self.max_concurrent = max_concurrent
        super().__init__(
            f"Bulkhead '{name}' is full (max_concurrent={max_concurrent})"
        )


class Bulkhead(BulkheadProtocol, ABC):
    """Abstract base class for bulkhead implementations."""

    def __init__(self, name: str, config: BulkheadConfig | None = None):
        """Initialize bulkhead.

        Args:
            name: Unique name for this bulkhead.
            config: Configuration options.
        """
        self._name = name
        self._config = config or BulkheadConfig()
        self._total_acquired = 0
        self._total_rejected = 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Get bulkhead name."""
        return self._name

    @property
    def config(self) -> BulkheadConfig:
        """Get configuration."""
        return self._config

    def get_metrics(self) -> dict[str, Any]:
        """Get bulkhead metrics."""
        with self._lock:
            return {
                "name": self._name,
                "max_concurrent": self._config.max_concurrent,
                "available_slots": self.available_slots(),
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
            }

    @contextmanager
    def limit(self, timeout: float | None = None) -> Generator[None, None, None]:
        """Context manager for bulkhead-limited execution."""
        effective_timeout = timeout if timeout is not None else self._config.max_wait_time

        if not self.acquire(effective_timeout):
            with self._lock:
                self._total_rejected += 1
            raise BulkheadFullError(self._name, self._config.max_concurrent)

        with self._lock:
            self._total_acquired += 1

        try:
            yield
        finally:
            self.release()

    def __call__(self, func: Callable[..., R]) -> Callable[..., R]:
        """Decorator for bulkhead-limited execution."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            with self.limit():
                return func(*args, **kwargs)
        return wrapper


class SemaphoreBulkhead(Bulkhead):
    """Semaphore-based bulkhead implementation.

    Uses a counting semaphore to limit concurrent executions.

    Example:
        bulkhead = SemaphoreBulkhead("database", BulkheadConfig(max_concurrent=10))

        @bulkhead
        def database_operation():
            return db.query(...)

        # Or as context manager
        with bulkhead.limit():
            result = db.query(...)
    """

    def __init__(self, name: str, config: BulkheadConfig | None = None):
        """Initialize semaphore bulkhead."""
        super().__init__(name, config)
        self._semaphore = threading.Semaphore(self._config.max_concurrent)
        self._active_count = 0

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a slot."""
        acquired = self._semaphore.acquire(blocking=True, timeout=timeout)
        if acquired:
            with self._lock:
                self._active_count += 1
        return acquired

    def release(self) -> None:
        """Release a slot."""
        with self._lock:
            self._active_count -= 1
        self._semaphore.release()

    def available_slots(self) -> int:
        """Get number of available slots."""
        with self._lock:
            return self._config.max_concurrent - self._active_count

    def reset(self) -> None:
        """Reset bulkhead state."""
        # Note: Cannot truly reset semaphore without tracking state
        with self._lock:
            # Release any held slots
            while self._active_count > 0:
                self._semaphore.release()
                self._active_count -= 1
            self._total_acquired = 0
            self._total_rejected = 0


class ThreadPoolBulkhead(Bulkhead):
    """Thread pool-based bulkhead implementation.

    Uses a thread pool to limit concurrent executions and provide
    isolation from the main thread.

    Example:
        bulkhead = ThreadPoolBulkhead("io-operations", BulkheadConfig(max_concurrent=5))

        @bulkhead
        def io_operation():
            return read_file(...)
    """

    def __init__(self, name: str, config: BulkheadConfig | None = None):
        """Initialize thread pool bulkhead."""
        super().__init__(name, config)
        self._executor = ThreadPoolExecutor(
            max_workers=self._config.max_concurrent,
            thread_name_prefix=f"bulkhead-{name}",
        )
        self._semaphore = threading.Semaphore(self._config.max_concurrent)
        self._active_count = 0

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a slot."""
        acquired = self._semaphore.acquire(blocking=True, timeout=timeout)
        if acquired:
            with self._lock:
                self._active_count += 1
        return acquired

    def release(self) -> None:
        """Release a slot."""
        with self._lock:
            self._active_count -= 1
        self._semaphore.release()

    def available_slots(self) -> int:
        """Get number of available slots."""
        with self._lock:
            return self._config.max_concurrent - self._active_count

    def submit(self, func: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        """Submit work to the thread pool with bulkhead protection.

        Args:
            func: Function to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the function execution.

        Raises:
            BulkheadFullError: If bulkhead is full.
        """
        if not self.acquire(self._config.max_wait_time):
            with self._lock:
                self._total_rejected += 1
            raise BulkheadFullError(self._name, self._config.max_concurrent)

        with self._lock:
            self._total_acquired += 1

        try:
            future = self._executor.submit(func, *args, **kwargs)
            return future.result()
        finally:
            self.release()

    def reset(self) -> None:
        """Reset bulkhead state."""
        with self._lock:
            while self._active_count > 0:
                self._semaphore.release()
                self._active_count -= 1
            self._total_acquired = 0
            self._total_rejected = 0

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=wait)

    def __del__(self) -> None:
        """Cleanup thread pool on deletion."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


def with_bulkhead(
    max_concurrent: int = 10,
    max_wait_time: float = 0.0,
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """Convenience decorator factory for bulkhead.

    Example:
        @with_bulkhead(max_concurrent=5)
        def limited_operation():
            return expensive_call()
    """
    config = BulkheadConfig(
        max_concurrent=max_concurrent,
        max_wait_time=max_wait_time,
    )
    bulkhead = SemaphoreBulkhead(f"inline-{id(config)}", config)

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        return bulkhead(func)

    return decorator
