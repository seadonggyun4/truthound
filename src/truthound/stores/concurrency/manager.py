"""Lock manager for coordinating file access.

This module provides high-level lock management with features like:
- Deadlock detection
- Lock timeout handling
- Reentrant locks
- Lock statistics and debugging

The FileLockManager is the main entry point for most use cases.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from truthound.stores.concurrency.locks import (
    LockHandle,
    LockMode,
    LockStrategy,
    get_default_lock_strategy,
)


class LockTimeout(Exception):
    """Raised when lock acquisition times out."""

    def __init__(self, path: Path, timeout: float) -> None:
        self.path = path
        self.timeout = timeout
        super().__init__(f"Timeout acquiring lock on {path} after {timeout}s")


class DeadlockError(Exception):
    """Raised when a potential deadlock is detected."""

    def __init__(self, path: Path, held_locks: list[Path]) -> None:
        self.path = path
        self.held_locks = held_locks
        super().__init__(
            f"Potential deadlock: trying to acquire {path} while holding {held_locks}"
        )


@dataclass
class LockStatistics:
    """Statistics about lock usage.

    Useful for debugging and performance monitoring.
    """

    total_acquisitions: int = 0
    total_releases: int = 0
    total_wait_time: float = 0.0
    total_hold_time: float = 0.0
    max_wait_time: float = 0.0
    max_hold_time: float = 0.0
    contentions: int = 0
    timeouts: int = 0
    deadlocks: int = 0

    def record_acquisition(self, wait_time: float, contended: bool) -> None:
        """Record a lock acquisition."""
        self.total_acquisitions += 1
        self.total_wait_time += wait_time
        self.max_wait_time = max(self.max_wait_time, wait_time)
        if contended:
            self.contentions += 1

    def record_release(self, hold_time: float) -> None:
        """Record a lock release."""
        self.total_releases += 1
        self.total_hold_time += hold_time
        self.max_hold_time = max(self.max_hold_time, hold_time)

    def record_timeout(self) -> None:
        """Record a timeout."""
        self.timeouts += 1

    def record_deadlock(self) -> None:
        """Record a deadlock detection."""
        self.deadlocks += 1

    @property
    def avg_wait_time(self) -> float:
        """Average wait time for lock acquisition."""
        if self.total_acquisitions == 0:
            return 0.0
        return self.total_wait_time / self.total_acquisitions

    @property
    def avg_hold_time(self) -> float:
        """Average time locks are held."""
        if self.total_releases == 0:
            return 0.0
        return self.total_hold_time / self.total_releases

    @property
    def contention_rate(self) -> float:
        """Percentage of acquisitions that were contended."""
        if self.total_acquisitions == 0:
            return 0.0
        return self.contentions / self.total_acquisitions * 100


@dataclass
class LockContext:
    """Context for a held lock.

    Tracks information about the lock for debugging and management.
    """

    handle: LockHandle
    acquired_at: float = field(default_factory=time.time)
    thread_id: int = field(default_factory=threading.get_ident)
    reentry_count: int = 1
    wait_time: float = 0.0

    @property
    def hold_time(self) -> float:
        """How long the lock has been held."""
        return time.time() - self.acquired_at


class FileLockManager:
    """High-level manager for file locking.

    Provides a thread-safe interface for acquiring and releasing locks
    with support for:
    - Multiple locking strategies
    - Reentrant locks (same thread can acquire multiple times)
    - Deadlock detection
    - Lock statistics
    - Automatic cleanup

    Example:
        >>> manager = FileLockManager()
        >>>
        >>> # Using context manager
        >>> with manager.acquire(Path("/path/to/file"), LockMode.EXCLUSIVE):
        ...     # File is locked
        ...     pass
        >>>
        >>> # Manual acquire/release
        >>> handle = manager.lock(Path("/path/to/file"), LockMode.SHARED)
        >>> try:
        ...     # Do work
        ...     pass
        ... finally:
        ...     manager.unlock(handle)
    """

    def __init__(
        self,
        strategy: LockStrategy | None = None,
        enable_deadlock_detection: bool = True,
        enable_statistics: bool = True,
        default_timeout: float | None = 30.0,
    ) -> None:
        """Initialize the lock manager.

        Args:
            strategy: Lock strategy to use. If None, auto-detects best strategy.
            enable_deadlock_detection: Whether to detect potential deadlocks.
            enable_statistics: Whether to collect lock statistics.
            default_timeout: Default timeout for lock acquisition (None = infinite).
        """
        self._strategy = strategy or get_default_lock_strategy()
        self._enable_deadlock_detection = enable_deadlock_detection
        self._enable_statistics = enable_statistics
        self._default_timeout = default_timeout

        # Thread-local storage for tracking held locks
        self._thread_locks: threading.local = threading.local()

        # Global lock tracking
        self._global_lock = threading.RLock()
        self._held_locks: dict[str, LockContext] = {}

        # Statistics
        self._stats = LockStatistics() if enable_statistics else None
        self._path_stats: dict[str, LockStatistics] = defaultdict(LockStatistics)

    @property
    def statistics(self) -> LockStatistics | None:
        """Get global lock statistics."""
        return self._stats

    def get_path_statistics(self, path: Path) -> LockStatistics:
        """Get statistics for a specific path."""
        return self._path_stats[str(path)]

    def _get_thread_locks(self) -> dict[str, LockContext]:
        """Get locks held by the current thread."""
        if not hasattr(self._thread_locks, "locks"):
            self._thread_locks.locks = {}
        return self._thread_locks.locks

    def _check_deadlock(self, path: Path) -> None:
        """Check for potential deadlock.

        Simple cycle detection: if thread already holds locks and is
        trying to acquire a new one, there's potential for deadlock.
        """
        if not self._enable_deadlock_detection:
            return

        thread_locks = self._get_thread_locks()
        if thread_locks and str(path) not in thread_locks:
            # Thread holds locks and is trying to acquire a new one
            held_paths = [Path(p) for p in thread_locks.keys()]

            # Check if any other thread is waiting for our locks
            # while holding the lock we want
            with self._global_lock:
                target_key = str(path)
                if target_key in self._held_locks:
                    holder = self._held_locks[target_key]
                    if holder.thread_id != threading.get_ident():
                        # Another thread holds the lock we want
                        # This is a potential deadlock scenario
                        if self._stats:
                            self._stats.record_deadlock()
                        raise DeadlockError(path, held_paths)

    def lock(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
        blocking: bool = True,
    ) -> LockHandle:
        """Acquire a lock on the specified path.

        Args:
            path: Path to lock.
            mode: Lock mode (SHARED or EXCLUSIVE).
            timeout: Timeout in seconds. None uses default timeout.
            blocking: If False, fail immediately if lock unavailable.

        Returns:
            LockHandle for the acquired lock.

        Raises:
            LockTimeout: If timeout expires.
            DeadlockError: If potential deadlock detected.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout
        path_key = str(path)
        thread_locks = self._get_thread_locks()

        # Check for reentry
        if path_key in thread_locks:
            context = thread_locks[path_key]
            # For reentry, must be same or lower lock mode
            if mode == LockMode.EXCLUSIVE and context.handle.mode == LockMode.SHARED:
                raise ValueError("Cannot upgrade shared lock to exclusive")
            context.reentry_count += 1
            return context.handle

        # Check for deadlock
        self._check_deadlock(path)

        # Acquire the lock
        start_time = time.time()
        contended = self._strategy.is_locked(path)

        try:
            handle = self._strategy.acquire(path, mode, effective_timeout, blocking)
        except LockTimeout:
            if self._stats:
                self._stats.record_timeout()
            raise

        wait_time = time.time() - start_time

        # Record the lock
        context = LockContext(handle=handle, wait_time=wait_time)
        thread_locks[path_key] = context

        with self._global_lock:
            self._held_locks[path_key] = context

        # Update statistics
        if self._stats:
            self._stats.record_acquisition(wait_time, contended)
            self._path_stats[path_key].record_acquisition(wait_time, contended)

        return handle

    def unlock(self, handle: LockHandle) -> None:
        """Release a previously acquired lock.

        Args:
            handle: The lock handle to release.

        Raises:
            ValueError: If the lock is not held.
        """
        path_key = str(handle.path)
        thread_locks = self._get_thread_locks()

        if path_key not in thread_locks:
            raise ValueError(f"Lock not held by this thread: {handle.path}")

        context = thread_locks[path_key]

        # Handle reentry
        context.reentry_count -= 1
        if context.reentry_count > 0:
            return

        # Actually release the lock
        hold_time = context.hold_time
        self._strategy.release(handle)

        # Clean up tracking
        del thread_locks[path_key]
        with self._global_lock:
            self._held_locks.pop(path_key, None)

        # Update statistics
        if self._stats:
            self._stats.record_release(hold_time)
            self._path_stats[path_key].record_release(hold_time)

    def try_lock(self, path: Path, mode: LockMode) -> LockHandle | None:
        """Try to acquire a lock without blocking.

        Args:
            path: Path to lock.
            mode: Lock mode.

        Returns:
            LockHandle if acquired, None otherwise.
        """
        try:
            return self.lock(path, mode, timeout=0, blocking=False)
        except (LockTimeout, DeadlockError):
            return None

    def is_locked(self, path: Path) -> bool:
        """Check if a path is currently locked.

        Args:
            path: Path to check.

        Returns:
            True if the path is locked by any thread/process.
        """
        return self._strategy.is_locked(path)

    def is_held_by_current_thread(self, path: Path) -> bool:
        """Check if current thread holds a lock on the path.

        Args:
            path: Path to check.

        Returns:
            True if current thread holds the lock.
        """
        return str(path) in self._get_thread_locks()

    def get_held_locks(self) -> list[LockContext]:
        """Get all locks held by the current thread.

        Returns:
            List of lock contexts.
        """
        return list(self._get_thread_locks().values())

    @contextmanager
    def acquire(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
    ) -> Iterator[LockHandle]:
        """Context manager for lock acquisition.

        Args:
            path: Path to lock.
            mode: Lock mode.
            timeout: Timeout in seconds.

        Yields:
            LockHandle for the acquired lock.

        Example:
            >>> with manager.acquire(path, LockMode.EXCLUSIVE) as handle:
            ...     # Safe to modify file
            ...     pass
        """
        handle = self.lock(path, mode, timeout)
        try:
            yield handle
        finally:
            self.unlock(handle)

    @contextmanager
    def read_lock(self, path: Path, timeout: float | None = None) -> Iterator[LockHandle]:
        """Convenience context manager for shared (read) lock.

        Args:
            path: Path to lock.
            timeout: Timeout in seconds.

        Yields:
            LockHandle for the acquired lock.
        """
        with self.acquire(path, LockMode.SHARED, timeout) as handle:
            yield handle

    @contextmanager
    def write_lock(self, path: Path, timeout: float | None = None) -> Iterator[LockHandle]:
        """Convenience context manager for exclusive (write) lock.

        Args:
            path: Path to lock.
            timeout: Timeout in seconds.

        Yields:
            LockHandle for the acquired lock.
        """
        with self.acquire(path, LockMode.EXCLUSIVE, timeout) as handle:
            yield handle

    def release_all(self) -> int:
        """Release all locks held by the current thread.

        Returns:
            Number of locks released.

        Warning:
            This should only be used in cleanup scenarios.
        """
        thread_locks = self._get_thread_locks()
        count = 0

        for path_key in list(thread_locks.keys()):
            context = thread_locks[path_key]
            # Force release regardless of reentry count
            context.reentry_count = 1
            self.unlock(context.handle)
            count += 1

        return count


# Module-level default manager for convenience
_default_manager: FileLockManager | None = None
_manager_lock = threading.Lock()


def get_default_manager() -> FileLockManager:
    """Get the default global lock manager.

    Returns:
        The default FileLockManager instance.
    """
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = FileLockManager()
    return _default_manager


def set_default_manager(manager: FileLockManager) -> None:
    """Set the default global lock manager.

    Args:
        manager: The manager to use as default.
    """
    global _default_manager
    with _manager_lock:
        _default_manager = manager
