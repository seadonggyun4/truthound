"""Lock strategies for filesystem concurrency control.

This module implements the Strategy pattern for file locking, allowing
different locking mechanisms to be used interchangeably based on platform
and requirements.

Supported strategies:
- FcntlLockStrategy: POSIX fcntl-based locking (Unix/Linux/macOS)
- PortalockerStrategy: Cross-platform locking via portalocker library
- FileLockStrategy: Cross-platform locking via filelock library
- NoOpLockStrategy: No-op for single-threaded scenarios

The module auto-detects the best available strategy for the current platform.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator, TypeVar

# Thread-local storage for lock tracking
_thread_local = threading.local()


class LockMode(Enum):
    """Lock acquisition modes."""

    SHARED = auto()  # Read lock - multiple readers allowed
    EXCLUSIVE = auto()  # Write lock - exclusive access


@dataclass(frozen=True)
class LockHandle:
    """Handle representing an acquired lock.

    This is an opaque handle that must be passed to release the lock.
    It contains information about the lock for debugging and management.

    Attributes:
        path: Path to the locked file.
        mode: Lock mode (shared or exclusive).
        fd: File descriptor (if applicable).
        timestamp: When the lock was acquired.
        thread_id: ID of the thread that acquired the lock.
        process_id: ID of the process that acquired the lock.
    """

    path: Path
    mode: LockMode
    fd: int | None = None
    timestamp: float = field(default_factory=time.time)
    thread_id: int = field(default_factory=threading.get_ident)
    process_id: int = field(default_factory=os.getpid)

    def __str__(self) -> str:
        mode_str = "SHARED" if self.mode == LockMode.SHARED else "EXCLUSIVE"
        return f"LockHandle({self.path}, {mode_str}, pid={self.process_id})"


class LockStrategy(ABC):
    """Abstract base class for lock strategies.

    Implementations must be thread-safe and handle process-level locking.
    The strategy pattern allows swapping locking mechanisms without changing
    the client code.
    """

    @abstractmethod
    def acquire(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
        blocking: bool = True,
    ) -> LockHandle:
        """Acquire a lock on the specified path.

        Args:
            path: Path to the file to lock.
            mode: Lock mode (shared or exclusive).
            timeout: Maximum time to wait for lock (None = infinite).
            blocking: If False, raise immediately if lock unavailable.

        Returns:
            LockHandle for the acquired lock.

        Raises:
            LockTimeout: If timeout expires before lock is acquired.
            OSError: If locking fails due to system error.
        """
        pass

    @abstractmethod
    def release(self, handle: LockHandle) -> None:
        """Release a previously acquired lock.

        Args:
            handle: The lock handle returned by acquire().

        Raises:
            ValueError: If the handle is invalid or already released.
        """
        pass

    @abstractmethod
    def try_acquire(self, path: Path, mode: LockMode) -> LockHandle | None:
        """Try to acquire a lock without blocking.

        Args:
            path: Path to the file to lock.
            mode: Lock mode (shared or exclusive).

        Returns:
            LockHandle if lock acquired, None otherwise.
        """
        pass

    @abstractmethod
    def is_locked(self, path: Path) -> bool:
        """Check if a path is currently locked by any process.

        Args:
            path: Path to check.

        Returns:
            True if the path is locked.
        """
        pass

    @contextmanager
    def lock(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
    ) -> Iterator[LockHandle]:
        """Context manager for lock acquisition.

        Args:
            path: Path to the file to lock.
            mode: Lock mode (shared or exclusive).
            timeout: Maximum time to wait for lock.

        Yields:
            LockHandle for the acquired lock.

        Example:
            >>> with strategy.lock(path, LockMode.EXCLUSIVE) as handle:
            ...     # Exclusive access to file
            ...     pass
        """
        handle = self.acquire(path, mode, timeout)
        try:
            yield handle
        finally:
            self.release(handle)


class FcntlLockStrategy(LockStrategy):
    """POSIX fcntl-based file locking strategy.

    This strategy uses the fcntl.flock() system call for file locking.
    It's available on Unix-like systems (Linux, macOS, BSD).

    Features:
    - Advisory locking (cooperative)
    - Automatic release on file descriptor close
    - Works across processes
    - Supports shared and exclusive modes
    """

    def __init__(self) -> None:
        """Initialize the fcntl lock strategy."""
        if sys.platform == "win32":
            raise RuntimeError("FcntlLockStrategy is not available on Windows")

        import fcntl

        self._fcntl = fcntl
        self._locks: dict[str, tuple[int, Any]] = {}  # path -> (fd, file_obj)
        self._lock = threading.RLock()

    def acquire(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
        blocking: bool = True,
    ) -> LockHandle:
        """Acquire a lock using fcntl.flock()."""
        lock_path = self._get_lock_path(path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine lock operation
        if mode == LockMode.SHARED:
            operation = self._fcntl.LOCK_SH
        else:
            operation = self._fcntl.LOCK_EX

        if not blocking:
            operation |= self._fcntl.LOCK_NB

        # Open lock file
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)

        try:
            if timeout is not None and blocking:
                # Implement timeout with polling
                start_time = time.time()
                while True:
                    try:
                        self._fcntl.flock(fd, operation | self._fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.time() - start_time >= timeout:
                            os.close(fd)
                            from truthound.stores.concurrency.manager import LockTimeout

                            raise LockTimeout(path, timeout)
                        time.sleep(0.01)  # 10ms sleep between retries
            else:
                self._fcntl.flock(fd, operation)

            with self._lock:
                self._locks[str(path)] = (fd, None)

            return LockHandle(path=path, mode=mode, fd=fd)

        except (BlockingIOError, OSError) as e:
            os.close(fd)
            if not blocking:
                from truthound.stores.concurrency.manager import LockTimeout

                raise LockTimeout(path, 0) from e
            raise

    def release(self, handle: LockHandle) -> None:
        """Release the fcntl lock."""
        with self._lock:
            key = str(handle.path)
            if key not in self._locks:
                raise ValueError(f"Lock not held: {handle.path}")

            fd, _ = self._locks.pop(key)

        try:
            self._fcntl.flock(fd, self._fcntl.LOCK_UN)
        finally:
            os.close(fd)

        # Clean up lock file if it exists and is empty
        lock_path = self._get_lock_path(handle.path)
        try:
            if lock_path.exists() and lock_path.stat().st_size == 0:
                lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def try_acquire(self, path: Path, mode: LockMode) -> LockHandle | None:
        """Try to acquire without blocking."""
        try:
            return self.acquire(path, mode, blocking=False)
        except Exception:
            return None

    def is_locked(self, path: Path) -> bool:
        """Check if path is locked."""
        lock_path = self._get_lock_path(path)
        if not lock_path.exists():
            return False

        try:
            fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
            try:
                self._fcntl.flock(fd, self._fcntl.LOCK_EX | self._fcntl.LOCK_NB)
                self._fcntl.flock(fd, self._fcntl.LOCK_UN)
                return False
            except BlockingIOError:
                return True
            finally:
                os.close(fd)
        except OSError:
            return False

    def _get_lock_path(self, path: Path) -> Path:
        """Get the lock file path for a given path."""
        return path.parent / f".{path.name}.lock"


class PortalockerStrategy(LockStrategy):
    """Cross-platform locking using portalocker library.

    This strategy provides cross-platform file locking using the portalocker
    library. It works on Windows, Linux, and macOS.

    Requires: pip install portalocker
    """

    def __init__(self) -> None:
        """Initialize the portalocker strategy."""
        try:
            import portalocker

            self._portalocker = portalocker
        except ImportError as e:
            raise ImportError(
                "portalocker is required for PortalockerStrategy. "
                "Install it with: pip install portalocker"
            ) from e

        self._locks: dict[str, Any] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
        blocking: bool = True,
    ) -> LockHandle:
        """Acquire a lock using portalocker."""
        lock_path = self._get_lock_path(path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine lock flags
        if mode == LockMode.SHARED:
            flags = self._portalocker.LOCK_SH
        else:
            flags = self._portalocker.LOCK_EX

        # Open and lock
        lock_file = open(lock_path, "w")

        try:
            if timeout is not None or not blocking:
                effective_timeout = timeout if timeout is not None else 0
                self._portalocker.lock(
                    lock_file,
                    flags | self._portalocker.LOCK_NB,
                    timeout=effective_timeout,
                )
            else:
                self._portalocker.lock(lock_file, flags)

            with self._lock:
                self._locks[str(path)] = lock_file

            return LockHandle(path=path, mode=mode, fd=lock_file.fileno())

        except self._portalocker.LockException as e:
            lock_file.close()
            from truthound.stores.concurrency.manager import LockTimeout

            raise LockTimeout(path, timeout or 0) from e
        except Exception:
            lock_file.close()
            raise

    def release(self, handle: LockHandle) -> None:
        """Release the portalocker lock."""
        with self._lock:
            key = str(handle.path)
            if key not in self._locks:
                raise ValueError(f"Lock not held: {handle.path}")

            lock_file = self._locks.pop(key)

        try:
            self._portalocker.unlock(lock_file)
        finally:
            lock_file.close()

    def try_acquire(self, path: Path, mode: LockMode) -> LockHandle | None:
        """Try to acquire without blocking."""
        try:
            return self.acquire(path, mode, timeout=0, blocking=False)
        except Exception:
            return None

    def is_locked(self, path: Path) -> bool:
        """Check if path is locked."""
        lock_path = self._get_lock_path(path)
        if not lock_path.exists():
            return False

        try:
            lock_file = open(lock_path, "w")
            try:
                self._portalocker.lock(
                    lock_file,
                    self._portalocker.LOCK_EX | self._portalocker.LOCK_NB,
                )
                self._portalocker.unlock(lock_file)
                return False
            except self._portalocker.LockException:
                return True
            finally:
                lock_file.close()
        except OSError:
            return False

    def _get_lock_path(self, path: Path) -> Path:
        """Get the lock file path for a given path."""
        return path.parent / f".{path.name}.lock"


class FileLockStrategy(LockStrategy):
    """Cross-platform locking using filelock library.

    This strategy provides cross-platform file locking using the filelock
    library. It's a simpler alternative to portalocker.

    Requires: pip install filelock
    """

    def __init__(self) -> None:
        """Initialize the filelock strategy."""
        try:
            import filelock

            self._filelock = filelock
        except ImportError as e:
            raise ImportError(
                "filelock is required for FileLockStrategy. "
                "Install it with: pip install filelock"
            ) from e

        self._locks: dict[str, Any] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
        blocking: bool = True,
    ) -> LockHandle:
        """Acquire a lock using filelock."""
        lock_path = self._get_lock_path(path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # filelock only supports exclusive locks
        lock = self._filelock.FileLock(lock_path)

        try:
            effective_timeout = timeout if blocking else 0
            lock.acquire(timeout=effective_timeout if effective_timeout else -1)

            with self._lock:
                self._locks[str(path)] = lock

            return LockHandle(path=path, mode=mode)

        except self._filelock.Timeout as e:
            from truthound.stores.concurrency.manager import LockTimeout

            raise LockTimeout(path, timeout or 0) from e

    def release(self, handle: LockHandle) -> None:
        """Release the filelock lock."""
        with self._lock:
            key = str(handle.path)
            if key not in self._locks:
                raise ValueError(f"Lock not held: {handle.path}")

            lock = self._locks.pop(key)

        lock.release()

    def try_acquire(self, path: Path, mode: LockMode) -> LockHandle | None:
        """Try to acquire without blocking."""
        try:
            return self.acquire(path, mode, timeout=0, blocking=False)
        except Exception:
            return None

    def is_locked(self, path: Path) -> bool:
        """Check if path is locked."""
        lock_path = self._get_lock_path(path)
        if not lock_path.exists():
            return False

        lock = self._filelock.FileLock(lock_path)
        try:
            lock.acquire(timeout=0)
            lock.release()
            return False
        except self._filelock.Timeout:
            return True

    def _get_lock_path(self, path: Path) -> Path:
        """Get the lock file path for a given path."""
        return path.parent / f".{path.name}.lock"


class NoOpLockStrategy(LockStrategy):
    """No-op lock strategy for single-threaded scenarios.

    This strategy provides no actual locking, useful for:
    - Single-threaded applications
    - Testing
    - When external locking is already in place
    """

    def acquire(
        self,
        path: Path,
        mode: LockMode,
        timeout: float | None = None,
        blocking: bool = True,
    ) -> LockHandle:
        """Return a lock handle without actual locking."""
        return LockHandle(path=path, mode=mode)

    def release(self, handle: LockHandle) -> None:
        """No-op release."""
        pass

    def try_acquire(self, path: Path, mode: LockMode) -> LockHandle | None:
        """Always succeeds."""
        return LockHandle(path=path, mode=mode)

    def is_locked(self, path: Path) -> bool:
        """Always returns False."""
        return False


def get_default_lock_strategy() -> LockStrategy:
    """Get the best available lock strategy for the current platform.

    Returns:
        The most appropriate LockStrategy for the current environment.

    Priority order:
    1. FcntlLockStrategy (Unix-like systems)
    2. FileLockStrategy (if filelock is installed)
    3. PortalockerStrategy (if portalocker is installed)
    4. NoOpLockStrategy (fallback)
    """
    # Try fcntl first on Unix-like systems
    if sys.platform != "win32":
        try:
            return FcntlLockStrategy()
        except Exception:
            pass

    # Try filelock (simpler API)
    try:
        return FileLockStrategy()
    except ImportError:
        pass

    # Try portalocker
    try:
        return PortalockerStrategy()
    except ImportError:
        pass

    # Fallback to no-op
    import warnings

    warnings.warn(
        "No file locking library available. Using NoOpLockStrategy. "
        "Install 'filelock' or 'portalocker' for proper concurrency control.",
        RuntimeWarning,
        stacklevel=2,
    )
    return NoOpLockStrategy()
