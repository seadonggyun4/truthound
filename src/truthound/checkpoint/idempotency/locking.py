"""Distributed Locking for Idempotency.

This module provides locking mechanisms to ensure exclusive execution
of idempotent operations across processes.
"""

from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator, Protocol, runtime_checkable
from uuid import uuid4


logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Raised when lock acquisition fails."""

    def __init__(
        self,
        key: str,
        reason: str = "Lock acquisition failed",
        holder: str | None = None,
    ) -> None:
        super().__init__(f"{reason}: {key}" + (f" (held by {holder})" if holder else ""))
        self.key = key
        self.reason = reason
        self.holder = holder


@dataclass
class LockInfo:
    """Information about a held lock.

    Attributes:
        key: Lock key.
        holder_id: ID of the lock holder.
        acquired_at: When the lock was acquired.
        expires_at: When the lock expires.
        metadata: Additional lock metadata.
    """

    key: str
    holder_id: str
    acquired_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def remaining_seconds(self) -> float | None:
        """Get remaining lock time in seconds."""
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, remaining)


@runtime_checkable
class DistributedLock(Protocol):
    """Protocol for distributed lock implementations."""

    def acquire(
        self,
        key: str,
        timeout: float = 30.0,
        holder_id: str | None = None,
    ) -> LockInfo:
        """Acquire a lock.

        Args:
            key: Lock key.
            timeout: Lock timeout in seconds.
            holder_id: Unique ID for the lock holder.

        Returns:
            LockInfo for the acquired lock.

        Raises:
            LockAcquisitionError: If lock cannot be acquired.
        """
        ...

    def release(self, key: str, holder_id: str) -> bool:
        """Release a lock.

        Args:
            key: Lock key.
            holder_id: ID of the lock holder.

        Returns:
            True if released, False if not held.
        """
        ...

    def is_locked(self, key: str) -> bool:
        """Check if a key is locked.

        Args:
            key: Lock key.

        Returns:
            True if locked.
        """
        ...

    def get_lock_info(self, key: str) -> LockInfo | None:
        """Get information about a lock.

        Args:
            key: Lock key.

        Returns:
            LockInfo if locked, None otherwise.
        """
        ...

    @contextmanager
    def lock(
        self,
        key: str,
        timeout: float = 30.0,
        holder_id: str | None = None,
    ) -> Generator[LockInfo, None, None]:
        """Context manager for acquiring and releasing a lock.

        Args:
            key: Lock key.
            timeout: Lock timeout.
            holder_id: Lock holder ID.

        Yields:
            LockInfo for the acquired lock.
        """
        ...


class InMemoryLock:
    """In-memory distributed lock implementation.

    Suitable for single-process scenarios. Uses threading for safety.

    Example:
        >>> lock = InMemoryLock()
        >>> with lock.lock("my-key") as info:
        ...     # Do exclusive work
        ...     pass
    """

    def __init__(
        self,
        default_timeout: float = 30.0,
        cleanup_interval: float = 60.0,
    ) -> None:
        """Initialize the lock.

        Args:
            default_timeout: Default lock timeout.
            cleanup_interval: Interval for cleaning up expired locks.
        """
        self._locks: dict[str, LockInfo] = {}
        self._lock = threading.RLock()
        self._default_timeout = default_timeout
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = datetime.now()

    def acquire(
        self,
        key: str,
        timeout: float | None = None,
        holder_id: str | None = None,
    ) -> LockInfo:
        with self._lock:
            self._maybe_cleanup()

            holder_id = holder_id or self._generate_holder_id()
            timeout = timeout or self._default_timeout

            # Check if already locked
            existing = self._locks.get(key)
            if existing and not existing.is_expired:
                if existing.holder_id == holder_id:
                    # Re-entrant lock
                    return existing
                raise LockAcquisitionError(
                    key,
                    "Lock already held",
                    existing.holder_id,
                )

            # Acquire lock
            lock_info = LockInfo(
                key=key,
                holder_id=holder_id,
                acquired_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=timeout),
            )
            self._locks[key] = lock_info

            logger.debug(f"Lock acquired: {key} by {holder_id}")
            return lock_info

    def release(self, key: str, holder_id: str) -> bool:
        with self._lock:
            existing = self._locks.get(key)
            if existing is None:
                return False

            if existing.holder_id != holder_id:
                logger.warning(
                    f"Cannot release lock {key}: held by {existing.holder_id}, "
                    f"release requested by {holder_id}"
                )
                return False

            del self._locks[key]
            logger.debug(f"Lock released: {key} by {holder_id}")
            return True

    def is_locked(self, key: str) -> bool:
        with self._lock:
            existing = self._locks.get(key)
            if existing is None:
                return False
            if existing.is_expired:
                del self._locks[key]
                return False
            return True

    def get_lock_info(self, key: str) -> LockInfo | None:
        with self._lock:
            existing = self._locks.get(key)
            if existing and existing.is_expired:
                del self._locks[key]
                return None
            return existing

    def extend(self, key: str, holder_id: str, additional_seconds: float) -> bool:
        """Extend a lock's timeout.

        Args:
            key: Lock key.
            holder_id: Lock holder ID.
            additional_seconds: Seconds to add to expiry.

        Returns:
            True if extended, False if not held.
        """
        with self._lock:
            existing = self._locks.get(key)
            if existing is None or existing.holder_id != holder_id:
                return False

            existing.expires_at = datetime.now() + timedelta(seconds=additional_seconds)
            return True

    @contextmanager
    def lock(
        self,
        key: str,
        timeout: float | None = None,
        holder_id: str | None = None,
    ) -> Generator[LockInfo, None, None]:
        holder_id = holder_id or self._generate_holder_id()
        lock_info = self.acquire(key, timeout, holder_id)
        try:
            yield lock_info
        finally:
            self.release(key, holder_id)

    def _generate_holder_id(self) -> str:
        """Generate a unique holder ID."""
        return f"{os.getpid()}-{threading.get_ident()}-{uuid4().hex[:8]}"

    def _maybe_cleanup(self) -> None:
        """Cleanup expired locks if interval has passed."""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() < self._cleanup_interval:
            return

        expired = [k for k, v in self._locks.items() if v.is_expired]
        for key in expired:
            del self._locks[key]

        self._last_cleanup = now

    def cleanup_expired(self) -> int:
        """Force cleanup of expired locks.

        Returns:
            Number of locks cleaned up.
        """
        with self._lock:
            expired = [k for k, v in self._locks.items() if v.is_expired]
            for key in expired:
                del self._locks[key]
            self._last_cleanup = datetime.now()
            return len(expired)

    def size(self) -> int:
        """Get number of active locks."""
        with self._lock:
            return len([k for k, v in self._locks.items() if not v.is_expired])


class FileLock:
    """File-based distributed lock implementation.

    Uses file system locks for cross-process coordination.
    Suitable for single-machine multi-process scenarios.

    Example:
        >>> lock = FileLock("/var/lock/truthound")
        >>> with lock.lock("my-key") as info:
        ...     # Do exclusive work across processes
        ...     pass
    """

    def __init__(
        self,
        lock_dir: str | Path,
        default_timeout: float = 30.0,
        stale_lock_timeout: float = 300.0,
    ) -> None:
        """Initialize the lock.

        Args:
            lock_dir: Directory for lock files.
            default_timeout: Default lock timeout.
            stale_lock_timeout: Time after which locks are considered stale.
        """
        self._lock_dir = Path(lock_dir)
        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._default_timeout = default_timeout
        self._stale_lock_timeout = stale_lock_timeout
        self._held_locks: dict[str, tuple[int, LockInfo]] = {}  # fd, info
        self._local_lock = threading.RLock()

    def _lock_file_path(self, key: str) -> Path:
        """Get lock file path for a key."""
        # Use hash to avoid filesystem issues
        import hashlib
        hashed = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._lock_dir / f"{hashed}.lock"

    def acquire(
        self,
        key: str,
        timeout: float | None = None,
        holder_id: str | None = None,
    ) -> LockInfo:
        with self._local_lock:
            holder_id = holder_id or self._generate_holder_id()
            timeout = timeout or self._default_timeout

            lock_path = self._lock_file_path(key)

            # Check for stale lock
            self._check_stale_lock(lock_path)

            # Try to acquire file lock
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    raise LockAcquisitionError(key, "Lock already held")

                # Write lock info
                lock_info = LockInfo(
                    key=key,
                    holder_id=holder_id,
                    acquired_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=timeout),
                )

                os.ftruncate(fd, 0)
                os.lseek(fd, 0, os.SEEK_SET)
                import json
                lock_data = {
                    "holder_id": holder_id,
                    "acquired_at": lock_info.acquired_at.isoformat(),
                    "expires_at": lock_info.expires_at.isoformat() if lock_info.expires_at else None,
                    "pid": os.getpid(),
                }
                os.write(fd, json.dumps(lock_data).encode())

                self._held_locks[key] = (fd, lock_info)

                logger.debug(f"File lock acquired: {key} by {holder_id}")
                return lock_info

            except Exception as e:
                if isinstance(e, LockAcquisitionError):
                    raise
                raise LockAcquisitionError(key, str(e))

    def release(self, key: str, holder_id: str) -> bool:
        with self._local_lock:
            if key not in self._held_locks:
                return False

            fd, lock_info = self._held_locks[key]

            if lock_info.holder_id != holder_id:
                logger.warning(f"Cannot release lock {key}: holder mismatch")
                return False

            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                lock_path = self._lock_file_path(key)
                if lock_path.exists():
                    lock_path.unlink()
            except Exception as e:
                logger.error(f"Error releasing file lock {key}: {e}")

            del self._held_locks[key]
            logger.debug(f"File lock released: {key} by {holder_id}")
            return True

    def is_locked(self, key: str) -> bool:
        lock_path = self._lock_file_path(key)
        if not lock_path.exists():
            return False

        # Try to acquire to check
        try:
            fd = os.open(str(lock_path), os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return False  # Lock was available
            except BlockingIOError:
                os.close(fd)
                return True  # Lock is held
        except Exception:
            return False

    def get_lock_info(self, key: str) -> LockInfo | None:
        lock_path = self._lock_file_path(key)
        if not lock_path.exists():
            return None

        try:
            import json
            data = json.loads(lock_path.read_text())
            return LockInfo(
                key=key,
                holder_id=data["holder_id"],
                acquired_at=datetime.fromisoformat(data["acquired_at"]),
                expires_at=(
                    datetime.fromisoformat(data["expires_at"])
                    if data.get("expires_at")
                    else None
                ),
                metadata={"pid": data.get("pid")},
            )
        except Exception:
            return None

    @contextmanager
    def lock(
        self,
        key: str,
        timeout: float | None = None,
        holder_id: str | None = None,
    ) -> Generator[LockInfo, None, None]:
        holder_id = holder_id or self._generate_holder_id()
        lock_info = self.acquire(key, timeout, holder_id)
        try:
            yield lock_info
        finally:
            self.release(key, holder_id)

    def _generate_holder_id(self) -> str:
        """Generate a unique holder ID."""
        return f"{os.getpid()}-{threading.get_ident()}-{uuid4().hex[:8]}"

    def _check_stale_lock(self, lock_path: Path) -> None:
        """Check and remove stale lock file."""
        if not lock_path.exists():
            return

        try:
            mtime = datetime.fromtimestamp(lock_path.stat().st_mtime)
            if (datetime.now() - mtime).total_seconds() > self._stale_lock_timeout:
                logger.warning(f"Removing stale lock file: {lock_path}")
                lock_path.unlink()
        except Exception as e:
            logger.error(f"Error checking stale lock: {e}")

    def cleanup_stale(self) -> int:
        """Clean up stale lock files.

        Returns:
            Number of stale locks removed.
        """
        removed = 0
        now = datetime.now()

        for lock_path in self._lock_dir.glob("*.lock"):
            try:
                mtime = datetime.fromtimestamp(lock_path.stat().st_mtime)
                if (now - mtime).total_seconds() > self._stale_lock_timeout:
                    lock_path.unlink()
                    removed += 1
            except Exception:
                pass

        return removed


class RetryingLock:
    """Wrapper that adds retry logic to any lock implementation.

    Example:
        >>> base_lock = InMemoryLock()
        >>> retrying = RetryingLock(base_lock, max_retries=5)
        >>> with retrying.lock("key") as info:
        ...     pass
    """

    def __init__(
        self,
        lock: DistributedLock,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        backoff_multiplier: float = 2.0,
    ) -> None:
        """Initialize the retrying lock.

        Args:
            lock: Underlying lock implementation.
            max_retries: Maximum retry attempts.
            retry_delay: Initial delay between retries.
            backoff_multiplier: Multiplier for exponential backoff.
        """
        self._lock = lock
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._backoff_multiplier = backoff_multiplier

    def acquire(
        self,
        key: str,
        timeout: float = 30.0,
        holder_id: str | None = None,
    ) -> LockInfo:
        last_error: Exception | None = None
        delay = self._retry_delay

        for attempt in range(self._max_retries + 1):
            try:
                return self._lock.acquire(key, timeout, holder_id)
            except LockAcquisitionError as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.debug(f"Lock acquisition failed, retrying in {delay}s")
                    time.sleep(delay)
                    delay *= self._backoff_multiplier

        raise last_error or LockAcquisitionError(key, "Max retries exceeded")

    def release(self, key: str, holder_id: str) -> bool:
        return self._lock.release(key, holder_id)

    def is_locked(self, key: str) -> bool:
        return self._lock.is_locked(key)

    def get_lock_info(self, key: str) -> LockInfo | None:
        return self._lock.get_lock_info(key)

    @contextmanager
    def lock(
        self,
        key: str,
        timeout: float = 30.0,
        holder_id: str | None = None,
    ) -> Generator[LockInfo, None, None]:
        holder_id = holder_id or f"{os.getpid()}-{threading.get_ident()}-{uuid4().hex[:8]}"
        lock_info = self.acquire(key, timeout, holder_id)
        try:
            yield lock_info
        finally:
            self.release(key, holder_id)
