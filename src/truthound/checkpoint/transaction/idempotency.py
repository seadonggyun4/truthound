"""Idempotency Support for Transactions.

This module provides idempotency key management to prevent duplicate
execution of the same transaction.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from truthound.checkpoint.transaction.base import TransactionResult


logger = logging.getLogger(__name__)


@dataclass
class IdempotencyKey:
    """Represents an idempotency key for transaction deduplication.

    Idempotency keys ensure that the same transaction is not executed
    multiple times, even if the request is retried.

    Attributes:
        key: The unique key string.
        created_at: When the key was first seen.
        expires_at: When the key expires.
        request_hash: Hash of the original request.
        result: Cached result if completed.
        status: Current status (pending, completed, expired).
    """

    key: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    request_hash: str | None = None
    result: TransactionResult | None = None
    status: str = "pending"  # pending, completed, expired

    def __post_init__(self) -> None:
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=1)

    @property
    def is_expired(self) -> bool:
        """Check if this key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_completed(self) -> bool:
        """Check if the associated request is completed."""
        return self.status == "completed" and self.result is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "request_hash": self.request_hash,
            "result": self.result.to_dict() if self.result else None,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdempotencyKey":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            request_hash=data.get("request_hash"),
            result=None,  # Result reconstruction requires more context
            status=data.get("status", "pending"),
        )


@runtime_checkable
class IdempotencyStore(Protocol):
    """Protocol for idempotency key storage.

    Implementations must provide thread-safe operations for
    storing and retrieving idempotency keys.
    """

    def get(self, key: str) -> IdempotencyKey | None:
        """Get an idempotency key by its key string.

        Args:
            key: The idempotency key string.

        Returns:
            IdempotencyKey if found, None otherwise.
        """
        ...

    def set(self, idempotency_key: IdempotencyKey) -> None:
        """Store an idempotency key.

        Args:
            idempotency_key: The key to store.
        """
        ...

    def update(self, idempotency_key: IdempotencyKey) -> None:
        """Update an existing idempotency key.

        Args:
            idempotency_key: The key to update.
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete an idempotency key.

        Args:
            key: The key string to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def cleanup_expired(self) -> int:
        """Remove expired keys.

        Returns:
            Number of keys removed.
        """
        ...


class InMemoryIdempotencyStore:
    """In-memory implementation of IdempotencyStore.

    Suitable for single-process, non-persistent scenarios.
    Thread-safe using a lock.

    Example:
        >>> store = InMemoryIdempotencyStore()
        >>> key = IdempotencyKey(key="req-123")
        >>> store.set(key)
        >>> retrieved = store.get("req-123")
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize the store.

        Args:
            max_size: Maximum number of keys to store.
        """
        self._store: dict[str, IdempotencyKey] = {}
        self._lock = threading.RLock()
        self._max_size = max_size

    def get(self, key: str) -> IdempotencyKey | None:
        """Get an idempotency key."""
        with self._lock:
            idempotency_key = self._store.get(key)
            if idempotency_key and idempotency_key.is_expired:
                del self._store[key]
                return None
            return idempotency_key

    def set(self, idempotency_key: IdempotencyKey) -> None:
        """Store an idempotency key."""
        with self._lock:
            # Evict old entries if at capacity
            if len(self._store) >= self._max_size:
                self._evict_oldest()
            self._store[idempotency_key.key] = idempotency_key

    def update(self, idempotency_key: IdempotencyKey) -> None:
        """Update an existing idempotency key."""
        with self._lock:
            if idempotency_key.key in self._store:
                self._store[idempotency_key.key] = idempotency_key

    def delete(self, key: str) -> bool:
        """Delete an idempotency key."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove expired keys."""
        with self._lock:
            now = datetime.now()
            expired = [
                k for k, v in self._store.items()
                if v.expires_at and v.expires_at < now
            ]
            for key in expired:
                del self._store[key]
            return len(expired)

    def _evict_oldest(self) -> None:
        """Evict oldest entries to make room."""
        # Remove expired first
        expired_count = self.cleanup_expired()
        if expired_count > 0:
            return

        # Remove oldest 10% if still at capacity
        if len(self._store) >= self._max_size:
            to_remove = max(1, self._max_size // 10)
            sorted_keys = sorted(
                self._store.keys(),
                key=lambda k: self._store[k].created_at,
            )
            for key in sorted_keys[:to_remove]:
                del self._store[key]

    def size(self) -> int:
        """Get the number of stored keys."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Clear all stored keys."""
        with self._lock:
            self._store.clear()


class FileIdempotencyStore:
    """File-based implementation of IdempotencyStore.

    Provides persistence across process restarts. Each key is stored
    as a separate JSON file.

    Example:
        >>> store = FileIdempotencyStore("/tmp/idempotency")
        >>> key = IdempotencyKey(key="req-123")
        >>> store.set(key)
    """

    def __init__(
        self,
        storage_path: str | Path,
        ttl_seconds: int = 3600,
    ) -> None:
        """Initialize the store.

        Args:
            storage_path: Directory for storing key files.
            ttl_seconds: Default TTL for keys.
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

    def _key_path(self, key: str) -> Path:
        """Get the file path for a key."""
        # Hash the key to avoid filesystem issues with special characters
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._storage_path / f"{hashed}.json"

    def get(self, key: str) -> IdempotencyKey | None:
        """Get an idempotency key."""
        with self._lock:
            path = self._key_path(key)
            if not path.exists():
                return None

            try:
                data = json.loads(path.read_text())
                idempotency_key = IdempotencyKey.from_dict(data)

                if idempotency_key.is_expired:
                    path.unlink()
                    return None

                return idempotency_key
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to read idempotency key {key}: {e}")
                return None

    def set(self, idempotency_key: IdempotencyKey) -> None:
        """Store an idempotency key."""
        with self._lock:
            path = self._key_path(idempotency_key.key)
            data = idempotency_key.to_dict()
            path.write_text(json.dumps(data, indent=2))

    def update(self, idempotency_key: IdempotencyKey) -> None:
        """Update an existing idempotency key."""
        with self._lock:
            path = self._key_path(idempotency_key.key)
            if path.exists():
                self.set(idempotency_key)

    def delete(self, key: str) -> bool:
        """Delete an idempotency key."""
        with self._lock:
            path = self._key_path(key)
            if path.exists():
                path.unlink()
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove expired keys."""
        with self._lock:
            removed = 0
            now = datetime.now()

            for path in self._storage_path.glob("*.json"):
                try:
                    data = json.loads(path.read_text())
                    if data.get("expires_at"):
                        expires = datetime.fromisoformat(data["expires_at"])
                        if expires < now:
                            path.unlink()
                            removed += 1
                except Exception:
                    # Remove corrupted files
                    path.unlink()
                    removed += 1

            return removed

    def size(self) -> int:
        """Get the number of stored keys."""
        return len(list(self._storage_path.glob("*.json")))

    def clear(self) -> None:
        """Clear all stored keys."""
        with self._lock:
            for path in self._storage_path.glob("*.json"):
                path.unlink()


class IdempotencyManager:
    """Manages idempotency for transaction execution.

    The IdempotencyManager wraps transaction execution to prevent
    duplicate processing based on idempotency keys.

    Example:
        >>> manager = IdempotencyManager(store=InMemoryIdempotencyStore())
        >>> result = manager.execute_with_idempotency(
        ...     key="req-123",
        ...     execute_fn=lambda: execute_transaction(),
        ... )
        >>> # Second call with same key returns cached result
        >>> result2 = manager.execute_with_idempotency(
        ...     key="req-123",
        ...     execute_fn=lambda: execute_transaction(),  # Not called
        ... )
    """

    def __init__(
        self,
        store: IdempotencyStore | None = None,
        ttl_seconds: int = 3600,
    ) -> None:
        """Initialize the manager.

        Args:
            store: Idempotency store implementation.
            ttl_seconds: Default TTL for keys.
        """
        self._store = store or InMemoryIdempotencyStore()
        self._ttl_seconds = ttl_seconds

    @property
    def store(self) -> IdempotencyStore:
        """Get the underlying store."""
        return self._store

    def generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate an idempotency key from arguments.

        Args:
            *args: Positional arguments to hash.
            **kwargs: Keyword arguments to hash.

        Returns:
            Generated idempotency key.
        """
        # Create deterministic hash from arguments
        data = json.dumps(
            {"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}},
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def execute_with_idempotency(
        self,
        key: str,
        execute_fn: Any,  # Callable[[], TransactionResult]
        request_hash: str | None = None,
    ) -> tuple[TransactionResult, bool]:
        """Execute a function with idempotency protection.

        If the key already exists and is completed, returns the cached
        result without re-executing.

        Args:
            key: Idempotency key.
            execute_fn: Function to execute.
            request_hash: Optional hash of the request for validation.

        Returns:
            Tuple of (TransactionResult, was_cached).
        """
        # Check for existing key
        existing = self._store.get(key)

        if existing:
            # Validate request hash if provided
            if request_hash and existing.request_hash != request_hash:
                logger.warning(
                    f"Idempotency key {key} hash mismatch. "
                    f"Expected {existing.request_hash}, got {request_hash}"
                )

            if existing.is_completed and existing.result:
                logger.info(f"Returning cached result for key {key}")
                return (existing.result, True)

            if existing.status == "pending":
                # Another execution in progress - wait or reject
                raise IdempotencyConflictError(
                    f"Transaction with key {key} is already in progress"
                )

        # Create new idempotency key
        idempotency_key = IdempotencyKey(
            key=key,
            request_hash=request_hash,
            expires_at=datetime.now() + timedelta(seconds=self._ttl_seconds),
        )
        self._store.set(idempotency_key)

        try:
            # Execute
            result = execute_fn()

            # Update key with result
            idempotency_key.result = result
            idempotency_key.status = "completed"
            self._store.update(idempotency_key)

            return (result, False)

        except Exception as e:
            # Remove key on failure to allow retry
            self._store.delete(key)
            raise

    def invalidate(self, key: str) -> bool:
        """Invalidate an idempotency key.

        Args:
            key: Key to invalidate.

        Returns:
            True if key was found and invalidated.
        """
        return self._store.delete(key)

    def cleanup(self) -> int:
        """Clean up expired keys.

        Returns:
            Number of keys cleaned up.
        """
        return self._store.cleanup_expired()


class IdempotencyConflictError(Exception):
    """Raised when a concurrent execution with the same key is detected."""

    pass


# =============================================================================
# Decorators
# =============================================================================


def idempotent(
    key_fn: Any | None = None,  # Callable[..., str]
    store: IdempotencyStore | None = None,
    ttl_seconds: int = 3600,
) -> Any:  # Callable[..., tuple[TransactionResult, bool]]
    """Decorator to make a function idempotent.

    Args:
        key_fn: Function to generate idempotency key from arguments.
        store: Idempotency store to use.
        ttl_seconds: TTL for idempotency keys.

    Example:
        >>> @idempotent(key_fn=lambda x, y: f"{x}-{y}")
        ... def process_order(order_id: str, amount: int) -> TransactionResult:
        ...     return do_processing()
    """
    manager = IdempotencyManager(store=store, ttl_seconds=ttl_seconds)

    def decorator(fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> tuple[TransactionResult, bool]:
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = manager.generate_key(*args, **kwargs)

            return manager.execute_with_idempotency(
                key=key,
                execute_fn=lambda: fn(*args, **kwargs),
            )

        return wrapper

    return decorator
