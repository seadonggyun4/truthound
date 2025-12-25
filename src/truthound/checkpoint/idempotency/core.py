"""Core types and classes for Idempotency Framework.

This module defines the fundamental types, enums, and data structures
used throughout the idempotency system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generic, TypeVar
from uuid import uuid4


T = TypeVar("T")


class IdempotencyStatus(str, Enum):
    """Status of an idempotency record."""

    PENDING = "pending"          # Execution in progress
    COMPLETED = "completed"      # Successfully completed
    FAILED = "failed"            # Execution failed
    EXPIRED = "expired"          # Record expired
    INVALIDATED = "invalidated"  # Manually invalidated

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (
            IdempotencyStatus.COMPLETED,
            IdempotencyStatus.FAILED,
            IdempotencyStatus.EXPIRED,
            IdempotencyStatus.INVALIDATED,
        )

    @property
    def allows_retry(self) -> bool:
        """Check if retry is allowed in this state."""
        return self in (
            IdempotencyStatus.FAILED,
            IdempotencyStatus.EXPIRED,
            IdempotencyStatus.INVALIDATED,
        )


@dataclass
class IdempotencyConfig:
    """Configuration for idempotency behavior.

    Attributes:
        enabled: Whether idempotency is enabled.
        ttl_seconds: Time-to-live for idempotency records.
        lock_timeout_seconds: Timeout for acquiring locks.
        retry_on_conflict: Whether to retry on conflict.
        max_retries: Maximum retry attempts.
        retry_delay_seconds: Delay between retries.
        validate_hash: Whether to validate request hash.
        store_result: Whether to store execution result.
        cleanup_interval_seconds: Interval for cleanup tasks.
    """

    enabled: bool = True
    ttl_seconds: int = 3600
    lock_timeout_seconds: float = 30.0
    retry_on_conflict: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    validate_hash: bool = True
    store_result: bool = True
    cleanup_interval_seconds: int = 300

    def with_ttl(self, seconds: int) -> "IdempotencyConfig":
        """Create a copy with different TTL."""
        return IdempotencyConfig(
            enabled=self.enabled,
            ttl_seconds=seconds,
            lock_timeout_seconds=self.lock_timeout_seconds,
            retry_on_conflict=self.retry_on_conflict,
            max_retries=self.max_retries,
            retry_delay_seconds=self.retry_delay_seconds,
            validate_hash=self.validate_hash,
            store_result=self.store_result,
            cleanup_interval_seconds=self.cleanup_interval_seconds,
        )


@dataclass
class IdempotencyRecord(Generic[T]):
    """Record of an idempotent operation.

    This record tracks the state of an idempotent operation,
    including its result if completed.

    Attributes:
        key: Unique idempotency key.
        status: Current status of the operation.
        request_hash: Hash of the original request for validation.
        result: Stored result if completed.
        error: Error message if failed.
        created_at: When the record was created.
        updated_at: When the record was last updated.
        expires_at: When the record expires.
        attempt_count: Number of execution attempts.
        locked_by: ID of the process holding the lock.
        locked_at: When the lock was acquired.
        metadata: Additional metadata.
    """

    key: str
    status: IdempotencyStatus = IdempotencyStatus.PENDING
    request_hash: str | None = None
    result: T | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    attempt_count: int = 0
    locked_by: str | None = None
    locked_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(hours=1)

    @property
    def is_expired(self) -> bool:
        """Check if this record has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def is_locked(self) -> bool:
        """Check if this record is currently locked."""
        return self.locked_by is not None

    @property
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.is_expired:
            return True
        if self.status == IdempotencyStatus.PENDING and self.is_locked:
            return False
        return self.status.allows_retry or self.status == IdempotencyStatus.PENDING

    @property
    def has_result(self) -> bool:
        """Check if a result is available."""
        return self.status == IdempotencyStatus.COMPLETED and self.result is not None

    def mark_pending(self, lock_id: str) -> None:
        """Mark as pending with lock."""
        self.status = IdempotencyStatus.PENDING
        self.locked_by = lock_id
        self.locked_at = datetime.now()
        self.attempt_count += 1
        self.updated_at = datetime.now()

    def mark_completed(self, result: T) -> None:
        """Mark as completed with result."""
        self.status = IdempotencyStatus.COMPLETED
        self.result = result
        self.error = None
        self.locked_by = None
        self.locked_at = None
        self.updated_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        """Mark as failed with error."""
        self.status = IdempotencyStatus.FAILED
        self.error = error
        self.locked_by = None
        self.locked_at = None
        self.updated_at = datetime.now()

    def release_lock(self) -> None:
        """Release the lock without changing status."""
        self.locked_by = None
        self.locked_at = None
        self.updated_at = datetime.now()

    def extend_expiry(self, seconds: int) -> None:
        """Extend the expiry time."""
        self.expires_at = datetime.now() + timedelta(seconds=seconds)
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "status": self.status.value,
            "request_hash": self.request_hash,
            "result": self._serialize_result(),
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "attempt_count": self.attempt_count,
            "locked_by": self.locked_by,
            "locked_at": self.locked_at.isoformat() if self.locked_at else None,
            "metadata": self.metadata,
        }

    def _serialize_result(self) -> Any:
        """Serialize the result for storage."""
        if self.result is None:
            return None
        if hasattr(self.result, "to_dict"):
            return self.result.to_dict()
        return self.result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdempotencyRecord[Any]":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            status=IdempotencyStatus(data["status"]),
            request_hash=data.get("request_hash"),
            result=data.get("result"),  # Result needs to be reconstructed by caller
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            attempt_count=data.get("attempt_count", 0),
            locked_by=data.get("locked_by"),
            locked_at=(
                datetime.fromisoformat(data["locked_at"])
                if data.get("locked_at")
                else None
            ),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Exceptions
# =============================================================================


class IdempotencyError(Exception):
    """Base exception for idempotency errors."""

    def __init__(self, message: str, key: str | None = None) -> None:
        super().__init__(message)
        self.key = key


class IdempotencyConflictError(IdempotencyError):
    """Raised when concurrent execution is detected."""

    def __init__(self, key: str, locked_by: str | None = None) -> None:
        message = f"Idempotency conflict: key '{key}' is already being processed"
        if locked_by:
            message += f" by {locked_by}"
        super().__init__(message, key)
        self.locked_by = locked_by


class IdempotencyExpiredError(IdempotencyError):
    """Raised when accessing an expired record."""

    def __init__(self, key: str, expired_at: datetime) -> None:
        super().__init__(
            f"Idempotency key '{key}' expired at {expired_at.isoformat()}",
            key,
        )
        self.expired_at = expired_at


class IdempotencyHashMismatchError(IdempotencyError):
    """Raised when request hash doesn't match stored hash."""

    def __init__(
        self,
        key: str,
        expected_hash: str,
        actual_hash: str,
    ) -> None:
        super().__init__(
            f"Request hash mismatch for key '{key}': "
            f"expected {expected_hash}, got {actual_hash}",
            key,
        )
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
