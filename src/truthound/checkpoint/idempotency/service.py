"""Idempotency Service and Middleware.

This module provides the main service for idempotent execution
and middleware for integrating with the action system.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable, Generator, Generic, TypeVar

from truthound.checkpoint.idempotency.core import (
    IdempotencyConfig,
    IdempotencyConflictError,
    IdempotencyError,
    IdempotencyExpiredError,
    IdempotencyHashMismatchError,
    IdempotencyRecord,
    IdempotencyStatus,
)
from truthound.checkpoint.idempotency.fingerprint import (
    RequestFingerprint,
    quick_fingerprint,
)
from truthound.checkpoint.idempotency.locking import (
    DistributedLock,
    InMemoryLock,
    LockAcquisitionError,
)
from truthound.checkpoint.idempotency.stores import (
    IdempotencyStore,
    InMemoryIdempotencyStore,
)

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionResult, BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult


logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class ExecutionResult(Generic[T]):
    """Result of an idempotent execution.

    Attributes:
        result: The execution result.
        was_cached: Whether result came from cache.
        record: The idempotency record.
        execution_time_ms: Time spent executing (0 if cached).
    """

    result: T
    was_cached: bool = False
    record: IdempotencyRecord[T] | None = None
    execution_time_ms: float = 0.0


class IdempotencyService:
    """Service for managing idempotent operations.

    The IdempotencyService provides a unified interface for executing
    operations with idempotency guarantees, including:
    - Automatic deduplication
    - Result caching
    - Distributed locking
    - Retry handling

    Example:
        >>> service = IdempotencyService()
        >>>
        >>> # Execute with idempotency
        >>> result = service.execute(
        ...     key="unique-operation-key",
        ...     execute_fn=lambda: perform_operation(),
        ... )
        >>>
        >>> # Check if cached
        >>> if result.was_cached:
        ...     print("Returned cached result")
    """

    def __init__(
        self,
        store: IdempotencyStore | None = None,
        lock: DistributedLock | None = None,
        config: IdempotencyConfig | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            store: Storage backend for idempotency records.
            lock: Distributed lock implementation.
            config: Idempotency configuration.
        """
        self._store = store or InMemoryIdempotencyStore()
        self._lock = lock or InMemoryLock()
        self._config = config or IdempotencyConfig()
        self._local = threading.local()

    @property
    def config(self) -> IdempotencyConfig:
        """Get the configuration."""
        return self._config

    @property
    def store(self) -> IdempotencyStore:
        """Get the storage backend."""
        return self._store

    def execute(
        self,
        key: str,
        execute_fn: Callable[[], T],
        request_hash: str | None = None,
        config: IdempotencyConfig | None = None,
    ) -> ExecutionResult[T]:
        """Execute a function with idempotency.

        Args:
            key: Unique idempotency key.
            execute_fn: Function to execute.
            request_hash: Optional hash for request validation.
            config: Optional config override.

        Returns:
            ExecutionResult with result and metadata.

        Raises:
            IdempotencyConflictError: If another execution is in progress.
            IdempotencyHashMismatchError: If request hash doesn't match.
        """
        config = config or self._config

        if not config.enabled:
            start_time = time.time()
            result = execute_fn()
            return ExecutionResult(
                result=result,
                was_cached=False,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        # Check for existing record
        existing = self._store.get(key)

        if existing is not None:
            # Validate hash if configured
            if config.validate_hash and request_hash and existing.request_hash:
                if existing.request_hash != request_hash:
                    raise IdempotencyHashMismatchError(
                        key,
                        existing.request_hash,
                        request_hash,
                    )

            # Return cached result if completed
            if existing.has_result:
                logger.debug(f"Returning cached result for key {key}")
                return ExecutionResult(
                    result=existing.result,
                    was_cached=True,
                    record=existing,
                )

            # Check if still being processed
            if existing.status == IdempotencyStatus.PENDING and existing.is_locked:
                if not existing.is_expired:
                    raise IdempotencyConflictError(key, existing.locked_by)

        # Acquire lock and execute
        holder_id = self._generate_holder_id()

        try:
            with self._lock.lock(key, config.lock_timeout_seconds, holder_id):
                return self._execute_with_lock(
                    key=key,
                    execute_fn=execute_fn,
                    request_hash=request_hash,
                    holder_id=holder_id,
                    config=config,
                )
        except LockAcquisitionError as e:
            raise IdempotencyConflictError(key, e.holder) from e

    def _execute_with_lock(
        self,
        key: str,
        execute_fn: Callable[[], T],
        request_hash: str | None,
        holder_id: str,
        config: IdempotencyConfig,
    ) -> ExecutionResult[T]:
        """Execute with lock held."""
        # Create or update record
        record: IdempotencyRecord[T] = IdempotencyRecord(
            key=key,
            request_hash=request_hash,
            expires_at=datetime.now() + timedelta(seconds=config.ttl_seconds),
        )
        record.mark_pending(holder_id)
        self._store.set(record)

        start_time = time.time()

        try:
            result = execute_fn()

            record.mark_completed(result)
            if config.store_result:
                self._store.update(record)

            return ExecutionResult(
                result=result,
                was_cached=False,
                record=record,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            record.mark_failed(str(e))
            self._store.update(record)
            raise

    def get_result(self, key: str) -> T | None:
        """Get cached result for a key.

        Args:
            key: Idempotency key.

        Returns:
            Cached result if available, None otherwise.
        """
        record = self._store.get(key)
        if record and record.has_result:
            return record.result
        return None

    def get_status(self, key: str) -> IdempotencyStatus | None:
        """Get status for a key.

        Args:
            key: Idempotency key.

        Returns:
            Status if found, None otherwise.
        """
        record = self._store.get(key)
        return record.status if record else None

    def invalidate(self, key: str) -> bool:
        """Invalidate an idempotency key.

        Args:
            key: Key to invalidate.

        Returns:
            True if invalidated.
        """
        return self._store.delete(key)

    def cleanup(self) -> int:
        """Cleanup expired records.

        Returns:
            Number of records cleaned up.
        """
        return self._store.cleanup_expired()

    @contextmanager
    def scope(
        self,
        prefix: str = "",
        config: IdempotencyConfig | None = None,
    ) -> Generator["ScopedIdempotencyService", None, None]:
        """Create a scoped service with a key prefix.

        Args:
            prefix: Prefix for all keys in this scope.
            config: Optional config override.

        Yields:
            Scoped service instance.

        Example:
            >>> with service.scope("user-123:") as scoped:
            ...     scoped.execute("action-1", lambda: do_action())
        """
        yield ScopedIdempotencyService(
            parent=self,
            prefix=prefix,
            config=config or self._config,
        )

    def _generate_holder_id(self) -> str:
        """Generate unique holder ID."""
        import os
        from uuid import uuid4
        return f"{os.getpid()}-{threading.get_ident()}-{uuid4().hex[:8]}"


class ScopedIdempotencyService:
    """Scoped idempotency service with key prefix."""

    def __init__(
        self,
        parent: IdempotencyService,
        prefix: str,
        config: IdempotencyConfig,
    ) -> None:
        self._parent = parent
        self._prefix = prefix
        self._config = config

    def execute(
        self,
        key: str,
        execute_fn: Callable[[], T],
        request_hash: str | None = None,
    ) -> ExecutionResult[T]:
        """Execute with prefixed key."""
        return self._parent.execute(
            key=f"{self._prefix}{key}",
            execute_fn=execute_fn,
            request_hash=request_hash,
            config=self._config,
        )

    def get_result(self, key: str) -> T | None:
        return self._parent.get_result(f"{self._prefix}{key}")

    def invalidate(self, key: str) -> bool:
        return self._parent.invalidate(f"{self._prefix}{key}")


class IdempotencyMiddleware:
    """Middleware for adding idempotency to action execution.

    This middleware wraps action execution to provide automatic
    idempotency based on action configuration.

    Example:
        >>> middleware = IdempotencyMiddleware()
        >>>
        >>> # Wrap action execution
        >>> @middleware.wrap
        ... def execute_action(action, checkpoint_result):
        ...     return action.execute(checkpoint_result)
        >>>
        >>> # Or use decorator on action class
        >>> @middleware.action(key_fn=lambda a, r: f"{a.name}-{r.id}")
        ... class MyAction(BaseAction):
        ...     pass
    """

    def __init__(
        self,
        service: IdempotencyService | None = None,
        default_config: IdempotencyConfig | None = None,
    ) -> None:
        """Initialize middleware.

        Args:
            service: Idempotency service to use.
            default_config: Default configuration.
        """
        self._service = service or IdempotencyService()
        self._default_config = default_config or IdempotencyConfig()

    @property
    def service(self) -> IdempotencyService:
        """Get the idempotency service."""
        return self._service

    def execute_action(
        self,
        action: "BaseAction[Any]",
        checkpoint_result: "CheckpointResult",
        key: str | None = None,
        key_fn: Callable[["BaseAction[Any]", "CheckpointResult"], str] | None = None,
    ) -> ExecutionResult["ActionResult"]:
        """Execute an action with idempotency.

        Args:
            action: Action to execute.
            checkpoint_result: Checkpoint result.
            key: Explicit idempotency key.
            key_fn: Function to generate key.

        Returns:
            ExecutionResult with action result.
        """
        # Generate key
        if key is None:
            if key_fn:
                key = key_fn(action, checkpoint_result)
            else:
                key = self._default_key(action, checkpoint_result)

        # Generate request hash
        request_hash = quick_fingerprint(
            action.name,
            action.action_type,
            checkpoint_result.status if hasattr(checkpoint_result, "status") else str(checkpoint_result),
        )

        return self._service.execute(
            key=key,
            execute_fn=lambda: action.execute(checkpoint_result),
            request_hash=request_hash,
        )

    def wrap(
        self,
        key_fn: Callable[..., str] | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., ExecutionResult[T]]]:
        """Decorator to wrap a function with idempotency.

        Args:
            key_fn: Function to generate idempotency key from arguments.

        Returns:
            Decorator function.
        """
        def decorator(fn: Callable[..., T]) -> Callable[..., ExecutionResult[T]]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> ExecutionResult[T]:
                if key_fn:
                    key = key_fn(*args, **kwargs)
                else:
                    key = quick_fingerprint(*args, **kwargs)

                return self._service.execute(
                    key=key,
                    execute_fn=lambda: fn(*args, **kwargs),
                )

            return wrapper

        return decorator

    def action(
        self,
        key_fn: Callable[["BaseAction[Any]", "CheckpointResult"], str] | None = None,
        enabled: bool = True,
    ) -> Callable[[type], type]:
        """Decorator for action classes to enable idempotency.

        Args:
            key_fn: Function to generate idempotency key.
            enabled: Whether idempotency is enabled.

        Returns:
            Class decorator.

        Example:
            >>> @middleware.action(key_fn=lambda a, r: f"{a.name}-{r.run_id}")
            ... class MyAction(BaseAction):
            ...     def _execute(self, result):
            ...         return do_work()
        """
        middleware = self

        def decorator(cls: type) -> type:
            if not enabled:
                return cls

            original_execute = cls.execute

            def idempotent_execute(
                self: Any,
                checkpoint_result: "CheckpointResult",
            ) -> "ActionResult":
                result = middleware.execute_action(
                    action=self,
                    checkpoint_result=checkpoint_result,
                    key_fn=key_fn,
                )
                return result.result

            cls.execute = idempotent_execute
            cls._idempotency_enabled = True

            return cls

        return decorator

    def _default_key(
        self,
        action: "BaseAction[Any]",
        checkpoint_result: "CheckpointResult",
    ) -> str:
        """Generate default idempotency key."""
        return quick_fingerprint(
            action.name,
            action.action_type,
            getattr(action, "config", {}),
        )


# =============================================================================
# Decorators
# =============================================================================


def idempotent(
    key: str | None = None,
    key_fn: Callable[..., str] | None = None,
    store: IdempotencyStore | None = None,
    ttl_seconds: int = 3600,
) -> Callable[[Callable[..., T]], Callable[..., ExecutionResult[T]]]:
    """Decorator to make a function idempotent.

    Args:
        key: Static idempotency key (use for fixed operations).
        key_fn: Function to generate key from arguments.
        store: Storage backend.
        ttl_seconds: Time-to-live for cached results.

    Returns:
        Decorator function.

    Example:
        >>> @idempotent(key_fn=lambda user_id, amount: f"charge-{user_id}-{amount}")
        ... def charge_user(user_id: str, amount: float) -> Receipt:
        ...     return process_payment(user_id, amount)
        >>>
        >>> result = charge_user("user-123", 99.99)
        >>> if result.was_cached:
        ...     print("Already charged!")
    """
    service = IdempotencyService(
        store=store,
        config=IdempotencyConfig(ttl_seconds=ttl_seconds),
    )

    def decorator(fn: Callable[..., T]) -> Callable[..., ExecutionResult[T]]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> ExecutionResult[T]:
            if key:
                idempotency_key = key
            elif key_fn:
                idempotency_key = key_fn(*args, **kwargs)
            else:
                idempotency_key = quick_fingerprint(fn.__name__, *args, **kwargs)

            return service.execute(
                key=idempotency_key,
                execute_fn=lambda: fn(*args, **kwargs),
            )

        # Attach service for testing/introspection
        wrapper._idempotency_service = service

        return wrapper

    return decorator


def idempotent_action(
    key_fn: Callable[["BaseAction[Any]", "CheckpointResult"], str] | None = None,
) -> Callable[[type], type]:
    """Decorator for action classes to enable idempotency.

    This is a convenience wrapper around IdempotencyMiddleware.action().

    Args:
        key_fn: Function to generate idempotency key.

    Returns:
        Class decorator.

    Example:
        >>> @idempotent_action(
        ...     key_fn=lambda action, result: f"{action.name}-{result.dataset_id}"
        ... )
        ... class StoreResultAction(BaseAction):
        ...     def _execute(self, result):
        ...         return store_result(result)
    """
    middleware = IdempotencyMiddleware()
    return middleware.action(key_fn=key_fn)
