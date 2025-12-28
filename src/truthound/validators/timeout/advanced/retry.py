"""Retry and rollback policies for timeout handling.

This module provides comprehensive retry and rollback mechanisms:
- Exponential backoff with jitter
- Configurable retry policies
- Rollback support for failed operations
- Dead letter queue for failed items

Example:
    from truthound.validators.timeout.advanced.retry import (
        RetryPolicy,
        with_retry,
        ExponentialBackoff,
    )

    # Create retry policy
    policy = RetryPolicy(
        max_attempts=3,
        backoff=ExponentialBackoff(base_ms=100, max_ms=5000),
    )

    # Use decorator
    @with_retry(max_attempts=3)
    def validate_with_retry(data):
        return validate(data)

    # Or use policy directly
    result = policy.execute(validate, data)
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Base class for retryable errors."""

    def __init__(self, message: str, should_retry: bool = True):
        super().__init__(message)
        self.should_retry = should_retry


class RetryExhaustedError(Exception):
    """Raised when all retries are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class BackoffStrategy(ABC):
    """Base class for backoff strategies."""

    @abstractmethod
    def get_delay_ms(self, attempt: int) -> float:
        """Get delay in milliseconds for attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in milliseconds
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the backoff state."""
        pass


class ExponentialBackoff(BackoffStrategy):
    """Exponential backoff with optional jitter.

    delay = min(max_ms, base_ms * multiplier^attempt) + jitter
    """

    def __init__(
        self,
        base_ms: float = 100.0,
        max_ms: float = 30000.0,
        multiplier: float = 2.0,
        jitter_ratio: float = 0.1,
    ):
        """Initialize exponential backoff.

        Args:
            base_ms: Base delay in milliseconds
            max_ms: Maximum delay in milliseconds
            multiplier: Exponential multiplier
            jitter_ratio: Jitter as ratio of delay (0.0-1.0)
        """
        self.base_ms = base_ms
        self.max_ms = max_ms
        self.multiplier = multiplier
        self.jitter_ratio = jitter_ratio

    def get_delay_ms(self, attempt: int) -> float:
        """Get delay with exponential backoff."""
        delay = self.base_ms * (self.multiplier ** attempt)
        delay = min(delay, self.max_ms)

        if self.jitter_ratio > 0:
            jitter = delay * self.jitter_ratio * random.random()
            delay += jitter

        return delay

    def reset(self) -> None:
        """No state to reset for simple exponential."""
        pass


class LinearBackoff(BackoffStrategy):
    """Linear backoff strategy.

    delay = min(max_ms, base_ms + increment_ms * attempt)
    """

    def __init__(
        self,
        base_ms: float = 100.0,
        increment_ms: float = 100.0,
        max_ms: float = 10000.0,
    ):
        """Initialize linear backoff.

        Args:
            base_ms: Base delay in milliseconds
            increment_ms: Increment per attempt
            max_ms: Maximum delay
        """
        self.base_ms = base_ms
        self.increment_ms = increment_ms
        self.max_ms = max_ms

    def get_delay_ms(self, attempt: int) -> float:
        """Get delay with linear backoff."""
        delay = self.base_ms + self.increment_ms * attempt
        return min(delay, self.max_ms)

    def reset(self) -> None:
        """No state to reset."""
        pass


class DecorrelatedJitter(BackoffStrategy):
    """Decorrelated jitter backoff (AWS recommended).

    More effective at spreading retry attempts than simple jitter.
    sleep = min(max_ms, random(base_ms, sleep * 3))
    """

    def __init__(
        self,
        base_ms: float = 100.0,
        max_ms: float = 30000.0,
    ):
        """Initialize decorrelated jitter.

        Args:
            base_ms: Base delay in milliseconds
            max_ms: Maximum delay
        """
        self.base_ms = base_ms
        self.max_ms = max_ms
        self._last_delay = base_ms

    def get_delay_ms(self, attempt: int) -> float:
        """Get delay with decorrelated jitter."""
        if attempt == 0:
            self._last_delay = self.base_ms

        delay = random.uniform(self.base_ms, self._last_delay * 3)
        delay = min(delay, self.max_ms)
        self._last_delay = delay

        return delay

    def reset(self) -> None:
        """Reset the last delay."""
        self._last_delay = self.base_ms


class ConstantBackoff(BackoffStrategy):
    """Constant delay backoff."""

    def __init__(self, delay_ms: float = 1000.0):
        """Initialize constant backoff.

        Args:
            delay_ms: Constant delay in milliseconds
        """
        self.delay_ms = delay_ms

    def get_delay_ms(self, attempt: int) -> float:
        """Get constant delay."""
        return self.delay_ms

    def reset(self) -> None:
        """No state to reset."""
        pass


@dataclass
class RetryResult(Generic[T]):
    """Result of a retry operation.

    Attributes:
        success: Whether operation succeeded
        value: Result value if successful
        attempts: Number of attempts made
        total_delay_ms: Total delay across retries
        errors: List of errors from failed attempts
        final_error: Final error if all retries failed
    """

    success: bool
    value: T | None = None
    attempts: int = 1
    total_delay_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    final_error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @classmethod
    def ok(cls, value: T, attempts: int, delay_ms: float) -> "RetryResult[T]":
        """Create success result."""
        return cls(
            success=True,
            value=value,
            attempts=attempts,
            total_delay_ms=delay_ms,
            completed_at=datetime.now(timezone.utc),
        )

    @classmethod
    def failed(
        cls,
        attempts: int,
        delay_ms: float,
        errors: list[str],
        final_error: str,
    ) -> "RetryResult[T]":
        """Create failure result."""
        return cls(
            success=False,
            attempts=attempts,
            total_delay_ms=delay_ms,
            errors=errors,
            final_error=final_error,
            completed_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_delay_ms": self.total_delay_ms,
            "errors": self.errors,
            "final_error": self.final_error,
        }


@dataclass
class RetryPolicy:
    """Policy for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts
        backoff: Backoff strategy
        retryable_exceptions: Exception types to retry
        non_retryable_exceptions: Exception types to not retry
        on_retry: Callback for retry events
        on_success: Callback for success
        on_failure: Callback for final failure
    """

    max_attempts: int = 3
    backoff: BackoffStrategy = field(default_factory=lambda: ExponentialBackoff())
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )
    non_retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (KeyboardInterrupt, SystemExit)
    )
    on_retry: Callable[[int, Exception], None] | None = None
    on_success: Callable[[Any, int], None] | None = None
    on_failure: Callable[[int, Exception], None] | None = None

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if should retry for this exception.

        Args:
            exception: The exception that occurred
            attempt: Current attempt number

        Returns:
            True if should retry
        """
        # Check attempt limit
        if attempt >= self.max_attempts:
            return False

        # Check non-retryable
        if isinstance(exception, self.non_retryable_exceptions):
            return False

        # Check RetryableError
        if isinstance(exception, RetryableError):
            return exception.should_retry

        # Check retryable
        return isinstance(exception, self.retryable_exceptions)

    def execute(
        self,
        operation: Callable[[], T],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult[T]:
        """Execute operation with retry policy.

        Args:
            operation: Operation to execute
            *args: Arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            RetryResult
        """
        errors: list[str] = []
        total_delay_ms = 0.0
        attempt = 0
        last_exception: Exception | None = None

        self.backoff.reset()

        while attempt < self.max_attempts:
            try:
                result = operation(*args, **kwargs)

                if self.on_success:
                    self.on_success(result, attempt + 1)

                return RetryResult.ok(result, attempt + 1, total_delay_ms)

            except Exception as e:
                last_exception = e
                errors.append(str(e))
                attempt += 1

                if not self.should_retry(e, attempt):
                    break

                if self.on_retry:
                    self.on_retry(attempt, e)

                # Apply backoff
                delay_ms = self.backoff.get_delay_ms(attempt - 1)
                total_delay_ms += delay_ms
                time.sleep(delay_ms / 1000)

        if self.on_failure and last_exception:
            self.on_failure(attempt, last_exception)

        return RetryResult.failed(
            attempt,
            total_delay_ms,
            errors,
            str(last_exception) if last_exception else "Unknown error",
        )

    async def execute_async(
        self,
        operation: Callable[[], T],
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult[T]:
        """Execute operation asynchronously with retry.

        Args:
            operation: Operation to execute
            *args: Arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            RetryResult
        """
        import asyncio

        errors: list[str] = []
        total_delay_ms = 0.0
        attempt = 0
        last_exception: Exception | None = None

        self.backoff.reset()

        while attempt < self.max_attempts:
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        lambda: operation(*args, **kwargs),
                    )

                if self.on_success:
                    self.on_success(result, attempt + 1)

                return RetryResult.ok(result, attempt + 1, total_delay_ms)

            except Exception as e:
                last_exception = e
                errors.append(str(e))
                attempt += 1

                if not self.should_retry(e, attempt):
                    break

                if self.on_retry:
                    self.on_retry(attempt, e)

                delay_ms = self.backoff.get_delay_ms(attempt - 1)
                total_delay_ms += delay_ms
                await asyncio.sleep(delay_ms / 1000)

        if self.on_failure and last_exception:
            self.on_failure(attempt, last_exception)

        return RetryResult.failed(
            attempt,
            total_delay_ms,
            errors,
            str(last_exception) if last_exception else "Unknown error",
        )


@dataclass
class RollbackAction:
    """Action to perform on rollback.

    Attributes:
        name: Action name
        action: Rollback function
        executed: Whether action was executed
        error: Error if rollback failed
    """

    name: str
    action: Callable[[], None]
    executed: bool = False
    error: str | None = None


class RollbackManager:
    """Manages rollback actions for failed operations.

    Use this to track actions that need to be undone if a
    multi-step operation fails.

    Example:
        with RollbackManager() as rollback:
            # Create file
            create_file()
            rollback.register("delete_file", delete_file)

            # If this fails, delete_file will be called
            process_file()
    """

    def __init__(self) -> None:
        """Initialize rollback manager."""
        self._actions: list[RollbackAction] = []
        self._lock = threading.Lock()
        self._committed = False

    def register(self, name: str, action: Callable[[], None]) -> None:
        """Register a rollback action.

        Args:
            name: Action name for logging
            action: Rollback function
        """
        with self._lock:
            self._actions.append(RollbackAction(name=name, action=action))

    def commit(self) -> None:
        """Commit the transaction (clear rollback actions)."""
        with self._lock:
            self._committed = True
            self._actions.clear()

    def rollback(self) -> list[RollbackAction]:
        """Execute all rollback actions in reverse order.

        Returns:
            List of executed rollback actions
        """
        with self._lock:
            if self._committed:
                return []

            results: list[RollbackAction] = []

            # Execute in reverse order
            for action in reversed(self._actions):
                try:
                    action.action()
                    action.executed = True
                except Exception as e:
                    action.error = str(e)
                    logger.warning(f"Rollback action '{action.name}' failed: {e}")
                results.append(action)

            self._actions.clear()
            return results

    def __enter__(self) -> "RollbackManager":
        """Enter context."""
        return self

    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_val: Exception | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context, rollback on exception."""
        if exc_type is not None and not self._committed:
            self.rollback()
        return False  # Don't suppress exception


class DeadLetterQueue(Generic[T]):
    """Queue for items that failed all retry attempts.

    Stores failed items for later analysis or manual processing.

    Example:
        dlq = DeadLetterQueue()

        # Add failed item
        dlq.add(item, error="Validation failed", attempts=3)

        # Process dead letters
        for item, meta in dlq.get_all():
            # Handle failed item
            ...
    """

    def __init__(self, max_size: int = 1000):
        """Initialize dead letter queue.

        Args:
            max_size: Maximum items to store
        """
        self.max_size = max_size
        self._items: list[tuple[T, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def add(
        self,
        item: T,
        error: str,
        attempts: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add item to dead letter queue.

        Args:
            item: Failed item
            error: Error message
            attempts: Number of attempts made
            metadata: Additional metadata
        """
        meta = {
            "error": error,
            "attempts": attempts,
            "added_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        with self._lock:
            self._items.append((item, meta))
            # Trim if over max size
            if len(self._items) > self.max_size:
                self._items = self._items[-self.max_size:]

    def get_all(self) -> list[tuple[T, dict[str, Any]]]:
        """Get all items in the queue.

        Returns:
            List of (item, metadata) tuples
        """
        with self._lock:
            return list(self._items)

    def pop(self) -> tuple[T, dict[str, Any]] | None:
        """Remove and return oldest item.

        Returns:
            (item, metadata) tuple or None if empty
        """
        with self._lock:
            if self._items:
                return self._items.pop(0)
            return None

    def clear(self) -> int:
        """Clear all items.

        Returns:
            Number of items cleared
        """
        with self._lock:
            count = len(self._items)
            self._items.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def with_retry(
    max_attempts: int = 3,
    backoff: BackoffStrategy | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add retry behavior to a function.

    Args:
        max_attempts: Maximum attempts
        backoff: Backoff strategy
        retryable_exceptions: Exceptions to retry

    Returns:
        Decorated function

    Example:
        @with_retry(max_attempts=3)
        def validate(data):
            return check(data)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            policy = RetryPolicy(
                max_attempts=max_attempts,
                backoff=backoff or ExponentialBackoff(),
                retryable_exceptions=retryable_exceptions,
            )

            result = policy.execute(lambda: func(*args, **kwargs))

            if result.success:
                return result.value  # type: ignore
            else:
                raise RetryExhaustedError(
                    f"All {result.attempts} attempts failed: {result.final_error}",
                    result.attempts,
                )

        return wrapper

    return decorator


def create_retry_policy(
    max_attempts: int = 3,
    base_delay_ms: float = 100.0,
    max_delay_ms: float = 30000.0,
    backoff_type: str = "exponential",
) -> RetryPolicy:
    """Create a retry policy with common defaults.

    Args:
        max_attempts: Maximum attempts
        base_delay_ms: Base delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds
        backoff_type: Type of backoff (exponential, linear, decorrelated)

    Returns:
        RetryPolicy
    """
    if backoff_type == "linear":
        backoff: BackoffStrategy = LinearBackoff(base_delay_ms, base_delay_ms, max_delay_ms)
    elif backoff_type == "decorrelated":
        backoff = DecorrelatedJitter(base_delay_ms, max_delay_ms)
    else:
        backoff = ExponentialBackoff(base_delay_ms, max_delay_ms)

    return RetryPolicy(
        max_attempts=max_attempts,
        backoff=backoff,
    )
