"""Deadline propagation and timeout budgeting for distributed validation.

This module implements deadline propagation patterns for distributed systems:
- DeadlineContext: Carries deadline information across service boundaries
- TimeoutBudget: Manages timeout allocation across multiple operations
- DeadlinePropagator: Propagates deadlines through async contexts

The key insight is that in distributed systems, the overall deadline must be
shared and tracked across all participating services to prevent cascading
timeouts and ensure consistent behavior.

Example:
    # Set up a deadline context
    with DeadlineContext.from_seconds(60) as ctx:
        # Allocate budget for sub-operations
        budget = ctx.allocate(validators=40, reporting=15, cleanup=5)

        # Execute with deadline awareness
        result = validate_with_deadline(data, budget.validators)

        # Check remaining time
        if ctx.remaining_seconds < 10:
            # Use fast path
            ...
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

T = TypeVar("T")

# Context variable for deadline propagation
_deadline_context: contextvars.ContextVar["DeadlineContext | None"] = (
    contextvars.ContextVar("deadline_context", default=None)
)


class DeadlineStatus(str, Enum):
    """Status of a deadline."""

    ACTIVE = "active"       # Deadline is in the future
    EXPIRED = "expired"     # Deadline has passed
    CANCELLED = "cancelled"  # Deadline was cancelled


@dataclass
class DeadlineContext:
    """Context for propagating deadlines across service boundaries.

    DeadlineContext provides a way to track and propagate deadline information
    in distributed systems. It supports:
    - Absolute deadlines (UTC timestamp)
    - Remaining time calculation
    - Budget allocation for sub-operations
    - Serialization for wire transmission

    Example:
        # Create from duration
        ctx = DeadlineContext.from_seconds(60)

        # Check time remaining
        print(f"Remaining: {ctx.remaining_seconds}s")

        # Create sub-context with portion of budget
        sub_ctx = ctx.with_budget_fraction(0.5)

        # Serialize for transmission
        data = ctx.to_dict()
        ctx2 = DeadlineContext.from_dict(data)
    """

    deadline_utc: datetime
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation_id: str = ""
    parent_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _status: DeadlineStatus = field(default=DeadlineStatus.ACTIVE, init=False)
    _cancelled: bool = field(default=False, init=False)

    @classmethod
    def from_seconds(
        cls,
        seconds: float,
        operation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "DeadlineContext":
        """Create deadline from seconds from now.

        Args:
            seconds: Seconds until deadline
            operation_id: Optional operation identifier
            metadata: Optional metadata

        Returns:
            New DeadlineContext
        """
        now = datetime.now(timezone.utc)
        deadline = now + timedelta(seconds=seconds)
        return cls(
            deadline_utc=deadline,
            created_at=now,
            operation_id=operation_id,
            metadata=metadata or {},
        )

    @classmethod
    def from_timestamp(
        cls,
        timestamp_utc: float,
        operation_id: str = "",
    ) -> "DeadlineContext":
        """Create deadline from UTC timestamp.

        Args:
            timestamp_utc: Unix timestamp in UTC
            operation_id: Optional operation identifier

        Returns:
            New DeadlineContext
        """
        deadline = datetime.fromtimestamp(timestamp_utc, tz=timezone.utc)
        return cls(deadline_utc=deadline, operation_id=operation_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeadlineContext":
        """Deserialize from dictionary (for wire transmission).

        Args:
            data: Dictionary with deadline data

        Returns:
            New DeadlineContext
        """
        deadline = datetime.fromisoformat(data["deadline_utc"])
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)

        created = data.get("created_at")
        if created:
            created = datetime.fromisoformat(created)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        else:
            created = datetime.now(timezone.utc)

        return cls(
            deadline_utc=deadline,
            created_at=created,
            operation_id=data.get("operation_id", ""),
            parent_id=data.get("parent_id", ""),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for wire transmission.

        Returns:
            Dictionary representation
        """
        return {
            "deadline_utc": self.deadline_utc.isoformat(),
            "created_at": self.created_at.isoformat(),
            "operation_id": self.operation_id,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "remaining_seconds": self.remaining_seconds,
        }

    @property
    def remaining_seconds(self) -> float:
        """Get remaining time until deadline in seconds."""
        if self._cancelled:
            return 0.0
        now = datetime.now(timezone.utc)
        remaining = (self.deadline_utc - now).total_seconds()
        return max(0.0, remaining)

    @property
    def remaining_timedelta(self) -> timedelta:
        """Get remaining time as timedelta."""
        return timedelta(seconds=self.remaining_seconds)

    @property
    def is_expired(self) -> bool:
        """Check if deadline has passed."""
        return self.remaining_seconds <= 0

    @property
    def is_cancelled(self) -> bool:
        """Check if deadline was cancelled."""
        return self._cancelled

    @property
    def status(self) -> DeadlineStatus:
        """Get current deadline status."""
        if self._cancelled:
            return DeadlineStatus.CANCELLED
        if self.is_expired:
            return DeadlineStatus.EXPIRED
        return DeadlineStatus.ACTIVE

    def cancel(self) -> None:
        """Cancel the deadline."""
        self._cancelled = True
        self._status = DeadlineStatus.CANCELLED

    def with_budget_fraction(
        self,
        fraction: float,
        operation_id: str = "",
    ) -> "DeadlineContext":
        """Create a sub-deadline with a fraction of remaining time.

        Args:
            fraction: Fraction of remaining time (0.0-1.0)
            operation_id: Operation ID for sub-context

        Returns:
            New DeadlineContext with reduced deadline
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"Fraction must be in (0, 1], got {fraction}")

        remaining = self.remaining_seconds * fraction
        return DeadlineContext.from_seconds(
            remaining,
            operation_id=operation_id or self.operation_id,
            metadata={**self.metadata, "parent_id": self.operation_id},
        )

    def allocate(self, **allocations: float) -> "BudgetAllocation":
        """Allocate remaining time to named sub-operations.

        Args:
            **allocations: Named allocations in seconds

        Returns:
            BudgetAllocation with named deadlines

        Example:
            allocation = ctx.allocate(
                validation=30,
                reporting=10,
                cleanup=5,
            )
            validate_with_deadline(data, allocation.validation)
        """
        return BudgetAllocation.from_context(self, allocations)

    def check(self) -> None:
        """Check if deadline has expired and raise if so.

        Raises:
            DeadlineExceededError: If deadline has passed
        """
        if self.is_expired:
            raise DeadlineExceededError(
                f"Deadline exceeded for operation '{self.operation_id}'"
            )

    def __enter__(self) -> "DeadlineContext":
        """Enter context and set as current deadline."""
        self._token = _deadline_context.set(self)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous deadline."""
        _deadline_context.reset(self._token)


class DeadlineExceededError(Exception):
    """Raised when a deadline has been exceeded."""

    def __init__(self, message: str, context: DeadlineContext | None = None):
        super().__init__(message)
        self.context = context


@dataclass
class BudgetAllocation:
    """Allocation of timeout budget across multiple operations.

    Provides named access to deadline contexts for sub-operations.

    Example:
        allocation = BudgetAllocation.from_context(ctx, {
            "validation": 30,
            "reporting": 10,
        })

        validate(data, allocation.validation.remaining_seconds)
    """

    allocations: dict[str, DeadlineContext] = field(default_factory=dict)
    total_allocated: float = 0.0
    remaining_unallocated: float = 0.0

    @classmethod
    def from_context(
        cls,
        parent: DeadlineContext,
        allocations: dict[str, float],
    ) -> "BudgetAllocation":
        """Create allocation from parent context and seconds per operation.

        Args:
            parent: Parent deadline context
            allocations: Mapping of operation names to seconds

        Returns:
            BudgetAllocation with named deadline contexts
        """
        total = sum(allocations.values())
        remaining = parent.remaining_seconds

        if total > remaining:
            # Scale down proportionally if over budget
            scale = remaining / total
            allocations = {k: v * scale for k, v in allocations.items()}
            total = remaining

        result = cls(
            total_allocated=total,
            remaining_unallocated=remaining - total,
        )

        for name, seconds in allocations.items():
            result.allocations[name] = DeadlineContext.from_seconds(
                seconds,
                operation_id=f"{parent.operation_id}/{name}",
            )

        return result

    def __getattr__(self, name: str) -> DeadlineContext:
        """Get allocation by name."""
        if name in self.allocations:
            return self.allocations[name]
        raise AttributeError(f"No allocation named '{name}'")

    def get(self, name: str, default: DeadlineContext | None = None) -> DeadlineContext | None:
        """Get allocation by name with default."""
        return self.allocations.get(name, default)


@dataclass
class TimeoutBudget:
    """Manages timeout budget across multiple operations.

    TimeoutBudget tracks total time available and allocates it across
    multiple operations, ensuring the total doesn't exceed the budget.

    Example:
        budget = TimeoutBudget(total_seconds=120)

        # Allocate time for operations
        validation_budget = budget.allocate("validation", 60)
        report_budget = budget.allocate("reporting", 30)

        # Check remaining
        print(f"Remaining: {budget.remaining_seconds}s")

        # Use allocated time
        with budget.use("validation") as ctx:
            validate(data)
    """

    total_seconds: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _allocations: dict[str, float] = field(default_factory=dict, init=False)
    _used: dict[str, float] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def elapsed_seconds(self) -> float:
        """Get seconds elapsed since budget creation."""
        now = datetime.now(timezone.utc)
        return (now - self.created_at).total_seconds()

    @property
    def remaining_seconds(self) -> float:
        """Get remaining seconds in budget."""
        return max(0.0, self.total_seconds - self.elapsed_seconds)

    @property
    def allocated_seconds(self) -> float:
        """Get total seconds allocated."""
        with self._lock:
            return sum(self._allocations.values())

    @property
    def unallocated_seconds(self) -> float:
        """Get seconds not yet allocated."""
        return max(0.0, self.remaining_seconds - self.allocated_seconds)

    def allocate(self, name: str, seconds: float) -> DeadlineContext:
        """Allocate time for a named operation.

        Args:
            name: Operation name
            seconds: Seconds to allocate

        Returns:
            DeadlineContext for the operation

        Raises:
            ValueError: If not enough time remaining
        """
        with self._lock:
            if seconds > self.unallocated_seconds:
                raise ValueError(
                    f"Cannot allocate {seconds}s for '{name}': "
                    f"only {self.unallocated_seconds}s remaining"
                )
            self._allocations[name] = seconds

        return DeadlineContext.from_seconds(seconds, operation_id=name)

    def allocate_fraction(self, name: str, fraction: float) -> DeadlineContext:
        """Allocate a fraction of remaining time.

        Args:
            name: Operation name
            fraction: Fraction of remaining time (0.0-1.0)

        Returns:
            DeadlineContext for the operation
        """
        seconds = self.unallocated_seconds * fraction
        return self.allocate(name, seconds)

    def record_usage(self, name: str, seconds: float) -> None:
        """Record actual time used for an operation.

        Args:
            name: Operation name
            seconds: Actual seconds used
        """
        with self._lock:
            self._used[name] = seconds

    @contextmanager
    def use(self, name: str, seconds: float | None = None) -> Generator[DeadlineContext, None, None]:
        """Context manager to allocate and track usage.

        Args:
            name: Operation name
            seconds: Seconds to allocate (None = use remaining)

        Yields:
            DeadlineContext for the operation
        """
        if seconds is None:
            seconds = self.unallocated_seconds * 0.5  # Default to half remaining

        ctx = self.allocate(name, seconds)
        start = time.time()

        try:
            yield ctx
        finally:
            actual = time.time() - start
            self.record_usage(name, actual)

    def get_summary(self) -> dict[str, Any]:
        """Get budget summary.

        Returns:
            Dictionary with budget statistics
        """
        with self._lock:
            return {
                "total_seconds": self.total_seconds,
                "elapsed_seconds": self.elapsed_seconds,
                "remaining_seconds": self.remaining_seconds,
                "allocated": dict(self._allocations),
                "used": dict(self._used),
                "unallocated_seconds": self.unallocated_seconds,
            }


class DeadlinePropagator:
    """Propagates deadlines through async and sync contexts.

    This class manages deadline context propagation for both synchronous
    and asynchronous code, ensuring deadline information is available
    throughout the call stack.

    Example:
        propagator = DeadlinePropagator()

        # Set deadline in async context
        async with propagator.async_context(60) as ctx:
            result = await some_async_operation()

        # Get current deadline
        current = propagator.current()
        if current and current.remaining_seconds < 10:
            # Use fast path
            ...
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @property
    def current(self) -> DeadlineContext | None:
        """Get current deadline context."""
        # Try context var first (works in async)
        ctx = _deadline_context.get()
        if ctx is not None:
            return ctx

        # Fall back to thread local
        return getattr(self._local, "deadline", None)

    @contextmanager
    def context(
        self,
        seconds: float,
        operation_id: str = "",
    ) -> Generator[DeadlineContext, None, None]:
        """Create a deadline context for synchronous code.

        Args:
            seconds: Seconds until deadline
            operation_id: Optional operation identifier

        Yields:
            DeadlineContext
        """
        ctx = DeadlineContext.from_seconds(seconds, operation_id)
        old = getattr(self._local, "deadline", None)
        self._local.deadline = ctx

        try:
            with ctx:
                yield ctx
        finally:
            self._local.deadline = old

    @contextmanager
    def async_context(
        self,
        seconds: float,
        operation_id: str = "",
    ) -> Generator[DeadlineContext, None, None]:
        """Create a deadline context for async code.

        Args:
            seconds: Seconds until deadline
            operation_id: Optional operation identifier

        Yields:
            DeadlineContext
        """
        ctx = DeadlineContext.from_seconds(seconds, operation_id)
        token = _deadline_context.set(ctx)

        try:
            yield ctx
        finally:
            _deadline_context.reset(token)

    def check(self) -> None:
        """Check current deadline and raise if expired.

        Raises:
            DeadlineExceededError: If current deadline has passed
        """
        ctx = self.current
        if ctx is not None:
            ctx.check()


# Module-level propagator instance
_propagator = DeadlinePropagator()


def get_current_deadline() -> DeadlineContext | None:
    """Get the current deadline context.

    Returns:
        Current DeadlineContext or None
    """
    return _propagator.current


def check_deadline() -> None:
    """Check current deadline and raise if expired.

    Raises:
        DeadlineExceededError: If current deadline has passed
    """
    _propagator.check()


def with_deadline(seconds: float, operation_id: str = "") -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add deadline to a function.

    Args:
        seconds: Seconds until deadline
        operation_id: Optional operation identifier

    Returns:
        Decorated function

    Example:
        @with_deadline(30, "my_operation")
        def process_data(data):
            # Deadline context is available
            ctx = get_current_deadline()
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with _propagator.context(seconds, operation_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def propagate_deadline(ctx: DeadlineContext) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to propagate an existing deadline context.

    Args:
        ctx: DeadlineContext to propagate

    Returns:
        Decorated function

    Example:
        ctx = DeadlineContext.from_seconds(60)

        @propagate_deadline(ctx)
        def sub_operation():
            remaining = get_current_deadline().remaining_seconds
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with ctx:
                return func(*args, **kwargs)
        return wrapper
    return decorator
