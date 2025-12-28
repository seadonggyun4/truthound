"""Cascading timeout handling for distributed validation.

This module implements cascading timeout patterns for handling timeouts
across multiple levels of a validation pipeline. When a timeout occurs,
it can trigger different actions at different levels.

Example:
    handler = CascadeTimeoutHandler(
        policies={
            CascadeLevel.VALIDATOR: CascadePolicy(
                action=TimeoutAction.SKIP,
                propagate=False,
            ),
            CascadeLevel.COLUMN: CascadePolicy(
                action=TimeoutAction.PARTIAL,
                propagate=True,
            ),
            CascadeLevel.TABLE: CascadePolicy(
                action=TimeoutAction.FAIL,
                propagate=True,
            ),
        }
    )

    result = handler.handle_timeout(
        level=CascadeLevel.VALIDATOR,
        context=deadline_ctx,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar

from truthound.validators.timeout.deadline import (
    DeadlineContext,
    DeadlineExceededError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CascadeLevel(str, Enum):
    """Levels in the validation cascade hierarchy."""

    EXPRESSION = "expression"   # Single Polars expression
    VALIDATOR = "validator"     # Single validator
    COLUMN = "column"          # Column-level validation
    TABLE = "table"            # Table-level validation
    DATASET = "dataset"        # Dataset/batch level
    JOB = "job"               # Entire validation job


class TimeoutAction(str, Enum):
    """Actions to take when timeout occurs."""

    CONTINUE = "continue"    # Continue without result
    SKIP = "skip"           # Skip current operation
    PARTIAL = "partial"     # Return partial results
    RETRY = "retry"         # Retry with extended timeout
    FAIL = "fail"           # Fail immediately
    ESCALATE = "escalate"   # Escalate to parent level


@dataclass
class CascadePolicy:
    """Policy for handling timeout at a specific level.

    Attributes:
        action: Action to take when timeout occurs
        propagate: Whether to propagate timeout to parent levels
        retry_count: Number of retries before escalating
        retry_multiplier: Timeout multiplier for retries
        max_retry_seconds: Maximum timeout for retries
        fallback_action: Action if retries exhausted
        notify: Whether to emit notification
        log_level: Logging level for timeout events
    """

    action: TimeoutAction = TimeoutAction.SKIP
    propagate: bool = False
    retry_count: int = 0
    retry_multiplier: float = 1.5
    max_retry_seconds: float = 300.0
    fallback_action: TimeoutAction = TimeoutAction.FAIL
    notify: bool = False
    log_level: int = logging.WARNING

    def should_retry(self, current_attempt: int) -> bool:
        """Check if retry should be attempted.

        Args:
            current_attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry
        """
        return current_attempt < self.retry_count

    def get_retry_timeout(self, base_timeout: float, attempt: int) -> float:
        """Get timeout for retry attempt.

        Args:
            base_timeout: Original timeout
            attempt: Retry attempt number

        Returns:
            Timeout seconds for retry
        """
        timeout = base_timeout * (self.retry_multiplier ** attempt)
        return min(timeout, self.max_retry_seconds)


@dataclass
class TimeoutCascadeResult:
    """Result of cascading timeout handling.

    Attributes:
        level: Level where timeout occurred
        action_taken: Action that was taken
        propagated: Whether timeout was propagated
        partial_result: Any partial result collected
        retries_attempted: Number of retries attempted
        error: Error message if failed
        metadata: Additional metadata
    """

    level: CascadeLevel
    action_taken: TimeoutAction
    propagated: bool = False
    partial_result: Any = None
    retries_attempted: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_success(self) -> bool:
        """Check if handling was successful (not failed)."""
        return self.action_taken != TimeoutAction.FAIL

    @property
    def has_partial_result(self) -> bool:
        """Check if partial result is available."""
        return self.partial_result is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "action_taken": self.action_taken.value,
            "propagated": self.propagated,
            "has_partial_result": self.has_partial_result,
            "retries_attempted": self.retries_attempted,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CascadeTimeoutHandler:
    """Handler for cascading timeout events.

    This handler manages timeout events across multiple levels of a
    validation pipeline, applying appropriate policies at each level.

    Example:
        handler = CascadeTimeoutHandler()

        # Configure policies
        handler.set_policy(CascadeLevel.VALIDATOR, CascadePolicy(
            action=TimeoutAction.SKIP,
            retry_count=2,
        ))

        # Handle timeout
        result = handler.handle_timeout(
            level=CascadeLevel.VALIDATOR,
            context=deadline_ctx,
            operation_name="null_check",
        )
    """

    policies: dict[CascadeLevel, CascadePolicy] = field(default_factory=dict)
    default_policy: CascadePolicy = field(default_factory=CascadePolicy)
    _handlers: dict[CascadeLevel, list[Callable]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Initialize with default policies if not provided."""
        # Set sensible defaults for each level
        defaults = {
            CascadeLevel.EXPRESSION: CascadePolicy(
                action=TimeoutAction.SKIP,
                propagate=False,
            ),
            CascadeLevel.VALIDATOR: CascadePolicy(
                action=TimeoutAction.SKIP,
                propagate=False,
                retry_count=1,
            ),
            CascadeLevel.COLUMN: CascadePolicy(
                action=TimeoutAction.PARTIAL,
                propagate=True,
            ),
            CascadeLevel.TABLE: CascadePolicy(
                action=TimeoutAction.PARTIAL,
                propagate=True,
            ),
            CascadeLevel.DATASET: CascadePolicy(
                action=TimeoutAction.FAIL,
                propagate=True,
                notify=True,
            ),
            CascadeLevel.JOB: CascadePolicy(
                action=TimeoutAction.FAIL,
                propagate=False,
                notify=True,
            ),
        }

        for level, policy in defaults.items():
            if level not in self.policies:
                self.policies[level] = policy

    def set_policy(self, level: CascadeLevel, policy: CascadePolicy) -> None:
        """Set policy for a cascade level.

        Args:
            level: Cascade level
            policy: Policy to apply
        """
        self.policies[level] = policy

    def get_policy(self, level: CascadeLevel) -> CascadePolicy:
        """Get policy for a cascade level.

        Args:
            level: Cascade level

        Returns:
            Policy for the level
        """
        return self.policies.get(level, self.default_policy)

    def register_handler(
        self,
        level: CascadeLevel,
        handler: Callable[[TimeoutCascadeResult], None],
    ) -> None:
        """Register a handler for timeout events at a level.

        Args:
            level: Cascade level
            handler: Handler function
        """
        if level not in self._handlers:
            self._handlers[level] = []
        self._handlers[level].append(handler)

    def handle_timeout(
        self,
        level: CascadeLevel,
        context: DeadlineContext | None = None,
        operation_name: str = "",
        partial_result: Any = None,
        error: Exception | None = None,
    ) -> TimeoutCascadeResult:
        """Handle a timeout event at the specified level.

        Args:
            level: Level where timeout occurred
            context: Deadline context
            operation_name: Name of the operation that timed out
            partial_result: Any partial result collected
            error: Original error if any

        Returns:
            TimeoutCascadeResult describing the handling outcome
        """
        policy = self.get_policy(level)
        retries = 0
        action = policy.action

        # Log the timeout
        logger.log(
            policy.log_level,
            f"Timeout at {level.value} level for '{operation_name}'",
        )

        # Create result
        result = TimeoutCascadeResult(
            level=level,
            action_taken=action,
            propagated=policy.propagate,
            partial_result=partial_result,
            retries_attempted=retries,
            error=str(error) if error else None,
            metadata={
                "operation": operation_name,
                "context_id": context.operation_id if context else None,
                "remaining_seconds": context.remaining_seconds if context else None,
            },
        )

        # Call registered handlers
        for handler in self._handlers.get(level, []):
            try:
                handler(result)
            except Exception as e:
                logger.warning(f"Timeout handler failed: {e}")

        # Handle notification
        if policy.notify:
            self._notify_timeout(result)

        return result

    def handle_with_retry(
        self,
        level: CascadeLevel,
        operation: Callable[[], T],
        base_timeout: float,
        operation_name: str = "",
    ) -> tuple[T | None, TimeoutCascadeResult | None]:
        """Execute operation with retry handling.

        Args:
            level: Cascade level
            operation: Operation to execute
            base_timeout: Base timeout in seconds
            operation_name: Name for logging

        Returns:
            Tuple of (result, cascade_result if timeout occurred)
        """
        policy = self.get_policy(level)
        attempt = 0

        while True:
            try:
                timeout = policy.get_retry_timeout(base_timeout, attempt)
                ctx = DeadlineContext.from_seconds(timeout, operation_name)

                with ctx:
                    result = operation()
                    return result, None

            except (DeadlineExceededError, TimeoutError) as e:
                if policy.should_retry(attempt):
                    attempt += 1
                    logger.info(
                        f"Retry {attempt}/{policy.retry_count} for '{operation_name}'"
                    )
                    continue

                # No more retries
                cascade_result = self.handle_timeout(
                    level=level,
                    context=ctx,
                    operation_name=operation_name,
                    error=e,
                )
                cascade_result.retries_attempted = attempt

                return None, cascade_result

    def _notify_timeout(self, result: TimeoutCascadeResult) -> None:
        """Send notification for timeout event.

        Override this method to customize notification behavior.

        Args:
            result: Timeout result to notify about
        """
        # Default implementation just logs
        logger.warning(
            f"Timeout notification: {result.level.value} - {result.action_taken.value}"
        )

    def get_parent_level(self, level: CascadeLevel) -> CascadeLevel | None:
        """Get the parent level in the cascade hierarchy.

        Args:
            level: Current level

        Returns:
            Parent level or None if at top
        """
        hierarchy = [
            CascadeLevel.EXPRESSION,
            CascadeLevel.VALIDATOR,
            CascadeLevel.COLUMN,
            CascadeLevel.TABLE,
            CascadeLevel.DATASET,
            CascadeLevel.JOB,
        ]

        try:
            idx = hierarchy.index(level)
            if idx < len(hierarchy) - 1:
                return hierarchy[idx + 1]
        except ValueError:
            pass

        return None

    def propagate_to_parent(
        self,
        result: TimeoutCascadeResult,
    ) -> TimeoutCascadeResult | None:
        """Propagate timeout to parent level if configured.

        Args:
            result: Result from current level

        Returns:
            New result from parent level, or None if not propagated
        """
        if not result.propagated:
            return None

        parent = self.get_parent_level(result.level)
        if parent is None:
            return None

        return self.handle_timeout(
            level=parent,
            operation_name=f"propagated_from_{result.level.value}",
            partial_result=result.partial_result,
        )


# Default handler instance
_default_handler = CascadeTimeoutHandler()


def get_cascade_handler() -> CascadeTimeoutHandler:
    """Get the default cascade handler.

    Returns:
        Default CascadeTimeoutHandler instance
    """
    return _default_handler


def handle_timeout_cascade(
    level: CascadeLevel,
    context: DeadlineContext | None = None,
    operation_name: str = "",
    partial_result: Any = None,
) -> TimeoutCascadeResult:
    """Handle timeout using default handler.

    Args:
        level: Level where timeout occurred
        context: Deadline context
        operation_name: Name of the operation
        partial_result: Any partial result

    Returns:
        TimeoutCascadeResult
    """
    return _default_handler.handle_timeout(
        level=level,
        context=context,
        operation_name=operation_name,
        partial_result=partial_result,
    )
