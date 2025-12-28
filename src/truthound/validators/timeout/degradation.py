"""Graceful degradation policies for timeout handling.

This module implements graceful degradation patterns for when validation
operations approach or exceed their time limits. Instead of failing
completely, the system can:
- Return partial results
- Use sampling to reduce workload
- Skip optional validations
- Apply fast-path algorithms

Example:
    degradation = GracefulDegradation(
        policy=DegradationPolicy(
            sample_threshold=0.3,  # Sample when 30% time remaining
            skip_threshold=0.1,    # Skip optional when 10% remaining
        )
    )

    result = degradation.apply(
        deadline_ctx,
        operation=full_validation,
        fallback=sampled_validation,
    )
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from truthound.validators.timeout.deadline import (
    DeadlineContext,
    DeadlineExceededError,
    get_current_deadline,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DegradationLevel(str, Enum):
    """Levels of degradation."""

    NONE = "none"           # Full quality
    LIGHT = "light"         # Minor optimizations
    MODERATE = "moderate"   # Sampling, skip optional
    HEAVY = "heavy"         # Aggressive shortcuts
    EMERGENCY = "emergency"  # Return immediately


class DegradationAction(str, Enum):
    """Actions for degradation."""

    CONTINUE = "continue"      # Continue normally
    SAMPLE = "sample"          # Use sampling
    SKIP_OPTIONAL = "skip"     # Skip optional operations
    FAST_PATH = "fast_path"    # Use fast algorithm
    CACHE_ONLY = "cache_only"  # Use cached results only
    ABORT = "abort"            # Abort operation


@dataclass
class DegradationPolicy:
    """Policy for graceful degradation.

    Attributes:
        sample_threshold: Remaining time fraction to trigger sampling
        skip_threshold: Remaining time fraction to skip optional ops
        fast_path_threshold: Remaining time fraction for fast path
        emergency_threshold: Remaining time fraction for emergency abort
        sample_ratio: Ratio of data to sample when sampling
        cache_enabled: Whether to use cached results as fallback
        notify_on_degrade: Whether to emit notifications
    """

    sample_threshold: float = 0.3       # 30% time remaining
    skip_threshold: float = 0.2         # 20% time remaining
    fast_path_threshold: float = 0.1    # 10% time remaining
    emergency_threshold: float = 0.05   # 5% time remaining
    sample_ratio: float = 0.1           # Sample 10% of data
    cache_enabled: bool = True
    notify_on_degrade: bool = True

    def get_degradation_level(
        self,
        remaining_fraction: float,
    ) -> DegradationLevel:
        """Get degradation level based on remaining time.

        Args:
            remaining_fraction: Fraction of time remaining (0.0-1.0)

        Returns:
            Appropriate DegradationLevel
        """
        if remaining_fraction <= self.emergency_threshold:
            return DegradationLevel.EMERGENCY
        elif remaining_fraction <= self.fast_path_threshold:
            return DegradationLevel.HEAVY
        elif remaining_fraction <= self.skip_threshold:
            return DegradationLevel.MODERATE
        elif remaining_fraction <= self.sample_threshold:
            return DegradationLevel.LIGHT
        return DegradationLevel.NONE

    def get_action(self, level: DegradationLevel) -> DegradationAction:
        """Get action for degradation level.

        Args:
            level: Current degradation level

        Returns:
            Recommended action
        """
        actions = {
            DegradationLevel.NONE: DegradationAction.CONTINUE,
            DegradationLevel.LIGHT: DegradationAction.SAMPLE,
            DegradationLevel.MODERATE: DegradationAction.SKIP_OPTIONAL,
            DegradationLevel.HEAVY: DegradationAction.FAST_PATH,
            DegradationLevel.EMERGENCY: DegradationAction.ABORT,
        }
        return actions.get(level, DegradationAction.CONTINUE)


@dataclass
class DegradationResult(Generic[T]):
    """Result of a degraded operation.

    Attributes:
        value: Result value
        level: Degradation level applied
        action: Action taken
        quality_score: Estimated quality (0.0-1.0)
        was_degraded: Whether degradation was applied
        fallback_used: Whether fallback was used
        metadata: Additional metadata
    """

    value: T | None
    level: DegradationLevel = DegradationLevel.NONE
    action: DegradationAction = DegradationAction.CONTINUE
    quality_score: float = 1.0
    was_degraded: bool = False
    fallback_used: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def success(cls, value: T, level: DegradationLevel = DegradationLevel.NONE) -> "DegradationResult[T]":
        """Create successful result.

        Args:
            value: Result value
            level: Degradation level applied

        Returns:
            Success result
        """
        quality = {
            DegradationLevel.NONE: 1.0,
            DegradationLevel.LIGHT: 0.9,
            DegradationLevel.MODERATE: 0.7,
            DegradationLevel.HEAVY: 0.5,
            DegradationLevel.EMERGENCY: 0.2,
        }
        return cls(
            value=value,
            level=level,
            action=DegradationAction.CONTINUE,
            quality_score=quality.get(level, 1.0),
            was_degraded=level != DegradationLevel.NONE,
        )

    @classmethod
    def fallback(
        cls,
        value: T | None,
        level: DegradationLevel,
        action: DegradationAction,
    ) -> "DegradationResult[T]":
        """Create fallback result.

        Args:
            value: Fallback value
            level: Degradation level
            action: Action taken

        Returns:
            Fallback result
        """
        quality = {
            DegradationLevel.LIGHT: 0.8,
            DegradationLevel.MODERATE: 0.6,
            DegradationLevel.HEAVY: 0.4,
            DegradationLevel.EMERGENCY: 0.1,
        }
        return cls(
            value=value,
            level=level,
            action=action,
            quality_score=quality.get(level, 0.5),
            was_degraded=True,
            fallback_used=True,
        )

    @classmethod
    def aborted(cls, error: str | None = None) -> "DegradationResult[T]":
        """Create aborted result.

        Args:
            error: Error message

        Returns:
            Aborted result
        """
        return cls(
            value=None,
            level=DegradationLevel.EMERGENCY,
            action=DegradationAction.ABORT,
            quality_score=0.0,
            was_degraded=True,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "action": self.action.value,
            "quality_score": self.quality_score,
            "was_degraded": self.was_degraded,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GracefulDegradation:
    """Manager for graceful degradation.

    This class manages the application of degradation policies to
    operations that may approach their time limits.

    Example:
        degradation = GracefulDegradation()

        # Check if degradation needed
        level = degradation.check(deadline_ctx)

        if level != DegradationLevel.NONE:
            # Use degraded operation
            result = fast_validate(data)
        else:
            # Full validation
            result = full_validate(data)

        # Or use apply() for automatic handling
        result = degradation.apply(
            deadline_ctx,
            full_operation=full_validate,
            fallback_operation=fast_validate,
        )
    """

    policy: DegradationPolicy = field(default_factory=DegradationPolicy)
    _cache: dict[str, Any] = field(default_factory=dict, init=False)
    _stats: dict[str, int] = field(default_factory=dict, init=False)

    def check(
        self,
        context: DeadlineContext | None = None,
    ) -> DegradationLevel:
        """Check current degradation level.

        Args:
            context: Deadline context (uses current if None)

        Returns:
            Current DegradationLevel
        """
        if context is None:
            context = get_current_deadline()

        if context is None:
            return DegradationLevel.NONE

        # Calculate remaining fraction
        elapsed = (
            datetime.now(timezone.utc) - context.created_at
        ).total_seconds()
        total = (context.deadline_utc - context.created_at).total_seconds()

        if total <= 0:
            return DegradationLevel.EMERGENCY

        remaining_fraction = context.remaining_seconds / total

        return self.policy.get_degradation_level(remaining_fraction)

    def get_action(
        self,
        context: DeadlineContext | None = None,
    ) -> DegradationAction:
        """Get recommended action based on current state.

        Args:
            context: Deadline context

        Returns:
            Recommended DegradationAction
        """
        level = self.check(context)
        return self.policy.get_action(level)

    def should_sample(self, context: DeadlineContext | None = None) -> bool:
        """Check if sampling should be used.

        Args:
            context: Deadline context

        Returns:
            True if sampling recommended
        """
        level = self.check(context)
        return level in (
            DegradationLevel.LIGHT,
            DegradationLevel.MODERATE,
        )

    def should_skip_optional(self, context: DeadlineContext | None = None) -> bool:
        """Check if optional operations should be skipped.

        Args:
            context: Deadline context

        Returns:
            True if skipping recommended
        """
        level = self.check(context)
        return level in (
            DegradationLevel.MODERATE,
            DegradationLevel.HEAVY,
            DegradationLevel.EMERGENCY,
        )

    def should_abort(self, context: DeadlineContext | None = None) -> bool:
        """Check if operation should abort.

        Args:
            context: Deadline context

        Returns:
            True if abort recommended
        """
        level = self.check(context)
        return level == DegradationLevel.EMERGENCY

    def apply(
        self,
        context: DeadlineContext | None,
        full_operation: Callable[[], T],
        fallback_operation: Callable[[], T] | None = None,
        cache_key: str | None = None,
        operation_name: str = "",
    ) -> DegradationResult[T]:
        """Apply degradation policy to an operation.

        Args:
            context: Deadline context
            full_operation: Full quality operation
            fallback_operation: Fallback operation for degradation
            cache_key: Key for caching results
            operation_name: Name for logging

        Returns:
            DegradationResult with value and metadata
        """
        level = self.check(context)
        action = self.policy.get_action(level)

        # Track statistics
        self._track_stat(f"level_{level.value}")
        self._track_stat(f"action_{action.value}")

        # Handle emergency abort
        if action == DegradationAction.ABORT:
            logger.warning(f"Emergency abort for '{operation_name}'")
            return DegradationResult.aborted(
                f"Emergency abort due to time pressure"
            )

        # Try cache if available
        if cache_key and self.policy.cache_enabled and action != DegradationAction.CONTINUE:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Using cached result for '{operation_name}'")
                return DegradationResult.fallback(
                    cached,
                    level,
                    DegradationAction.CACHE_ONLY,
                )

        # Use fallback if degraded
        if level != DegradationLevel.NONE and fallback_operation is not None:
            try:
                result = fallback_operation()
                if cache_key:
                    self._cache[cache_key] = result
                return DegradationResult.fallback(result, level, action)
            except Exception as e:
                logger.warning(f"Fallback failed: {e}")
                # Fall through to try full operation

        # Try full operation
        try:
            result = full_operation()
            if cache_key:
                self._cache[cache_key] = result
            return DegradationResult.success(result, level)
        except DeadlineExceededError as e:
            logger.warning(f"Deadline exceeded in '{operation_name}': {e}")
            return DegradationResult.aborted(str(e))
        except TimeoutError as e:
            logger.warning(f"Timeout in '{operation_name}': {e}")
            return DegradationResult.aborted(str(e))
        except Exception as e:
            logger.error(f"Error in '{operation_name}': {e}")
            raise

    def _track_stat(self, key: str) -> None:
        """Track a statistic.

        Args:
            key: Statistic key
        """
        self._stats[key] = self._stats.get(key, 0) + 1

    def get_stats(self) -> dict[str, int]:
        """Get degradation statistics.

        Returns:
            Dictionary of statistics
        """
        return dict(self._stats)

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()

    def clear_stats(self) -> None:
        """Clear statistics."""
        self._stats.clear()


def with_graceful_degradation(
    fallback: Callable[..., T] | None = None,
    policy: DegradationPolicy | None = None,
) -> Callable[[Callable[..., T]], Callable[..., DegradationResult[T]]]:
    """Decorator to add graceful degradation to a function.

    Args:
        fallback: Fallback function to use when degraded
        policy: Degradation policy

    Returns:
        Decorated function

    Example:
        @with_graceful_degradation(fallback=fast_validate)
        def full_validate(data):
            ...
    """
    degradation = GracefulDegradation(policy or DegradationPolicy())

    def decorator(func: Callable[..., T]) -> Callable[..., DegradationResult[T]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> DegradationResult[T]:
            ctx = get_current_deadline()

            full_op = lambda: func(*args, **kwargs)
            fallback_op = (
                (lambda: fallback(*args, **kwargs)) if fallback else None
            )

            return degradation.apply(
                context=ctx,
                full_operation=full_op,
                fallback_operation=fallback_op,
                operation_name=func.__name__,
            )

        return wrapper

    return decorator


# Default degradation instance
_default_degradation = GracefulDegradation()


def check_degradation_level() -> DegradationLevel:
    """Check current degradation level using default instance.

    Returns:
        Current DegradationLevel
    """
    return _default_degradation.check()


def should_degrade() -> bool:
    """Check if any degradation should be applied.

    Returns:
        True if degradation recommended
    """
    level = check_degradation_level()
    return level != DegradationLevel.NONE
