"""Compensatable action interface and implementations.

This module defines the Compensatable protocol that actions can implement
to support transaction rollback/compensation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

from truthound.checkpoint.transaction.base import (
    CompensationResult,
    TransactionContext,
)
from truthound.checkpoint.actions.base import (
    ActionConfig,
    ActionResult,
    ActionStatus,
    BaseAction,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


ConfigT = TypeVar("ConfigT", bound=ActionConfig)


@runtime_checkable
class Compensatable(Protocol):
    """Protocol for compensatable actions.

    Actions that implement this protocol can have their effects
    undone/rolled back when a subsequent action fails.

    The compensation should be idempotent - calling it multiple times
    should have the same effect as calling it once.
    """

    def compensate(
        self,
        checkpoint_result: "CheckpointResult",
        action_result: ActionResult,
        context: TransactionContext,
    ) -> CompensationResult:
        """Execute compensation to undo the action's effects.

        Args:
            checkpoint_result: The original checkpoint result.
            action_result: The result from the original action execution.
            context: Transaction context.

        Returns:
            CompensationResult indicating success or failure.
        """
        ...

    def can_compensate(self, action_result: ActionResult) -> bool:
        """Check if this action can be compensated.

        Some actions may not be compensatable in certain states.

        Args:
            action_result: The result from the original execution.

        Returns:
            True if compensation is possible.
        """
        ...


@dataclass
class CompensationConfig:
    """Configuration for compensation behavior.

    Attributes:
        enabled: Whether compensation is enabled for this action.
        max_retries: Maximum retry attempts for compensation.
        retry_delay_seconds: Delay between retry attempts.
        timeout_seconds: Timeout for compensation execution.
        require_success: Whether original action must succeed for compensation.
        idempotent: Whether compensation is idempotent.
        priority: Priority for compensation ordering (higher = first).
    """

    enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 30.0
    require_success: bool = True
    idempotent: bool = True
    priority: int = 0


class CompensatableAction(BaseAction[ConfigT], ABC, Generic[ConfigT]):
    """Abstract base class for actions that support compensation.

    This extends BaseAction with compensation capabilities. Actions
    inheriting from this class must implement the _compensate method.

    Example:
        >>> class StoreResultAction(CompensatableAction[StoreConfig]):
        ...     action_type = "store_result"
        ...
        ...     def _execute(self, result: CheckpointResult) -> ActionResult:
        ...         # Store the result
        ...         self._stored_path = self._save_to_storage(result)
        ...         return ActionResult(...)
        ...
        ...     def _compensate(
        ...         self,
        ...         result: CheckpointResult,
        ...         action_result: ActionResult,
        ...         context: TransactionContext,
        ...     ) -> CompensationResult:
        ...         # Delete the stored result
        ...         self._delete_from_storage(self._stored_path)
        ...         return CompensationResult(action_name=self.name, success=True)
    """

    # Compensation configuration
    _compensation_config: CompensationConfig = field(default_factory=CompensationConfig)

    def __init__(
        self,
        config: ConfigT | None = None,
        compensation_config: CompensationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the compensatable action.

        Args:
            config: Action configuration.
            compensation_config: Compensation-specific configuration.
            **kwargs: Additional configuration options.
        """
        super().__init__(config, **kwargs)
        self._compensation_config = compensation_config or CompensationConfig()
        self._execution_state: dict[str, Any] = {}

    @property
    def compensation_enabled(self) -> bool:
        """Check if compensation is enabled."""
        return self._compensation_config.enabled

    def can_compensate(self, action_result: ActionResult) -> bool:
        """Check if this action can be compensated.

        Default implementation requires:
        1. Compensation is enabled
        2. If require_success is True, action must have succeeded

        Args:
            action_result: Result from original execution.

        Returns:
            True if compensation is possible.
        """
        if not self._compensation_config.enabled:
            return False

        if self._compensation_config.require_success and not action_result.success:
            return False

        return True

    def compensate(
        self,
        checkpoint_result: "CheckpointResult",
        action_result: ActionResult,
        context: TransactionContext,
    ) -> CompensationResult:
        """Execute compensation with retry logic.

        Args:
            checkpoint_result: Original checkpoint result.
            action_result: Result from original execution.
            context: Transaction context.

        Returns:
            CompensationResult indicating outcome.
        """
        import time

        started_at = datetime.now()
        config = self._compensation_config
        last_error: Exception | None = None

        # Check if compensation is possible
        if not self.can_compensate(action_result):
            return CompensationResult(
                action_name=self.name,
                success=False,
                started_at=started_at,
                completed_at=datetime.now(),
                error="Compensation not available for this action state",
            )

        # Execute with retries
        for attempt in range(config.max_retries + 1):
            try:
                result = self._compensate(checkpoint_result, action_result, context)
                result.started_at = started_at
                result.completed_at = datetime.now()
                result.duration_ms = (result.completed_at - started_at).total_seconds() * 1000
                return result
            except Exception as e:
                last_error = e
                if attempt < config.max_retries:
                    time.sleep(config.retry_delay_seconds)

        # All retries failed
        completed_at = datetime.now()
        return CompensationResult(
            action_name=self.name,
            success=False,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=(completed_at - started_at).total_seconds() * 1000,
            error=f"Compensation failed after {config.max_retries + 1} attempts: {last_error}",
        )

    @abstractmethod
    def _compensate(
        self,
        checkpoint_result: "CheckpointResult",
        action_result: ActionResult,
        context: TransactionContext,
    ) -> CompensationResult:
        """Execute the compensation logic.

        Subclasses must implement this to define how to undo the action.

        Args:
            checkpoint_result: Original checkpoint result.
            action_result: Result from original execution.
            context: Transaction context.

        Returns:
            CompensationResult indicating outcome.
        """
        pass

    def capture_state(self, key: str, value: Any) -> None:
        """Capture state for use during compensation.

        Call this during _execute to save information needed
        for compensation later.

        Args:
            key: State key.
            value: State value.
        """
        self._execution_state[key] = value

    def get_captured_state(self, key: str, default: Any = None) -> Any:
        """Get captured state.

        Args:
            key: State key.
            default: Default value if not found.

        Returns:
            The captured state value.
        """
        return self._execution_state.get(key, default)

    def clear_state(self) -> None:
        """Clear all captured state."""
        self._execution_state.clear()


class CompensationWrapper(BaseAction[ActionConfig]):
    """Wraps a non-compensatable action with compensation logic.

    Use this to add compensation to existing actions without modifying them.

    Example:
        >>> original_action = ExistingAction()
        >>> compensatable = CompensationWrapper(
        ...     action=original_action,
        ...     compensation_fn=lambda result, ar, ctx: cleanup_logic(),
        ... )
    """

    action_type = "compensation_wrapper"

    def __init__(
        self,
        action: BaseAction[Any],
        compensation_fn: Callable[
            ["CheckpointResult", ActionResult, TransactionContext],
            CompensationResult | bool,
        ],
        can_compensate_fn: Callable[[ActionResult], bool] | None = None,
        compensation_config: CompensationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the wrapper.

        Args:
            action: The action to wrap.
            compensation_fn: Function to execute for compensation.
                Can return CompensationResult or bool.
            can_compensate_fn: Optional function to check if compensation is possible.
            compensation_config: Compensation configuration.
            **kwargs: Additional config options.
        """
        # Use the wrapped action's config
        super().__init__(action.config, **kwargs)
        self._wrapped_action = action
        self._compensation_fn = compensation_fn
        self._can_compensate_fn = can_compensate_fn
        self._compensation_config = compensation_config or CompensationConfig()
        self._last_action_result: ActionResult | None = None

    @classmethod
    def _default_config(cls) -> ActionConfig:
        return ActionConfig()

    @property
    def name(self) -> str:
        return f"compensatable_{self._wrapped_action.name}"

    @property
    def wrapped_action(self) -> BaseAction[Any]:
        """Get the wrapped action."""
        return self._wrapped_action

    def _execute(self, checkpoint_result: "CheckpointResult") -> ActionResult:
        """Execute the wrapped action."""
        result = self._wrapped_action.execute(checkpoint_result)
        self._last_action_result = result
        return result

    def can_compensate(self, action_result: ActionResult) -> bool:
        """Check if compensation is possible."""
        if self._can_compensate_fn:
            return self._can_compensate_fn(action_result)
        return action_result.success

    def compensate(
        self,
        checkpoint_result: "CheckpointResult",
        action_result: ActionResult,
        context: TransactionContext,
    ) -> CompensationResult:
        """Execute compensation."""
        started_at = datetime.now()

        if not self.can_compensate(action_result):
            return CompensationResult(
                action_name=self.name,
                success=False,
                started_at=started_at,
                completed_at=datetime.now(),
                error="Compensation not available",
            )

        try:
            result = self._compensation_fn(checkpoint_result, action_result, context)

            # Handle bool return
            if isinstance(result, bool):
                return CompensationResult(
                    action_name=self.name,
                    success=result,
                    started_at=started_at,
                    completed_at=datetime.now(),
                )

            result.started_at = started_at
            result.completed_at = datetime.now()
            result.duration_ms = (result.completed_at - started_at).total_seconds() * 1000
            return result

        except Exception as e:
            return CompensationResult(
                action_name=self.name,
                success=False,
                started_at=started_at,
                completed_at=datetime.now(),
                error=str(e),
            )


# =============================================================================
# Decorators
# =============================================================================


def compensatable(
    compensation_fn: Callable[
        ["CheckpointResult", ActionResult, TransactionContext],
        CompensationResult | bool,
    ],
) -> Callable[[type[BaseAction[Any]]], type[BaseAction[Any]]]:
    """Class decorator to add compensation to an action class.

    Example:
        >>> @compensatable(my_compensation_logic)
        ... class MyAction(BaseAction[MyConfig]):
        ...     def _execute(self, result):
        ...         ...
    """

    def decorator(cls: type[BaseAction[Any]]) -> type[BaseAction[Any]]:
        original_init = cls.__init__

        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self._compensation_fn = compensation_fn
            self._compensation_config = kwargs.get(
                "compensation_config", CompensationConfig()
            )

        def compensate(
            self: Any,
            checkpoint_result: "CheckpointResult",
            action_result: ActionResult,
            context: TransactionContext,
        ) -> CompensationResult:
            started_at = datetime.now()
            try:
                result = self._compensation_fn(checkpoint_result, action_result, context)
                if isinstance(result, bool):
                    return CompensationResult(
                        action_name=self.name,
                        success=result,
                        started_at=started_at,
                        completed_at=datetime.now(),
                    )
                return result
            except Exception as e:
                return CompensationResult(
                    action_name=self.name,
                    success=False,
                    started_at=started_at,
                    completed_at=datetime.now(),
                    error=str(e),
                )

        def can_compensate(self: Any, action_result: ActionResult) -> bool:
            return action_result.success

        cls.__init__ = new_init
        cls.compensate = compensate
        cls.can_compensate = can_compensate

        return cls

    return decorator


def with_compensation(
    compensation_fn: Callable[..., CompensationResult | bool],
) -> Callable[
    [Callable[..., ActionResult]],
    Callable[..., ActionResult],
]:
    """Method decorator to add compensation to an action's _execute method.

    The compensation function receives the same arguments as _execute,
    plus the action_result and transaction context.

    Example:
        >>> class MyAction(BaseAction[MyConfig]):
        ...     @with_compensation(my_cleanup_fn)
        ...     def _execute(self, checkpoint_result):
        ...         ...
    """

    def decorator(
        execute_fn: Callable[..., ActionResult],
    ) -> Callable[..., ActionResult]:
        @wraps(execute_fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> ActionResult:
            result = execute_fn(self, *args, **kwargs)

            # Store compensation function for later use
            if not hasattr(self, "_compensation_fns"):
                self._compensation_fns = []
            self._compensation_fns.append((compensation_fn, args, kwargs))

            return result

        return wrapper

    return decorator


# =============================================================================
# Utility Functions
# =============================================================================


def is_compensatable(action: BaseAction[Any]) -> bool:
    """Check if an action supports compensation.

    Args:
        action: The action to check.

    Returns:
        True if the action implements Compensatable protocol.
    """
    return isinstance(action, Compensatable)


def wrap_with_compensation(
    action: BaseAction[Any],
    compensation_fn: Callable[
        ["CheckpointResult", ActionResult, TransactionContext],
        CompensationResult | bool,
    ],
    **kwargs: Any,
) -> CompensationWrapper:
    """Wrap an action with compensation logic.

    Convenience function to create a CompensationWrapper.

    Args:
        action: Action to wrap.
        compensation_fn: Compensation function.
        **kwargs: Additional wrapper options.

    Returns:
        CompensationWrapper instance.
    """
    return CompensationWrapper(
        action=action,
        compensation_fn=compensation_fn,
        **kwargs,
    )
