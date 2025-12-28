"""Saga Builder Module.

This module provides a fluent builder API for constructing sagas
in a declarative and readable manner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from truthound.checkpoint.transaction.saga.definition import (
    DependencyType,
    RetryConfig,
    RetryPolicy,
    SagaDefinition,
    SagaStepDefinition,
    StepDependency,
    StepExecutionMode,
    TimeoutConfig,
)

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import BaseAction
    from truthound.checkpoint.transaction.base import (
        CompensationResult,
        TransactionContext,
    )
    from truthound.checkpoint.checkpoint import CheckpointResult


T = TypeVar("T")


class StepBuilder:
    """Fluent builder for saga steps.

    Example:
        >>> step = (
        ...     StepBuilder("process_payment")
        ...     .action(PaymentAction())
        ...     .compensate_with(RefundAction())
        ...     .depends_on("validate_order")
        ...     .with_timeout(30)
        ...     .with_retry(max_attempts=3)
        ...     .build()
        ... )
    """

    def __init__(self, step_id: str, saga_builder: "SagaBuilder | None" = None) -> None:
        """Initialize the step builder.

        Args:
            step_id: Unique step identifier.
            saga_builder: Parent saga builder for chaining.
        """
        self._step_id = step_id
        self._saga_builder = saga_builder
        self._name = step_id
        self._action: "BaseAction[Any]" | None = None
        self._compensation_action: "BaseAction[Any]" | None = None
        self._compensation_fn: Callable[..., Any] | None = None
        self._dependencies: list[StepDependency] = []
        self._retry_config = RetryConfig()
        self._timeout_config = TimeoutConfig()
        self._execution_mode = StepExecutionMode.SYNC
        self._condition: Callable[["CheckpointResult", "TransactionContext"], bool] | None = None
        self._metadata: dict[str, Any] = {}
        self._is_pivot = False
        self._is_countermeasure = False
        self._semantic_undo = False
        self._required = True

    def named(self, name: str) -> "StepBuilder":
        """Set human-readable name for the step.

        Args:
            name: Step name.

        Returns:
            Self for chaining.
        """
        self._name = name
        return self

    def action(self, action: "BaseAction[Any]") -> "StepBuilder":
        """Set the action to execute.

        Args:
            action: Action to execute.

        Returns:
            Self for chaining.
        """
        self._action = action
        return self

    def compensate_with(
        self,
        action: "BaseAction[Any] | None" = None,
        *,
        fn: Callable[..., "CompensationResult | bool"] | None = None,
    ) -> "StepBuilder":
        """Set compensation for this step.

        Args:
            action: Compensation action.
            fn: Compensation function (alternative to action).

        Returns:
            Self for chaining.
        """
        self._compensation_action = action
        self._compensation_fn = fn
        return self

    def semantic_compensation(self) -> "StepBuilder":
        """Mark compensation as semantic (not exact undo).

        Semantic compensation indicates that the compensation action
        doesn't strictly undo the original action but provides a
        semantically equivalent result (e.g., credit instead of refund).

        Returns:
            Self for chaining.
        """
        self._semantic_undo = True
        return self

    def depends_on(
        self,
        step_id: str,
        dependency_type: DependencyType = DependencyType.REQUIRES_SUCCESS,
        condition: Callable[["CheckpointResult", "TransactionContext"], bool] | None = None,
    ) -> "StepBuilder":
        """Add a dependency on another step.

        Args:
            step_id: ID of the step to depend on.
            dependency_type: Type of dependency.
            condition: Optional condition for dependency.

        Returns:
            Self for chaining.
        """
        self._dependencies.append(
            StepDependency(
                step_id=step_id,
                dependency_type=dependency_type,
                condition=condition,
            )
        )
        return self

    def depends_on_all(self, *step_ids: str) -> "StepBuilder":
        """Add dependencies on multiple steps.

        Args:
            step_ids: IDs of steps to depend on.

        Returns:
            Self for chaining.
        """
        for step_id in step_ids:
            self.depends_on(step_id)
        return self

    def parallel_with(self, step_id: str) -> "StepBuilder":
        """Mark this step as parallel with another step.

        Args:
            step_id: ID of the step to run in parallel with.

        Returns:
            Self for chaining.
        """
        return self.depends_on(step_id, DependencyType.PARALLEL)

    def with_timeout(
        self,
        execution_seconds: float = 30,
        compensation_seconds: float = 30,
        total_seconds: float | None = None,
        on_timeout: str = "fail",
    ) -> "StepBuilder":
        """Configure timeout for the step.

        Args:
            execution_seconds: Execution timeout in seconds.
            compensation_seconds: Compensation timeout in seconds.
            total_seconds: Total timeout including retries.
            on_timeout: Behavior on timeout ("fail", "compensate", "skip").

        Returns:
            Self for chaining.
        """
        self._timeout_config = TimeoutConfig(
            execution_timeout=timedelta(seconds=execution_seconds),
            compensation_timeout=timedelta(seconds=compensation_seconds),
            total_timeout=timedelta(seconds=total_seconds) if total_seconds else None,
            on_timeout=on_timeout,
        )
        return self

    def with_retry(
        self,
        max_attempts: int = 3,
        policy: RetryPolicy | str = RetryPolicy.EXPONENTIAL,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,
        multiplier: float = 2.0,
        jitter: float = 0.1,
    ) -> "StepBuilder":
        """Configure retry for the step.

        Args:
            max_attempts: Maximum retry attempts.
            policy: Retry policy.
            initial_delay: Initial delay in seconds.
            max_delay: Maximum delay in seconds.
            multiplier: Backoff multiplier.
            jitter: Jitter factor (0.0 - 1.0).

        Returns:
            Self for chaining.
        """
        if isinstance(policy, str):
            policy = RetryPolicy(policy)

        self._retry_config = RetryConfig(
            policy=policy,
            max_attempts=max_attempts,
            initial_delay=timedelta(seconds=initial_delay),
            max_delay=timedelta(seconds=max_delay),
            multiplier=multiplier,
            jitter=jitter,
        )
        return self

    def no_retry(self) -> "StepBuilder":
        """Disable retry for this step.

        Returns:
            Self for chaining.
        """
        self._retry_config = RetryConfig(policy=RetryPolicy.NONE, max_attempts=0)
        return self

    def retry_on(self, *exception_types: type[Exception]) -> "StepBuilder":
        """Set exception types to retry on.

        Args:
            exception_types: Exception types to retry on.

        Returns:
            Self for chaining.
        """
        self._retry_config.retry_on = exception_types
        return self

    def abort_on(self, *exception_types: type[Exception]) -> "StepBuilder":
        """Set exception types to abort on (no retry).

        Args:
            exception_types: Exception types to abort on.

        Returns:
            Self for chaining.
        """
        self._retry_config.abort_on = exception_types
        return self

    def async_execution(self) -> "StepBuilder":
        """Execute this step asynchronously.

        Returns:
            Self for chaining.
        """
        self._execution_mode = StepExecutionMode.ASYNC
        return self

    def deferred_execution(self) -> "StepBuilder":
        """Defer execution until triggered.

        Returns:
            Self for chaining.
        """
        self._execution_mode = StepExecutionMode.DEFERRED
        return self

    def when(
        self,
        condition: Callable[["CheckpointResult", "TransactionContext"], bool],
    ) -> "StepBuilder":
        """Add execution condition.

        Args:
            condition: Condition function.

        Returns:
            Self for chaining.
        """
        self._condition = condition
        return self

    def optional(self) -> "StepBuilder":
        """Mark this step as optional (failure won't trigger rollback).

        Returns:
            Self for chaining.
        """
        self._required = False
        return self

    def as_pivot(self) -> "StepBuilder":
        """Mark this step as the pivot transaction.

        A pivot transaction is the point of no return - once it succeeds,
        the saga cannot be compensated. Instead, forward recovery must
        be used.

        Returns:
            Self for chaining.
        """
        self._is_pivot = True
        self._compensation_action = None
        self._compensation_fn = None
        return self

    def as_countermeasure(self) -> "StepBuilder":
        """Mark this step as a countermeasure.

        A countermeasure is an action that helps correct issues
        rather than undoing previous actions.

        Returns:
            Self for chaining.
        """
        self._is_countermeasure = True
        return self

    def with_metadata(self, **kwargs: Any) -> "StepBuilder":
        """Add metadata to the step.

        Args:
            kwargs: Metadata key-value pairs.

        Returns:
            Self for chaining.
        """
        self._metadata.update(kwargs)
        return self

    def build(self) -> SagaStepDefinition:
        """Build the step definition.

        Returns:
            Completed step definition.
        """
        return SagaStepDefinition(
            step_id=self._step_id,
            name=self._name,
            action=self._action,
            compensation_action=self._compensation_action,
            compensation_fn=self._compensation_fn,
            dependencies=self._dependencies,
            retry_config=self._retry_config,
            timeout_config=self._timeout_config,
            execution_mode=self._execution_mode,
            condition=self._condition,
            metadata=self._metadata,
            is_pivot=self._is_pivot,
            is_countermeasure=self._is_countermeasure,
            semantic_undo=self._semantic_undo,
            required=self._required,
        )

    def end_step(self) -> "SagaBuilder":
        """Complete step configuration and return to saga builder.

        Returns:
            Parent saga builder.

        Raises:
            ValueError: If no parent saga builder.
        """
        if self._saga_builder is None:
            raise ValueError("No parent saga builder")

        step = self.build()
        self._saga_builder._steps.append(step)
        return self._saga_builder


class SagaBuilder:
    """Fluent builder for saga definitions.

    Example:
        >>> saga = (
        ...     SagaBuilder("order_processing")
        ...     .description("Process customer orders")
        ...     .step("validate_order")
        ...         .action(ValidateAction())
        ...         .compensate_with(RejectOrderAction())
        ...     .end_step()
        ...     .step("reserve_inventory")
        ...         .action(ReserveAction())
        ...         .compensate_with(ReleaseInventoryAction())
        ...         .depends_on("validate_order")
        ...         .with_timeout(30)
        ...     .end_step()
        ...     .step("process_payment")
        ...         .action(PaymentAction())
        ...         .compensate_with(RefundAction())
        ...         .depends_on("reserve_inventory")
        ...         .with_retry(max_attempts=3)
        ...     .end_step()
        ...     .step("ship_order")
        ...         .action(ShipAction())
        ...         .as_pivot()
        ...         .depends_on("process_payment")
        ...     .end_step()
        ...     .with_timeout(300)
        ...     .with_policy("backward")
        ...     .build()
        ... )
    """

    def __init__(self, name: str) -> None:
        """Initialize the saga builder.

        Args:
            name: Saga name.
        """
        self._name = name
        self._description = ""
        self._version = "1.0.0"
        self._steps: list[SagaStepDefinition] = []
        self._global_timeout: timedelta | None = None
        self._compensation_policy = "backward"
        self._metadata: dict[str, Any] = {}

    def description(self, desc: str) -> "SagaBuilder":
        """Set saga description.

        Args:
            desc: Description text.

        Returns:
            Self for chaining.
        """
        self._description = desc
        return self

    def version(self, version: str) -> "SagaBuilder":
        """Set saga version.

        Args:
            version: Version string.

        Returns:
            Self for chaining.
        """
        self._version = version
        return self

    def step(self, step_id: str) -> StepBuilder:
        """Start building a new step.

        Args:
            step_id: Unique step identifier.

        Returns:
            Step builder for configuration.
        """
        return StepBuilder(step_id, saga_builder=self)

    def add_step(self, step: SagaStepDefinition) -> "SagaBuilder":
        """Add a pre-built step to the saga.

        Args:
            step: Step definition to add.

        Returns:
            Self for chaining.
        """
        self._steps.append(step)
        return self

    def with_timeout(self, seconds: float) -> "SagaBuilder":
        """Set global saga timeout.

        Args:
            seconds: Timeout in seconds.

        Returns:
            Self for chaining.
        """
        self._global_timeout = timedelta(seconds=seconds)
        return self

    def with_policy(self, policy: str) -> "SagaBuilder":
        """Set compensation policy.

        Args:
            policy: Policy name ("backward", "semantic", "pivot", "countermeasure").

        Returns:
            Self for chaining.
        """
        self._compensation_policy = policy
        return self

    def with_metadata(self, **kwargs: Any) -> "SagaBuilder":
        """Add metadata to the saga.

        Args:
            kwargs: Metadata key-value pairs.

        Returns:
            Self for chaining.
        """
        self._metadata.update(kwargs)
        return self

    def build(self) -> SagaDefinition:
        """Build the saga definition.

        Returns:
            Completed saga definition.

        Raises:
            ValueError: If saga definition is invalid.
        """
        saga = SagaDefinition(
            name=self._name,
            description=self._description,
            version=self._version,
            steps=self._steps,
            global_timeout=self._global_timeout,
            compensation_policy=self._compensation_policy,
            metadata=self._metadata,
        )

        # Assign step order
        for i, step in enumerate(saga.steps):
            step.order = i

        # Validate
        errors = saga.validate()
        if errors:
            raise ValueError(f"Invalid saga definition: {'; '.join(errors)}")

        return saga


# =============================================================================
# Convenience Functions
# =============================================================================


def saga(name: str) -> SagaBuilder:
    """Create a new saga builder.

    Args:
        name: Saga name.

    Returns:
        Saga builder instance.
    """
    return SagaBuilder(name)


def step(step_id: str) -> StepBuilder:
    """Create a standalone step builder.

    Args:
        step_id: Step identifier.

    Returns:
        Step builder instance.
    """
    return StepBuilder(step_id)
