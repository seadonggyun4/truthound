"""Transaction Coordinator and Saga Orchestrator.

This module provides the core transaction coordination logic,
implementing the Saga pattern for distributed transactions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterator

from truthound.checkpoint.transaction.base import (
    CompensationResult,
    CompensationStrategy,
    TransactionConfig,
    TransactionContext,
    TransactionPhase,
    TransactionResult,
    TransactionState,
)
from truthound.checkpoint.transaction.compensatable import (
    Compensatable,
    is_compensatable,
)
from truthound.checkpoint.actions.base import (
    ActionResult,
    ActionStatus,
    BaseAction,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


logger = logging.getLogger(__name__)


@dataclass
class ActionEntry:
    """Entry in the transaction log.

    Tracks an action's execution and compensation state.
    """

    action: BaseAction[Any]
    result: ActionResult | None = None
    compensation_result: CompensationResult | None = None
    compensatable: bool = False
    executed: bool = False
    compensated: bool = False

    def __post_init__(self) -> None:
        self.compensatable = is_compensatable(self.action)


class TransactionCoordinator:
    """Coordinates transaction execution and compensation.

    The TransactionCoordinator manages the lifecycle of a transaction,
    handling action execution, failure detection, and compensation.

    Key responsibilities:
    1. Execute actions in order
    2. Detect failures
    3. Trigger compensations when needed
    4. Track transaction state
    5. Generate audit logs

    Example:
        >>> coordinator = TransactionCoordinator(config=TransactionConfig())
        >>> result = coordinator.execute(
        ...     actions=[action1, action2, action3],
        ...     checkpoint_result=checkpoint_result,
        ... )
        >>> if not result.success:
        ...     print(f"Transaction failed: {result.error}")
        ...     print(f"Compensations: {result.compensations_completed}")
    """

    def __init__(
        self,
        config: TransactionConfig | None = None,
        on_action_complete: Callable[[ActionEntry], None] | None = None,
        on_compensation_complete: Callable[[ActionEntry], None] | None = None,
        on_state_change: Callable[[TransactionState, TransactionState], None] | None = None,
    ) -> None:
        """Initialize the coordinator.

        Args:
            config: Transaction configuration.
            on_action_complete: Callback after each action.
            on_compensation_complete: Callback after each compensation.
            on_state_change: Callback on state transitions.
        """
        self._config = config or TransactionConfig()
        self._on_action_complete = on_action_complete
        self._on_compensation_complete = on_compensation_complete
        self._on_state_change = on_state_change

    @property
    def config(self) -> TransactionConfig:
        """Get the transaction configuration."""
        return self._config

    def execute(
        self,
        actions: list[BaseAction[Any]],
        checkpoint_result: "CheckpointResult",
        context: TransactionContext | None = None,
    ) -> TransactionResult:
        """Execute actions as a transaction.

        Args:
            actions: List of actions to execute.
            checkpoint_result: Checkpoint result to pass to actions.
            context: Optional transaction context.

        Returns:
            TransactionResult with complete outcome.
        """
        context = context or TransactionContext()
        entries = [ActionEntry(action=action) for action in actions]

        # Start transaction
        self._transition_state(context, TransactionState.ACTIVE)
        context.phase = TransactionPhase.EXECUTE

        result = TransactionResult(
            transaction_id=context.transaction_id,
            state=TransactionState.ACTIVE,
            started_at=context.started_at,
        )

        try:
            # Execute each action
            failed_index = self._execute_actions(
                entries, checkpoint_result, context, result
            )

            if failed_index is not None:
                # Action failed - trigger compensation
                result.error = f"Action '{entries[failed_index].action.name}' failed"
                context.failed_action = entries[failed_index].action.name

                if self._config.rollback_on_failure:
                    self._transition_state(context, TransactionState.ROLLING_BACK)
                    context.phase = TransactionPhase.COMPENSATE

                    self._compensate_actions(
                        entries[:failed_index],  # Only compensate completed actions
                        checkpoint_result,
                        context,
                        result,
                    )

                    if all(e.compensated or not e.compensatable for e in entries[:failed_index]):
                        self._transition_state(context, TransactionState.COMPENSATED)
                    else:
                        self._transition_state(context, TransactionState.FAILED)
                else:
                    self._transition_state(context, TransactionState.FAILED)
            else:
                # All actions succeeded
                self._transition_state(context, TransactionState.COMMITTED)

        except Exception as e:
            logger.exception("Transaction execution error")
            result.error = str(e)
            self._transition_state(context, TransactionState.FAILED)

        # Finalize
        result.state = context.state
        result.completed_at = datetime.now()
        result.duration_ms = (result.completed_at - result.started_at).total_seconds() * 1000
        result.metadata["context"] = context.to_dict()

        return result

    def _execute_actions(
        self,
        entries: list[ActionEntry],
        checkpoint_result: "CheckpointResult",
        context: TransactionContext,
        result: TransactionResult,
    ) -> int | None:
        """Execute actions and return index of first failure.

        Returns:
            Index of failed action, or None if all succeeded.
        """
        for i, entry in enumerate(entries):
            # Check for rollback request
            if context.rollback_requested:
                logger.info(f"Rollback requested, stopping at action {i}")
                return i

            # Create savepoint if enabled
            if self._config.savepoint_enabled and entry.compensatable:
                context.create_savepoint(
                    name=f"before_{entry.action.name}",
                    state_snapshot={"action_index": i},
                )

            # Execute action
            try:
                action_result = entry.action.execute(checkpoint_result)
                entry.result = action_result
                entry.executed = True

                result.action_results.append(action_result)
                context.mark_action_completed(entry.action.name)

                # Callback
                if self._on_action_complete:
                    self._on_action_complete(entry)

                # Check for failure
                if not action_result.success:
                    # Check if action failure should trigger rollback
                    if entry.action.config.fail_checkpoint_on_error:
                        return i

            except Exception as e:
                logger.error(f"Action execution error: {e}")
                entry.result = ActionResult(
                    action_name=entry.action.name,
                    action_type=entry.action.action_type,
                    status=ActionStatus.ERROR,
                    error=str(e),
                )
                entry.executed = True
                result.action_results.append(entry.result)
                return i

        return None

    def _compensate_actions(
        self,
        entries: list[ActionEntry],
        checkpoint_result: "CheckpointResult",
        context: TransactionContext,
        result: TransactionResult,
    ) -> None:
        """Execute compensations for completed actions.

        Args:
            entries: Actions to compensate (in execution order).
            checkpoint_result: Original checkpoint result.
            context: Transaction context.
            result: Transaction result to update.
        """
        strategy = self._config.compensation_strategy

        if strategy == CompensationStrategy.BACKWARD:
            compensation_order = reversed(entries)
        elif strategy == CompensationStrategy.PARALLEL:
            # For parallel, we'd use asyncio - fall back to backward for sync
            compensation_order = reversed(entries)
        else:
            compensation_order = entries

        for entry in compensation_order:
            if not entry.executed or not entry.compensatable:
                continue

            if entry.result is None:
                continue

            action = entry.action
            if not isinstance(action, Compensatable):
                continue

            # Check if compensation is possible
            if not action.can_compensate(entry.result):
                logger.info(f"Skipping compensation for {action.name}: not compensatable")
                continue

            try:
                comp_result = action.compensate(
                    checkpoint_result,
                    entry.result,
                    context,
                )
                entry.compensation_result = comp_result
                entry.compensated = comp_result.success
                result.compensation_results.append(comp_result)
                context.mark_action_compensated(action.name)

                # Callback
                if self._on_compensation_complete:
                    self._on_compensation_complete(entry)

                if not comp_result.success and not self._config.continue_on_compensation_failure:
                    logger.error(f"Compensation failed for {action.name}, stopping")
                    break

            except Exception as e:
                logger.error(f"Compensation error for {action.name}: {e}")
                comp_result = CompensationResult(
                    action_name=action.name,
                    success=False,
                    error=str(e),
                )
                entry.compensation_result = comp_result
                result.compensation_results.append(comp_result)

                if not self._config.continue_on_compensation_failure:
                    break

    def _transition_state(
        self,
        context: TransactionContext,
        new_state: TransactionState,
    ) -> None:
        """Transition to a new state.

        Args:
            context: Transaction context.
            new_state: Target state.
        """
        old_state = context.state
        context.state = new_state

        if self._on_state_change:
            self._on_state_change(old_state, new_state)

        if self._config.audit_enabled:
            logger.info(
                f"Transaction {context.transaction_id} state: "
                f"{old_state.value} -> {new_state.value}"
            )

    def rollback_to_savepoint(
        self,
        entries: list[ActionEntry],
        savepoint_name: str,
        checkpoint_result: "CheckpointResult",
        context: TransactionContext,
        result: TransactionResult,
    ) -> bool:
        """Rollback to a specific savepoint.

        Args:
            entries: All action entries.
            savepoint_name: Name of savepoint to rollback to.
            checkpoint_result: Checkpoint result.
            context: Transaction context.
            result: Transaction result to update.

        Returns:
            True if rollback succeeded.
        """
        savepoint = context.get_savepoint(savepoint_name)
        if not savepoint:
            logger.error(f"Savepoint not found: {savepoint_name}")
            return False

        # Only compensate actions after the savepoint
        actions_to_compensate = entries[savepoint.action_index:]
        self._compensate_actions(
            list(reversed(actions_to_compensate)),
            checkpoint_result,
            context,
            result,
        )

        result.savepoints_used.append(savepoint_name)
        return True


class SagaOrchestrator:
    """Orchestrates complex multi-step sagas.

    A Saga is a sequence of transactions where each step can
    trigger a local transaction. If a step fails, all previous
    steps are compensated in reverse order.

    This orchestrator supports:
    - Sequential steps
    - Parallel step groups
    - Conditional steps
    - Nested sagas

    Example:
        >>> saga = SagaOrchestrator()
        >>> saga.add_step("validate", validate_action)
        >>> saga.add_step("store", store_action)
        >>> saga.add_step("notify", notify_action, compensatable=True)
        >>> result = saga.execute(checkpoint_result)
    """

    def __init__(
        self,
        name: str = "saga",
        config: TransactionConfig | None = None,
    ) -> None:
        """Initialize the saga orchestrator.

        Args:
            name: Name of this saga.
            config: Transaction configuration.
        """
        self._name = name
        self._config = config or TransactionConfig()
        self._steps: list[SagaStep] = []
        self._coordinator = TransactionCoordinator(config=self._config)

    @property
    def name(self) -> str:
        """Get saga name."""
        return self._name

    def add_step(
        self,
        name: str,
        action: BaseAction[Any],
        compensation: Callable[..., CompensationResult | bool] | None = None,
        condition: Callable[["CheckpointResult", TransactionContext], bool] | None = None,
        required: bool = True,
    ) -> "SagaOrchestrator":
        """Add a step to the saga.

        Args:
            name: Step name.
            action: Action to execute.
            compensation: Optional compensation function.
            condition: Optional condition for execution.
            required: Whether step failure should trigger rollback.

        Returns:
            Self for chaining.
        """
        step = SagaStep(
            name=name,
            action=action,
            compensation=compensation,
            condition=condition,
            required=required,
        )
        self._steps.append(step)
        return self

    def add_parallel_steps(
        self,
        name: str,
        actions: list[tuple[str, BaseAction[Any]]],
        required: bool = True,
    ) -> "SagaOrchestrator":
        """Add a group of parallel steps.

        Note: Parallel execution requires async support.
        In sync mode, these execute sequentially.

        Args:
            name: Group name.
            actions: List of (name, action) tuples.
            required: Whether any failure should trigger rollback.

        Returns:
            Self for chaining.
        """
        step = SagaStep(
            name=name,
            action=None,
            parallel_actions=actions,
            required=required,
        )
        self._steps.append(step)
        return self

    def execute(
        self,
        checkpoint_result: "CheckpointResult",
        context: TransactionContext | None = None,
    ) -> TransactionResult:
        """Execute the saga.

        Args:
            checkpoint_result: Checkpoint result.
            context: Optional transaction context.

        Returns:
            TransactionResult with saga outcome.
        """
        context = context or TransactionContext()
        context.metadata["saga_name"] = self._name

        # Convert steps to actions
        actions = []
        for step in self._steps:
            if step.should_execute(checkpoint_result, context):
                if step.action:
                    actions.append(step.get_action())
                elif step.parallel_actions:
                    # Add all parallel actions (execute sequentially in sync mode)
                    for _, action in step.parallel_actions:
                        actions.append(action)

        return self._coordinator.execute(
            actions=actions,
            checkpoint_result=checkpoint_result,
            context=context,
        )

    def visualize(self) -> str:
        """Generate ASCII visualization of the saga.

        Returns:
            String visualization.
        """
        lines = [f"Saga: {self._name}", "=" * 40]

        for i, step in enumerate(self._steps, 1):
            prefix = "├─" if i < len(self._steps) else "└─"
            comp_indicator = " [C]" if step.has_compensation else ""
            req_indicator = " (required)" if step.required else " (optional)"

            if step.parallel_actions:
                lines.append(f"{prefix} {step.name} (parallel){req_indicator}")
                for j, (name, _) in enumerate(step.parallel_actions):
                    sub_prefix = "│  ├─" if j < len(step.parallel_actions) - 1 else "│  └─"
                    lines.append(f"{sub_prefix} {name}")
            else:
                lines.append(f"{prefix} {step.name}{comp_indicator}{req_indicator}")

        return "\n".join(lines)


@dataclass
class SagaStep:
    """A step in a saga."""

    name: str
    action: BaseAction[Any] | None = None
    compensation: Callable[..., CompensationResult | bool] | None = None
    condition: Callable[["CheckpointResult", TransactionContext], bool] | None = None
    required: bool = True
    parallel_actions: list[tuple[str, BaseAction[Any]]] | None = None

    @property
    def has_compensation(self) -> bool:
        """Check if step has compensation defined."""
        if self.compensation:
            return True
        if self.action and is_compensatable(self.action):
            return True
        return False

    def should_execute(
        self,
        checkpoint_result: "CheckpointResult",
        context: TransactionContext,
    ) -> bool:
        """Check if step should execute."""
        if self.condition is None:
            return True
        return self.condition(checkpoint_result, context)

    def get_action(self) -> BaseAction[Any]:
        """Get the action, wrapped with compensation if needed."""
        if self.action is None:
            raise ValueError("No action defined for step")

        if self.compensation and not is_compensatable(self.action):
            from truthound.checkpoint.transaction.compensatable import (
                CompensationWrapper,
            )

            return CompensationWrapper(
                action=self.action,
                compensation_fn=self.compensation,
            )

        return self.action
