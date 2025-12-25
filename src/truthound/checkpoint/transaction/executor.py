"""Transactional Executor.

This module provides the TransactionalExecutor class that wraps
checkpoint execution with transaction support.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterator

from truthound.checkpoint.transaction.base import (
    CompensationResult,
    CompensationStrategy,
    TransactionConfig,
    TransactionContext,
    TransactionPhase,
    TransactionResult,
    TransactionState,
)
from truthound.checkpoint.transaction.coordinator import (
    ActionEntry,
    TransactionCoordinator,
)
from truthound.checkpoint.actions.base import (
    ActionResult,
    BaseAction,
)

if TYPE_CHECKING:
    from truthound.checkpoint.checkpoint import CheckpointResult


logger = logging.getLogger(__name__)


@dataclass
class TransactionBoundary:
    """Defines a transaction boundary for execution.

    Transaction boundaries allow grouping multiple actions into
    a single atomic unit that succeeds or fails together.

    Attributes:
        name: Boundary identifier.
        actions: Actions within this boundary.
        config: Transaction configuration.
        nested: Whether this is a nested transaction.
        parent: Parent boundary for nested transactions.
    """

    name: str
    actions: list[BaseAction[Any]] = field(default_factory=list)
    config: TransactionConfig = field(default_factory=TransactionConfig)
    nested: bool = False
    parent: TransactionBoundary | None = None

    def add_action(self, action: BaseAction[Any]) -> "TransactionBoundary":
        """Add an action to this boundary.

        Args:
            action: Action to add.

        Returns:
            Self for chaining.
        """
        self.actions.append(action)
        return self

    def create_nested(self, name: str) -> "TransactionBoundary":
        """Create a nested transaction boundary.

        Args:
            name: Name for the nested boundary.

        Returns:
            New nested boundary.
        """
        return TransactionBoundary(
            name=name,
            config=self.config,
            nested=True,
            parent=self,
        )


class TransactionalExecutor:
    """Executor that wraps actions in transactions.

    The TransactionalExecutor provides transaction management for
    action execution, including:
    - Automatic transaction boundaries
    - Nested transaction support
    - Manual transaction control
    - Transaction hooks

    Example:
        >>> executor = TransactionalExecutor()
        >>> with executor.transaction("my_txn") as txn:
        ...     txn.add_action(action1)
        ...     txn.add_action(action2)
        >>> result = executor.execute(checkpoint_result)
        >>> if not result.success:
        ...     print(f"Transaction failed: {result.error}")
    """

    def __init__(
        self,
        config: TransactionConfig | None = None,
        on_transaction_start: Callable[[TransactionContext], None] | None = None,
        on_transaction_complete: Callable[[TransactionResult], None] | None = None,
        on_compensation_needed: Callable[[TransactionContext, str], None] | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            config: Default transaction configuration.
            on_transaction_start: Callback when transaction starts.
            on_transaction_complete: Callback when transaction completes.
            on_compensation_needed: Callback when compensation is triggered.
        """
        self._config = config or TransactionConfig()
        self._on_transaction_start = on_transaction_start
        self._on_transaction_complete = on_transaction_complete
        self._on_compensation_needed = on_compensation_needed

        self._boundaries: list[TransactionBoundary] = []
        self._current_boundary: TransactionBoundary | None = None
        self._coordinator = TransactionCoordinator(config=self._config)

    @property
    def config(self) -> TransactionConfig:
        """Get the default transaction configuration."""
        return self._config

    @contextmanager
    def transaction(
        self,
        name: str = "default",
        config: TransactionConfig | None = None,
    ) -> Generator[TransactionBoundary, None, None]:
        """Create a transaction boundary context.

        Use this to define a group of actions that should execute
        as an atomic unit.

        Args:
            name: Transaction boundary name.
            config: Optional configuration override.

        Yields:
            TransactionBoundary to add actions to.

        Example:
            >>> with executor.transaction("process_data") as txn:
            ...     txn.add_action(validate_action)
            ...     txn.add_action(store_action)
            ...     txn.add_action(notify_action)
        """
        boundary = TransactionBoundary(
            name=name,
            config=config or self._config,
            nested=self._current_boundary is not None,
            parent=self._current_boundary,
        )

        previous = self._current_boundary
        self._current_boundary = boundary

        try:
            yield boundary
            self._boundaries.append(boundary)
        finally:
            self._current_boundary = previous

    def add_action(self, action: BaseAction[Any]) -> "TransactionalExecutor":
        """Add an action to the current or default boundary.

        Args:
            action: Action to add.

        Returns:
            Self for chaining.
        """
        if self._current_boundary:
            self._current_boundary.add_action(action)
        else:
            # Create implicit boundary
            boundary = TransactionBoundary(
                name="implicit",
                config=self._config,
            )
            boundary.add_action(action)
            self._boundaries.append(boundary)
        return self

    def execute(
        self,
        checkpoint_result: "CheckpointResult",
        context: TransactionContext | None = None,
    ) -> TransactionResult:
        """Execute all transaction boundaries.

        Args:
            checkpoint_result: Checkpoint result to pass to actions.
            context: Optional transaction context.

        Returns:
            TransactionResult with complete outcome.
        """
        context = context or TransactionContext()

        # Start callback
        if self._on_transaction_start:
            self._on_transaction_start(context)

        results: list[TransactionResult] = []

        for boundary in self._boundaries:
            if not boundary.actions:
                continue

            # Create coordinator for this boundary
            coordinator = TransactionCoordinator(
                config=boundary.config,
                on_state_change=self._create_state_handler(boundary),
            )

            # Execute
            result = coordinator.execute(
                actions=boundary.actions,
                checkpoint_result=checkpoint_result,
                context=context,
            )
            results.append(result)

            # Handle failure with compensation notification
            if not result.success and self._on_compensation_needed:
                self._on_compensation_needed(context, boundary.name)

            # Stop if boundary failed and not configured to continue
            if not result.success and boundary.config.rollback_on_failure:
                break

        # Aggregate results
        final_result = self._aggregate_results(results, context)

        # Complete callback
        if self._on_transaction_complete:
            self._on_transaction_complete(final_result)

        # Clear boundaries for next execution
        self._boundaries.clear()

        return final_result

    def execute_single(
        self,
        actions: list[BaseAction[Any]],
        checkpoint_result: "CheckpointResult",
        context: TransactionContext | None = None,
        config: TransactionConfig | None = None,
    ) -> TransactionResult:
        """Execute actions as a single transaction.

        Convenience method for executing a list of actions without
        explicitly creating transaction boundaries.

        Args:
            actions: Actions to execute.
            checkpoint_result: Checkpoint result.
            context: Optional context.
            config: Optional configuration.

        Returns:
            TransactionResult.
        """
        context = context or TransactionContext()
        config = config or self._config

        coordinator = TransactionCoordinator(config=config)
        return coordinator.execute(
            actions=actions,
            checkpoint_result=checkpoint_result,
            context=context,
        )

    def _create_state_handler(
        self,
        boundary: TransactionBoundary,
    ) -> Callable[[TransactionState, TransactionState], None]:
        """Create state change handler for a boundary."""

        def handler(old_state: TransactionState, new_state: TransactionState) -> None:
            logger.debug(
                f"Boundary '{boundary.name}': {old_state.value} -> {new_state.value}"
            )

        return handler

    def _aggregate_results(
        self,
        results: list[TransactionResult],
        context: TransactionContext,
    ) -> TransactionResult:
        """Aggregate multiple transaction results."""
        if not results:
            return TransactionResult(
                transaction_id=context.transaction_id,
                state=TransactionState.COMMITTED,
                started_at=context.started_at,
                completed_at=datetime.now(),
            )

        # Combine all action and compensation results
        all_action_results: list[ActionResult] = []
        all_compensation_results: list[CompensationResult] = []
        all_savepoints: list[str] = []
        errors: list[str] = []

        for result in results:
            all_action_results.extend(result.action_results)
            all_compensation_results.extend(result.compensation_results)
            all_savepoints.extend(result.savepoints_used)
            if result.error:
                errors.append(result.error)

        # Determine final state
        if all(r.success for r in results):
            final_state = TransactionState.COMMITTED
        elif all_compensation_results and all(c.success for c in all_compensation_results):
            final_state = TransactionState.COMPENSATED
        else:
            final_state = TransactionState.FAILED

        completed_at = datetime.now()
        return TransactionResult(
            transaction_id=context.transaction_id,
            state=final_state,
            started_at=context.started_at,
            completed_at=completed_at,
            duration_ms=(completed_at - context.started_at).total_seconds() * 1000,
            action_results=all_action_results,
            compensation_results=all_compensation_results,
            savepoints_used=all_savepoints,
            error="; ".join(errors) if errors else None,
            metadata={"boundary_count": len(results)},
        )


class TransactionManager:
    """Manages transaction lifecycle across multiple executors.

    The TransactionManager provides a higher-level interface for
    managing transactions, including:
    - Transaction registry
    - Distributed transaction coordination
    - Transaction recovery
    - Audit logging

    Example:
        >>> manager = TransactionManager()
        >>> txn_id = manager.begin_transaction()
        >>> try:
        ...     manager.execute_action(txn_id, action)
        ...     manager.commit(txn_id)
        ... except Exception:
        ...     manager.rollback(txn_id)
    """

    def __init__(
        self,
        config: TransactionConfig | None = None,
    ) -> None:
        """Initialize the manager.

        Args:
            config: Default transaction configuration.
        """
        self._config = config or TransactionConfig()
        self._active_transactions: dict[str, TransactionContext] = {}
        self._transaction_history: list[TransactionResult] = []
        self._action_log: dict[str, list[ActionEntry]] = {}

    def begin_transaction(
        self,
        context: TransactionContext | None = None,
    ) -> str:
        """Begin a new transaction.

        Args:
            context: Optional pre-configured context.

        Returns:
            Transaction ID.
        """
        context = context or TransactionContext()
        context.state = TransactionState.ACTIVE
        context.phase = TransactionPhase.EXECUTE

        self._active_transactions[context.transaction_id] = context
        self._action_log[context.transaction_id] = []

        logger.info(f"Transaction started: {context.transaction_id}")
        return context.transaction_id

    def get_transaction(self, transaction_id: str) -> TransactionContext | None:
        """Get an active transaction by ID.

        Args:
            transaction_id: Transaction ID.

        Returns:
            Transaction context, or None if not found.
        """
        return self._active_transactions.get(transaction_id)

    def add_action(
        self,
        transaction_id: str,
        action: BaseAction[Any],
    ) -> None:
        """Add an action to a transaction.

        Args:
            transaction_id: Transaction ID.
            action: Action to add.

        Raises:
            ValueError: If transaction not found or not active.
        """
        context = self._active_transactions.get(transaction_id)
        if not context:
            raise ValueError(f"Transaction not found: {transaction_id}")
        if context.state != TransactionState.ACTIVE:
            raise ValueError(f"Transaction not active: {transaction_id}")

        entry = ActionEntry(action=action)
        self._action_log[transaction_id].append(entry)

    def execute_pending(
        self,
        transaction_id: str,
        checkpoint_result: "CheckpointResult",
    ) -> list[ActionResult]:
        """Execute pending actions in a transaction.

        Args:
            transaction_id: Transaction ID.
            checkpoint_result: Checkpoint result.

        Returns:
            List of action results.
        """
        context = self._active_transactions.get(transaction_id)
        if not context:
            raise ValueError(f"Transaction not found: {transaction_id}")

        entries = self._action_log.get(transaction_id, [])
        results: list[ActionResult] = []

        for entry in entries:
            if entry.executed:
                continue

            try:
                result = entry.action.execute(checkpoint_result)
                entry.result = result
                entry.executed = True
                results.append(result)

                if not result.success:
                    context.failed_action = entry.action.name
                    break

            except Exception as e:
                logger.error(f"Action execution error: {e}")
                context.failed_action = entry.action.name
                break

        return results

    def commit(
        self,
        transaction_id: str,
    ) -> TransactionResult:
        """Commit a transaction.

        Args:
            transaction_id: Transaction ID.

        Returns:
            TransactionResult.
        """
        context = self._active_transactions.get(transaction_id)
        if not context:
            raise ValueError(f"Transaction not found: {transaction_id}")

        entries = self._action_log.get(transaction_id, [])

        # Check all actions completed successfully
        all_success = all(
            entry.result and entry.result.success
            for entry in entries
            if entry.executed
        )

        if all_success:
            context.state = TransactionState.COMMITTED
        else:
            context.state = TransactionState.FAILED

        result = self._create_result(context, entries)
        self._finalize_transaction(transaction_id, result)
        return result

    def rollback(
        self,
        transaction_id: str,
        checkpoint_result: "CheckpointResult",
    ) -> TransactionResult:
        """Rollback a transaction.

        Args:
            transaction_id: Transaction ID.
            checkpoint_result: Checkpoint result for compensation.

        Returns:
            TransactionResult.
        """
        context = self._active_transactions.get(transaction_id)
        if not context:
            raise ValueError(f"Transaction not found: {transaction_id}")

        context.state = TransactionState.ROLLING_BACK
        context.phase = TransactionPhase.COMPENSATE

        entries = self._action_log.get(transaction_id, [])

        # Compensate executed actions
        coordinator = TransactionCoordinator(config=self._config)
        result = TransactionResult(
            transaction_id=transaction_id,
            state=TransactionState.ROLLING_BACK,
            started_at=context.started_at,
        )

        executed_entries = [e for e in entries if e.executed]
        coordinator._compensate_actions(
            list(reversed(executed_entries)),
            checkpoint_result,
            context,
            result,
        )

        # Update state
        if all(e.compensated or not e.compensatable for e in executed_entries):
            context.state = TransactionState.COMPENSATED
        else:
            context.state = TransactionState.FAILED

        result.state = context.state
        result.completed_at = datetime.now()
        result.duration_ms = (result.completed_at - result.started_at).total_seconds() * 1000

        self._finalize_transaction(transaction_id, result)
        return result

    def _create_result(
        self,
        context: TransactionContext,
        entries: list[ActionEntry],
    ) -> TransactionResult:
        """Create transaction result from entries."""
        completed_at = datetime.now()
        return TransactionResult(
            transaction_id=context.transaction_id,
            state=context.state,
            started_at=context.started_at,
            completed_at=completed_at,
            duration_ms=(completed_at - context.started_at).total_seconds() * 1000,
            action_results=[e.result for e in entries if e.result],
            error=f"Failed at: {context.failed_action}" if context.failed_action else None,
            metadata=context.to_dict(),
        )

    def _finalize_transaction(
        self,
        transaction_id: str,
        result: TransactionResult,
    ) -> None:
        """Clean up after transaction completion."""
        self._transaction_history.append(result)
        del self._active_transactions[transaction_id]
        del self._action_log[transaction_id]

        logger.info(
            f"Transaction {transaction_id} finalized: {result.state.value}"
        )

    def get_history(
        self,
        limit: int | None = None,
    ) -> list[TransactionResult]:
        """Get transaction history.

        Args:
            limit: Maximum number of results.

        Returns:
            List of historical transaction results.
        """
        if limit:
            return self._transaction_history[-limit:]
        return list(self._transaction_history)

    def clear_history(self) -> None:
        """Clear transaction history."""
        self._transaction_history.clear()
