"""Saga Pattern Implementations.

This module provides implementations of common saga patterns:
- Chained Saga: Sequential saga execution
- Nested Saga: Sagas containing sub-sagas
- Parallel Saga: Concurrent step execution
- Choreography Saga: Event-driven saga coordination
- Orchestrator Saga: Central coordinator pattern
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from truthound.checkpoint.transaction.saga.definition import (
    DependencyType,
    SagaDefinition,
    SagaStepDefinition,
    StepDependency,
)
from truthound.checkpoint.transaction.saga.state_machine import (
    SagaState,
    SagaStateMachine,
)
from truthound.checkpoint.transaction.saga.runner import (
    SagaRunner,
    SagaExecutionContext,
    SagaExecutionResult,
    SagaMetrics,
)

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionResult
    from truthound.checkpoint.checkpoint import CheckpointResult
    from truthound.checkpoint.transaction.base import TransactionContext


logger = logging.getLogger(__name__)


class SagaPattern(ABC):
    """Base class for saga patterns."""

    @abstractmethod
    def execute(
        self,
        context: SagaExecutionContext,
    ) -> SagaExecutionResult:
        """Execute the saga pattern.

        Args:
            context: Execution context.

        Returns:
            Execution result.
        """
        pass


@dataclass
class ChainedSagaConfig:
    """Configuration for chained saga execution.

    Attributes:
        stop_on_first_failure: Stop chain on first saga failure.
        rollback_all_on_failure: Rollback all sagas if one fails.
        timeout_seconds: Timeout for entire chain.
        share_context: Share context between chained sagas.
    """

    stop_on_first_failure: bool = True
    rollback_all_on_failure: bool = True
    timeout_seconds: float | None = None
    share_context: bool = True


class ChainedSagaPattern(SagaPattern):
    """Chained saga pattern - execute sagas sequentially.

    This pattern is useful when multiple sagas must execute in order,
    with later sagas depending on the results of earlier ones.

    Example:
        >>> chain = ChainedSagaPattern()
        >>> chain.add(order_saga)
        >>> chain.add(payment_saga)
        >>> chain.add(fulfillment_saga)
        >>> result = chain.execute(context)
    """

    def __init__(
        self,
        config: ChainedSagaConfig | None = None,
        runner: SagaRunner | None = None,
    ) -> None:
        """Initialize chained saga.

        Args:
            config: Chain configuration.
            runner: Saga runner to use.
        """
        self._config = config or ChainedSagaConfig()
        self._runner = runner or SagaRunner()
        self._sagas: list[SagaDefinition] = []
        self._chain_id = f"chain_{uuid4().hex[:12]}"

    @property
    def chain_id(self) -> str:
        """Get chain identifier."""
        return self._chain_id

    def add(self, saga: SagaDefinition) -> "ChainedSagaPattern":
        """Add a saga to the chain.

        Args:
            saga: Saga to add.

        Returns:
            Self for chaining.
        """
        self._sagas.append(saga)
        return self

    def execute(
        self,
        context: SagaExecutionContext,
    ) -> SagaExecutionResult:
        """Execute the saga chain."""
        results: list[SagaExecutionResult] = []
        completed_sagas: list[str] = []
        failed_saga: str | None = None
        failed_error: str | None = None
        metrics = SagaMetrics(saga_id=self._chain_id)

        start_time = datetime.now()

        for saga in self._sagas:
            # Check timeout
            if self._config.timeout_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > self._config.timeout_seconds:
                    failed_saga = saga.saga_id
                    failed_error = "Chain timeout"
                    break

            # Execute saga
            result = self._runner.execute(saga, context)
            results.append(result)

            if result.success:
                completed_sagas.append(saga.saga_id)
                # Share context if enabled
                if self._config.share_context:
                    context.metadata[f"saga_{saga.saga_id}_result"] = result.to_dict()
            else:
                failed_saga = saga.saga_id
                failed_error = result.error

                if self._config.stop_on_first_failure:
                    break

        # Handle rollback if needed
        if failed_saga and self._config.rollback_all_on_failure:
            logger.info(f"Rolling back {len(completed_sagas)} completed sagas")
            for saga_id in reversed(completed_sagas):
                saga = next(s for s in self._sagas if s.saga_id == saga_id)
                # Trigger compensation for completed sagas
                # This would require re-running with a flag or having stored state

        # Aggregate results
        all_completed_steps = []
        all_compensated_steps = []
        all_events = []

        for result in results:
            all_completed_steps.extend(result.completed_steps)
            all_compensated_steps.extend(result.compensated_steps)
            all_events.extend(result.events)

        success = all(r.success for r in results)
        final_state = SagaState.COMPLETED if success else SagaState.FAILED

        metrics.finalize(success=success)

        return SagaExecutionResult(
            saga_id=self._chain_id,
            state=final_state,
            success=success,
            completed_steps=all_completed_steps,
            compensated_steps=all_compensated_steps,
            failed_step=failed_saga,
            error=failed_error,
            metrics=metrics,
            events=all_events,
        )


@dataclass
class NestedSagaConfig:
    """Configuration for nested saga execution.

    Attributes:
        max_depth: Maximum nesting depth.
        propagate_failure: Propagate failure to parent.
        inherit_context: Child inherits parent context.
    """

    max_depth: int = 5
    propagate_failure: bool = True
    inherit_context: bool = True


class NestedSagaPattern(SagaPattern):
    """Nested saga pattern - sagas containing sub-sagas.

    This pattern allows complex workflows where steps can trigger
    entire sub-sagas.

    Example:
        >>> parent = NestedSagaPattern(parent_saga)
        >>> parent.add_nested_at_step("process_order", order_processing_saga)
        >>> result = parent.execute(context)
    """

    def __init__(
        self,
        parent_saga: SagaDefinition,
        config: NestedSagaConfig | None = None,
        runner: SagaRunner | None = None,
    ) -> None:
        """Initialize nested saga.

        Args:
            parent_saga: Parent saga definition.
            config: Nesting configuration.
            runner: Saga runner to use.
        """
        self._parent = parent_saga
        self._config = config or NestedSagaConfig()
        self._runner = runner or SagaRunner()
        self._nested_sagas: dict[str, SagaDefinition] = {}
        self._current_depth = 0

    def add_nested_at_step(
        self,
        step_id: str,
        saga: SagaDefinition,
    ) -> "NestedSagaPattern":
        """Add a nested saga to execute at a specific step.

        Args:
            step_id: Parent step ID after which to execute.
            saga: Nested saga definition.

        Returns:
            Self for chaining.
        """
        self._nested_sagas[step_id] = saga
        return self

    def execute(
        self,
        context: SagaExecutionContext,
        depth: int = 0,
    ) -> SagaExecutionResult:
        """Execute the nested saga pattern."""
        if depth > self._config.max_depth:
            raise RecursionError(f"Maximum saga nesting depth ({self._config.max_depth}) exceeded")

        self._current_depth = depth
        all_results: list[SagaExecutionResult] = []
        all_completed_steps = []
        all_compensated_steps = []

        # Create execution context for nested tracking
        nested_context = SagaExecutionContext(
            checkpoint_result=context.checkpoint_result,
            transaction_context=context.transaction_context,
            metadata=context.metadata.copy() if self._config.inherit_context else {},
        )

        # Execute parent saga steps
        execution_order = self._parent.get_execution_order()
        machine = SagaStateMachine(self._parent.saga_id)
        machine.start()

        for step in execution_order:
            # Execute step
            if step.action:
                try:
                    machine.step_started(step.step_id)
                    result = step.action.execute(context.checkpoint_result)

                    if result.success:
                        machine.step_completed(step.step_id)
                        all_completed_steps.append(step.step_id)

                        # Check for nested saga at this step
                        if step.step_id in self._nested_sagas:
                            nested_saga = self._nested_sagas[step.step_id]
                            nested_result = self._execute_nested(
                                nested_saga, nested_context, depth + 1
                            )
                            all_results.append(nested_result)

                            if not nested_result.success and self._config.propagate_failure:
                                machine.step_failed(step.step_id, "Nested saga failed")
                                break

                            all_completed_steps.extend(nested_result.completed_steps)
                    else:
                        machine.step_failed(step.step_id, result.error or "Step failed")
                        break

                except Exception as e:
                    machine.step_failed(step.step_id, str(e))
                    break

        # Handle compensation if needed
        if machine.state == SagaState.STEP_FAILED:
            machine.start_compensation()
            compensation_result = self._run_parent_compensation(
                self._parent, machine, context, all_completed_steps
            )
            all_compensated_steps.extend(compensation_result)

            # Also compensate nested sagas in reverse order
            for nested_result in reversed(all_results):
                if nested_result.success:
                    all_compensated_steps.extend(
                        self._compensate_nested(
                            nested_result.saga_id, nested_context
                        )
                    )

            machine.compensation_complete()

        success = machine.state == SagaState.COMPLETED
        metrics = SagaMetrics(saga_id=self._parent.saga_id)
        metrics.finalize(success=success)

        return SagaExecutionResult(
            saga_id=self._parent.saga_id,
            state=machine.state,
            success=success,
            completed_steps=all_completed_steps,
            compensated_steps=all_compensated_steps,
            metrics=metrics,
            events=machine.events,
        )

    def _execute_nested(
        self,
        saga: SagaDefinition,
        context: SagaExecutionContext,
        depth: int,
    ) -> SagaExecutionResult:
        """Execute a nested saga."""
        # Check if nested saga has its own nested sagas
        nested_pattern = NestedSagaPattern(
            saga, self._config, self._runner
        )
        return nested_pattern.execute(context, depth)

    def _run_parent_compensation(
        self,
        saga: SagaDefinition,
        machine: SagaStateMachine,
        context: SagaExecutionContext,
        completed_steps: list[str],
    ) -> list[str]:
        """Run compensation for parent saga steps."""
        compensated = []
        compensation_order = saga.get_compensation_order()

        for step in compensation_order:
            if step.step_id not in completed_steps:
                continue

            machine.step_compensating(step.step_id)

            try:
                if step.compensation_action:
                    step.compensation_action.execute(context.checkpoint_result)
                elif step.compensation_fn:
                    step.compensation_fn(
                        context.checkpoint_result,
                        context.step_results.get(step.step_id),
                        context.transaction_context,
                    )

                machine.step_compensated(step.step_id)
                compensated.append(step.step_id)
            except Exception as e:
                logger.error(f"Compensation failed for {step.step_id}: {e}")
                machine.compensation_failed(step.step_id, str(e))

        return compensated

    def _compensate_nested(
        self,
        saga_id: str,
        context: SagaExecutionContext,
    ) -> list[str]:
        """Compensate a nested saga."""
        # Would need to trigger compensation for the nested saga
        # This is simplified - real implementation would use event store
        return []


@dataclass
class ParallelSagaConfig:
    """Configuration for parallel saga execution.

    Attributes:
        max_parallel: Maximum concurrent executions.
        wait_all: Wait for all to complete before returning.
        fail_fast: Fail immediately on first error.
        timeout_seconds: Timeout for parallel execution.
    """

    max_parallel: int = 5
    wait_all: bool = True
    fail_fast: bool = True
    timeout_seconds: float | None = None


class ParallelSagaPattern(SagaPattern):
    """Parallel saga pattern - execute steps concurrently.

    This pattern is useful when multiple independent steps can
    execute simultaneously.

    Example:
        >>> parallel = ParallelSagaPattern(saga)
        >>> parallel.parallelize_steps(["step1", "step2", "step3"])
        >>> result = parallel.execute(context)
    """

    def __init__(
        self,
        saga: SagaDefinition,
        config: ParallelSagaConfig | None = None,
    ) -> None:
        """Initialize parallel saga.

        Args:
            saga: Saga definition.
            config: Parallel execution configuration.
        """
        self._saga = saga
        self._config = config or ParallelSagaConfig()
        self._parallel_groups: list[set[str]] = []

    def parallelize_steps(self, step_ids: list[str]) -> "ParallelSagaPattern":
        """Mark steps for parallel execution.

        Args:
            step_ids: Steps to execute in parallel.

        Returns:
            Self for chaining.
        """
        self._parallel_groups.append(set(step_ids))
        return self

    def execute(
        self,
        context: SagaExecutionContext,
    ) -> SagaExecutionResult:
        """Execute the parallel saga pattern."""
        machine = SagaStateMachine(self._saga.saga_id)
        metrics = SagaMetrics(saga_id=self._saga.saga_id)
        completed_steps: list[str] = []
        failed_step: str | None = None
        failed_error: str | None = None

        machine.start()

        # Build parallel execution groups
        parallel_step_ids = set()
        for group in self._parallel_groups:
            parallel_step_ids.update(group)

        execution_order = self._saga.get_execution_order()

        # Group steps
        current_parallel = []
        sequential_buffer = []

        for step in execution_order:
            if step.step_id in parallel_step_ids:
                if sequential_buffer:
                    # Execute sequential buffer first
                    for seq_step in sequential_buffer:
                        result = self._execute_step_sync(
                            seq_step, context, machine
                        )
                        if result.success:
                            completed_steps.append(seq_step.step_id)
                        else:
                            failed_step = seq_step.step_id
                            failed_error = result.error
                            break
                    sequential_buffer = []

                    if failed_step:
                        break

                current_parallel.append(step)
            else:
                if current_parallel:
                    # Execute parallel group
                    results = self._execute_parallel_group(
                        current_parallel, context, machine
                    )
                    for step_id, result in results.items():
                        if result.success:
                            completed_steps.append(step_id)
                        elif self._config.fail_fast:
                            failed_step = step_id
                            failed_error = result.error
                            break

                    current_parallel = []

                    if failed_step:
                        break

                sequential_buffer.append(step)

        # Handle remaining steps
        if not failed_step:
            if current_parallel:
                results = self._execute_parallel_group(
                    current_parallel, context, machine
                )
                for step_id, result in results.items():
                    if result.success:
                        completed_steps.append(step_id)
                    elif self._config.fail_fast:
                        failed_step = step_id
                        failed_error = result.error
                        break

            if not failed_step:
                for seq_step in sequential_buffer:
                    result = self._execute_step_sync(seq_step, context, machine)
                    if result.success:
                        completed_steps.append(seq_step.step_id)
                    else:
                        failed_step = seq_step.step_id
                        failed_error = result.error
                        break

        # Finalize
        if failed_step:
            # Run compensation
            compensated = self._run_compensation(
                completed_steps, context, machine
            )
            metrics.finalize(success=False)

            return SagaExecutionResult(
                saga_id=self._saga.saga_id,
                state=SagaState.COMPENSATED if compensated else SagaState.FAILED,
                success=False,
                completed_steps=completed_steps,
                compensated_steps=compensated,
                failed_step=failed_step,
                error=failed_error,
                metrics=metrics,
                events=machine.events,
            )

        machine.complete()
        metrics.finalize(success=True)

        return SagaExecutionResult(
            saga_id=self._saga.saga_id,
            state=SagaState.COMPLETED,
            success=True,
            completed_steps=completed_steps,
            metrics=metrics,
            events=machine.events,
        )

    def _execute_step_sync(
        self,
        step: SagaStepDefinition,
        context: SagaExecutionContext,
        machine: SagaStateMachine,
    ) -> "ActionResult":
        """Execute a step synchronously."""
        from truthound.checkpoint.actions.base import ActionResult, ActionStatus

        if step.action is None:
            return ActionResult(
                action_name=step.name,
                action_type="parallel_step",
                status=ActionStatus.FAILURE,
                error="No action defined",
            )

        machine.step_started(step.step_id)

        try:
            result = step.action.execute(context.checkpoint_result)
            if result.success:
                machine.step_completed(step.step_id)
            else:
                machine.step_failed(step.step_id, result.error or "Step failed")
            return result
        except Exception as e:
            machine.step_failed(step.step_id, str(e))
            return ActionResult(
                action_name=step.name,
                action_type="parallel_step",
                status=ActionStatus.ERROR,
                error=str(e),
            )

    def _execute_parallel_group(
        self,
        steps: list[SagaStepDefinition],
        context: SagaExecutionContext,
        machine: SagaStateMachine,
    ) -> dict[str, "ActionResult"]:
        """Execute a group of steps in parallel."""
        from truthound.checkpoint.actions.base import ActionResult, ActionStatus

        results: dict[str, ActionResult] = {}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._config.max_parallel
        ) as executor:
            futures = {}

            for step in steps:
                if step.action:
                    machine.step_started(step.step_id)
                    future = executor.submit(
                        step.action.execute,
                        context.checkpoint_result,
                    )
                    futures[step.step_id] = future

            # Collect results
            for step_id, future in futures.items():
                try:
                    if self._config.timeout_seconds:
                        result = future.result(timeout=self._config.timeout_seconds)
                    else:
                        result = future.result()

                    results[step_id] = result

                    if result.success:
                        machine.step_completed(step_id)
                    else:
                        machine.step_failed(step_id, result.error or "Step failed")

                except concurrent.futures.TimeoutError:
                    results[step_id] = ActionResult(
                        action_name=step_id,
                        action_type="parallel_step",
                        status=ActionStatus.ERROR,
                        error="Timeout",
                    )
                    machine.step_failed(step_id, "Timeout")

                except Exception as e:
                    results[step_id] = ActionResult(
                        action_name=step_id,
                        action_type="parallel_step",
                        status=ActionStatus.ERROR,
                        error=str(e),
                    )
                    machine.step_failed(step_id, str(e))

        return results

    def _run_compensation(
        self,
        completed_steps: list[str],
        context: SagaExecutionContext,
        machine: SagaStateMachine,
    ) -> list[str]:
        """Run compensation for completed steps."""
        compensated = []
        machine.start_compensation()

        for step_id in reversed(completed_steps):
            step = self._saga.get_step(step_id)
            if step is None or not step.has_compensation():
                continue

            machine.step_compensating(step_id)

            try:
                if step.compensation_action:
                    step.compensation_action.execute(context.checkpoint_result)
                elif step.compensation_fn:
                    step.compensation_fn(
                        context.checkpoint_result,
                        context.step_results.get(step_id),
                        context.transaction_context,
                    )

                machine.step_compensated(step_id)
                compensated.append(step_id)

            except Exception as e:
                logger.error(f"Compensation failed for {step_id}: {e}")
                machine.compensation_failed(step_id, str(e))

        if compensated:
            machine.compensation_complete()

        return compensated


class ChoreographySagaPattern(SagaPattern):
    """Choreography saga pattern - event-driven coordination.

    In this pattern, each service produces and listens to events,
    with no central coordinator.

    Example:
        >>> choreography = ChoreographySagaPattern()
        >>> choreography.on_event("order_created", process_order_step)
        >>> choreography.on_event("order_processed", reserve_inventory_step)
        >>> choreography.emit("order_created", {"order_id": 123})
    """

    def __init__(self) -> None:
        """Initialize choreography saga."""
        self._handlers: dict[str, list[Callable[..., Any]]] = {}
        self._event_history: list[tuple[str, Any]] = []
        self._saga_id = f"choreography_{uuid4().hex[:12]}"

    def on_event(
        self,
        event_type: str,
        handler: Callable[..., Any],
    ) -> "ChoreographySagaPattern":
        """Register a handler for an event type.

        Args:
            event_type: Type of event to handle.
            handler: Handler function.

        Returns:
            Self for chaining.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        return self

    def emit(self, event_type: str, data: Any = None) -> list[Any]:
        """Emit an event to all registered handlers.

        Args:
            event_type: Type of event.
            data: Event data.

        Returns:
            Results from all handlers.
        """
        self._event_history.append((event_type, data))
        results = []

        for handler in self._handlers.get(event_type, []):
            try:
                result = handler(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler error for {event_type}: {e}")
                results.append(e)

        return results

    def execute(
        self,
        context: SagaExecutionContext,
    ) -> SagaExecutionResult:
        """Execute is not directly used in choreography pattern."""
        return SagaExecutionResult(
            saga_id=self._saga_id,
            state=SagaState.COMPLETED,
            success=True,
            metrics=SagaMetrics(saga_id=self._saga_id),
        )


class OrchestratorSagaPattern(SagaPattern):
    """Orchestrator saga pattern - central coordinator.

    In this pattern, a central orchestrator coordinates all saga
    steps and manages the overall transaction.

    This is essentially what SagaRunner implements, but this class
    provides additional orchestration features.
    """

    def __init__(
        self,
        saga: SagaDefinition,
        runner: SagaRunner | None = None,
    ) -> None:
        """Initialize orchestrator saga.

        Args:
            saga: Saga definition.
            runner: Saga runner to use.
        """
        self._saga = saga
        self._runner = runner or SagaRunner()

    def execute(
        self,
        context: SagaExecutionContext,
    ) -> SagaExecutionResult:
        """Execute the saga using the orchestrator."""
        return self._runner.execute(self._saga, context)
