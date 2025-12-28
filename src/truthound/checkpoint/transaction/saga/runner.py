"""Saga Runner Module.

This module provides the execution engine for running sagas,
with support for various execution modes and recovery options.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

from truthound.checkpoint.transaction.saga.definition import (
    RetryPolicy,
    SagaDefinition,
    SagaStepDefinition,
    StepExecutionMode,
)
from truthound.checkpoint.transaction.saga.state_machine import (
    SagaEvent,
    SagaEventType,
    SagaState,
    SagaStateMachine,
    InvalidTransitionError,
)
from truthound.checkpoint.transaction.saga.strategies import (
    CompensationPlan,
    CompensationPlanner,
    CompensationPolicy,
)
from truthound.checkpoint.transaction.saga.event_store import (
    InMemorySagaEventStore,
    SagaEventStore,
    SagaSnapshot,
)

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionResult, BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult
    from truthound.checkpoint.transaction.base import (
        CompensationResult,
        TransactionContext,
    )


logger = logging.getLogger(__name__)


@dataclass
class SagaMetrics:
    """Metrics collected during saga execution.

    Attributes:
        saga_id: Saga identifier.
        started_at: When saga started.
        completed_at: When saga completed.
        total_duration_ms: Total execution time.
        step_durations: Duration of each step.
        compensation_durations: Duration of each compensation.
        retry_counts: Retry count per step.
        error_count: Total error count.
        success: Whether saga succeeded.
    """

    saga_id: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    total_duration_ms: float = 0.0
    step_durations: dict[str, float] = field(default_factory=dict)
    compensation_durations: dict[str, float] = field(default_factory=dict)
    retry_counts: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    success: bool = False

    def record_step_duration(self, step_id: str, duration_ms: float) -> None:
        """Record step duration."""
        self.step_durations[step_id] = duration_ms

    def record_compensation_duration(self, step_id: str, duration_ms: float) -> None:
        """Record compensation duration."""
        self.compensation_durations[step_id] = duration_ms

    def record_retry(self, step_id: str) -> None:
        """Record a retry for a step."""
        self.retry_counts[step_id] = self.retry_counts.get(step_id, 0) + 1

    def record_error(self) -> None:
        """Record an error."""
        self.error_count += 1

    def finalize(self, success: bool) -> None:
        """Finalize metrics."""
        self.completed_at = datetime.now()
        self.total_duration_ms = (
            self.completed_at - self.started_at
        ).total_seconds() * 1000
        self.success = success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "saga_id": self.saga_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "step_durations": self.step_durations,
            "compensation_durations": self.compensation_durations,
            "retry_counts": self.retry_counts,
            "error_count": self.error_count,
            "success": self.success,
        }


@dataclass
class SagaExecutionContext:
    """Context for saga execution.

    Attributes:
        checkpoint_result: Checkpoint result to pass to actions.
        transaction_context: Transaction context.
        step_results: Results from each step.
        compensation_results: Results from compensations.
        metadata: Additional context metadata.
    """

    checkpoint_result: "CheckpointResult"
    transaction_context: "TransactionContext | None" = None
    step_results: dict[str, "ActionResult"] = field(default_factory=dict)
    compensation_results: dict[str, "CompensationResult"] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SagaExecutionResult:
    """Result of saga execution.

    Attributes:
        saga_id: Saga identifier.
        state: Final saga state.
        success: Whether saga succeeded.
        completed_steps: IDs of completed steps.
        compensated_steps: IDs of compensated steps.
        failed_step: ID of the failed step.
        error: Error message if failed.
        step_results: Results from each step.
        compensation_results: Results from compensations.
        metrics: Execution metrics.
        events: All saga events.
    """

    saga_id: str
    state: SagaState
    success: bool
    completed_steps: list[str] = field(default_factory=list)
    compensated_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None
    error: str | None = None
    step_results: dict[str, Any] = field(default_factory=dict)
    compensation_results: dict[str, Any] = field(default_factory=dict)
    metrics: SagaMetrics = field(default_factory=SagaMetrics)
    events: list[SagaEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "saga_id": self.saga_id,
            "state": self.state.value,
            "success": self.success,
            "completed_steps": self.completed_steps,
            "compensated_steps": self.compensated_steps,
            "failed_step": self.failed_step,
            "error": self.error,
            "metrics": self.metrics.to_dict(),
            "event_count": len(self.events),
        }


class SagaRunner:
    """Executes saga definitions with full lifecycle management.

    The SagaRunner handles:
    - Step execution with retry and timeout
    - Compensation on failure
    - State machine management
    - Event persistence
    - Metrics collection

    Example:
        >>> runner = SagaRunner(event_store=InMemorySagaEventStore())
        >>> result = runner.execute(saga_definition, context)
        >>> if not result.success:
        ...     print(f"Saga failed: {result.error}")
        ...     print(f"Compensated: {result.compensated_steps}")
    """

    def __init__(
        self,
        event_store: SagaEventStore | None = None,
        compensation_policy: CompensationPolicy = CompensationPolicy.BACKWARD,
        default_timeout: timedelta = timedelta(seconds=30),
        max_parallel_steps: int = 5,
        on_step_complete: Callable[[str, "ActionResult"], None] | None = None,
        on_step_failed: Callable[[str, Exception], None] | None = None,
        on_compensation: Callable[[str, "CompensationResult"], None] | None = None,
    ) -> None:
        """Initialize the saga runner.

        Args:
            event_store: Store for saga events (default: in-memory).
            compensation_policy: Default compensation policy.
            default_timeout: Default timeout for steps.
            max_parallel_steps: Maximum parallel step executions.
            on_step_complete: Callback when step completes.
            on_step_failed: Callback when step fails.
            on_compensation: Callback when compensation completes.
        """
        self._event_store = event_store or InMemorySagaEventStore()
        self._compensation_policy = compensation_policy
        self._default_timeout = default_timeout
        self._max_parallel_steps = max_parallel_steps
        self._on_step_complete = on_step_complete
        self._on_step_failed = on_step_failed
        self._on_compensation = on_compensation
        self._compensation_planner = CompensationPlanner(compensation_policy)

    def execute(
        self,
        saga: SagaDefinition,
        context: SagaExecutionContext,
    ) -> SagaExecutionResult:
        """Execute a saga synchronously.

        Args:
            saga: Saga definition to execute.
            context: Execution context.

        Returns:
            Execution result.
        """
        # Initialize state machine
        machine = SagaStateMachine(saga.saga_id)
        metrics = SagaMetrics(saga_id=saga.saga_id)

        # Register event store callback
        machine.on_state_change(
            lambda old, new, evt: self._event_store.append(evt)
        )

        try:
            # Start saga
            machine.start(data={"saga_name": saga.name})

            # Execute steps
            execution_order = saga.get_execution_order()
            failed_step = None
            failed_error = None

            for step in execution_order:
                # Check if step should be executed
                if not step.can_execute(
                    context.step_results,
                    context.checkpoint_result,
                    context.transaction_context,
                ):
                    logger.debug(f"Skipping step {step.step_id}: condition not met")
                    continue

                # Check global timeout
                if saga.global_timeout:
                    elapsed = datetime.now() - metrics.started_at
                    if elapsed > saga.global_timeout:
                        machine.timeout()
                        failed_step = step.step_id
                        failed_error = "Saga timeout"
                        break

                # Execute step
                step_start = time.time()
                try:
                    machine.step_started(step.step_id)
                    result = self._execute_step(step, context, metrics)
                    context.step_results[step.step_id] = result

                    step_duration = (time.time() - step_start) * 1000
                    metrics.record_step_duration(step.step_id, step_duration)

                    if result.success:
                        machine.step_completed(
                            step.step_id,
                            data={"duration_ms": step_duration},
                        )
                        if self._on_step_complete:
                            self._on_step_complete(step.step_id, result)
                    else:
                        if step.required:
                            machine.step_failed(step.step_id, result.error or "Step failed")
                            failed_step = step.step_id
                            failed_error = result.error
                            if self._on_step_failed:
                                self._on_step_failed(
                                    step.step_id,
                                    Exception(result.error),
                                )
                            break
                        else:
                            # Optional step - continue
                            machine.step_completed(
                                step.step_id,
                                data={"optional_failed": True},
                            )

                except Exception as e:
                    step_duration = (time.time() - step_start) * 1000
                    metrics.record_step_duration(step.step_id, step_duration)
                    metrics.record_error()

                    if step.required:
                        machine.step_failed(step.step_id, str(e))
                        failed_step = step.step_id
                        failed_error = str(e)
                        if self._on_step_failed:
                            self._on_step_failed(step.step_id, e)
                        break

            # Complete or compensate
            if failed_step is None:
                machine.complete()
                metrics.finalize(success=True)
            else:
                # Run compensation
                compensation_result = self._run_compensation(
                    saga, machine, context, metrics
                )
                context.compensation_results.update(compensation_result)
                metrics.finalize(success=False)

            return SagaExecutionResult(
                saga_id=saga.saga_id,
                state=machine.state,
                success=machine.state.is_success,
                completed_steps=machine.get_completed_steps(),
                compensated_steps=machine.get_compensated_steps(),
                failed_step=failed_step,
                error=failed_error,
                step_results=context.step_results,
                compensation_results=context.compensation_results,
                metrics=metrics,
                events=machine.events,
            )

        except InvalidTransitionError as e:
            logger.error(f"Invalid state transition: {e}")
            metrics.finalize(success=False)
            return SagaExecutionResult(
                saga_id=saga.saga_id,
                state=machine.state,
                success=False,
                error=str(e),
                metrics=metrics,
                events=machine.events,
            )

    def _execute_step(
        self,
        step: SagaStepDefinition,
        context: SagaExecutionContext,
        metrics: SagaMetrics,
    ) -> "ActionResult":
        """Execute a single step with retry logic.

        Args:
            step: Step definition.
            context: Execution context.
            metrics: Metrics to update.

        Returns:
            Action result.
        """
        from truthound.checkpoint.actions.base import ActionResult, ActionStatus

        if step.action is None:
            return ActionResult(
                action_name=step.name,
                action_type="saga_step",
                status=ActionStatus.FAILURE,
                error="No action defined for step",
            )

        retry_config = step.retry_config
        timeout_config = step.timeout_config
        last_exception: Exception | None = None

        for attempt in range(retry_config.max_attempts + 1):
            try:
                # Execute with timeout
                timeout_seconds = timeout_config.execution_timeout.total_seconds()

                # Use threading for timeout
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        step.action.execute,
                        context.checkpoint_result,
                    )
                    try:
                        result = future.result(timeout=timeout_seconds)
                        if result.success:
                            return result
                        else:
                            # Check if should retry
                            if attempt < retry_config.max_attempts:
                                metrics.record_retry(step.step_id)
                                delay = retry_config.calculate_delay(attempt)
                                time.sleep(delay.total_seconds())
                                continue
                            return result
                    except concurrent.futures.TimeoutError:
                        if timeout_config.on_timeout == "skip":
                            return ActionResult(
                                action_name=step.name,
                                action_type="saga_step",
                                status=ActionStatus.SKIPPED,
                                message="Step skipped due to timeout",
                            )
                        raise TimeoutError(
                            f"Step {step.step_id} timed out after {timeout_seconds}s"
                        )

            except Exception as e:
                last_exception = e
                metrics.record_error()

                if not retry_config.should_retry(e, attempt):
                    break

                metrics.record_retry(step.step_id)
                delay = retry_config.calculate_delay(attempt)
                logger.info(
                    f"Step {step.step_id} failed (attempt {attempt + 1}), "
                    f"retrying in {delay.total_seconds():.1f}s"
                )
                time.sleep(delay.total_seconds())

        return ActionResult(
            action_name=step.name,
            action_type="saga_step",
            status=ActionStatus.ERROR,
            error=str(last_exception) if last_exception else "Unknown error",
        )

    def _run_compensation(
        self,
        saga: SagaDefinition,
        machine: SagaStateMachine,
        context: SagaExecutionContext,
        metrics: SagaMetrics,
    ) -> dict[str, "CompensationResult"]:
        """Run compensation for failed saga.

        Args:
            saga: Saga definition.
            machine: State machine.
            context: Execution context.
            metrics: Metrics to update.

        Returns:
            Compensation results by step ID.
        """
        from truthound.checkpoint.transaction.base import CompensationResult

        results: dict[str, CompensationResult] = {}
        completed_steps = machine.get_completed_steps()

        # Create compensation plan
        plan = self._compensation_planner.create_plan(
            saga,
            completed_steps=completed_steps,
            failed_step=machine.current_step,
            policy=CompensationPolicy(saga.compensation_policy),
        )

        if not plan.steps:
            logger.info(f"No compensation needed for saga {saga.saga_id}")
            return results

        # Start compensation
        machine.start_compensation()

        # Execute compensation steps
        for plan_step in plan.get_execution_order():
            step_def = saga.get_step(plan_step.step_id)
            if step_def is None:
                continue

            comp_start = time.time()
            machine.step_compensating(plan_step.step_id)

            try:
                action = step_def.compensation_action
                comp_fn = step_def.compensation_fn

                if action:
                    action.execute(context.checkpoint_result)
                    comp_result = CompensationResult(
                        action_name=plan_step.step_name,
                        success=True,
                    )
                elif comp_fn:
                    result = comp_fn(
                        context.checkpoint_result,
                        context.step_results.get(plan_step.step_id),
                        context.transaction_context,
                    )
                    if isinstance(result, bool):
                        comp_result = CompensationResult(
                            action_name=plan_step.step_name,
                            success=result,
                        )
                    else:
                        comp_result = result
                else:
                    comp_result = CompensationResult(
                        action_name=plan_step.step_name,
                        success=False,
                        error="No compensation defined",
                    )

                comp_duration = (time.time() - comp_start) * 1000
                metrics.record_compensation_duration(plan_step.step_id, comp_duration)

                if comp_result.success:
                    machine.step_compensated(plan_step.step_id)
                    if self._on_compensation:
                        self._on_compensation(plan_step.step_id, comp_result)
                else:
                    machine.compensation_failed(
                        plan_step.step_id,
                        comp_result.error or "Compensation failed",
                    )
                    if plan.policy != CompensationPolicy.BEST_EFFORT:
                        results[plan_step.step_id] = comp_result
                        break

                results[plan_step.step_id] = comp_result

            except Exception as e:
                comp_duration = (time.time() - comp_start) * 1000
                metrics.record_compensation_duration(plan_step.step_id, comp_duration)
                metrics.record_error()

                comp_result = CompensationResult(
                    action_name=plan_step.step_name,
                    success=False,
                    error=str(e),
                )
                results[plan_step.step_id] = comp_result

                machine.compensation_failed(plan_step.step_id, str(e))

                if plan.policy != CompensationPolicy.BEST_EFFORT:
                    break

        # Finalize compensation state
        if all(r.success for r in results.values()):
            machine.compensation_complete()
        elif not any(r.success for r in results.values()):
            machine.fail("All compensations failed")

        return results

    def resume(
        self,
        saga_id: str,
        saga: SagaDefinition,
        context: SagaExecutionContext,
    ) -> SagaExecutionResult:
        """Resume a suspended or failed saga.

        Args:
            saga_id: Saga identifier.
            saga: Saga definition.
            context: Execution context.

        Returns:
            Execution result.
        """
        # Replay state from event store
        machine = self._event_store.replay_from_snapshot(saga_id)

        if machine.state.is_terminal:
            logger.warning(f"Saga {saga_id} is already in terminal state: {machine.state}")
            return SagaExecutionResult(
                saga_id=saga_id,
                state=machine.state,
                success=machine.state.is_success,
                completed_steps=machine.get_completed_steps(),
                compensated_steps=machine.get_compensated_steps(),
                events=machine.events,
                metrics=SagaMetrics(saga_id=saga_id),
            )

        # Resume from current state
        if machine.state == SagaState.SUSPENDED:
            machine.resume()

        # Re-execute using existing runner logic
        # Find where we left off and continue
        completed_steps = set(machine.get_completed_steps())
        execution_order = saga.get_execution_order()
        remaining_steps = [s for s in execution_order if s.step_id not in completed_steps]

        metrics = SagaMetrics(saga_id=saga_id)
        failed_step = None
        failed_error = None

        for step in remaining_steps:
            if not step.can_execute(
                context.step_results,
                context.checkpoint_result,
                context.transaction_context,
            ):
                continue

            step_start = time.time()
            try:
                machine.step_started(step.step_id)
                result = self._execute_step(step, context, metrics)
                context.step_results[step.step_id] = result

                step_duration = (time.time() - step_start) * 1000
                metrics.record_step_duration(step.step_id, step_duration)

                if result.success:
                    machine.step_completed(step.step_id)
                else:
                    if step.required:
                        machine.step_failed(step.step_id, result.error or "Step failed")
                        failed_step = step.step_id
                        failed_error = result.error
                        break
                    else:
                        machine.step_completed(step.step_id)

            except Exception as e:
                step_duration = (time.time() - step_start) * 1000
                metrics.record_step_duration(step.step_id, step_duration)

                if step.required:
                    machine.step_failed(step.step_id, str(e))
                    failed_step = step.step_id
                    failed_error = str(e)
                    break

        if failed_step is None:
            machine.complete()
            metrics.finalize(success=True)
        else:
            self._run_compensation(saga, machine, context, metrics)
            metrics.finalize(success=False)

        return SagaExecutionResult(
            saga_id=saga_id,
            state=machine.state,
            success=machine.state.is_success,
            completed_steps=machine.get_completed_steps(),
            compensated_steps=machine.get_compensated_steps(),
            failed_step=failed_step,
            error=failed_error,
            step_results=context.step_results,
            compensation_results=context.compensation_results,
            metrics=metrics,
            events=machine.events,
        )

    def suspend(self, saga_id: str, reason: str = "") -> bool:
        """Suspend a running saga.

        Args:
            saga_id: Saga identifier.
            reason: Suspension reason.

        Returns:
            True if saga was suspended.
        """
        machine = self._event_store.replay(saga_id)

        if machine.state.is_terminal:
            return False

        try:
            machine.suspend(reason=reason)
            return True
        except InvalidTransitionError:
            return False

    def abort(self, saga_id: str, reason: str = "") -> bool:
        """Abort a running saga.

        Args:
            saga_id: Saga identifier.
            reason: Abort reason.

        Returns:
            True if saga was aborted.
        """
        machine = self._event_store.replay(saga_id)

        if machine.state.is_terminal:
            return False

        try:
            machine.abort(reason=reason)
            return True
        except InvalidTransitionError:
            return False
