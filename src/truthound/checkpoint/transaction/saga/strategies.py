"""Advanced Compensation Strategies.

This module provides advanced compensation strategies for enterprise
saga patterns, including semantic compensation, pivot transactions,
and countermeasure strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import ActionResult, BaseAction
    from truthound.checkpoint.checkpoint import CheckpointResult
    from truthound.checkpoint.transaction.base import CompensationResult, TransactionContext
    from truthound.checkpoint.transaction.saga.definition import SagaDefinition, SagaStepDefinition


logger = logging.getLogger(__name__)


class CompensationPolicy(str, Enum):
    """Compensation policy types."""

    # Standard backward compensation (default Saga pattern)
    BACKWARD = "backward"

    # Forward recovery - try to complete remaining steps
    FORWARD = "forward"

    # Semantic compensation - compensate with semantically equivalent actions
    SEMANTIC = "semantic"

    # Pivot transaction - commit point after which no compensation
    PIVOT = "pivot"

    # Countermeasure - corrective actions instead of undo
    COUNTERMEASURE = "countermeasure"

    # Parallel compensation - compensate multiple steps in parallel
    PARALLEL = "parallel"

    # Selective compensation - only compensate specific steps
    SELECTIVE = "selective"

    # Best effort - try to compensate, continue on failure
    BEST_EFFORT = "best_effort"

    def __str__(self) -> str:
        return self.value


class CompensationPriority(str, Enum):
    """Priority for compensation execution."""

    CRITICAL = "critical"  # Must compensate, fail saga if not possible
    HIGH = "high"  # Highly important, retry multiple times
    NORMAL = "normal"  # Standard compensation
    LOW = "low"  # Can skip if necessary
    OPTIONAL = "optional"  # Best effort, ignore failures


@dataclass
class CompensationPlanStep:
    """A single step in a compensation plan.

    Attributes:
        step_id: ID of the step to compensate.
        step_name: Name of the step.
        priority: Compensation priority.
        strategy: Strategy for this step.
        dependencies: Steps that must be compensated first.
        action: Compensation action or function.
        estimated_duration_ms: Estimated compensation duration.
        can_parallel: Whether can run in parallel with others.
    """

    step_id: str
    step_name: str
    priority: CompensationPriority = CompensationPriority.NORMAL
    strategy: str = "backward"
    dependencies: list[str] = field(default_factory=list)
    action: "BaseAction[Any] | Callable[..., Any]" | None = None
    estimated_duration_ms: float = 0.0
    can_parallel: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompensationPlan:
    """Complete compensation plan for a saga.

    Attributes:
        saga_id: ID of the saga.
        policy: Overall compensation policy.
        steps: Ordered list of compensation steps.
        created_at: When the plan was created.
        pivot_step_id: ID of pivot step (if using pivot strategy).
        metadata: Additional plan metadata.
    """

    saga_id: str
    policy: CompensationPolicy
    steps: list[CompensationPlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    pivot_step_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: CompensationPlanStep) -> "CompensationPlan":
        """Add a step to the plan.

        Args:
            step: Compensation step to add.

        Returns:
            Self for chaining.
        """
        self.steps.append(step)
        return self

    def get_execution_order(self) -> list[CompensationPlanStep]:
        """Get steps in execution order based on dependencies.

        Returns:
            Steps in order of execution.
        """
        # Simple topological sort
        result = []
        remaining = list(self.steps)
        completed = set()

        while remaining:
            for step in list(remaining):
                deps_satisfied = all(
                    dep in completed for dep in step.dependencies
                )
                if deps_satisfied:
                    result.append(step)
                    completed.add(step.step_id)
                    remaining.remove(step)
                    break
            else:
                # No progress - might be cycle or missing dep
                result.extend(remaining)
                break

        return result

    def get_parallel_groups(self) -> list[list[CompensationPlanStep]]:
        """Get steps grouped for parallel execution.

        Returns:
            List of step groups that can run in parallel.
        """
        groups = []
        current_group = []
        completed = set()

        for step in self.get_execution_order():
            # Check if all dependencies are complete
            deps_complete = all(d in completed for d in step.dependencies)

            if step.can_parallel and deps_complete:
                current_group.append(step)
            else:
                if current_group:
                    groups.append(current_group)
                    completed.update(s.step_id for s in current_group)
                    current_group = []
                groups.append([step])
                completed.add(step.step_id)

        if current_group:
            groups.append(current_group)

        return groups

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "saga_id": self.saga_id,
            "policy": self.policy.value,
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_name": s.step_name,
                    "priority": s.priority.value,
                    "strategy": s.strategy,
                    "dependencies": s.dependencies,
                    "can_parallel": s.can_parallel,
                }
                for s in self.steps
            ],
            "pivot_step_id": self.pivot_step_id,
            "created_at": self.created_at.isoformat(),
        }


class CompensationPlanner:
    """Generates compensation plans for sagas.

    This class analyzes a saga definition and generates an appropriate
    compensation plan based on the configured policy and step characteristics.
    """

    def __init__(self, policy: CompensationPolicy = CompensationPolicy.BACKWARD) -> None:
        """Initialize the planner.

        Args:
            policy: Default compensation policy.
        """
        self._default_policy = policy

    def create_plan(
        self,
        saga: "SagaDefinition",
        completed_steps: list[str],
        failed_step: str | None = None,
        policy: CompensationPolicy | None = None,
    ) -> CompensationPlan:
        """Create a compensation plan for the saga.

        Args:
            saga: Saga definition.
            completed_steps: List of completed step IDs.
            failed_step: ID of the failed step (if any).
            policy: Override policy for this plan.

        Returns:
            Compensation plan.
        """
        effective_policy = policy or self._default_policy

        plan = CompensationPlan(
            saga_id=saga.saga_id,
            policy=effective_policy,
        )

        # Find pivot step
        pivot_step = saga.get_pivot_step()
        if pivot_step:
            plan.pivot_step_id = pivot_step.step_id

        # Get steps to compensate
        steps_to_compensate = self._determine_compensatable_steps(
            saga, completed_steps, failed_step, effective_policy, pivot_step
        )

        # Build plan based on policy
        if effective_policy == CompensationPolicy.BACKWARD:
            self._build_backward_plan(plan, saga, steps_to_compensate)
        elif effective_policy == CompensationPolicy.FORWARD:
            self._build_forward_plan(plan, saga, completed_steps, failed_step)
        elif effective_policy == CompensationPolicy.SEMANTIC:
            self._build_semantic_plan(plan, saga, steps_to_compensate)
        elif effective_policy == CompensationPolicy.PIVOT:
            self._build_pivot_plan(plan, saga, completed_steps, pivot_step)
        elif effective_policy == CompensationPolicy.COUNTERMEASURE:
            self._build_countermeasure_plan(plan, saga, steps_to_compensate)
        elif effective_policy == CompensationPolicy.PARALLEL:
            self._build_parallel_plan(plan, saga, steps_to_compensate)
        elif effective_policy == CompensationPolicy.SELECTIVE:
            self._build_selective_plan(plan, saga, steps_to_compensate, failed_step)
        elif effective_policy == CompensationPolicy.BEST_EFFORT:
            self._build_best_effort_plan(plan, saga, steps_to_compensate)

        return plan

    def _determine_compensatable_steps(
        self,
        saga: "SagaDefinition",
        completed_steps: list[str],
        failed_step: str | None,
        policy: CompensationPolicy,
        pivot_step: "SagaStepDefinition | None",
    ) -> list["SagaStepDefinition"]:
        """Determine which steps need compensation."""
        compensatable = []

        # Check if we passed the pivot point
        if pivot_step and pivot_step.step_id in completed_steps:
            # Past pivot - no compensation allowed
            logger.info(
                f"Saga passed pivot point at {pivot_step.step_id}, "
                "compensation not possible"
            )
            return []

        for step in saga.steps:
            # Only compensate completed steps
            if step.step_id not in completed_steps:
                continue

            # Skip the failed step itself (wasn't completed)
            if step.step_id == failed_step:
                continue

            # Must have compensation defined
            if not step.has_compensation():
                continue

            compensatable.append(step)

        return compensatable

    def _build_backward_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        steps: list["SagaStepDefinition"],
    ) -> None:
        """Build backward compensation plan (reverse order)."""
        # Sort by execution order, then reverse
        step_order = {s.step_id: i for i, s in enumerate(saga.steps)}
        sorted_steps = sorted(steps, key=lambda s: step_order.get(s.step_id, 0), reverse=True)

        prev_step_id = None
        for step in sorted_steps:
            plan_step = CompensationPlanStep(
                step_id=step.step_id,
                step_name=step.name,
                priority=CompensationPriority.NORMAL,
                strategy="backward",
                dependencies=[prev_step_id] if prev_step_id else [],
                action=step.compensation_action or step.compensation_fn,
            )
            plan.add_step(plan_step)
            prev_step_id = step.step_id

    def _build_forward_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        completed_steps: list[str],
        failed_step: str | None,
    ) -> None:
        """Build forward recovery plan (try to complete)."""
        # Get remaining steps after failure
        execution_order = saga.get_execution_order()
        failed_index = next(
            (i for i, s in enumerate(execution_order) if s.step_id == failed_step),
            len(execution_order),
        )

        for step in execution_order[failed_index:]:
            if step.step_id in completed_steps:
                continue

            # Add as forward recovery step
            plan_step = CompensationPlanStep(
                step_id=step.step_id,
                step_name=step.name,
                priority=CompensationPriority.NORMAL,
                strategy="forward",
                action=step.action,
                metadata={"recovery": True},
            )
            plan.add_step(plan_step)

    def _build_semantic_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        steps: list["SagaStepDefinition"],
    ) -> None:
        """Build semantic compensation plan."""
        for step in reversed(steps):
            priority = (
                CompensationPriority.HIGH if step.required else CompensationPriority.NORMAL
            )
            strategy = "semantic" if step.semantic_undo else "backward"

            plan_step = CompensationPlanStep(
                step_id=step.step_id,
                step_name=step.name,
                priority=priority,
                strategy=strategy,
                action=step.compensation_action or step.compensation_fn,
                metadata={"semantic_undo": step.semantic_undo},
            )
            plan.add_step(plan_step)

    def _build_pivot_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        completed_steps: list[str],
        pivot_step: "SagaStepDefinition | None",
    ) -> None:
        """Build compensation plan with pivot transaction support."""
        if pivot_step and pivot_step.step_id in completed_steps:
            # Past pivot - use forward recovery
            plan.policy = CompensationPolicy.FORWARD
            plan.metadata["pivot_passed"] = True
            logger.info("Pivot passed, switching to forward recovery")
            return

        # Normal backward compensation up to pivot
        compensatable = []
        for step in saga.steps:
            if step.step_id not in completed_steps:
                continue
            if step.is_pivot:
                break
            if step.has_compensation():
                compensatable.append(step)

        self._build_backward_plan(plan, saga, compensatable)

    def _build_countermeasure_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        steps: list["SagaStepDefinition"],
    ) -> None:
        """Build countermeasure compensation plan."""
        for step in saga.steps:
            if not step.is_countermeasure:
                continue

            plan_step = CompensationPlanStep(
                step_id=step.step_id,
                step_name=step.name,
                priority=CompensationPriority.HIGH,
                strategy="countermeasure",
                action=step.action,
                metadata={"is_countermeasure": True},
            )
            plan.add_step(plan_step)

    def _build_parallel_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        steps: list["SagaStepDefinition"],
    ) -> None:
        """Build parallel compensation plan."""
        for step in steps:
            # All steps marked as parallel-capable
            plan_step = CompensationPlanStep(
                step_id=step.step_id,
                step_name=step.name,
                priority=CompensationPriority.NORMAL,
                strategy="parallel",
                action=step.compensation_action or step.compensation_fn,
                can_parallel=True,
                # Dependencies only from step definition
                dependencies=[d.step_id for d in step.dependencies],
            )
            plan.add_step(plan_step)

    def _build_selective_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        steps: list["SagaStepDefinition"],
        failed_step: str | None,
    ) -> None:
        """Build selective compensation plan."""
        # Only compensate steps related to the failed step
        if not failed_step:
            return

        failed_step_def = saga.get_step(failed_step)
        if not failed_step_def:
            return

        # Find dependent steps
        dependent_ids = set()
        for step in saga.steps:
            for dep in step.dependencies:
                if dep.step_id == failed_step:
                    dependent_ids.add(step.step_id)

        # Compensate only related steps
        for step in reversed(steps):
            is_related = (
                step.step_id in dependent_ids
                or any(d.step_id == failed_step for d in step.dependencies)
            )

            if is_related and step.has_compensation():
                plan_step = CompensationPlanStep(
                    step_id=step.step_id,
                    step_name=step.name,
                    priority=CompensationPriority.NORMAL,
                    strategy="selective",
                    action=step.compensation_action or step.compensation_fn,
                )
                plan.add_step(plan_step)

    def _build_best_effort_plan(
        self,
        plan: CompensationPlan,
        saga: "SagaDefinition",
        steps: list["SagaStepDefinition"],
    ) -> None:
        """Build best-effort compensation plan."""
        for step in reversed(steps):
            plan_step = CompensationPlanStep(
                step_id=step.step_id,
                step_name=step.name,
                priority=CompensationPriority.OPTIONAL,
                strategy="best_effort",
                action=step.compensation_action or step.compensation_fn,
                can_parallel=True,  # All can run in parallel
            )
            plan.add_step(plan_step)


# =============================================================================
# Strategy Implementations
# =============================================================================


class CompensationStrategyBase(ABC):
    """Base class for compensation strategy implementations."""

    @abstractmethod
    def execute(
        self,
        plan: CompensationPlan,
        checkpoint_result: "CheckpointResult",
        context: "TransactionContext",
    ) -> list["CompensationResult"]:
        """Execute the compensation plan.

        Args:
            plan: Compensation plan to execute.
            checkpoint_result: Original checkpoint result.
            context: Transaction context.

        Returns:
            List of compensation results.
        """
        pass


class SemanticCompensation(CompensationStrategyBase):
    """Semantic compensation strategy.

    Semantic compensation allows for compensation actions that don't
    strictly undo the original action but provide a semantically
    equivalent outcome.

    Example:
        - Original: Reserve 10 seats
        - Strict undo: Release 10 seats
        - Semantic: Add 10 seats to waiting list
    """

    def __init__(
        self,
        fallback_to_strict: bool = True,
        require_acknowledgment: bool = False,
    ) -> None:
        """Initialize semantic compensation.

        Args:
            fallback_to_strict: Fall back to strict undo if semantic fails.
            require_acknowledgment: Require external acknowledgment.
        """
        self._fallback_to_strict = fallback_to_strict
        self._require_acknowledgment = require_acknowledgment

    def execute(
        self,
        plan: CompensationPlan,
        checkpoint_result: "CheckpointResult",
        context: "TransactionContext",
    ) -> list["CompensationResult"]:
        """Execute semantic compensation."""
        from truthound.checkpoint.transaction.base import CompensationResult

        results = []

        for step in plan.get_execution_order():
            is_semantic = step.metadata.get("semantic_undo", False)

            try:
                if step.action:
                    if callable(step.action):
                        result = step.action(checkpoint_result, None, context)
                    else:
                        result = step.action.execute(checkpoint_result)

                    comp_result = CompensationResult(
                        action_name=step.step_name,
                        success=True,
                        details={"semantic": is_semantic},
                    )
                else:
                    comp_result = CompensationResult(
                        action_name=step.step_name,
                        success=False,
                        error="No compensation action defined",
                    )

                results.append(comp_result)

            except Exception as e:
                if self._fallback_to_strict and is_semantic:
                    logger.info(f"Semantic compensation failed, trying strict: {e}")
                    # Could try strict undo here
                    comp_result = CompensationResult(
                        action_name=step.step_name,
                        success=False,
                        error=f"Semantic compensation failed: {e}",
                    )
                else:
                    comp_result = CompensationResult(
                        action_name=step.step_name,
                        success=False,
                        error=str(e),
                    )
                results.append(comp_result)

        return results


class PivotTransaction(CompensationStrategyBase):
    """Pivot transaction strategy.

    A pivot transaction is a step in a saga that represents a point
    of no return. Once the pivot succeeds, the saga cannot be
    compensated and must proceed forward.

    Example:
        - Step 1: Reserve seats (compensatable)
        - Step 2: Charge card (compensatable)
        - Step 3: Issue ticket (PIVOT - irreversible)
        - Step 4: Send confirmation (must complete)

    If step 4 fails, compensation stops at step 3.
    """

    def __init__(self, pivot_step_id: str) -> None:
        """Initialize pivot transaction.

        Args:
            pivot_step_id: ID of the pivot step.
        """
        self._pivot_step_id = pivot_step_id

    def execute(
        self,
        plan: CompensationPlan,
        checkpoint_result: "CheckpointResult",
        context: "TransactionContext",
    ) -> list["CompensationResult"]:
        """Execute pivot-aware compensation."""
        from truthound.checkpoint.transaction.base import CompensationResult

        results = []

        for step in plan.get_execution_order():
            # Stop at pivot
            if step.step_id == self._pivot_step_id:
                logger.info(f"Reached pivot step {step.step_id}, stopping compensation")
                break

            if step.action:
                try:
                    if callable(step.action):
                        step.action(checkpoint_result, None, context)
                    else:
                        step.action.execute(checkpoint_result)

                    results.append(
                        CompensationResult(
                            action_name=step.step_name,
                            success=True,
                        )
                    )
                except Exception as e:
                    results.append(
                        CompensationResult(
                            action_name=step.step_name,
                            success=False,
                            error=str(e),
                        )
                    )

        return results


class CountermeasureStrategy(CompensationStrategyBase):
    """Countermeasure compensation strategy.

    Instead of undoing completed actions, countermeasures apply
    corrective actions to handle the failure state.

    Example:
        - Original failure: Payment declined after inventory reserved
        - Undo: Release inventory (traditional)
        - Countermeasure: Send payment retry notification,
                          hold inventory for 24 hours
    """

    def __init__(
        self,
        countermeasures: dict[str, Callable[..., Any]] | None = None,
        apply_in_order: bool = True,
    ) -> None:
        """Initialize countermeasure strategy.

        Args:
            countermeasures: Map of step ID to countermeasure function.
            apply_in_order: Apply countermeasures in saga order.
        """
        self._countermeasures = countermeasures or {}
        self._apply_in_order = apply_in_order

    def add_countermeasure(
        self,
        step_id: str,
        action: Callable[..., Any],
    ) -> "CountermeasureStrategy":
        """Add a countermeasure for a step.

        Args:
            step_id: Step to apply countermeasure for.
            action: Countermeasure action.

        Returns:
            Self for chaining.
        """
        self._countermeasures[step_id] = action
        return self

    def execute(
        self,
        plan: CompensationPlan,
        checkpoint_result: "CheckpointResult",
        context: "TransactionContext",
    ) -> list["CompensationResult"]:
        """Execute countermeasure compensation."""
        from truthound.checkpoint.transaction.base import CompensationResult

        results = []

        # Find countermeasure steps
        countermeasure_steps = [
            s for s in plan.steps
            if s.metadata.get("is_countermeasure") or s.step_id in self._countermeasures
        ]

        for step in countermeasure_steps:
            action = self._countermeasures.get(step.step_id, step.action)

            if action:
                try:
                    if callable(action):
                        action(checkpoint_result, None, context)
                    else:
                        action.execute(checkpoint_result)

                    results.append(
                        CompensationResult(
                            action_name=step.step_name,
                            success=True,
                            details={"type": "countermeasure"},
                        )
                    )
                except Exception as e:
                    results.append(
                        CompensationResult(
                            action_name=step.step_name,
                            success=False,
                            error=str(e),
                            details={"type": "countermeasure"},
                        )
                    )

        return results
