"""Saga Definition Module.

This module provides the core data structures for defining sagas,
including steps, dependencies, and execution constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from truthound.checkpoint.actions.base import BaseAction
    from truthound.checkpoint.transaction.base import (
        CompensationResult,
        TransactionContext,
    )
    from truthound.checkpoint.checkpoint import CheckpointResult


T = TypeVar("T")


class DependencyType(str, Enum):
    """Type of dependency between saga steps."""

    REQUIRES = "requires"  # Must complete before this step
    REQUIRES_SUCCESS = "requires_success"  # Must succeed before this step
    OPTIONAL = "optional"  # Can proceed if dependency fails
    PARALLEL = "parallel"  # Can run in parallel with dependency


class StepExecutionMode(str, Enum):
    """Execution mode for a saga step."""

    SYNC = "sync"  # Synchronous execution
    ASYNC = "async"  # Asynchronous execution
    DEFERRED = "deferred"  # Execution deferred until trigger


class RetryPolicy(str, Enum):
    """Retry policy for failed steps."""

    NONE = "none"  # No retry
    FIXED = "fixed"  # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff


@dataclass
class RetryConfig:
    """Configuration for step retry behavior.

    Attributes:
        policy: Retry policy type.
        max_attempts: Maximum number of retry attempts.
        initial_delay: Initial delay before first retry.
        max_delay: Maximum delay between retries.
        multiplier: Multiplier for exponential/linear backoff.
        jitter: Add random jitter to delays (0.0 - 1.0).
        retry_on: Exception types to retry on.
        abort_on: Exception types to abort on (no retry).
    """

    policy: RetryPolicy = RetryPolicy.EXPONENTIAL
    max_attempts: int = 3
    initial_delay: timedelta = field(default_factory=lambda: timedelta(seconds=1))
    max_delay: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    multiplier: float = 2.0
    jitter: float = 0.1
    retry_on: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))
    abort_on: tuple[type[Exception], ...] = field(default_factory=tuple)

    def calculate_delay(self, attempt: int) -> timedelta:
        """Calculate delay for a given attempt number.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay before next retry.
        """
        import random

        if self.policy == RetryPolicy.NONE:
            return timedelta(0)
        elif self.policy == RetryPolicy.FIXED:
            base_delay = self.initial_delay
        elif self.policy == RetryPolicy.LINEAR:
            base_delay = self.initial_delay * (attempt + 1)
        else:  # EXPONENTIAL
            base_delay = self.initial_delay * (self.multiplier**attempt)

        # Apply max delay cap
        if base_delay > self.max_delay:
            base_delay = self.max_delay

        # Apply jitter
        if self.jitter > 0:
            jitter_range = base_delay.total_seconds() * self.jitter
            jitter = random.uniform(-jitter_range, jitter_range)
            base_delay = timedelta(seconds=base_delay.total_seconds() + jitter)

        return base_delay

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Check if retry should be attempted.

        Args:
            exception: The exception that occurred.
            attempt: Current attempt number.

        Returns:
            True if retry should be attempted.
        """
        if attempt >= self.max_attempts:
            return False

        if self.abort_on and isinstance(exception, self.abort_on):
            return False

        if self.retry_on:
            return isinstance(exception, self.retry_on)

        return True


@dataclass
class TimeoutConfig:
    """Configuration for step timeout behavior.

    Attributes:
        execution_timeout: Maximum execution time for the step.
        compensation_timeout: Maximum time for compensation.
        total_timeout: Total timeout including retries.
        on_timeout: Behavior on timeout.
    """

    execution_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    compensation_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    total_timeout: timedelta | None = None
    on_timeout: str = "fail"  # "fail", "compensate", "skip"


@dataclass
class StepDependency:
    """Represents a dependency on another saga step.

    Attributes:
        step_id: ID of the dependent step.
        dependency_type: Type of dependency.
        condition: Optional condition for the dependency.
    """

    step_id: str
    dependency_type: DependencyType = DependencyType.REQUIRES_SUCCESS
    condition: Callable[["CheckpointResult", "TransactionContext"], bool] | None = None

    def is_satisfied(
        self,
        step_results: dict[str, Any],
        checkpoint_result: "CheckpointResult",
        context: "TransactionContext",
    ) -> bool:
        """Check if this dependency is satisfied.

        Args:
            step_results: Results from executed steps.
            checkpoint_result: Current checkpoint result.
            context: Transaction context.

        Returns:
            True if dependency is satisfied.
        """
        if self.step_id not in step_results:
            return self.dependency_type == DependencyType.OPTIONAL

        result = step_results[self.step_id]

        if self.dependency_type == DependencyType.REQUIRES:
            # Step completed (regardless of success)
            return True
        elif self.dependency_type == DependencyType.REQUIRES_SUCCESS:
            # Step completed successfully
            return result.success if hasattr(result, "success") else bool(result)
        elif self.dependency_type == DependencyType.OPTIONAL:
            # Always satisfied
            return True
        elif self.dependency_type == DependencyType.PARALLEL:
            # Can run in parallel, doesn't need to wait
            return True

        # Check custom condition if provided
        if self.condition:
            return self.condition(checkpoint_result, context)

        return True


@dataclass
class SagaStepDefinition:
    """Definition of a single step in a saga.

    Attributes:
        step_id: Unique identifier for the step.
        name: Human-readable step name.
        action: Action to execute.
        compensation_action: Optional compensation action.
        compensation_fn: Optional compensation function.
        dependencies: Steps this step depends on.
        retry_config: Retry configuration.
        timeout_config: Timeout configuration.
        execution_mode: Sync, async, or deferred.
        condition: Optional execution condition.
        metadata: Additional step metadata.
        is_pivot: Whether this is a pivot transaction.
        is_countermeasure: Whether this is a countermeasure step.
        semantic_undo: Whether compensation is semantic (not exact undo).
    """

    step_id: str = field(default_factory=lambda: f"step_{uuid4().hex[:8]}")
    name: str = ""
    action: "BaseAction[Any]" | None = None
    compensation_action: "BaseAction[Any]" | None = None
    compensation_fn: Callable[..., "CompensationResult | bool"] | None = None
    dependencies: list[StepDependency] = field(default_factory=list)
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    timeout_config: TimeoutConfig = field(default_factory=TimeoutConfig)
    execution_mode: StepExecutionMode = StepExecutionMode.SYNC
    condition: Callable[["CheckpointResult", "TransactionContext"], bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    is_pivot: bool = False
    is_countermeasure: bool = False
    semantic_undo: bool = False
    required: bool = True
    order: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.step_id

    def has_compensation(self) -> bool:
        """Check if step has compensation defined."""
        return self.compensation_action is not None or self.compensation_fn is not None

    def can_execute(
        self,
        step_results: dict[str, Any],
        checkpoint_result: "CheckpointResult",
        context: "TransactionContext",
    ) -> bool:
        """Check if step can be executed.

        Args:
            step_results: Results from executed steps.
            checkpoint_result: Current checkpoint result.
            context: Transaction context.

        Returns:
            True if step can be executed.
        """
        # Check condition
        if self.condition and not self.condition(checkpoint_result, context):
            return False

        # Check all dependencies
        for dep in self.dependencies:
            if not dep.is_satisfied(step_results, checkpoint_result, context):
                return False

        return True

    def get_dependency_ids(self) -> list[str]:
        """Get IDs of all dependencies."""
        return [dep.step_id for dep in self.dependencies]


@dataclass
class SagaDefinition:
    """Complete definition of a saga.

    Attributes:
        saga_id: Unique saga identifier.
        name: Human-readable saga name.
        description: Saga description.
        version: Saga definition version.
        steps: Ordered list of step definitions.
        global_timeout: Overall saga timeout.
        compensation_policy: How to handle compensation.
        metadata: Additional saga metadata.
    """

    saga_id: str = field(default_factory=lambda: f"saga_{uuid4().hex[:12]}")
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    steps: list[SagaStepDefinition] = field(default_factory=list)
    global_timeout: timedelta | None = None
    compensation_policy: str = "backward"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.saga_id

    def add_step(self, step: SagaStepDefinition) -> "SagaDefinition":
        """Add a step to the saga.

        Args:
            step: Step definition to add.

        Returns:
            Self for chaining.
        """
        step.order = len(self.steps)
        self.steps.append(step)
        return self

    def get_step(self, step_id: str) -> SagaStepDefinition | None:
        """Get a step by ID.

        Args:
            step_id: Step ID to find.

        Returns:
            Step definition or None if not found.
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_execution_order(self) -> list[SagaStepDefinition]:
        """Get steps in execution order based on dependencies.

        Uses topological sort to determine execution order.

        Returns:
            Steps in execution order.
        """
        # Build dependency graph: step_id -> set of step_ids it depends on
        dependencies: dict[str, set[str]] = {step.step_id: set() for step in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                if dep.dependency_type != DependencyType.PARALLEL:
                    dependencies[step.step_id].add(dep.step_id)

        # Calculate in-degree (number of dependencies each step has)
        in_degree = {step_id: len(deps) for step_id, deps in dependencies.items()}

        # Start with nodes that have no dependencies (in_degree == 0)
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort by original order for determinism
            queue.sort(key=lambda x: next(
                (s.order for s in self.steps if s.step_id == x), 0
            ))
            current = queue.pop(0)
            step = self.get_step(current)
            if step:
                result.append(step)

            # For each step that depends on 'current', decrement its in-degree
            for step_id, deps in dependencies.items():
                if current in deps:
                    in_degree[step_id] -= 1
                    if in_degree[step_id] == 0:
                        queue.append(step_id)

        # Check for cycles
        if len(result) != len(self.steps):
            missing = set(s.step_id for s in self.steps) - set(s.step_id for s in result)
            raise ValueError(f"Cycle detected in saga dependencies: {missing}")

        return result

    def get_compensation_order(self) -> list[SagaStepDefinition]:
        """Get steps in compensation order (reverse of execution).

        Returns:
            Steps in compensation order.
        """
        execution_order = self.get_execution_order()
        return list(reversed([s for s in execution_order if s.has_compensation()]))

    def get_pivot_step(self) -> SagaStepDefinition | None:
        """Get the pivot transaction step if defined.

        Returns:
            Pivot step or None.
        """
        for step in self.steps:
            if step.is_pivot:
                return step
        return None

    def validate(self) -> list[str]:
        """Validate the saga definition.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Check for duplicate step IDs
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")

        # Check dependencies reference existing steps
        for step in self.steps:
            for dep in step.dependencies:
                if dep.step_id not in step_ids:
                    errors.append(
                        f"Step '{step.step_id}' depends on non-existent step '{dep.step_id}'"
                    )

        # Check for cycles
        try:
            self.get_execution_order()
        except ValueError as e:
            errors.append(str(e))

        # Check pivot transaction rules
        pivot_steps = [s for s in self.steps if s.is_pivot]
        if len(pivot_steps) > 1:
            errors.append("Only one pivot transaction allowed per saga")

        # Pivot should not have compensation
        for step in pivot_steps:
            if step.has_compensation():
                errors.append(
                    f"Pivot step '{step.step_id}' should not have compensation"
                )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.name,
                    "has_compensation": s.has_compensation(),
                    "dependencies": [d.step_id for d in s.dependencies],
                    "is_pivot": s.is_pivot,
                    "is_countermeasure": s.is_countermeasure,
                    "required": s.required,
                    "order": s.order,
                }
                for s in self.steps
            ],
            "compensation_policy": self.compensation_policy,
            "metadata": self.metadata,
        }

    def visualize(self) -> str:
        """Generate ASCII visualization of the saga.

        Returns:
            String visualization.
        """
        lines = [
            f"Saga: {self.name} (v{self.version})",
            "=" * 50,
        ]

        if self.description:
            lines.append(f"Description: {self.description}")
            lines.append("-" * 50)

        execution_order = self.get_execution_order()

        for i, step in enumerate(execution_order):
            is_last = i == len(execution_order) - 1
            prefix = "└─" if is_last else "├─"

            # Build step line
            indicators = []
            if step.has_compensation():
                indicators.append("[C]")
            if step.is_pivot:
                indicators.append("[P]")
            if step.is_countermeasure:
                indicators.append("[M]")
            if not step.required:
                indicators.append("(opt)")

            indicator_str = " ".join(indicators)
            step_line = f"{prefix} {step.name}"
            if indicator_str:
                step_line += f" {indicator_str}"

            lines.append(step_line)

            # Show dependencies
            if step.dependencies:
                dep_prefix = "│  " if not is_last else "   "
                dep_str = ", ".join(d.step_id for d in step.dependencies)
                lines.append(f"{dep_prefix}← depends on: {dep_str}")

        lines.append("")
        lines.append("Legend: [C]=Compensatable, [P]=Pivot, [M]=Countermeasure, (opt)=Optional")

        return "\n".join(lines)
