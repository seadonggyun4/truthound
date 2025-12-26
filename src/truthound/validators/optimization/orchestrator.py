"""DAG-based Validator Orchestration System.

This module provides a dependency-aware execution framework for validators
with support for parallel execution, caching, and multiple execution strategies.

Key Features:
    - Dependency-based topological ordering
    - Parallel execution of independent validators
    - Result caching for dependent validators
    - Multiple execution strategies (Sequential, Parallel, Adaptive)
    - Execution metrics and profiling

Usage:
    from truthound.validators.optimization.orchestrator import (
        ValidatorDAG,
        ExecutionPlan,
        ParallelExecutionStrategy,
    )

    # Build DAG from validators
    dag = ValidatorDAG()
    dag.add_validators(validators)

    # Create execution plan
    plan = dag.build_execution_plan()

    # Execute with parallel strategy
    strategy = ParallelExecutionStrategy(max_workers=4)
    results = plan.execute(lf, strategy)

Architecture:
    ValidatorDAG
        │
        ├── ValidatorNode (wraps Validator with metadata)
        │   ├── dependencies: set[str]
        │   ├── provides: set[str]
        │   └── priority: int
        │
        ├── build_execution_plan() -> ExecutionPlan
        │   └── Topological sort into execution levels
        │
        └── ExecutionPlan
            ├── levels: list[ExecutionLevel]
            │   └── validators in same level can run in parallel
            │
            └── execute(lf, strategy) -> ExecutionResult
                ├── SequentialExecutionStrategy
                ├── ParallelExecutionStrategy
                └── AdaptiveExecutionStrategy
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeVar, Generic, Iterator

import polars as pl

from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorExecutionResult,
    ValidationResult,
    ErrorContext,
    _validate_safe,
)
from truthound.validators.optimization.graph import TopologicalSort


logger = logging.getLogger("truthound.orchestrator")


# ============================================================================
# Validator Categories for Dependency Resolution
# ============================================================================

class ValidatorPhase(Enum):
    """Execution phases for validators.

    Validators in earlier phases must complete before later phases begin.
    Within a phase, validators can run in parallel if they don't have
    explicit dependencies.
    """
    SCHEMA = auto()       # Schema validation (column existence, types)
    COMPLETENESS = auto() # Null checks, missing values
    UNIQUENESS = auto()   # Duplicate detection, key validation
    FORMAT = auto()       # Pattern matching, format validation
    RANGE = auto()        # Value range, distribution checks
    STATISTICAL = auto()  # Aggregate statistics, outliers
    CROSS_TABLE = auto()  # Multi-table validation
    CUSTOM = auto()       # User-defined validators


# Default phase mapping for built-in validator categories
CATEGORY_TO_PHASE: dict[str, ValidatorPhase] = {
    "schema": ValidatorPhase.SCHEMA,
    "completeness": ValidatorPhase.COMPLETENESS,
    "uniqueness": ValidatorPhase.UNIQUENESS,
    "string": ValidatorPhase.FORMAT,
    "datetime": ValidatorPhase.FORMAT,
    "distribution": ValidatorPhase.RANGE,
    "aggregate": ValidatorPhase.STATISTICAL,
    "anomaly": ValidatorPhase.STATISTICAL,
    "cross_table": ValidatorPhase.CROSS_TABLE,
    "referential": ValidatorPhase.CROSS_TABLE,
    "general": ValidatorPhase.CUSTOM,
}


# ============================================================================
# Validator Node (Wrapper with Dependency Metadata)
# ============================================================================

@dataclass
class ValidatorNode:
    """Wrapper for Validator with dependency and execution metadata.

    Attributes:
        validator: The actual Validator instance
        node_id: Unique identifier (defaults to validator.name)
        dependencies: Set of node_ids this validator depends on
        provides: Set of capabilities this validator provides
        phase: Execution phase for ordering
        priority: Priority within phase (lower = earlier)
        estimated_cost: Estimated execution cost (for adaptive scheduling)
    """
    validator: Validator
    node_id: str = ""
    dependencies: set[str] = field(default_factory=set)
    provides: set[str] = field(default_factory=set)
    phase: ValidatorPhase = ValidatorPhase.CUSTOM
    priority: int = 100
    estimated_cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = self.validator.name

        # Auto-detect phase from category
        category = getattr(self.validator, "category", "general")
        if self.phase == ValidatorPhase.CUSTOM and category in CATEGORY_TO_PHASE:
            self.phase = CATEGORY_TO_PHASE[category]

        # Auto-populate provides if not set
        if not self.provides:
            self.provides = {self.node_id}

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ValidatorNode):
            return self.node_id == other.node_id
        return False


# ============================================================================
# Execution Level (Group of Parallel-Safe Validators)
# ============================================================================

@dataclass
class ExecutionLevel:
    """A group of validators that can execute in parallel.

    All validators in a level have no dependencies on each other,
    only on validators in previous levels.
    """
    level_index: int
    nodes: list[ValidatorNode]
    phase: ValidatorPhase

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    def __iter__(self) -> Iterator[ValidatorNode]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


# ============================================================================
# Execution Result
# ============================================================================

@dataclass
class NodeExecutionResult:
    """Result of executing a single validator node."""
    node_id: str
    result: ValidatorExecutionResult
    start_time: float
    end_time: float

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def issues(self) -> list[ValidationIssue]:
        return self.result.issues

    @property
    def status(self) -> ValidationResult:
        return self.result.status


@dataclass
class LevelExecutionResult:
    """Result of executing an entire level."""
    level_index: int
    node_results: list[NodeExecutionResult]
    start_time: float
    end_time: float

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def all_issues(self) -> list[ValidationIssue]:
        issues = []
        for node_result in self.node_results:
            issues.extend(node_result.issues)
        return issues

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.node_results if r.status == ValidationResult.SUCCESS)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.node_results if r.status == ValidationResult.FAILED)


@dataclass
class ExecutionResult:
    """Complete result of executing the entire DAG."""
    level_results: list[LevelExecutionResult]
    total_start_time: float
    total_end_time: float
    strategy_name: str

    @property
    def total_duration_ms(self) -> float:
        return (self.total_end_time - self.total_start_time) * 1000

    @property
    def all_issues(self) -> list[ValidationIssue]:
        issues = []
        for level_result in self.level_results:
            issues.extend(level_result.all_issues)
        return issues

    @property
    def node_results(self) -> list[NodeExecutionResult]:
        results = []
        for level_result in self.level_results:
            results.extend(level_result.node_results)
        return results

    @property
    def total_validators(self) -> int:
        return sum(len(lr.node_results) for lr in self.level_results)

    @property
    def success_count(self) -> int:
        return sum(lr.success_count for lr in self.level_results)

    @property
    def failure_count(self) -> int:
        return sum(lr.failure_count for lr in self.level_results)

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics summary."""
        return {
            "total_duration_ms": self.total_duration_ms,
            "total_validators": self.total_validators,
            "total_issues": len(self.all_issues),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "levels": len(self.level_results),
            "strategy": self.strategy_name,
            "parallelism_factor": self._compute_parallelism_factor(),
        }

    def _compute_parallelism_factor(self) -> float:
        """Compute how much parallelism was achieved."""
        if not self.node_results:
            return 1.0

        sequential_time = sum(r.duration_ms for r in self.node_results)
        if sequential_time == 0:
            return 1.0

        return sequential_time / self.total_duration_ms


# ============================================================================
# Execution Strategies
# ============================================================================

class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""

    name: str = "base"

    @abstractmethod
    def execute_level(
        self,
        level: ExecutionLevel,
        lf: pl.LazyFrame,
        context: ExecutionContext,
    ) -> LevelExecutionResult:
        """Execute all validators in a level."""
        pass


@dataclass
class ExecutionContext:
    """Shared context for execution."""
    previous_results: dict[str, NodeExecutionResult] = field(default_factory=dict)
    cached_data: dict[str, Any] = field(default_factory=dict)
    skip_on_error: bool = True
    log_errors: bool = True

    def get_result(self, node_id: str) -> NodeExecutionResult | None:
        return self.previous_results.get(node_id)

    def add_result(self, result: NodeExecutionResult) -> None:
        self.previous_results[result.node_id] = result


class SequentialExecutionStrategy(ExecutionStrategy):
    """Execute validators one at a time.

    Simplest strategy, useful for debugging and low-resource environments.
    """

    name = "sequential"

    def execute_level(
        self,
        level: ExecutionLevel,
        lf: pl.LazyFrame,
        context: ExecutionContext,
    ) -> LevelExecutionResult:
        level_start = time.time()
        node_results: list[NodeExecutionResult] = []

        for node in level:
            start = time.time()
            result = _validate_safe(
                node.validator,
                lf,
                skip_on_error=context.skip_on_error,
                log_errors=context.log_errors,
            )
            end = time.time()

            node_result = NodeExecutionResult(
                node_id=node.node_id,
                result=result,
                start_time=start,
                end_time=end,
            )
            node_results.append(node_result)
            context.add_result(node_result)

        return LevelExecutionResult(
            level_index=level.level_index,
            node_results=node_results,
            start_time=level_start,
            end_time=time.time(),
        )


class ParallelExecutionStrategy(ExecutionStrategy):
    """Execute validators in parallel using ThreadPoolExecutor.

    Best for I/O-bound validators or when using Polars' streaming mode.
    """

    name = "parallel"

    def __init__(self, max_workers: int | None = None):
        """Initialize parallel strategy.

        Args:
            max_workers: Maximum number of worker threads.
                        None = min(32, cpu_count + 4)
        """
        self.max_workers = max_workers

    def execute_level(
        self,
        level: ExecutionLevel,
        lf: pl.LazyFrame,
        context: ExecutionContext,
    ) -> LevelExecutionResult:
        level_start = time.time()
        node_results: list[NodeExecutionResult] = []

        # For single validator, no need for thread pool
        if len(level) <= 1:
            for node in level:
                start = time.time()
                result = _validate_safe(
                    node.validator,
                    lf,
                    skip_on_error=context.skip_on_error,
                    log_errors=context.log_errors,
                )
                end = time.time()

                node_result = NodeExecutionResult(
                    node_id=node.node_id,
                    result=result,
                    start_time=start,
                    end_time=end,
                )
                node_results.append(node_result)
                context.add_result(node_result)
        else:
            # Execute in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_node = {}

                for node in level:
                    future = executor.submit(
                        self._execute_node,
                        node,
                        lf,
                        context.skip_on_error,
                        context.log_errors,
                    )
                    future_to_node[future] = node

                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    try:
                        node_result = future.result()
                        node_results.append(node_result)
                        context.add_result(node_result)
                    except Exception as e:
                        logger.error(f"Error executing {node.node_id}: {e}")
                        node_result = NodeExecutionResult(
                            node_id=node.node_id,
                            result=ValidatorExecutionResult(
                                validator_name=node.validator.name,
                                status=ValidationResult.FAILED,
                                issues=[],
                                error_message=str(e),
                                error_context=ErrorContext(type(e).__name__, str(e)),
                            ),
                            start_time=time.time(),
                            end_time=time.time(),
                        )
                        node_results.append(node_result)
                        context.add_result(node_result)

        return LevelExecutionResult(
            level_index=level.level_index,
            node_results=node_results,
            start_time=level_start,
            end_time=time.time(),
        )

    def _execute_node(
        self,
        node: ValidatorNode,
        lf: pl.LazyFrame,
        skip_on_error: bool,
        log_errors: bool,
    ) -> NodeExecutionResult:
        """Execute a single node (for thread pool)."""
        start = time.time()
        result = _validate_safe(
            node.validator,
            lf,
            skip_on_error=skip_on_error,
            log_errors=log_errors,
        )
        end = time.time()

        return NodeExecutionResult(
            node_id=node.node_id,
            result=result,
            start_time=start,
            end_time=end,
        )


class AdaptiveExecutionStrategy(ExecutionStrategy):
    """Dynamically choose between sequential and parallel execution.

    Uses heuristics based on:
    - Number of validators in level
    - Estimated cost of validators
    - System resource availability
    """

    name = "adaptive"

    def __init__(
        self,
        parallel_threshold: int = 3,
        max_workers: int | None = None,
    ):
        """Initialize adaptive strategy.

        Args:
            parallel_threshold: Minimum validators in level to use parallel
            max_workers: Maximum workers for parallel execution
        """
        self.parallel_threshold = parallel_threshold
        self.sequential = SequentialExecutionStrategy()
        self.parallel = ParallelExecutionStrategy(max_workers=max_workers)

    def execute_level(
        self,
        level: ExecutionLevel,
        lf: pl.LazyFrame,
        context: ExecutionContext,
    ) -> LevelExecutionResult:
        if len(level) >= self.parallel_threshold:
            return self.parallel.execute_level(level, lf, context)
        else:
            return self.sequential.execute_level(level, lf, context)


# ============================================================================
# Execution Plan
# ============================================================================

@dataclass
class ExecutionPlan:
    """Executable plan for running validators in dependency order.

    The plan consists of levels, where each level contains validators
    that can run in parallel. Levels are executed sequentially.
    """
    levels: list[ExecutionLevel]
    total_nodes: int
    has_cycles: bool = False
    cycle_info: str | None = None

    def execute(
        self,
        lf: pl.LazyFrame,
        strategy: ExecutionStrategy | None = None,
        skip_on_error: bool = True,
        log_errors: bool = True,
    ) -> ExecutionResult:
        """Execute the plan.

        Args:
            lf: LazyFrame to validate
            strategy: Execution strategy (default: AdaptiveExecutionStrategy)
            skip_on_error: Continue on validator errors
            log_errors: Log validation errors

        Returns:
            ExecutionResult with all validation results
        """
        if strategy is None:
            strategy = AdaptiveExecutionStrategy()

        total_start = time.time()
        context = ExecutionContext(
            skip_on_error=skip_on_error,
            log_errors=log_errors,
        )
        level_results: list[LevelExecutionResult] = []

        for level in self.levels:
            level_result = strategy.execute_level(level, lf, context)
            level_results.append(level_result)

        return ExecutionResult(
            level_results=level_results,
            total_start_time=total_start,
            total_end_time=time.time(),
            strategy_name=strategy.name,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get plan summary."""
        return {
            "total_nodes": self.total_nodes,
            "total_levels": len(self.levels),
            "has_cycles": self.has_cycles,
            "levels": [
                {
                    "index": level.level_index,
                    "size": level.size,
                    "phase": level.phase.name,
                    "nodes": level.node_ids,
                }
                for level in self.levels
            ],
        }

    def __repr__(self) -> str:
        return (
            f"ExecutionPlan(nodes={self.total_nodes}, "
            f"levels={len(self.levels)}, has_cycles={self.has_cycles})"
        )


# ============================================================================
# Validator DAG
# ============================================================================

class ValidatorDAG:
    """Directed Acyclic Graph for validator dependency management.

    Builds an execution plan from a set of validators based on:
    1. Explicit dependencies (validator.dependencies)
    2. Phase ordering (schema -> completeness -> uniqueness -> ...)
    3. Priority within phase

    Example:
        dag = ValidatorDAG()

        # Add validators with automatic dependency detection
        dag.add_validator(NullValidator())
        dag.add_validator(DuplicateValidator())

        # Add with explicit dependencies
        dag.add_validator(
            RangeValidator(),
            dependencies={"null"},  # Must run after NullValidator
        )

        # Build and execute plan
        plan = dag.build_execution_plan()
        result = plan.execute(lf)
    """

    def __init__(self):
        self.nodes: dict[str, ValidatorNode] = {}
        self._dependency_graph: dict[str, set[str]] = {}

    def add_validator(
        self,
        validator: Validator,
        dependencies: set[str] | None = None,
        provides: set[str] | None = None,
        phase: ValidatorPhase | None = None,
        priority: int = 100,
        estimated_cost: float = 1.0,
    ) -> ValidatorNode:
        """Add a validator to the DAG.

        Args:
            validator: Validator instance
            dependencies: Set of node_ids this depends on
            provides: Set of capabilities this provides
            phase: Execution phase override
            priority: Priority within phase (lower = earlier)
            estimated_cost: Estimated execution cost

        Returns:
            The created ValidatorNode
        """
        node_id = validator.name

        # Check for explicit dependencies on validator class
        if dependencies is None:
            dependencies = getattr(validator, "dependencies", set())
            if dependencies is None:
                dependencies = set()

        # Auto-detect phase from category
        if phase is None:
            category = getattr(validator, "category", "general")
            phase = CATEGORY_TO_PHASE.get(category, ValidatorPhase.CUSTOM)

        node = ValidatorNode(
            validator=validator,
            node_id=node_id,
            dependencies=set(dependencies),
            provides=provides or {node_id},
            phase=phase,
            priority=priority,
            estimated_cost=estimated_cost,
        )

        self.nodes[node_id] = node
        return node

    def add_validators(
        self,
        validators: list[Validator],
    ) -> list[ValidatorNode]:
        """Add multiple validators.

        Args:
            validators: List of Validator instances

        Returns:
            List of created ValidatorNodes
        """
        return [self.add_validator(v) for v in validators]

    def add_dependency(self, from_id: str, to_id: str) -> None:
        """Add a dependency edge.

        Args:
            from_id: Node that depends
            to_id: Node that is depended upon
        """
        if from_id in self.nodes:
            self.nodes[from_id].dependencies.add(to_id)

    def build_execution_plan(self) -> ExecutionPlan:
        """Build an execution plan from the DAG.

        Returns:
            ExecutionPlan with validators organized into levels
        """
        if not self.nodes:
            return ExecutionPlan(levels=[], total_nodes=0)

        # Build full dependency graph including phase dependencies
        adjacency = self._build_adjacency_with_phases()

        # Check for cycles
        try:
            sorter = TopologicalSort(adjacency)
            sorted_ids = sorter.sort()
        except ValueError as e:
            logger.warning(f"Cycle detected in validator dependencies: {e}")
            # Fallback to phase-only ordering
            sorted_ids = self._sort_by_phase_only()
            return ExecutionPlan(
                levels=self._group_into_levels(sorted_ids),
                total_nodes=len(self.nodes),
                has_cycles=True,
                cycle_info=str(e),
            )

        # Group into execution levels
        levels = self._group_into_levels(sorted_ids)

        return ExecutionPlan(
            levels=levels,
            total_nodes=len(self.nodes),
        )

    def _build_adjacency_with_phases(self) -> dict[str, list[str]]:
        """Build adjacency list including implicit phase dependencies."""
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}

        # Add explicit dependencies
        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in self.nodes:
                    adjacency[dep].append(node_id)

        # Add implicit phase dependencies
        # Validators in later phases depend on validators in earlier phases
        phase_to_nodes: dict[ValidatorPhase, list[str]] = {}
        for node_id, node in self.nodes.items():
            if node.phase not in phase_to_nodes:
                phase_to_nodes[node.phase] = []
            phase_to_nodes[node.phase].append(node_id)

        # Sort phases by value
        sorted_phases = sorted(phase_to_nodes.keys(), key=lambda p: p.value)

        # Add edges from each phase to the next
        for i in range(len(sorted_phases) - 1):
            current_phase = sorted_phases[i]
            next_phase = sorted_phases[i + 1]

            # Each node in next phase depends on all nodes in current phase
            for current_node in phase_to_nodes[current_phase]:
                for next_node in phase_to_nodes[next_phase]:
                    if next_node not in adjacency[current_node]:
                        adjacency[current_node].append(next_node)

        return adjacency

    def _sort_by_phase_only(self) -> list[str]:
        """Fallback sort using only phases (ignores explicit dependencies)."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: (n.phase.value, n.priority, n.node_id),
        )
        return [n.node_id for n in sorted_nodes]

    def _group_into_levels(self, sorted_ids: list[str]) -> list[ExecutionLevel]:
        """Group sorted node IDs into execution levels.

        Nodes with no dependencies on each other can be in the same level.
        """
        if not sorted_ids:
            return []

        levels: list[ExecutionLevel] = []
        assigned: set[str] = set()
        remaining = list(sorted_ids)

        while remaining:
            # Find all nodes whose dependencies are already assigned
            current_level_nodes: list[ValidatorNode] = []
            current_phase = None

            for node_id in remaining:
                node = self.nodes[node_id]
                deps_satisfied = all(
                    dep in assigned or dep not in self.nodes
                    for dep in node.dependencies
                )

                if deps_satisfied:
                    # Check phase compatibility - only group same phase
                    if current_phase is None:
                        current_phase = node.phase

                    if node.phase == current_phase:
                        current_level_nodes.append(node)

            if not current_level_nodes:
                # Shouldn't happen if graph is acyclic, but handle gracefully
                logger.warning("Could not find nodes for next level")
                # Take the first remaining node
                node_id = remaining[0]
                current_level_nodes = [self.nodes[node_id]]
                current_phase = self.nodes[node_id].phase

            # Sort within level by priority
            current_level_nodes.sort(key=lambda n: (n.priority, n.node_id))

            # Create level
            level = ExecutionLevel(
                level_index=len(levels),
                nodes=current_level_nodes,
                phase=current_phase or ValidatorPhase.CUSTOM,
            )
            levels.append(level)

            # Mark as assigned
            for node in current_level_nodes:
                assigned.add(node.node_id)
                remaining.remove(node.node_id)

        return levels

    def get_dependency_chain(self, node_id: str) -> list[str]:
        """Get the full dependency chain for a node.

        Args:
            node_id: Node to get dependencies for

        Returns:
            List of node_ids in dependency order
        """
        if node_id not in self.nodes:
            return []

        visited: set[str] = set()
        chain: list[str] = []

        def visit(nid: str) -> None:
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)

            for dep in self.nodes[nid].dependencies:
                visit(dep)

            chain.append(nid)

        visit(node_id)
        return chain

    def visualize(self) -> str:
        """Create ASCII visualization of the DAG.

        Returns:
            ASCII art representation of the DAG
        """
        if not self.nodes:
            return "Empty DAG"

        lines = ["ValidatorDAG:"]

        # Group by phase
        phase_to_nodes: dict[ValidatorPhase, list[ValidatorNode]] = {}
        for node in self.nodes.values():
            if node.phase not in phase_to_nodes:
                phase_to_nodes[node.phase] = []
            phase_to_nodes[node.phase].append(node)

        for phase in sorted(phase_to_nodes.keys(), key=lambda p: p.value):
            lines.append(f"\n  [{phase.name}]")
            nodes = sorted(phase_to_nodes[phase], key=lambda n: n.priority)

            for node in nodes:
                deps = ", ".join(sorted(node.dependencies)) if node.dependencies else "none"
                lines.append(f"    - {node.node_id} (deps: {deps})")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ValidatorDAG(nodes={len(self.nodes)})"

    def __len__(self) -> int:
        return len(self.nodes)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_execution_plan(
    validators: list[Validator],
    dependencies: dict[str, set[str]] | None = None,
) -> ExecutionPlan:
    """Create an execution plan from validators.

    Args:
        validators: List of validators
        dependencies: Optional explicit dependencies {validator_name: {dep_names}}

    Returns:
        ExecutionPlan ready for execution
    """
    dag = ValidatorDAG()

    for validator in validators:
        deps = None
        if dependencies and validator.name in dependencies:
            deps = dependencies[validator.name]
        dag.add_validator(validator, dependencies=deps)

    return dag.build_execution_plan()


def execute_validators(
    validators: list[Validator],
    lf: pl.LazyFrame,
    strategy: ExecutionStrategy | None = None,
    dependencies: dict[str, set[str]] | None = None,
) -> ExecutionResult:
    """Execute validators with DAG-based ordering.

    Args:
        validators: List of validators
        lf: LazyFrame to validate
        strategy: Execution strategy (default: AdaptiveExecutionStrategy)
        dependencies: Optional explicit dependencies

    Returns:
        ExecutionResult with all validation results
    """
    plan = create_execution_plan(validators, dependencies)
    return plan.execute(lf, strategy)
