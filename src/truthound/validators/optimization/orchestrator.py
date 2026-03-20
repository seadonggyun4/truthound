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

from truthound.types import Severity
from truthound.validators.base import (
    Validator,
    ValidationIssue,
    ValidatorExecutionResult,
    ValidationResult,
    ErrorContext,
    ExceptionInfo,
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

    @property
    def skipped_count(self) -> int:
        return sum(
            1
            for nr in self.node_results
            if nr.status == ValidationResult.SKIPPED
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics summary."""
        return {
            "total_duration_ms": self.total_duration_ms,
            "total_validators": self.total_validators,
            "total_issues": len(self.all_issues),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
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
    """Shared context for DAG execution with result propagation.

    Tracks completed results, critical columns, and skip decisions
    so that downstream Validators can make informed skip/filter choices.
    """
    previous_results: dict[str, NodeExecutionResult] = field(default_factory=dict)
    cached_data: dict[str, Any] = field(default_factory=dict)
    skip_on_error: bool = True
    log_errors: bool = True

    # PHASE 4: result propagation fields
    critical_columns: set[str] = field(default_factory=set)
    failed_validators: set[str] = field(default_factory=set)
    skipped_validators: set[str] = field(default_factory=set)

    def get_result(self, node_id: str) -> NodeExecutionResult | None:
        return self.previous_results.get(node_id)

    def add_result(self, result: NodeExecutionResult) -> None:
        self.previous_results[result.node_id] = result

    def record_result(self, node_id: str, result: ValidatorExecutionResult) -> None:
        """Record a validator result and update propagation state."""
        if result.status in (ValidationResult.FAILED, ValidationResult.TIMEOUT):
            self.failed_validators.add(node_id)
        elif result.status == ValidationResult.SKIPPED:
            self.skipped_validators.add(node_id)

        for issue in result.issues:
            if issue.severity == Severity.CRITICAL:
                self.critical_columns.add(issue.column)

    def get_completed_execution_results(self) -> dict[str, ValidatorExecutionResult]:
        """Return ``{node_id: ValidatorExecutionResult}`` for ``should_skip()``."""
        return {
            nid: nr.result for nid, nr in self.previous_results.items()
        }

    def is_column_critical(self, column: str) -> bool:
        return column in self.critical_columns


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
            v_cfg = getattr(node.validator, "config", None)
            result = _validate_safe(
                node.validator,
                lf,
                skip_on_error=context.skip_on_error,
                log_errors=context.log_errors,
                max_retries=getattr(v_cfg, "max_retries", 0) if v_cfg else 0,
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
                v_cfg = getattr(node.validator, "config", None)
                result = _validate_safe(
                    node.validator,
                    lf,
                    skip_on_error=context.skip_on_error,
                    log_errors=context.log_errors,
                    max_retries=getattr(v_cfg, "max_retries", 0) if v_cfg else 0,
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
                        exc_info = ExceptionInfo.from_exception(
                            e, validator_name=node.validator.name,
                        )
                        node_result = NodeExecutionResult(
                            node_id=node.node_id,
                            result=ValidatorExecutionResult(
                                validator_name=node.validator.name,
                                status=ValidationResult.FAILED,
                                issues=[],
                                error_message=str(e),
                                error_context=exc_info.to_error_context(),
                                exception_info=exc_info,
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
        v_cfg = getattr(node.validator, "config", None)
        result = _validate_safe(
            node.validator,
            lf,
            skip_on_error=skip_on_error,
            log_errors=log_errors,
            max_retries=getattr(v_cfg, "max_retries", 0) if v_cfg else 0,
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
        """Execute the plan with dependency-based conditional skipping.

        For each level the executor:

        1. Evaluates ``Validator.should_skip()`` against prior results.
        2. Builds a *skipped* result for validators that should be skipped.
        3. Delegates the remaining validators to the chosen strategy.
        4. Records all results in the shared :class:`ExecutionContext` so
           that subsequent levels can use them for skip decisions and
           column-level filtering.

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
            level_start = time.time()
            nodes_to_run: list[ValidatorNode] = []
            skipped_results: list[NodeExecutionResult] = []

            prior = context.get_completed_execution_results()

            for node in level:
                should_skip, reason = node.validator.should_skip(prior)

                if should_skip:
                    logger.info(
                        "Skipping validator '%s': %s",
                        node.node_id,
                        reason,
                    )
                    now = time.time()
                    skip_result = ValidatorExecutionResult(
                        validator_name=node.validator.name,
                        status=ValidationResult.SKIPPED,
                        issues=[],
                        error_message=reason,
                    )
                    nr = NodeExecutionResult(
                        node_id=node.node_id,
                        result=skip_result,
                        start_time=now,
                        end_time=now,
                    )
                    skipped_results.append(nr)
                    context.add_result(nr)
                    context.record_result(node.node_id, skip_result)
                else:
                    nodes_to_run.append(node)

            if nodes_to_run:
                run_level = ExecutionLevel(
                    level_index=level.level_index,
                    nodes=nodes_to_run,
                    phase=level.phase,
                )
                level_result = strategy.execute_level(run_level, lf, context)

                # Propagate results to context
                for nr in level_result.node_results:
                    context.record_result(nr.node_id, nr.result)

                # Merge skipped results into level result
                level_result.node_results.extend(skipped_results)
            else:
                level_result = LevelExecutionResult(
                    level_index=level.level_index,
                    node_results=skipped_results,
                    start_time=level_start,
                    end_time=time.time(),
                )

            level_results.append(level_result)

        return ExecutionResult(
            level_results=level_results,
            total_start_time=total_start,
            total_end_time=time.time(),
            strategy_name=strategy.name,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get plan summary."""
        max_parallelism = max((lvl.size for lvl in self.levels), default=0)
        return {
            "total_nodes": self.total_nodes,
            "total_levels": len(self.levels),
            "max_parallelism": max_parallelism,
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
        priority: int | None = None,
        estimated_cost: float = 1.0,
    ) -> ValidatorNode:
        """Add a validator to the DAG.

        Class-level ``dependencies``, ``provides``, and ``priority``
        attributes are read from the Validator when the corresponding
        parameter is ``None``.

        Args:
            validator: Validator instance
            dependencies: Override set of node_ids this depends on
            provides: Override set of capabilities this provides
            phase: Execution phase override
            priority: Override priority (lower = earlier)
            estimated_cost: Estimated execution cost

        Returns:
            The created ValidatorNode
        """
        node_id = validator.name

        # Read class-level declarations when not overridden
        if dependencies is None:
            dependencies = getattr(validator, "dependencies", None) or set()
        if provides is None:
            provides = getattr(validator, "provides", None) or {node_id}
        if priority is None:
            priority = getattr(validator, "priority", 100)

        # Auto-detect phase from category
        if phase is None:
            category = getattr(validator, "category", "general")
            phase = CATEGORY_TO_PHASE.get(category, ValidatorPhase.CUSTOM)

        node = ValidatorNode(
            validator=validator,
            node_id=node_id,
            dependencies=set(dependencies),
            provides=set(provides),
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

    def _build_provides_map(self) -> dict[str, set[str]]:
        """Map each ``provides`` tag to the set of node_ids that provide it."""
        provides_map: dict[str, set[str]] = {}
        for node_id, node in self.nodes.items():
            for tag in node.provides:
                provides_map.setdefault(tag, set()).add(node_id)
        return provides_map

    def _resolve_dependencies(self) -> dict[str, set[str]]:
        """Resolve ``dependencies`` to concrete node_ids.

        A dependency string is resolved in order:

        1. Direct node_id match (``"null"`` → the ``null`` node).
        2. ``provides`` tag match (``"null_checked"`` → all nodes that
           include ``"null_checked"`` in their ``provides``).
        3. Unresolvable references are silently ignored (the upstream
           Validator may not be part of this particular run).
        """
        provides_map = self._build_provides_map()
        resolved: dict[str, set[str]] = {}

        for node_id, node in self.nodes.items():
            deps: set[str] = set()
            for dep in node.dependencies:
                if dep in self.nodes:
                    deps.add(dep)
                elif dep in provides_map:
                    deps.update(provides_map[dep])
            # Never depend on yourself
            deps.discard(node_id)
            resolved[node_id] = deps

        return resolved

    def _build_adjacency_with_phases(self) -> dict[str, list[str]]:
        """Build adjacency list from resolved dependencies + phase ordering."""
        adjacency: dict[str, list[str]] = {nid: [] for nid in self.nodes}

        resolved = self._resolve_dependencies()

        # Add resolved explicit dependencies
        for node_id, deps in resolved.items():
            for dep_id in deps:
                if dep_id in adjacency and node_id not in adjacency[dep_id]:
                    adjacency[dep_id].append(node_id)

        # Add implicit phase dependencies
        phase_to_nodes: dict[ValidatorPhase, list[str]] = {}
        for node_id, node in self.nodes.items():
            phase_to_nodes.setdefault(node.phase, []).append(node_id)

        sorted_phases = sorted(phase_to_nodes.keys(), key=lambda p: p.value)

        for i in range(len(sorted_phases) - 1):
            current_phase = sorted_phases[i]
            next_phase = sorted_phases[i + 1]

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
        """Group topologically-sorted node IDs into dependency-based levels.

        Two nodes share a level only when:
        - Neither depends on the other (directly or transitively).
        - They belong to the same execution phase.

        Validators within a level can safely run in parallel.
        """
        if not sorted_ids:
            return []

        resolved_deps = self._resolve_dependencies()

        # Compute each node's level index (longest-path from roots)
        node_level: dict[str, int] = {}
        for node_id in sorted_ids:
            deps = resolved_deps.get(node_id, set())
            if not deps:
                node_level[node_id] = 0
            else:
                max_dep = max(
                    (node_level.get(d, 0) for d in deps if d in node_level),
                    default=0,
                )
                node_level[node_id] = max_dep + 1

        # Group by (level_index, phase) to keep phases separate
        from collections import defaultdict
        groups: dict[tuple[int, ValidatorPhase], list[ValidatorNode]] = defaultdict(list)
        for node_id in sorted_ids:
            node = self.nodes[node_id]
            key = (node_level[node_id], node.phase)
            groups[key].append(node)

        # Sort groups by (level_index, phase.value) and build ExecutionLevels
        levels: list[ExecutionLevel] = []
        for (lvl_idx, phase) in sorted(groups.keys(), key=lambda k: (k[0], k[1].value)):
            nodes = sorted(groups[(lvl_idx, phase)], key=lambda n: (n.priority, n.node_id))
            levels.append(
                ExecutionLevel(
                    level_index=len(levels),
                    nodes=nodes,
                    phase=phase,
                )
            )

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
        """Create an ASCII visualization of the execution DAG.

        Example output::

            ValidatorDAG: 6 validators, 4 levels

            Level 0 [SCHEMA]:
              ├─ column_exists (provides: schema_validated, column_exists)

            Level 1 [COMPLETENESS]:
              ├─ null (depends: column_exists | provides: null_checked, null)
              ├─ unique (depends: column_exists | provides: uniqueness_checked, unique)

            Level 2 [RANGE]:
              ├─ between (depends: column_exists, null_checked | provides: range_checked, between)

            Level 3 [STATISTICAL]:
              └─ outlier (depends: column_exists, null_checked, range_checked)
        """
        if not self.nodes:
            return "Empty DAG"

        plan = self.build_execution_plan()
        lines = [f"ValidatorDAG: {len(self.nodes)} validators, {len(plan.levels)} levels"]

        for level in plan.levels:
            lines.append(f"\n  Level {level.level_index} [{level.phase.name}]:")
            sorted_nodes = sorted(level.nodes, key=lambda n: (n.priority, n.node_id))

            for i, node in enumerate(sorted_nodes):
                is_last = i == len(sorted_nodes) - 1
                prefix = "  └─ " if is_last else "  ├─ "

                parts = [node.node_id]

                deps = sorted(node.dependencies)
                if deps:
                    parts.append(f"depends: {', '.join(deps)}")

                # Only show provides if it's more than just {node_id}
                extra_provides = sorted(node.provides - {node.node_id})
                if extra_provides:
                    parts.append(f"provides: {', '.join(extra_provides)}")

                if len(parts) > 1:
                    lines.append(f"  {prefix}{parts[0]} ({' | '.join(parts[1:])})")
                else:
                    lines.append(f"  {prefix}{parts[0]}")

        return "\n".join(lines)

    def get_execution_summary(self, plan: ExecutionPlan | None = None) -> dict[str, Any]:
        """Return a summary of the execution plan.

        Args:
            plan: Pre-built plan. If *None*, ``build_execution_plan()``
                is called internally.

        Returns:
            Dict with keys ``total_validators``, ``total_levels``,
            ``max_parallelism``, ``dependency_chains``,
            ``estimated_speedup``.
        """
        if plan is None:
            plan = self.build_execution_plan()

        max_parallelism = max((len(lvl) for lvl in plan.levels), default=0)

        # Find longest dependency chains
        chains: list[list[str]] = []
        leaf_nodes = [
            nid
            for nid in self.nodes
            if not any(nid in n.dependencies for n in self.nodes.values())
        ]
        for leaf_id in leaf_nodes:
            chain = self.get_dependency_chain(leaf_id)
            if len(chain) > 1:
                chains.append(chain)
        # Keep top-5 longest chains
        chains.sort(key=len, reverse=True)
        chains = chains[:5]

        # Rough speedup estimate: total_validators / total_levels
        estimated_speedup = (
            len(self.nodes) / len(plan.levels) if plan.levels else 1.0
        )

        return {
            "total_validators": len(self.nodes),
            "total_levels": len(plan.levels),
            "max_parallelism": max_parallelism,
            "dependency_chains": chains,
            "estimated_speedup": round(estimated_speedup, 2),
        }

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
