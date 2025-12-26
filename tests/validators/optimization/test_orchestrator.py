"""Tests for DAG-based validator orchestration system.

This module tests:
    - ValidatorDAG construction and dependency resolution
    - ExecutionPlan building with topological ordering
    - Execution strategies (Sequential, Parallel, Adaptive)
    - Phase-based ordering
    - Cycle detection
    - Parallel execution correctness
"""

import time
import pytest
import polars as pl

from truthound.validators.base import Validator, ValidationIssue, ValidatorConfig
from truthound.validators.optimization.orchestrator import (
    ValidatorDAG,
    ValidatorNode,
    ValidatorPhase,
    ExecutionPlan,
    ExecutionLevel,
    ExecutionResult,
    ExecutionContext,
    SequentialExecutionStrategy,
    ParallelExecutionStrategy,
    AdaptiveExecutionStrategy,
    create_execution_plan,
    execute_validators,
    CATEGORY_TO_PHASE,
)
from truthound.types import Severity


# ============================================================================
# Test Fixtures
# ============================================================================

class MockValidator(Validator):
    """Mock validator for testing."""

    name = "mock"
    category = "general"

    def __init__(
        self,
        name: str = "mock",
        category: str = "general",
        dependencies: set[str] | None = None,
        sleep_time: float = 0.0,
        issues_to_return: list[ValidationIssue] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.category = category
        self.dependencies = dependencies or set()
        self.sleep_time = sleep_time
        self.issues_to_return = issues_to_return or []
        self.execution_count = 0
        self.execution_times: list[float] = []

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        start = time.time()
        self.execution_count += 1

        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        self.execution_times.append(time.time() - start)
        return self.issues_to_return


class SchemaValidator(MockValidator):
    """Mock schema validator."""
    name = "schema"
    category = "schema"


class NullValidator(MockValidator):
    """Mock null validator."""
    name = "null"
    category = "completeness"


class UniqueValidator(MockValidator):
    """Mock unique validator."""
    name = "unique"
    category = "uniqueness"


class RangeValidator(MockValidator):
    """Mock range validator."""
    name = "range"
    category = "distribution"


class OutlierValidator(MockValidator):
    """Mock outlier validator."""
    name = "outlier"
    category = "anomaly"
    dependencies = {"null", "range"}


@pytest.fixture
def sample_lf() -> pl.LazyFrame:
    """Create a sample LazyFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", None, "Eve"],
        "age": [25, 30, 35, 40, 45],
        "score": [85.5, 92.3, 78.1, 88.9, 95.2],
    }).lazy()


@pytest.fixture
def basic_validators() -> list[Validator]:
    """Create basic validators without dependencies."""
    return [
        MockValidator(name="v1", category="completeness"),
        MockValidator(name="v2", category="completeness"),
        MockValidator(name="v3", category="completeness"),
    ]


@pytest.fixture
def phased_validators() -> list[Validator]:
    """Create validators with different phases."""
    return [
        MockValidator(name="schema", category="schema"),
        MockValidator(name="null", category="completeness"),
        MockValidator(name="unique", category="uniqueness"),
        MockValidator(name="range", category="distribution"),
        MockValidator(name="outlier", category="anomaly"),
    ]


@pytest.fixture
def dependent_validators() -> list[Validator]:
    """Create validators with explicit dependencies."""
    v1 = MockValidator(name="v1", category="completeness")
    v2 = MockValidator(name="v2", category="completeness", dependencies={"v1"})
    v3 = MockValidator(name="v3", category="completeness", dependencies={"v1", "v2"})
    return [v1, v2, v3]


# ============================================================================
# ValidatorNode Tests
# ============================================================================

class TestValidatorNode:
    """Tests for ValidatorNode."""

    def test_node_creation_basic(self):
        """Test basic node creation."""
        validator = MockValidator(name="test")
        node = ValidatorNode(validator=validator)

        assert node.node_id == "test"
        assert node.validator == validator
        assert node.dependencies == set()
        assert node.provides == {"test"}
        assert node.priority == 100

    def test_node_auto_phase_detection(self):
        """Test automatic phase detection from category."""
        schema_v = MockValidator(name="s", category="schema")
        node = ValidatorNode(validator=schema_v)
        assert node.phase == ValidatorPhase.SCHEMA

        null_v = MockValidator(name="n", category="completeness")
        node = ValidatorNode(validator=null_v)
        assert node.phase == ValidatorPhase.COMPLETENESS

    def test_node_explicit_dependencies(self):
        """Test node with explicit dependencies."""
        validator = MockValidator(name="test")
        node = ValidatorNode(
            validator=validator,
            dependencies={"dep1", "dep2"},
        )

        assert node.dependencies == {"dep1", "dep2"}

    def test_node_hash_and_equality(self):
        """Test node hashing and equality."""
        v1 = MockValidator(name="test")
        v2 = MockValidator(name="test")

        node1 = ValidatorNode(validator=v1)
        node2 = ValidatorNode(validator=v2)

        assert node1 == node2
        assert hash(node1) == hash(node2)


# ============================================================================
# ValidatorDAG Tests
# ============================================================================

class TestValidatorDAG:
    """Tests for ValidatorDAG."""

    def test_empty_dag(self):
        """Test empty DAG."""
        dag = ValidatorDAG()
        assert len(dag) == 0

        plan = dag.build_execution_plan()
        assert plan.total_nodes == 0
        assert len(plan.levels) == 0

    def test_single_validator(self):
        """Test DAG with single validator."""
        dag = ValidatorDAG()
        dag.add_validator(MockValidator(name="single"))

        assert len(dag) == 1

        plan = dag.build_execution_plan()
        assert plan.total_nodes == 1
        assert len(plan.levels) == 1
        assert plan.levels[0].size == 1

    def test_add_validators_bulk(self, basic_validators):
        """Test adding multiple validators at once."""
        dag = ValidatorDAG()
        nodes = dag.add_validators(basic_validators)

        assert len(dag) == 3
        assert len(nodes) == 3
        assert all(isinstance(n, ValidatorNode) for n in nodes)

    def test_phase_based_ordering(self, phased_validators):
        """Test validators are ordered by phase."""
        dag = ValidatorDAG()
        dag.add_validators(phased_validators)

        plan = dag.build_execution_plan()

        # Should have multiple levels based on phases
        assert plan.total_nodes == 5
        assert len(plan.levels) >= 1

        # Verify phase ordering
        level_phases = [level.phase for level in plan.levels]
        phase_values = [p.value for p in level_phases]
        assert phase_values == sorted(phase_values), "Phases should be in order"

    def test_explicit_dependencies(self, dependent_validators):
        """Test explicit dependency resolution."""
        dag = ValidatorDAG()
        dag.add_validators(dependent_validators)

        plan = dag.build_execution_plan()

        # v1 should be first, then v2, then v3
        all_node_ids = []
        for level in plan.levels:
            all_node_ids.extend(level.node_ids)

        v1_idx = all_node_ids.index("v1")
        v2_idx = all_node_ids.index("v2")
        v3_idx = all_node_ids.index("v3")

        assert v1_idx < v2_idx, "v1 should come before v2"
        assert v2_idx < v3_idx, "v2 should come before v3"

    def test_parallel_safe_grouping(self, basic_validators):
        """Test that independent validators are grouped together."""
        dag = ValidatorDAG()
        dag.add_validators(basic_validators)

        plan = dag.build_execution_plan()

        # All have same category/phase, no deps, should be in same level
        assert len(plan.levels) == 1
        assert plan.levels[0].size == 3

    def test_add_dependency_manually(self):
        """Test manually adding dependencies."""
        dag = ValidatorDAG()
        dag.add_validator(MockValidator(name="a", category="completeness"))
        dag.add_validator(MockValidator(name="b", category="completeness"))
        dag.add_dependency("b", "a")

        plan = dag.build_execution_plan()

        all_node_ids = []
        for level in plan.levels:
            all_node_ids.extend(level.node_ids)

        assert all_node_ids.index("a") < all_node_ids.index("b")

    def test_get_dependency_chain(self, dependent_validators):
        """Test getting full dependency chain."""
        dag = ValidatorDAG()
        dag.add_validators(dependent_validators)

        chain = dag.get_dependency_chain("v3")

        assert "v1" in chain
        assert "v2" in chain
        assert "v3" in chain
        assert chain.index("v1") < chain.index("v2")
        assert chain.index("v2") < chain.index("v3")

    def test_visualize(self, phased_validators):
        """Test DAG visualization."""
        dag = ValidatorDAG()
        dag.add_validators(phased_validators)

        viz = dag.visualize()

        assert "ValidatorDAG:" in viz
        assert "SCHEMA" in viz
        assert "COMPLETENESS" in viz


# ============================================================================
# ExecutionPlan Tests
# ============================================================================

class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_plan_summary(self, phased_validators):
        """Test plan summary generation."""
        dag = ValidatorDAG()
        dag.add_validators(phased_validators)
        plan = dag.build_execution_plan()

        summary = plan.get_summary()

        assert summary["total_nodes"] == 5
        assert "levels" in summary
        assert all("nodes" in level for level in summary["levels"])

    def test_plan_repr(self, basic_validators):
        """Test plan string representation."""
        dag = ValidatorDAG()
        dag.add_validators(basic_validators)
        plan = dag.build_execution_plan()

        repr_str = repr(plan)
        assert "ExecutionPlan" in repr_str
        assert "nodes=3" in repr_str


# ============================================================================
# Execution Strategy Tests
# ============================================================================

class TestSequentialExecutionStrategy:
    """Tests for SequentialExecutionStrategy."""

    def test_sequential_execution(self, sample_lf, basic_validators):
        """Test sequential execution of validators."""
        dag = ValidatorDAG()
        dag.add_validators(basic_validators)
        plan = dag.build_execution_plan()

        strategy = SequentialExecutionStrategy()
        result = plan.execute(sample_lf, strategy)

        assert result.total_validators == 3
        assert result.strategy_name == "sequential"
        assert result.success_count == 3
        assert result.failure_count == 0

    def test_sequential_respects_order(self, sample_lf, dependent_validators):
        """Test that sequential execution respects dependency order."""
        execution_order = []

        class OrderTrackingValidator(MockValidator):
            def validate(self, lf):
                execution_order.append(self.name)
                return []

        validators = [
            OrderTrackingValidator(name="v1", category="completeness"),
            OrderTrackingValidator(name="v2", category="completeness", dependencies={"v1"}),
            OrderTrackingValidator(name="v3", category="completeness", dependencies={"v2"}),
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        strategy = SequentialExecutionStrategy()
        plan.execute(sample_lf, strategy)

        assert execution_order == ["v1", "v2", "v3"]


class TestParallelExecutionStrategy:
    """Tests for ParallelExecutionStrategy."""

    def test_parallel_execution(self, sample_lf, basic_validators):
        """Test parallel execution of validators."""
        dag = ValidatorDAG()
        dag.add_validators(basic_validators)
        plan = dag.build_execution_plan()

        strategy = ParallelExecutionStrategy(max_workers=3)
        result = plan.execute(sample_lf, strategy)

        assert result.total_validators == 3
        assert result.strategy_name == "parallel"
        assert result.success_count == 3

    def test_parallel_faster_than_sequential(self, sample_lf):
        """Test that parallel execution is faster for slow validators."""
        # Create validators with sleep
        slow_validators = [
            MockValidator(name=f"slow_{i}", category="completeness", sleep_time=0.05)
            for i in range(4)
        ]

        dag = ValidatorDAG()
        dag.add_validators(slow_validators)
        plan = dag.build_execution_plan()

        # Sequential timing
        start = time.time()
        seq_strategy = SequentialExecutionStrategy()
        seq_result = plan.execute(sample_lf, seq_strategy)
        seq_time = time.time() - start

        # Parallel timing
        start = time.time()
        par_strategy = ParallelExecutionStrategy(max_workers=4)
        par_result = plan.execute(sample_lf, par_strategy)
        par_time = time.time() - start

        # Parallel should be faster (with some margin for overhead)
        assert par_time < seq_time * 0.8, (
            f"Parallel ({par_time:.3f}s) should be faster than sequential ({seq_time:.3f}s)"
        )

    def test_parallel_with_max_workers(self, sample_lf, basic_validators):
        """Test parallel execution with limited workers."""
        dag = ValidatorDAG()
        dag.add_validators(basic_validators)
        plan = dag.build_execution_plan()

        strategy = ParallelExecutionStrategy(max_workers=1)
        result = plan.execute(sample_lf, strategy)

        assert result.success_count == 3


class TestAdaptiveExecutionStrategy:
    """Tests for AdaptiveExecutionStrategy."""

    def test_adaptive_uses_sequential_for_small(self, sample_lf):
        """Test that adaptive uses sequential for small levels."""
        validators = [
            MockValidator(name="v1", category="completeness"),
            MockValidator(name="v2", category="completeness"),
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        strategy = AdaptiveExecutionStrategy(parallel_threshold=3)
        result = plan.execute(sample_lf, strategy)

        assert result.strategy_name == "adaptive"
        assert result.success_count == 2

    def test_adaptive_uses_parallel_for_large(self, sample_lf):
        """Test that adaptive uses parallel for large levels."""
        validators = [
            MockValidator(name=f"v{i}", category="completeness")
            for i in range(5)
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        strategy = AdaptiveExecutionStrategy(parallel_threshold=3)
        result = plan.execute(sample_lf, strategy)

        assert result.success_count == 5


# ============================================================================
# ExecutionResult Tests
# ============================================================================

class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_result_metrics(self, sample_lf, basic_validators):
        """Test result metrics calculation."""
        dag = ValidatorDAG()
        dag.add_validators(basic_validators)
        plan = dag.build_execution_plan()

        result = plan.execute(sample_lf)

        metrics = result.get_metrics()

        assert "total_duration_ms" in metrics
        assert "total_validators" in metrics
        assert "success_count" in metrics
        assert "parallelism_factor" in metrics
        assert metrics["total_validators"] == 3

    def test_result_collects_issues(self, sample_lf):
        """Test that result collects all issues."""
        issue1 = ValidationIssue(
            column="col1",
            issue_type="test",
            count=1,
            severity=Severity.LOW,
        )
        issue2 = ValidationIssue(
            column="col2",
            issue_type="test",
            count=2,
            severity=Severity.MEDIUM,
        )

        validators = [
            MockValidator(name="v1", category="completeness", issues_to_return=[issue1]),
            MockValidator(name="v2", category="completeness", issues_to_return=[issue2]),
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        result = plan.execute(sample_lf)

        assert len(result.all_issues) == 2
        assert issue1 in result.all_issues
        assert issue2 in result.all_issues


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_execution_plan(self, basic_validators):
        """Test create_execution_plan function."""
        plan = create_execution_plan(basic_validators)

        assert isinstance(plan, ExecutionPlan)
        assert plan.total_nodes == 3

    def test_create_execution_plan_with_deps(self):
        """Test create_execution_plan with dependencies."""
        validators = [
            MockValidator(name="a", category="completeness"),
            MockValidator(name="b", category="completeness"),
            MockValidator(name="c", category="completeness"),
        ]

        dependencies = {
            "b": {"a"},
            "c": {"a", "b"},
        }

        plan = create_execution_plan(validators, dependencies)

        all_node_ids = []
        for level in plan.levels:
            all_node_ids.extend(level.node_ids)

        assert all_node_ids.index("a") < all_node_ids.index("b")
        assert all_node_ids.index("b") < all_node_ids.index("c")

    def test_execute_validators(self, sample_lf, basic_validators):
        """Test execute_validators function."""
        result = execute_validators(basic_validators, sample_lf)

        assert isinstance(result, ExecutionResult)
        assert result.total_validators == 3


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full orchestration system."""

    def test_full_pipeline_with_real_validators(self, sample_lf):
        """Test full pipeline with realistic validator setup."""
        validators = [
            MockValidator(name="schema", category="schema"),
            MockValidator(name="null", category="completeness"),
            MockValidator(name="not_null", category="completeness"),
            MockValidator(name="unique", category="uniqueness"),
            MockValidator(name="duplicate", category="uniqueness"),
            MockValidator(name="range", category="distribution"),
            MockValidator(name="outlier", category="anomaly", dependencies={"range"}),
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        # Verify phase ordering
        phase_order = [level.phase for level in plan.levels]
        phase_values = [p.value for p in phase_order]
        assert phase_values == sorted(phase_values)

        # Execute
        result = plan.execute(sample_lf)
        assert result.success_count == 7

    def test_error_handling(self, sample_lf):
        """Test error handling during execution."""

        class FailingValidator(MockValidator):
            def validate(self, lf):
                raise ValueError("Test error")

        validators = [
            MockValidator(name="good", category="completeness"),
            FailingValidator(name="bad", category="completeness"),
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        result = plan.execute(sample_lf, skip_on_error=True)

        # Should have 1 success, 1 failure
        assert result.success_count == 1
        assert result.failure_count == 1

    def test_phase_category_mapping(self):
        """Test that all standard categories map to phases."""
        expected_mappings = {
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
        }

        for category, expected_phase in expected_mappings.items():
            assert CATEGORY_TO_PHASE.get(category) == expected_phase


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_validator_with_unknown_dependency(self, sample_lf):
        """Test validator with dependency that doesn't exist."""
        validators = [
            MockValidator(name="v1", category="completeness", dependencies={"nonexistent"}),
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        # Should still work - unknown deps are ignored
        result = plan.execute(sample_lf)
        assert result.success_count == 1

    def test_self_referencing_dependency(self, sample_lf):
        """Test validator that depends on itself."""
        validators = [
            MockValidator(name="v1", category="completeness", dependencies={"v1"}),
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)
        plan = dag.build_execution_plan()

        # Should handle gracefully
        result = plan.execute(sample_lf)
        assert result.total_validators == 1

    def test_empty_lazyframe(self):
        """Test execution with empty LazyFrame."""
        empty_lf = pl.DataFrame({"col": []}).lazy()

        validators = [
            MockValidator(name="v1", category="completeness"),
        ]

        result = execute_validators(validators, empty_lf)
        assert result.success_count == 1

    def test_duplicate_validator_names(self, sample_lf):
        """Test handling of duplicate validator names."""
        validators = [
            MockValidator(name="dup", category="completeness"),
            MockValidator(name="dup", category="completeness"),  # Same name
        ]

        dag = ValidatorDAG()
        dag.add_validators(validators)

        # Second should overwrite first
        assert len(dag) == 1

        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf)
        assert result.total_validators == 1
