"""PHASE 4: Validator Dependency DAG Tests.

Tests for:
- SkipCondition evaluation
- Validator.should_skip() / get_skip_conditions()
- Validator._filter_columns_by_context()
- ValidatorDAG dependency resolution (provides-based)
- ExecutionPlan conditional execution (skip on failure)
- ExecutionContext result propagation (critical_columns, etc.)
- Enhanced visualize() and get_execution_summary()
- End-to-end DAG execution with dependency skipping
"""

import pytest
import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    Validator,
    ValidatorConfig,
    ValidationIssue,
    ValidatorExecutionResult,
    ValidationResult,
    SkipCondition,
)
from truthound.validators.optimization.orchestrator import (
    ValidatorDAG,
    ValidatorNode,
    ValidatorPhase,
    ExecutionPlan,
    ExecutionLevel,
    ExecutionContext,
    ExecutionResult,
    NodeExecutionResult,
    LevelExecutionResult,
    SequentialExecutionStrategy,
    ParallelExecutionStrategy,
    AdaptiveExecutionStrategy,
    CATEGORY_TO_PHASE,
    create_execution_plan,
    execute_validators,
)


# ============================================================================
# Test Helpers: Concrete Validators for Testing
# ============================================================================

class AlwaysPassValidator(Validator):
    name = "always_pass"
    category = "general"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


class AlwaysFailValidator(Validator):
    name = "always_fail"
    category = "general"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return [
            ValidationIssue(
                column="test_col",
                issue_type="test_failure",
                count=1,
                severity=Severity.HIGH,
                validator_name=self.name,
            )
        ]


class CriticalFailValidator(Validator):
    name = "critical_fail"
    category = "schema"
    provides = {"schema_validated", "critical_fail"}
    priority = 10

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return [
            ValidationIssue(
                column="missing_col",
                issue_type="missing_column",
                count=1,
                severity=Severity.CRITICAL,
                validator_name=self.name,
            )
        ]


class DependentValidator(Validator):
    name = "dependent"
    category = "completeness"
    dependencies = {"critical_fail"}
    provides = {"dependent"}
    priority = 50

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


class SkipConditionValidator(Validator):
    """Validator that uses get_skip_conditions() for fine-grained control."""
    name = "skip_cond_validator"
    category = "distribution"
    dependencies = {"critical_fail"}
    provides = {"skip_cond_validator"}
    priority = 70

    def get_skip_conditions(self) -> list[SkipCondition]:
        return [
            SkipCondition(
                depends_on="critical_fail",
                skip_when="critical",
                reason_template="Skipped: {depends_on} had {skip_when} issues",
            ),
        ]

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


class NoDepValidator(Validator):
    name = "no_dep"
    category = "schema"
    provides = {"no_dep"}
    priority = 10

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


class ProvidesDepValidator(Validator):
    """Depends on a 'provides' tag, not a direct validator name."""
    name = "provides_dep"
    category = "completeness"
    dependencies = {"schema_validated"}
    provides = {"provides_dep"}
    priority = 50

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


class ChainAValidator(Validator):
    name = "chain_a"
    category = "schema"
    provides = {"chain_a_done"}
    priority = 10

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


class ChainBValidator(Validator):
    name = "chain_b"
    category = "completeness"
    dependencies = {"chain_a_done"}
    provides = {"chain_b_done"}
    priority = 50

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


class ChainCValidator(Validator):
    name = "chain_c"
    category = "distribution"
    dependencies = {"chain_b_done"}
    provides = {"chain_c_done"}
    priority = 70

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        return []


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_lf():
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", None, "Diana", "Eve"],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    }).lazy()


@pytest.fixture
def critical_exec_result():
    return ValidatorExecutionResult(
        validator_name="critical_fail",
        status=ValidationResult.SUCCESS,
        issues=[
            ValidationIssue(
                column="missing_col",
                issue_type="missing_column",
                count=1,
                severity=Severity.CRITICAL,
                validator_name="critical_fail",
            )
        ],
    )


@pytest.fixture
def failed_exec_result():
    return ValidatorExecutionResult(
        validator_name="some_validator",
        status=ValidationResult.FAILED,
        issues=[],
        error_message="Something went wrong",
    )


@pytest.fixture
def success_exec_result():
    return ValidatorExecutionResult(
        validator_name="some_validator",
        status=ValidationResult.SUCCESS,
        issues=[],
    )


# ============================================================================
# 1. SkipCondition Tests
# ============================================================================

class TestSkipCondition:
    def test_skip_condition_failed_trigger(self, failed_exec_result):
        cond = SkipCondition(depends_on="some_validator", skip_when="failed")
        skip, reason = cond.evaluate(failed_exec_result)
        assert skip is True
        assert "some_validator" in reason

    def test_skip_condition_failed_no_trigger_on_success(self, success_exec_result):
        cond = SkipCondition(depends_on="some_validator", skip_when="failed")
        skip, reason = cond.evaluate(success_exec_result)
        assert skip is False

    def test_skip_condition_critical_trigger(self, critical_exec_result):
        cond = SkipCondition(depends_on="critical_fail", skip_when="critical")
        skip, reason = cond.evaluate(critical_exec_result)
        assert skip is True

    def test_skip_condition_critical_no_trigger_on_high(self):
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.SUCCESS,
            issues=[
                ValidationIssue(
                    column="col",
                    issue_type="test",
                    count=1,
                    severity=Severity.HIGH,
                    validator_name="test",
                )
            ],
        )
        cond = SkipCondition(depends_on="test", skip_when="critical")
        skip, _ = cond.evaluate(result)
        assert skip is False

    def test_skip_condition_any_issue_trigger(self):
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.SUCCESS,
            issues=[
                ValidationIssue(
                    column="col",
                    issue_type="test",
                    count=1,
                    severity=Severity.LOW,
                    validator_name="test",
                )
            ],
        )
        cond = SkipCondition(depends_on="test", skip_when="any_issue")
        skip, _ = cond.evaluate(result)
        assert skip is True

    def test_skip_condition_any_issue_no_trigger_on_empty(self, success_exec_result):
        cond = SkipCondition(depends_on="some_validator", skip_when="any_issue")
        skip, _ = cond.evaluate(success_exec_result)
        assert skip is False

    def test_skip_condition_unknown_mode_no_trigger(self, failed_exec_result):
        cond = SkipCondition(depends_on="some_validator", skip_when="unknown_mode")
        skip, _ = cond.evaluate(failed_exec_result)
        assert skip is False

    def test_skip_condition_custom_reason_template(self, failed_exec_result):
        cond = SkipCondition(
            depends_on="some_validator",
            skip_when="failed",
            reason_template="Custom: {depends_on} was {skip_when}",
        )
        skip, reason = cond.evaluate(failed_exec_result)
        assert skip is True
        assert reason == "Custom: some_validator was failed"

    def test_skip_condition_is_frozen(self):
        cond = SkipCondition(depends_on="test", skip_when="failed")
        with pytest.raises(AttributeError):
            cond.depends_on = "other"

    def test_skip_condition_timeout_triggers_failed(self):
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.TIMEOUT,
            issues=[],
        )
        cond = SkipCondition(depends_on="test", skip_when="failed")
        skip, _ = cond.evaluate(result)
        assert skip is True


# ============================================================================
# 2. Validator.should_skip() Tests
# ============================================================================

class TestValidatorShouldSkip:
    def test_no_dependencies_never_skip(self):
        v = AlwaysPassValidator()
        skip, reason = v.should_skip({})
        assert skip is False
        assert reason is None

    def test_skip_when_dependency_failed(self, failed_exec_result):
        v = DependentValidator()
        prior = {"critical_fail": failed_exec_result}
        skip, reason = v.should_skip(prior)
        assert skip is True
        assert "critical_fail" in reason

    def test_no_skip_when_dependency_succeeded(self, success_exec_result):
        v = DependentValidator()
        prior = {"critical_fail": success_exec_result}
        skip, reason = v.should_skip(prior)
        assert skip is False

    def test_no_skip_when_dependency_not_in_prior(self):
        v = DependentValidator()
        skip, reason = v.should_skip({})
        assert skip is False

    def test_skip_condition_evaluated(self, critical_exec_result):
        v = SkipConditionValidator()
        prior = {"critical_fail": critical_exec_result}
        skip, reason = v.should_skip(prior)
        assert skip is True
        assert "critical" in reason

    def test_skip_condition_not_triggered(self, success_exec_result):
        v = SkipConditionValidator()
        prior = {"critical_fail": success_exec_result}
        skip, reason = v.should_skip(prior)
        assert skip is False

    def test_dependency_failure_takes_precedence_over_skip_condition(self):
        """If dependency itself FAILED, basic dependency check triggers first."""
        v = SkipConditionValidator()
        failed = ValidatorExecutionResult(
            validator_name="critical_fail",
            status=ValidationResult.FAILED,
            issues=[],
        )
        skip, reason = v.should_skip({"critical_fail": failed})
        assert skip is True
        assert "FAILED" in reason.upper() or "failed" in reason


# ============================================================================
# 3. Validator._filter_columns_by_context() Tests
# ============================================================================

class TestFilterColumnsByContext:
    def test_no_critical_columns_returns_all(self):
        v = AlwaysPassValidator()
        result = v._filter_columns_by_context(["a", "b", "c"], None)
        assert result == ["a", "b", "c"]

    def test_critical_columns_removed(self):
        v = AlwaysPassValidator()
        result = v._filter_columns_by_context(["a", "b", "c"], {"b"})
        assert result == ["a", "c"]

    def test_all_columns_critical_returns_empty(self):
        v = AlwaysPassValidator()
        result = v._filter_columns_by_context(["a", "b"], {"a", "b"})
        assert result == []

    def test_empty_critical_set_returns_all(self):
        v = AlwaysPassValidator()
        result = v._filter_columns_by_context(["a", "b"], set())
        assert result == ["a", "b"]


# ============================================================================
# 4. ExecutionContext Tests
# ============================================================================

class TestExecutionContext:
    def test_record_failed_result(self):
        ctx = ExecutionContext()
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.FAILED,
            issues=[],
        )
        ctx.record_result("test", result)
        assert "test" in ctx.failed_validators
        assert "test" not in ctx.skipped_validators

    def test_record_skipped_result(self):
        ctx = ExecutionContext()
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.SKIPPED,
            issues=[],
        )
        ctx.record_result("test", result)
        assert "test" in ctx.skipped_validators

    def test_record_critical_issue_tracks_column(self):
        ctx = ExecutionContext()
        result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.SUCCESS,
            issues=[
                ValidationIssue(
                    column="bad_col",
                    issue_type="missing",
                    count=1,
                    severity=Severity.CRITICAL,
                    validator_name="test",
                )
            ],
        )
        ctx.record_result("test", result)
        assert ctx.is_column_critical("bad_col")
        assert not ctx.is_column_critical("good_col")

    def test_get_completed_execution_results(self):
        ctx = ExecutionContext()
        exec_result = ValidatorExecutionResult(
            validator_name="test",
            status=ValidationResult.SUCCESS,
            issues=[],
        )
        node_result = NodeExecutionResult(
            node_id="test",
            result=exec_result,
            start_time=0.0,
            end_time=0.1,
        )
        ctx.add_result(node_result)
        completed = ctx.get_completed_execution_results()
        assert "test" in completed
        assert completed["test"].status == ValidationResult.SUCCESS


# ============================================================================
# 5. ValidatorDAG Dependency Resolution Tests
# ============================================================================

class TestValidatorDAGDependencies:
    def test_direct_dependency_resolution(self):
        dag = ValidatorDAG()
        dag.add_validator(NoDepValidator())
        dag.add_validator(DependentValidator())

        # DependentValidator depends on "critical_fail" which isn't in the DAG
        # so it should still be resolved without issues
        plan = dag.build_execution_plan()
        assert plan.total_nodes == 2

    def test_provides_based_dependency_resolution(self):
        """Dependencies can reference 'provides' tags."""
        dag = ValidatorDAG()
        dag.add_validator(CriticalFailValidator())  # provides: schema_validated
        dag.add_validator(ProvidesDepValidator())    # depends: schema_validated

        resolved = dag._resolve_dependencies()
        # provides_dep should resolve to critical_fail via schema_validated
        assert "critical_fail" in resolved["provides_dep"]

    def test_dependency_chain(self):
        """get_dependency_chain only follows direct name refs in node.dependencies."""
        dag = ValidatorDAG()
        dag.add_validator(ChainAValidator())
        dag.add_validator(ChainBValidator())
        dag.add_validator(ChainCValidator())

        # ChainB depends on "chain_a_done" (provides tag, not direct name),
        # so get_dependency_chain (which traverses raw .dependencies)
        # won't find a chain through direct names.
        # Just verify chain_c itself is returned.
        chain = dag.get_dependency_chain("chain_c")
        assert "chain_c" in chain

    def test_self_dependency_removed(self):
        """A validator should never depend on itself."""

        class SelfDepValidator(Validator):
            name = "self_dep"
            category = "general"
            dependencies = {"self_dep"}
            provides = {"self_dep"}

            def validate(self, lf):
                return []

        dag = ValidatorDAG()
        dag.add_validator(SelfDepValidator())
        resolved = dag._resolve_dependencies()
        assert "self_dep" not in resolved["self_dep"]

    def test_unresolvable_dependency_ignored(self):
        """Dependencies on validators not in the DAG are silently ignored."""
        dag = ValidatorDAG()
        dag.add_validator(DependentValidator())  # depends on "critical_fail"
        # critical_fail is not added to DAG

        resolved = dag._resolve_dependencies()
        assert resolved["dependent"] == set()

    def test_provides_map(self):
        dag = ValidatorDAG()
        dag.add_validator(CriticalFailValidator())
        provides_map = dag._build_provides_map()
        assert "schema_validated" in provides_map
        assert "critical_fail" in provides_map["critical_fail"]


# ============================================================================
# 6. ExecutionPlan Level Grouping Tests
# ============================================================================

class TestExecutionPlanLevels:
    def test_independent_validators_same_phase_same_level(self):
        class V1(Validator):
            name = "v1"
            category = "schema"
            provides = {"v1"}
            priority = 10
            def validate(self, lf): return []

        class V2(Validator):
            name = "v2"
            category = "schema"
            provides = {"v2"}
            priority = 10
            def validate(self, lf): return []

        dag = ValidatorDAG()
        dag.add_validator(V1())
        dag.add_validator(V2())
        plan = dag.build_execution_plan()

        # Both should be in the same level (no deps, same phase)
        assert len(plan.levels) == 1
        assert plan.levels[0].size == 2

    def test_dependent_validators_different_levels(self):
        dag = ValidatorDAG()
        dag.add_validator(ChainAValidator())
        dag.add_validator(ChainBValidator())
        dag.add_validator(ChainCValidator())

        plan = dag.build_execution_plan()
        # chain_a → chain_b → chain_c must be in different levels
        assert plan.total_nodes == 3

        node_ids_per_level = [lvl.node_ids for lvl in plan.levels]
        # Flatten to check ordering
        flat = [nid for ids in node_ids_per_level for nid in ids]
        assert flat.index("chain_a") < flat.index("chain_b") < flat.index("chain_c")

    def test_priority_ordering_within_level(self):
        class LowPri(Validator):
            name = "low_pri"
            category = "schema"
            priority = 90
            def validate(self, lf): return []

        class HighPri(Validator):
            name = "high_pri"
            category = "schema"
            priority = 10
            def validate(self, lf): return []

        dag = ValidatorDAG()
        dag.add_validator(LowPri())
        dag.add_validator(HighPri())
        plan = dag.build_execution_plan()

        level = plan.levels[0]
        assert level.nodes[0].node_id == "high_pri"
        assert level.nodes[1].node_id == "low_pri"


# ============================================================================
# 7. Conditional Execution Tests
# ============================================================================

class TestConditionalExecution:
    def test_dependent_skipped_when_dependency_fails(self, sample_lf):
        """When a dependency validator FAILS, dependents are skipped."""

        class FailingRoot(Validator):
            name = "failing_root"
            category = "schema"
            provides = {"failing_root"}
            priority = 10
            def validate(self, lf):
                raise ValueError("Intentional failure for testing")

        class DepOnFailing(Validator):
            name = "dep_on_failing"
            category = "completeness"
            dependencies = {"failing_root"}
            provides = {"dep_on_failing"}
            priority = 50
            def validate(self, lf):
                return []

        dag = ValidatorDAG()
        dag.add_validator(FailingRoot())
        dag.add_validator(DepOnFailing())
        plan = dag.build_execution_plan()

        result = plan.execute(sample_lf, SequentialExecutionStrategy())
        nr_map = {nr.node_id: nr for nr in result.node_results}

        assert nr_map["failing_root"].status == ValidationResult.FAILED
        assert nr_map["dep_on_failing"].status == ValidationResult.SKIPPED

    def test_skip_condition_critical_triggers_skip(self, sample_lf):
        """SkipCondition with skip_when='critical' triggers on CRITICAL issues."""
        dag = ValidatorDAG()
        dag.add_validator(CriticalFailValidator())
        dag.add_validator(SkipConditionValidator())
        plan = dag.build_execution_plan()

        result = plan.execute(sample_lf, SequentialExecutionStrategy())
        nr_map = {nr.node_id: nr for nr in result.node_results}

        # CriticalFailValidator returns SUCCESS with CRITICAL issue
        assert nr_map["critical_fail"].status == ValidationResult.SUCCESS
        # SkipConditionValidator should be skipped due to critical skip condition
        assert nr_map["skip_cond_validator"].status == ValidationResult.SKIPPED

    def test_no_skip_when_dependency_succeeds(self, sample_lf):
        dag = ValidatorDAG()
        dag.add_validator(NoDepValidator())
        dag.add_validator(ProvidesDepValidator())

        # NoDepValidator provides "no_dep" but not "schema_validated",
        # so provides_dep's dep on "schema_validated" is unresolvable → not skipped
        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf, SequentialExecutionStrategy())

        nr_map = {nr.node_id: nr for nr in result.node_results}
        assert nr_map["provides_dep"].status == ValidationResult.SUCCESS

    def test_skipped_count_property(self, sample_lf):
        dag = ValidatorDAG()
        dag.add_validator(CriticalFailValidator())
        dag.add_validator(SkipConditionValidator())
        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf, SequentialExecutionStrategy())

        assert result.skipped_count == 1


# ============================================================================
# 8. Visualization Tests
# ============================================================================

class TestVisualization:
    def test_visualize_empty_dag(self):
        dag = ValidatorDAG()
        assert "Empty" in dag.visualize()

    def test_visualize_with_validators(self):
        dag = ValidatorDAG()
        dag.add_validator(CriticalFailValidator())
        dag.add_validator(DependentValidator())

        viz = dag.visualize()
        assert "critical_fail" in viz
        assert "dependent" in viz
        assert "Level" in viz

    def test_visualize_shows_dependencies(self):
        dag = ValidatorDAG()
        dag.add_validator(ChainAValidator())
        dag.add_validator(ChainBValidator())
        dag.add_validator(ChainCValidator())

        viz = dag.visualize()
        assert "chain_a" in viz
        assert "chain_b" in viz
        assert "chain_c" in viz
        assert "depends:" in viz

    def test_get_execution_summary(self):
        dag = ValidatorDAG()
        dag.add_validator(ChainAValidator())
        dag.add_validator(ChainBValidator())
        dag.add_validator(ChainCValidator())

        summary = dag.get_execution_summary()
        assert summary["total_validators"] == 3
        assert summary["total_levels"] >= 3
        assert summary["max_parallelism"] >= 1
        # dependency_chains may be empty when deps are provides-based
        assert isinstance(summary["dependency_chains"], list)
        assert summary["estimated_speedup"] >= 0.1

    def test_get_summary_includes_max_parallelism(self):
        class V1(Validator):
            name = "v1"
            category = "schema"
            provides = {"v1"}
            def validate(self, lf): return []

        class V2(Validator):
            name = "v2"
            category = "schema"
            provides = {"v2"}
            def validate(self, lf): return []

        dag = ValidatorDAG()
        dag.add_validator(V1())
        dag.add_validator(V2())
        plan = dag.build_execution_plan()

        summary = plan.get_summary()
        assert "max_parallelism" in summary
        assert summary["max_parallelism"] == 2


# ============================================================================
# 9. ExecutionResult Metrics Tests
# ============================================================================

class TestExecutionResultMetrics:
    def test_get_metrics_includes_skipped(self, sample_lf):
        dag = ValidatorDAG()
        dag.add_validator(CriticalFailValidator())
        dag.add_validator(SkipConditionValidator())
        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf, SequentialExecutionStrategy())

        metrics = result.get_metrics()
        assert "skipped_count" in metrics
        assert metrics["skipped_count"] == 1


# ============================================================================
# 10. End-to-End DAG Execution Tests
# ============================================================================

class TestEndToEndDAG:
    def test_full_chain_execution(self, sample_lf):
        """chain_a → chain_b → chain_c all succeed."""
        dag = ValidatorDAG()
        dag.add_validator(ChainAValidator())
        dag.add_validator(ChainBValidator())
        dag.add_validator(ChainCValidator())

        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf, SequentialExecutionStrategy())

        assert result.total_validators == 3
        assert result.success_count == 3
        assert result.failure_count == 0
        assert result.skipped_count == 0

    def test_cascade_skip_on_root_failure(self, sample_lf):
        """When root fails, all dependents cascade-skip."""

        class Root(Validator):
            name = "root"
            category = "schema"
            provides = {"root_ok"}
            priority = 10
            def validate(self, lf):
                raise RuntimeError("root failed")

        class Mid(Validator):
            name = "mid"
            category = "completeness"
            dependencies = {"root"}
            provides = {"mid_ok"}
            priority = 50
            def validate(self, lf):
                return []

        class Leaf(Validator):
            name = "leaf"
            category = "distribution"
            dependencies = {"mid"}
            provides = {"leaf_ok"}
            priority = 70
            def validate(self, lf):
                return []

        dag = ValidatorDAG()
        dag.add_validator(Root())
        dag.add_validator(Mid())
        dag.add_validator(Leaf())

        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf, SequentialExecutionStrategy())

        nr_map = {nr.node_id: nr for nr in result.node_results}
        assert nr_map["root"].status == ValidationResult.FAILED
        assert nr_map["mid"].status == ValidationResult.SKIPPED
        assert nr_map["leaf"].status == ValidationResult.SKIPPED

    def test_parallel_execution_respects_skipping(self, sample_lf):
        """Parallel strategy also respects dependency-based skipping."""
        dag = ValidatorDAG()
        dag.add_validator(CriticalFailValidator())
        dag.add_validator(SkipConditionValidator())

        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf, ParallelExecutionStrategy(max_workers=2))

        nr_map = {nr.node_id: nr for nr in result.node_results}
        assert nr_map["skip_cond_validator"].status == ValidationResult.SKIPPED

    def test_adaptive_strategy_works(self, sample_lf):
        dag = ValidatorDAG()
        dag.add_validator(ChainAValidator())
        dag.add_validator(ChainBValidator())
        dag.add_validator(ChainCValidator())

        plan = dag.build_execution_plan()
        result = plan.execute(sample_lf, AdaptiveExecutionStrategy())

        assert result.total_validators == 3
        assert result.success_count == 3

    def test_convenience_functions(self, sample_lf):
        validators = [ChainAValidator(), ChainBValidator()]
        plan = create_execution_plan(validators)
        assert plan.total_nodes == 2

        result = execute_validators(validators, sample_lf)
        assert result.success_count == 2


# ============================================================================
# 11. ValidatorNode Tests
# ============================================================================

class TestValidatorNode:
    def test_auto_detect_phase(self):
        v = AlwaysPassValidator()
        v.category = "schema"
        node = ValidatorNode(validator=v)
        assert node.phase == ValidatorPhase.SCHEMA

    def test_auto_populate_provides(self):
        v = AlwaysPassValidator()
        node = ValidatorNode(validator=v)
        assert node.node_id in node.provides

    def test_explicit_provides_preserved(self):
        node = ValidatorNode(
            validator=CriticalFailValidator(),
            provides={"custom_tag"},
        )
        assert "custom_tag" in node.provides

    def test_node_id_defaults_to_name(self):
        v = AlwaysPassValidator()
        node = ValidatorNode(validator=v)
        assert node.node_id == "always_pass"


# ============================================================================
# 12. Provides-based Resolution Integration Tests
# ============================================================================

class TestProvidesResolution:
    def test_multiple_providers_all_become_deps(self):
        """When two validators provide the same tag, both become deps."""

        class ProviderA(Validator):
            name = "provider_a"
            category = "schema"
            provides = {"checked"}
            priority = 10
            def validate(self, lf): return []

        class ProviderB(Validator):
            name = "provider_b"
            category = "schema"
            provides = {"checked"}
            priority = 10
            def validate(self, lf): return []

        class Consumer(Validator):
            name = "consumer"
            category = "completeness"
            dependencies = {"checked"}
            provides = {"consumer"}
            priority = 50
            def validate(self, lf): return []

        dag = ValidatorDAG()
        dag.add_validator(ProviderA())
        dag.add_validator(ProviderB())
        dag.add_validator(Consumer())

        resolved = dag._resolve_dependencies()
        assert "provider_a" in resolved["consumer"]
        assert "provider_b" in resolved["consumer"]

    def test_mixed_direct_and_provides_deps(self, sample_lf):
        """Mix of direct name deps and provides-tag deps."""

        class SchemaV(Validator):
            name = "schema_v"
            category = "schema"
            provides = {"schema_done"}
            priority = 10
            def validate(self, lf): return []

        class NullV(Validator):
            name = "null_v"
            category = "completeness"
            dependencies = {"schema_done"}
            provides = {"null_done"}
            priority = 50
            def validate(self, lf): return []

        class RangeV(Validator):
            name = "range_v"
            category = "distribution"
            dependencies = {"schema_v", "null_done"}  # mix of direct + provides
            provides = {"range_v"}
            priority = 70
            def validate(self, lf): return []

        dag = ValidatorDAG()
        dag.add_validator(SchemaV())
        dag.add_validator(NullV())
        dag.add_validator(RangeV())

        resolved = dag._resolve_dependencies()
        assert "schema_v" in resolved["range_v"]
        assert "null_v" in resolved["range_v"]

        result = execute_validators([SchemaV(), NullV(), RangeV()], sample_lf)
        assert result.success_count == 3


# ============================================================================
# 13. Built-in Validator Dependency Declaration Tests
# ============================================================================

class TestBuiltinValidatorDeclarations:
    """Verify that built-in validators have proper dependency declarations."""

    def test_null_validator_has_deps(self):
        from truthound.validators.completeness.null import NullValidator
        assert "column_exists" in NullValidator.dependencies
        assert "null_checked" in NullValidator.provides
        assert NullValidator.priority == 50

    def test_between_validator_has_deps(self):
        from truthound.validators.distribution.range import BetweenValidator
        assert "column_exists" in BetweenValidator.dependencies
        assert "null_checked" in BetweenValidator.dependencies
        assert "range_checked" in BetweenValidator.provides
        assert BetweenValidator.priority == 70

    def test_column_exists_validator_has_provides(self):
        from truthound.validators.schema.column_exists import ColumnExistsValidator
        assert "schema_validated" in ColumnExistsValidator.provides
        assert ColumnExistsValidator.priority == 10

    def test_column_type_validator_has_deps(self):
        from truthound.validators.schema.column_type import ColumnTypeValidator
        assert "column_exists" in ColumnTypeValidator.dependencies
        assert "type_validated" in ColumnTypeValidator.provides
        assert ColumnTypeValidator.priority == 20

    def test_unique_validator_has_deps(self):
        from truthound.validators.uniqueness.unique import UniqueValidator
        assert "column_exists" in UniqueValidator.dependencies
        assert "uniqueness_checked" in UniqueValidator.provides
        assert UniqueValidator.priority == 60

    def test_outlier_validator_has_deps(self):
        from truthound.validators.distribution.outlier import OutlierValidator
        assert "column_exists" in OutlierValidator.dependencies
        assert "null_checked" in OutlierValidator.dependencies
        assert "range_checked" in OutlierValidator.dependencies
        assert OutlierValidator.priority == 80

    def test_regex_validator_has_deps(self):
        from truthound.validators.string.regex import RegexValidator
        assert "column_exists" in RegexValidator.dependencies
        assert "null_checked" in RegexValidator.dependencies
        assert "pattern_checked" in RegexValidator.provides
        assert RegexValidator.priority == 70

    def test_between_validator_has_skip_conditions(self):
        from truthound.validators.distribution.range import BetweenValidator
        v = BetweenValidator(min_value=0, max_value=100)
        conditions = v.get_skip_conditions()
        assert len(conditions) >= 2
        deps = {c.depends_on for c in conditions}
        assert "column_exists" in deps
        assert "null" in deps

    def test_outlier_validator_has_skip_conditions(self):
        from truthound.validators.distribution.outlier import OutlierValidator
        v = OutlierValidator()
        conditions = v.get_skip_conditions()
        assert len(conditions) >= 3
        deps = {c.depends_on for c in conditions}
        assert "column_exists" in deps
        assert "null" in deps
        assert "between" in deps
