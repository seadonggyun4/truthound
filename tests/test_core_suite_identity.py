"""Named/repeated checks retain actual instance outcomes across execution modes."""

import polars as pl
import pytest

from truthound.core import ScanPlanner, ValidationRuntime, build_validation_asset
from truthound.core.suite import CheckSpec, EvidencePolicy, ValidationSuite
from truthound.types import ResultFormat, ResultFormatConfig, Severity
from truthound.validators import get_validator
from truthound.validators.base import Validator

pytestmark = pytest.mark.contract


def test_direct_dag_replacement_compatibility_and_explicit_instance_retention():
    from truthound.validators.optimization.orchestrator import ValidatorDAG

    first = get_validator("between")(columns=["x"], min_value=0)
    second = get_validator("between")(columns=["x"], max_value=0)
    legacy = ValidatorDAG()
    legacy.add_validators([first, second])
    assert len(legacy) == 1
    assert legacy.nodes[second.name].validator is second
    retained = ValidatorDAG(preserve_instances=True)
    retained.add_validators([first, second])
    assert [node.validator for node in retained.nodes.values()] == [first, second]
    execution = retained.build_execution_plan().execute(pl.DataFrame({"x": [-1, 1]}).lazy())
    assert execution.total_validators == 2
    assert len(execution.all_issues) == 2


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("names", [("lower", "upper"), ("same", "same")])
def test_repeated_algorithm_issues_belong_to_exact_suite_instances(parallel, names):
    frame = build_validation_asset(pl.DataFrame({"x": [-1, 1]}))
    suite = ValidationSuite(
        checks=(
            CheckSpec(
                id="minimum",
                name=names[0],
                category="distribution",
                factory=lambda: get_validator("between")(columns=["x"], min_value=0),
            ),
            CheckSpec(
                id="maximum",
                name=names[1],
                category="distribution",
                factory=lambda: get_validator("between")(columns=["x"], max_value=0),
            ),
        ),
        evidence_policy=EvidencePolicy.from_any("complete"),
    )
    run = ValidationRuntime().execute(
        asset=frame,
        plan=ScanPlanner().plan(suite=suite, asset=frame, parallel=parallel, max_workers=2),
    )
    assert len(run.checks) == 2 and len(run.issues) == 2
    assert [check.name for check in run.checks] == list(names)
    assert [check.metadata["_truthound_check_id"] for check in run.checks] == ["minimum", "maximum"]
    assert [check.issue_count for check in run.checks] == [1, 1]
    assert [check.issues[0].sample_values for check in run.checks] == [[-1], [1]]
    assert not any(check.success for check in run.checks)
    assert not run.execution_issues


class Broken(Validator):
    name = "synthetic_error"

    def validate(self, lf):
        raise ValueError("synthetic native error")


class BrokenPushdown(Broken):
    from truthound.validators.pushdown_support import PushdownLevel

    pushdown_level = PushdownLevel.FULL

    def get_pushdown_queries(self, table, columns, dialect):
        from truthound.validators.pushdown_support import PushdownQuery

        # Real SQLite execution fails; a silently missing result must not be
        # passed to a processor as if the SQL check had succeeded.
        return [
            PushdownQuery(
                sql=f"SELECT absent_column FROM {table}", column="x", check_type="missing"
            )
        ]

    def process_pushdown_results(self, results):
        raise AssertionError("Incomplete SQL results must not be processed")


@pytest.mark.parametrize("parallel", [False, True])
def test_same_display_name_does_not_spread_one_execution_failure_after_filter(parallel):
    frame = build_validation_asset(pl.DataFrame({"x": [1, 2]}))
    suite = ValidationSuite(
        checks=(
            CheckSpec(id="broken", name="shared", category="general", factory=Broken),
            CheckSpec(
                id="valid",
                name="shared",
                category="general",
                factory=lambda: get_validator("null")(columns=["x"]),
            ),
        )
    )
    run = ValidationRuntime().execute(
        asset=frame,
        plan=ScanPlanner().plan(suite=suite, asset=frame, parallel=parallel, max_workers=2),
    )
    assert [check.success for check in run.checks] == [False, True]
    assert len(run.execution_issues) == 1
    assert [check.success for check in run.filter_by_severity(Severity.CRITICAL).checks] == [
        False,
        True,
    ]
    assert not run.success


def test_suite_evidence_policy_runs_as_configured_without_mutating_factory_object():
    stored = get_validator("between")(columns=["x"], min_value=0, max_value=10)
    original_config = stored.config
    frame = build_validation_asset(pl.DataFrame({"x": [-1, 20, 20, 30]}))
    evidence = EvidencePolicy(
        ResultFormatConfig(
            format=ResultFormat.COMPLETE, partial_unexpected_count=1, include_unexpected_rows=True
        )
    )
    suite = ValidationSuite(
        checks=(
            CheckSpec(id="bounded", name="custom", category="distribution", factory=lambda: stored),
        ),
        evidence_policy=evidence,
    )
    run = ValidationRuntime().execute(
        asset=frame, plan=ScanPlanner().plan(suite=suite, asset=frame)
    )
    detail = run.checks[0].issues[0].result
    assert run.result_format == ResultFormat.COMPLETE
    assert detail.partial_unexpected_counts == [{"value": 20, "count": 2}]
    assert detail.unexpected_rows.height == 4
    assert stored.config == original_config


@pytest.mark.fault
@pytest.mark.parametrize("mode", ["regular", "failed_pushdown"])
def test_actual_sql_source_fallback_error_is_not_an_empty_success(tmp_path, mode):
    import sqlite3

    import truthound as th
    from truthound.context import TruthoundContext
    from truthound.datasources.sql.sqlite import SQLiteDataSource

    path = tmp_path / "source.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE facts (x TEXT)")
        # Preserve the native statistical minimum-sample requirement. A
        # one-row string fixture returns early and never exercises the error.
        connection.executemany("INSERT INTO facts VALUES (?)", [("not numeric",)] * 100)
    with SQLiteDataSource(database=str(path), query='SELECT * FROM "facts"') as source:
        run = th.check(
            source=source,
            validators=[
                get_validator("iqr_anomaly")(column="x") if mode == "regular" else BrokenPushdown()
            ],
            context=TruthoundContext(tmp_path / "direct"),
        )
    assert run.execution_issues and not run.success
    assert len(run.checks) == 1 and not run.checks[0].success
    assert not run.filter_by_severity(Severity.CRITICAL).success
