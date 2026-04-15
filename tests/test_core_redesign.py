import truthound as th
from truthound.core import ScanPlanner, ValidationRuntime, ValidationSuite, build_validation_asset
from truthound.core.results import CheckResult, ValidationRunResult
from truthound.types import ResultFormat, Severity
from truthound.validators.base import ValidationIssue


def test_check_returns_validation_run_result():
    run = th.check({'name': ['Alice', None], 'age': [1, 2]}, validators=['null'])

    assert isinstance(run, ValidationRunResult)
    assert run.source == 'dict'
    assert run.execution_mode in {'sequential', 'threadpool', 'parallel', 'pushdown'}
    assert run.planned_execution_mode in {'sequential', 'parallel', 'pushdown'}
    assert any(check.name == 'null' for check in run.checks)


def test_validation_suite_from_legacy_produces_specs():
    suite = ValidationSuite.from_legacy(validators=['null', 'unique'])

    assert [check.name for check in suite.checks] == ['null', 'unique']
    assert suite.evidence_policy.result_format.format.value == 'summary'


def test_scan_planner_prefers_parallel_when_requested():
    suite = ValidationSuite.from_legacy(validators=['null', 'unique'])
    asset = build_validation_asset({'id': [1, 2], 'value': [None, 2]})

    plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=True)

    assert plan.execution_mode == 'parallel'


def test_scan_planner_counts_duplicate_checks():
    suite = ValidationSuite.from_legacy(validators=['null', 'null'])
    asset = build_validation_asset({'name': ['Alice', None]})

    plan = ScanPlanner().plan(suite=suite, asset=asset)

    assert plan.duplicate_check_count == 1


def test_validation_run_result_filters_by_severity():
    low_issue = ValidationIssue(
        column='name',
        issue_type='null',
        count=1,
        severity=Severity.LOW,
        validator_name='null',
    )
    high_issue = ValidationIssue(
        column='age',
        issue_type='range',
        count=2,
        severity=Severity.HIGH,
        validator_name='range',
    )
    result = ValidationRunResult(
        suite_name='suite',
        source='dict',
        row_count=2,
        column_count=2,
        result_format=ResultFormat.SUMMARY,
        checks=(
            CheckResult(
                name='null',
                category='completeness',
                success=False,
                issue_count=1,
                issues=(low_issue,),
            ),
            CheckResult(
                name='range',
                category='validity',
                success=False,
                issue_count=1,
                issues=(high_issue,),
            ),
        ),
        issues=(low_issue, high_issue),
    )

    filtered = result.filter_by_severity(Severity.HIGH)

    assert [issue.validator_name for issue in filtered.issues] == ['range']
    assert [check.name for check in filtered.checks if check.issue_count > 0] == ['range']


def test_legacy_check_facade_matches_core_runtime():
    data = {'name': ['Alice', None], 'age': [21, 22]}
    suite = ValidationSuite.from_legacy(validators=['null'])
    asset = build_validation_asset(data)
    plan = ScanPlanner().plan(suite=suite, asset=asset)
    run_result = ValidationRuntime().execute(asset=asset, plan=plan)

    report = th.check(data, validators=['null'])

    expected = sorted(
        (issue.validator_name, issue.issue_type, issue.column, issue.count)
        for issue in run_result.issues
    )
    actual = sorted(
        (issue.validator_name, issue.issue_type, issue.column, issue.count)
        for issue in report.issues
    )

    assert actual == expected
    attached = report.to_dict()
    expected_run = run_result.to_dict()
    attached.pop('run_id', None)
    attached.pop('run_time', None)
    expected_run.pop('run_id', None)
    expected_run.pop('run_time', None)
    expected_metadata = expected_run.pop('metadata', {})
    attached_metadata = attached.pop('metadata', {})
    assert attached == expected_run
    for key, value in expected_metadata.items():
        assert attached_metadata.get(key) == value


def test_validation_runtime_keeps_sequential_as_actual_mode():
    suite = ValidationSuite.from_legacy(validators=["null", "null", "null", "null", "null"])
    asset = build_validation_asset({"id": [1, 2, 3], "email": [None, "a@example.com", "b@example.com"]})
    plan = ScanPlanner().plan(suite=suite, asset=asset)

    run_result = ValidationRuntime().execute(asset=asset, plan=plan)

    assert plan.planned_execution_mode == "sequential"
    assert run_result.planned_execution_mode == "sequential"
    assert run_result.execution_mode == "sequential"


def test_validation_runtime_reports_parallel_mode_when_requested():
    suite = ValidationSuite.from_legacy(validators=["null", "unique"])
    asset = build_validation_asset({"id": [1, 2], "email": [None, "a@example.com"]})
    plan = ScanPlanner().plan(suite=suite, asset=asset, parallel=True)

    run_result = ValidationRuntime().execute(asset=asset, plan=plan)

    assert plan.planned_execution_mode == "parallel"
    assert run_result.planned_execution_mode == "parallel"
    assert run_result.execution_mode == "parallel"
