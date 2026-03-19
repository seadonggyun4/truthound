import json
from datetime import datetime

from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.datadocs import ValidationDocsBuilder, generate_validation_report
from truthound.html_reporter import generate_html_from_validation_result
from truthound.reporters.json_reporter import JSONReporter
from truthound.reporters.markdown_reporter import MarkdownReporter
from truthound.stores.results import ValidationResult
from truthound.types import ResultFormat, Severity
from truthound.validators.base import ValidationIssue


def _sample_run_result() -> ValidationRunResult:
    null_issue = ValidationIssue(
        column="email",
        issue_type="null_values",
        count=5,
        severity=Severity.HIGH,
        details="Found 5 null values in email column",
        validator_name="null_check",
        sample_values=["", None],
    )
    range_issue = ValidationIssue(
        column="age",
        issue_type="out_of_range",
        count=2,
        severity=Severity.MEDIUM,
        details="Found 2 values outside the expected range",
        validator_name="range_check",
    )
    return ValidationRunResult(
        run_id="run_20260319_120000_sample",
        run_time=datetime(2026, 3, 19, 12, 0, 0),
        suite_name="customer_suite",
        source="customers.csv",
        row_count=100,
        column_count=4,
        result_format=ResultFormat.SUMMARY,
        execution_mode="parallel",
        checks=(
            CheckResult(
                name="null_check",
                category="completeness",
                success=False,
                issue_count=1,
                issues=(null_issue,),
            ),
            CheckResult(
                name="range_check",
                category="validity",
                success=False,
                issue_count=1,
                issues=(range_issue,),
            ),
        ),
        issues=(null_issue, range_issue),
        execution_issues=(
            ExecutionIssue(
                check_name="schema_check",
                message="Schema introspection retried once",
                exception_type="TimeoutError",
                failure_category="transient",
                retry_count=1,
            ),
        ),
        metadata={
            "execution_time_ms": 125.5,
            "tags": {"env": "test"},
            "runtime_environment": {"backend": "polars"},
        },
    )


def test_json_reporter_accepts_validation_run_result_directly():
    run_result = _sample_run_result()
    reporter = JSONReporter()

    rendered = json.loads(reporter.render(run_result))

    assert rendered["result"]["run_id"] == run_result.run_id
    assert rendered["result"]["data_asset"] == "customers.csv"
    assert rendered["statistics"]["total_validators"] == 2
    assert rendered["issues"][0]["validator_name"] == "null_check"


def test_persistence_result_round_trips_through_validation_run_result():
    run_result = _sample_run_result()

    stored = ValidationResult.from_validation_run_result(run_result)
    restored = stored.to_validation_run_result()

    assert restored.to_dict() == run_result.to_dict()


def test_markdown_and_html_helpers_support_direct_result_model():
    run_result = _sample_run_result()

    markdown = MarkdownReporter().render(run_result)
    html = generate_html_from_validation_result(run_result)

    assert "customers.csv" in markdown
    assert "null_values" in markdown
    assert "customers.csv" in html
    assert "out_of_range" in html


def test_validation_datadocs_builder_renders_validation_run_result():
    run_result = _sample_run_result()

    html = ValidationDocsBuilder().build(run_result, title="Validation Data Docs")

    assert "Validation Data Docs" in html
    assert "customers.csv" in html
    assert "Execution Issues" in html
    assert "null_check" in html


def test_generate_validation_report_accepts_direct_run_result():
    run_result = _sample_run_result()

    html = generate_validation_report(run_result, title="Quality Overview")

    assert "Quality Overview" in html
    assert run_result.run_id in html
