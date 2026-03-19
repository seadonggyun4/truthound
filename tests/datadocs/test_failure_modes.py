from __future__ import annotations

from datetime import datetime

import pytest

from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.datadocs.engine.context import ReportContext, ReportData
from truthound.datadocs.engine.pipeline import ReportPipeline
from truthound.datadocs.engine.registry import ComponentRegistry
from truthound.datadocs.validation import ValidationDataConverter, ValidationDocsBuilder
from truthound.types import ResultFormat, Severity
from truthound.validators.base import ValidationIssue


pytestmark = pytest.mark.contract


def _build_run_result(
    *,
    source: str = "warehouse/orders.csv",
    issue_message: str = "Column contained <script>alert(1)</script>",
    execution_issue_only: bool = False,
    metadata: dict[str, object] | None = None,
) -> ValidationRunResult:
    issue = ValidationIssue(
        column="email",
        issue_type="null_values",
        count=2,
        severity=Severity.HIGH,
        details=issue_message,
        validator_name="null_check",
        sample_values=["", "<script>alert(1)</script>"],
    )
    checks = ()
    issues = ()
    if not execution_issue_only:
        checks = (
            CheckResult(
                name="null_check",
                category="completeness",
                success=False,
                issue_count=1,
                issues=(issue,),
            ),
        )
        issues = (issue,)

    return ValidationRunResult(
        run_id="run_20260319_120000_docs",
        run_time=datetime(2026, 3, 19, 12, 0, 0),
        suite_name="orders_suite",
        source=source,
        row_count=50,
        column_count=3,
        result_format=ResultFormat.SUMMARY,
        execution_mode="parallel",
        checks=checks,
        issues=issues,
        execution_issues=(
            ExecutionIssue(
                check_name="schema_check",
                message="schema loader crashed",
                exception_type="RuntimeError",
                failure_category="partial_failure",
                retry_count=2,
            ),
        ),
        metadata=metadata or {},
    )


@pytest.mark.fault
def test_validation_docs_builder_renders_execution_issue_only_runs():
    run_result = _build_run_result(execution_issue_only=True)

    html = ValidationDocsBuilder().build(run_result, title="Ops Run")

    assert "Ops Run" in html
    assert "Execution Issues" in html
    assert "schema loader crashed" in html
    assert "0" in html


@pytest.mark.fault
def test_validation_docs_builder_escapes_hostile_payloads():
    run_result = _build_run_result(source="<script>orders</script>")

    html = ValidationDocsBuilder().build(run_result, title="Escaping Audit")

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;script&gt;orders&lt;/script&gt;" in html


@pytest.mark.fault
def test_validation_data_converter_survives_missing_runtime_metadata():
    run_result = _build_run_result(metadata={})

    context = ValidationDataConverter(run_result).build_context()
    rows = dict(context.data.sections["metadata"]["rows"])

    assert rows["Runtime Environment"] == "-"
    assert rows["Execution Issues"] == "1"


@pytest.mark.fault
def test_report_pipeline_returns_failed_result_when_transformer_crashes():
    class BrokenTransformer:
        def transform(self, ctx: ReportContext) -> ReportContext:
            raise RuntimeError("transform blew up")

    ctx = ReportContext(
        data=ReportData(sections={"overview": {"row_count": 10}}),
    )

    result = ReportPipeline().transform(BrokenTransformer()).generate(ctx)

    assert result.success is False
    assert result.error == "transform blew up"
    assert result.content == ""


@pytest.mark.fault
def test_component_registry_missing_renderer_reports_available_choices():
    registry = ComponentRegistry()

    class StableRenderer:
        def render(self, ctx: ReportContext, theme: object | None = None) -> str:
            return "<p>ok</p>"

    registry.register_renderer("stable", StableRenderer)

    with pytest.raises(KeyError) as exc_info:
        registry.get_renderer("missing")

    message = str(exc_info.value)
    assert "missing" in message
    assert "stable" in message
