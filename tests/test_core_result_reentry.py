"""Canonical re-entry must retain detail, exception and retry evidence."""

from pathlib import Path

import polars as pl
import pytest

import truthound as th
from truthound.context import TruthoundContext
from truthound.core.results import CheckResult, ExecutionIssue, ValidationRunResult
from truthound.types import ResultFormat, ResultFormatConfig, Severity, ValidationDetail
from truthound.validators import get_validator
from truthound.validators.base import ExceptionInfo, ValidationIssue

pytestmark = pytest.mark.contract


def test_actual_complete_core_result_survives_native_reentry_with_full_rows(tmp_path):
    run = th.check(
        pl.DataFrame({"age": [-1, 20, 200]}),
        validators=[get_validator("range")(columns=["age"])],
        result_format=ResultFormatConfig(
            format=ResultFormat.COMPLETE,
            include_unexpected_rows=True,
            include_unexpected_index=True,
        ),
        context=TruthoundContext(Path(tmp_path)),
    )
    saved = run.to_dict()
    restored = ValidationRunResult.from_dict(saved)
    assert restored.to_dict() == saved
    assert restored.issues[0].result is not None
    assert restored.issues[0].unexpected_rows.equals(run.issues[0].unexpected_rows)
    assert restored.issues[0].unexpected_percent == run.issues[0].unexpected_percent


@pytest.mark.fault
def test_exception_retry_partial_details_and_empty_samples_survive_reentry():
    info = ExceptionInfo(
        raised_exception=True,
        exception_type="TimeoutError",
        exception_message="synthetic",
        exception_traceback="synthetic captured trace",
        retry_count=2,
        max_retries=2,
        is_retryable=True,
        validator_name="synthetic",
        column="x",
        expression_alias="test",
        failure_category="transient",
    )
    issue = ValidationIssue(
        column="x",
        issue_type="validator_error",
        count=0,
        severity=Severity.LOW,
        validator_name="synthetic",
        sample_values=[],
        exception_info=info,
        result=ValidationDetail(element_count=3, unexpected_count=1, unexpected_list=[None]),
    )
    run = ValidationRunResult(
        suite_name="reentry",
        source="synthetic",
        row_count=3,
        column_count=1,
        checks=(CheckResult(name="synthetic", success=False, issue_count=1, issues=(issue,)),),
        issues=(issue,),
        execution_issues=(
            ExecutionIssue(check_name="synthetic", message="synthetic", retry_count=2),
        ),
    )
    restored = ValidationRunResult.from_dict(run.to_dict())
    assert restored.to_dict() == run.to_dict()
    assert restored.issues[0].sample_values == []
    assert restored.issues[0].exception_info == info
    assert not restored.filter_by_severity(Severity.CRITICAL).success


@pytest.mark.fault
def test_malformed_complete_failure_rows_are_rejected_instead_of_discarded():
    with pytest.raises(TypeError, match="row records"):
        ValidationDetail.from_dict({"unexpected_rows": ["not a row"]})
