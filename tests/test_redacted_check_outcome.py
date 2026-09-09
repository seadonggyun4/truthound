"""An explicit failed check remains failed after detail redaction."""

import pytest

from truthound.core.results import CheckResult, ValidationRunResult


def test_failed_check_without_issue_details_is_not_success():
    result = ValidationRunResult(
        suite_name="synthetic",
        source="synthetic",
        row_count=1,
        column_count=1,
        checks=(CheckResult(name="synthetic", success=False, issue_count=2),),
    )
    assert result.has_failures
    assert not result.success
    assert not ValidationRunResult.from_dict(result.to_dict()).success


def test_empty_or_successful_checks_keep_existing_success_semantics():
    assert ValidationRunResult.from_dict({}).success
    assert ValidationRunResult.from_dict(
        {"checks": [{"name": "synthetic", "success": True}]}
    ).success


@pytest.mark.parametrize("locale", ["en", "ko"])
def test_report_preserves_actual_check_name_and_escapes_it(locale):
    from truthound.datadocs import generate_validation_report

    result = ValidationRunResult.from_dict(
        {
            "checks": [{"name": "Synthetic <check>", "success": False, "issue_count": 2}],
        }
    )
    rendered = generate_validation_report(result, locale=locale)
    assert "<td>Synthetic &lt;check&gt;</td>" in rendered
    assert "<td>failed</td>" in rendered
    assert "<td>2</td>" in rendered
    assert "<check>" not in rendered


def test_execution_table_preserves_check_and_retry_count():
    from truthound.datadocs import generate_validation_report

    result = ValidationRunResult.from_dict(
        {
            "execution_issues": [
                {"check_name": "Synthetic execution", "message": "Unavailable", "retry_count": 3}
            ],
        }
    )
    rendered = generate_validation_report(result)
    assert "<td>Synthetic execution</td>" in rendered
    assert "<td>3</td>" in rendered
