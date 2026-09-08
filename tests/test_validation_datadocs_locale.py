"""Validation Data Docs must localize content, not only the HTML language tag."""

import pytest

from truthound import ValidationRunResult
from truthound.datadocs import ValidationDocsBuilder, generate_validation_report


def sample_result():
    return ValidationRunResult.from_dict({
        "passed": True, "row_count": 2, "column_count": 1,
        "run_time": "2026-01-02T03:04:05", "source": "Status",
        "checks": [], "issues": [],
    })


def test_korean_validation_content_is_localized_without_changing_result():
    result = sample_result()
    before = result.to_dict()
    rendered = generate_validation_report(result, locale="ko")
    assert '<html lang="ko">' in rendered
    for label in ("개요", "검사", "실행 오류", "상태", "행 수", "열 수", "데이터가 없습니다."):
        assert label in rendered
    assert "<h2>Overview</h2>" not in rendered
    assert "SUCCESS" in rendered
    assert "<code>Status</code>" in rendered  # Source content is not translated.
    assert result.to_dict() == before


def test_english_validation_default_is_unchanged():
    result = sample_result()
    assert generate_validation_report(result) == generate_validation_report(result, locale="en")


def test_builder_and_converter_use_the_same_korean_locale():
    rendered = ValidationDocsBuilder(locale="ko").build(sample_result())
    assert '<html lang="ko">' in rendered and "<h2>개요</h2>" in rendered


def test_korean_failure_alert_retains_canonical_issue_count_and_status():
    result = ValidationRunResult.from_dict({
        "passed": False, "row_count": 2, "column_count": 1,
        "issues": [{"column": "synthetic", "issue_type": "null_values",
                    "count": 1, "severity": "high", "validator_name": "synthetic_check"}],
    })
    before = result.to_dict()
    rendered = generate_validation_report(result, locale="ko", title="<script>synthetic</script>")
    assert "품질 문제가 발견되었습니다" in rendered
    assert "품질 문제 1건을 검토해야 합니다." in rendered
    assert "FAILURE" in rendered and "synthetic_check" in rendered
    assert "&lt;script&gt;synthetic&lt;/script&gt;" in rendered
    assert result.to_dict() == before


@pytest.mark.parametrize("locale", ["xx", "<script>", ""])
def test_unsupported_validation_locale_fails_explicitly(locale):
    with pytest.raises(ValueError, match="Unsupported validation report locale"):
        generate_validation_report(sample_result(), locale=locale)
