"""Tests for html_reporter module.

This module tests the HTML report generation bridge functionality
that connects the CLI's Report objects with the HTMLReporter system.
"""

from datetime import datetime
import tempfile
from pathlib import Path

import pytest

from truthound.html_reporter import (
    generate_html_report,
    write_html_report,
    generate_html_from_validation_result,
    generate_pii_html_report,
    write_pii_html_report,
    HTMLReportConfig,
    HAS_JINJA2,
)
from truthound.report import Report, PIIReport
from truthound.validators.base import ValidationIssue
from truthound.types import Severity
from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
)


# Skip all tests if jinja2 is not installed
pytestmark = pytest.mark.skipif(
    not HAS_JINJA2,
    reason="jinja2 is required for HTML report tests",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_report() -> Report:
    """Create a sample Report for testing."""
    issues = [
        ValidationIssue(
            column="email",
            issue_type="null_values",
            count=5,
            severity=Severity.HIGH,
            details="Found 5 null values in email column",
            expected="No null values",
            actual="5 null values found",
        ),
        ValidationIssue(
            column="age",
            issue_type="out_of_range",
            count=3,
            severity=Severity.MEDIUM,
            details="3 values outside valid range [0, 120]",
            expected="Values between 0 and 120",
            actual="Found values -5, 150, 999",
        ),
        ValidationIssue(
            column="status",
            issue_type="invalid_format",
            count=10,
            severity=Severity.LOW,
            details="10 values with invalid format",
        ),
    ]
    return Report(
        issues=issues,
        source="test_data.csv",
        row_count=1000,
        column_count=15,
    )


@pytest.fixture
def empty_report() -> Report:
    """Create an empty Report (no issues) for testing."""
    return Report(
        issues=[],
        source="clean_data.csv",
        row_count=500,
        column_count=10,
    )


@pytest.fixture
def sample_pii_report() -> PIIReport:
    """Create a sample PIIReport for testing."""
    findings = [
        {
            "column": "email",
            "pii_type": "EMAIL",
            "count": 950,
            "confidence": 98,
        },
        {
            "column": "phone",
            "pii_type": "PHONE_NUMBER",
            "count": 800,
            "confidence": 85,
        },
        {
            "column": "ssn",
            "pii_type": "SSN",
            "count": 100,
            "confidence": 95,
        },
    ]
    return PIIReport(
        findings=findings,
        source="customers.csv",
        row_count=1000,
    )


@pytest.fixture
def empty_pii_report() -> PIIReport:
    """Create a PIIReport with no findings."""
    return PIIReport(
        findings=[],
        source="clean_data.csv",
        row_count=500,
    )


@pytest.fixture
def sample_validation_result() -> ValidationResult:
    """Create a sample ValidationResult for testing."""
    return ValidationResult(
        run_id="test_run_001",
        run_time=datetime(2024, 1, 15, 10, 30, 0),
        data_asset="customers.csv",
        status=ResultStatus.FAILURE,
        results=[
            ValidatorResult(
                validator_name="null_check",
                success=False,
                column="email",
                issue_type="null_values",
                count=5,
                severity="high",
                message="Found 5 null values in email column",
            ),
            ValidatorResult(
                validator_name="range_check",
                success=False,
                column="age",
                issue_type="out_of_range",
                count=3,
                severity="medium",
                message="3 values outside valid range",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=3,
            passed_validators=1,
            failed_validators=2,
            total_rows=1000,
            total_columns=10,
            total_issues=2,
            high_issues=1,
            medium_issues=1,
            execution_time_ms=125.5,
        ),
        tags={"env": "production"},
        suite_name="customer_validation",
    )


# =============================================================================
# generate_html_report Tests
# =============================================================================


class TestGenerateHtmlReport:
    """Tests for generate_html_report function."""

    def test_basic_generation(self, sample_report: Report) -> None:
        """Test basic HTML report generation."""
        html = generate_html_report(sample_report)

        # Check basic HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

        # Check content is included
        assert "test_data.csv" in html
        assert "1,000" in html  # row count formatted

    def test_includes_issues(self, sample_report: Report) -> None:
        """Test that issues are included in the report."""
        html = generate_html_report(sample_report)

        assert "email" in html
        assert "null_values" in html
        assert "age" in html
        assert "out_of_range" in html

    def test_empty_report(self, empty_report: Report) -> None:
        """Test generation with no issues."""
        html = generate_html_report(empty_report)

        assert "<!DOCTYPE html>" in html
        assert "clean_data.csv" in html
        # Should show success status
        assert "Passed" in html or "success" in html.lower()

    def test_custom_title(self, sample_report: Report) -> None:
        """Test with custom title."""
        html = generate_html_report(sample_report, title="My Custom Report")

        assert "My Custom Report" in html

    def test_with_config(self, sample_report: Report) -> None:
        """Test with HTMLReportConfig."""
        config = HTMLReportConfig(
            title="Config Report",
            theme="light",
        )
        html = generate_html_report(sample_report, config=config)

        assert "Config Report" in html

    def test_severity_ordering(self, sample_report: Report) -> None:
        """Test that issues are sorted by severity."""
        html = generate_html_report(sample_report)

        # HIGH severity should appear before MEDIUM and LOW
        high_pos = html.find("high")
        medium_pos = html.find("medium")
        low_pos = html.find("low")

        # All severities should be present
        assert high_pos != -1
        assert medium_pos != -1
        assert low_pos != -1

    def test_statistics_included(self, sample_report: Report) -> None:
        """Test that statistics are included."""
        html = generate_html_report(sample_report)

        # Row count
        assert "1,000" in html
        # Column count
        assert "15" in html

    def test_footer_present(self, sample_report: Report) -> None:
        """Test that footer with Truthound attribution is present."""
        html = generate_html_report(sample_report)

        assert "Truthound" in html
        assert "Generated by" in html


# =============================================================================
# write_html_report Tests
# =============================================================================


class TestWriteHtmlReport:
    """Tests for write_html_report function."""

    def test_writes_to_file(self, sample_report: Report) -> None:
        """Test writing HTML to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            result_path = write_html_report(sample_report, output_path)

            assert result_path == output_path
            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content

    def test_writes_with_custom_title(self, sample_report: Report) -> None:
        """Test writing HTML with custom title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_report.html"

            write_html_report(sample_report, output_path, title="Custom Title")

            content = output_path.read_text(encoding="utf-8")
            assert "Custom Title" in content

    def test_writes_with_string_path(self, sample_report: Report) -> None:
        """Test writing HTML with string path instead of Path object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "report.html")

            result_path = write_html_report(sample_report, output_path)

            assert Path(output_path).exists()
            assert isinstance(result_path, Path)


# =============================================================================
# generate_html_from_validation_result Tests
# =============================================================================


class TestGenerateHtmlFromValidationResult:
    """Tests for generate_html_from_validation_result function."""

    def test_basic_generation(
        self, sample_validation_result: ValidationResult
    ) -> None:
        """Test basic HTML generation from ValidationResult."""
        html = generate_html_from_validation_result(sample_validation_result)

        assert "<!DOCTYPE html>" in html
        assert "customers.csv" in html

    def test_includes_run_info(
        self, sample_validation_result: ValidationResult
    ) -> None:
        """Test that run information is included."""
        html = generate_html_from_validation_result(sample_validation_result)

        assert "test_run_001" in html

    def test_with_config(
        self, sample_validation_result: ValidationResult
    ) -> None:
        """Test with HTMLReportConfig."""
        config = HTMLReportConfig(
            title="Direct Result Report",
        )
        html = generate_html_from_validation_result(
            sample_validation_result, config=config
        )

        assert "Direct Result Report" in html


# =============================================================================
# generate_pii_html_report Tests
# =============================================================================


class TestGeneratePiiHtmlReport:
    """Tests for generate_pii_html_report function."""

    def test_basic_generation(self, sample_pii_report: PIIReport) -> None:
        """Test basic PII HTML report generation."""
        html = generate_pii_html_report(sample_pii_report)

        assert "<!DOCTYPE html>" in html
        assert "customers.csv" in html

    def test_includes_findings(self, sample_pii_report: PIIReport) -> None:
        """Test that PII findings are included."""
        html = generate_pii_html_report(sample_pii_report)

        assert "email" in html
        assert "EMAIL" in html
        assert "phone" in html
        assert "PHONE_NUMBER" in html
        assert "ssn" in html
        assert "SSN" in html

    def test_includes_confidence(self, sample_pii_report: PIIReport) -> None:
        """Test that confidence levels are displayed."""
        html = generate_pii_html_report(sample_pii_report)

        assert "98%" in html
        assert "85%" in html
        assert "95%" in html

    def test_empty_findings(self, empty_pii_report: PIIReport) -> None:
        """Test generation with no PII findings."""
        html = generate_pii_html_report(empty_pii_report)

        assert "<!DOCTYPE html>" in html
        assert "No PII Found" in html
        assert "clean" in html  # Clean status badge class

    def test_custom_title(self, sample_pii_report: PIIReport) -> None:
        """Test with custom title."""
        html = generate_pii_html_report(
            sample_pii_report, title="PII Audit Report"
        )

        assert "PII Audit Report" in html

    def test_statistics_present(self, sample_pii_report: PIIReport) -> None:
        """Test that statistics are shown."""
        html = generate_pii_html_report(sample_pii_report)

        # Row count
        assert "1,000" in html
        # PII columns count (3 findings)
        assert "3" in html

    def test_confidence_styling(self, sample_pii_report: PIIReport) -> None:
        """Test that confidence levels have appropriate styling."""
        html = generate_pii_html_report(sample_pii_report)

        # High confidence (>=90) should have high class
        assert "confidence-high" in html
        # Medium confidence (>=70, <90) should have medium class
        assert "confidence-medium" in html


# =============================================================================
# write_pii_html_report Tests
# =============================================================================


class TestWritePiiHtmlReport:
    """Tests for write_pii_html_report function."""

    def test_writes_to_file(self, sample_pii_report: PIIReport) -> None:
        """Test writing PII HTML report to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pii_report.html"

            result_path = write_pii_html_report(sample_pii_report, output_path)

            assert result_path == output_path
            assert output_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "EMAIL" in content

    def test_writes_with_custom_title(
        self, sample_pii_report: PIIReport
    ) -> None:
        """Test writing PII HTML with custom title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_pii.html"

            write_pii_html_report(
                sample_pii_report, output_path, title="Custom PII Report"
            )

            content = output_path.read_text(encoding="utf-8")
            assert "Custom PII Report" in content


# =============================================================================
# HTMLReportConfig Tests
# =============================================================================


class TestHTMLReportConfig:
    """Tests for HTMLReportConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = HTMLReportConfig()

        assert config.title == "Truthound Validation Report"
        assert config.theme == "light"
        assert config.custom_css == ""
        assert config.include_metadata is True
        assert config.include_statistics is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = HTMLReportConfig(
            title="Custom Report",
            theme="dark",
            custom_css=".custom { color: red; }",
            include_metadata=False,
            include_statistics=False,
        )

        assert config.title == "Custom Report"
        assert config.theme == "dark"
        assert ".custom" in config.custom_css
        assert config.include_metadata is False
        assert config.include_statistics is False


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_report_with_special_characters(self) -> None:
        """Test handling of special characters in data."""
        issues = [
            ValidationIssue(
                column="description<script>",
                issue_type="injection_test",
                count=1,
                severity=Severity.HIGH,
                details="Test <html> & 'quotes' \"double\"",
            ),
        ]
        report = Report(
            issues=issues,
            source="file<with>special.csv",
            row_count=100,
            column_count=5,
        )

        html = generate_html_report(report)

        # HTML should be escaped
        assert "<!DOCTYPE html>" in html
        # The content should be properly escaped (no raw script tags)
        assert "<script>" not in html.split("<style>")[0]

    def test_report_with_very_large_count(self) -> None:
        """Test handling of very large issue counts."""
        issues = [
            ValidationIssue(
                column="col",
                issue_type="test",
                count=999_999_999,
                severity=Severity.LOW,
            ),
        ]
        report = Report(
            issues=issues,
            source="large.csv",
            row_count=1_000_000_000,
            column_count=100,
        )

        html = generate_html_report(report)

        assert "<!DOCTYPE html>" in html
        # Numbers should be formatted with commas
        assert "999,999,999" in html

    def test_report_with_unicode(self) -> None:
        """Test handling of Unicode characters."""
        issues = [
            ValidationIssue(
                column="名前",  # Japanese
                issue_type="유효성_검사",  # Korean
                count=10,
                severity=Severity.MEDIUM,
                details="Ümläut and émojis 🎉",
            ),
        ]
        report = Report(
            issues=issues,
            source="데이터.csv",
            row_count=100,
            column_count=5,
        )

        html = generate_html_report(report)

        assert "名前" in html
        assert "유효성_검사" in html

    def test_pii_report_with_low_confidence(self) -> None:
        """Test PII report with low confidence findings."""
        pii_report = PIIReport(
            findings=[
                {
                    "column": "notes",
                    "pii_type": "POTENTIAL_NAME",
                    "count": 50,
                    "confidence": 45,
                },
            ],
            source="data.csv",
            row_count=1000,
        )

        html = generate_pii_html_report(pii_report)

        assert "45%" in html
        assert "confidence-low" in html

    def test_report_with_critical_severity(self) -> None:
        """Test report with critical severity issues."""
        issues = [
            ValidationIssue(
                column="password",
                issue_type="exposed_credentials",
                count=1,
                severity=Severity.CRITICAL,
                details="Plain text password detected",
            ),
        ]
        report = Report(
            issues=issues,
            source="sensitive.csv",
            row_count=100,
            column_count=5,
        )

        html = generate_html_report(report)

        assert "critical" in html.lower()
