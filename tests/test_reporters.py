"""Tests for reporters module."""

from datetime import datetime
import tempfile
from pathlib import Path

import pytest

from truthound.reporters import get_reporter, ReporterError
from truthound.reporters.json_reporter import JSONReporter
from truthound.reporters.console_reporter import ConsoleReporter
from truthound.reporters.markdown_reporter import MarkdownReporter
from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_result() -> ValidationResult:
    """Create a sample validation result for testing."""
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
            ValidatorResult(
                validator_name="type_check",
                success=True,
                column="id",
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
        tags={"env": "production", "version": "1.0.0"},
        suite_name="customer_validation",
    )


@pytest.fixture
def success_result() -> ValidationResult:
    """Create a successful validation result for testing."""
    return ValidationResult(
        run_id="success_run_001",
        run_time=datetime(2024, 1, 15, 10, 30, 0),
        data_asset="orders.csv",
        status=ResultStatus.SUCCESS,
        results=[
            ValidatorResult(
                validator_name="null_check",
                success=True,
                column="order_id",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=1,
            passed_validators=1,
            failed_validators=0,
            total_rows=500,
            total_columns=8,
            total_issues=0,
        ),
    )


# =============================================================================
# JSONReporter Tests
# =============================================================================


class TestJSONReporter:
    """Tests for JSON reporter."""

    def test_render_basic(self, sample_result: ValidationResult) -> None:
        """Test basic JSON rendering."""
        reporter = JSONReporter()
        json_str = reporter.render(sample_result)

        assert '"run_id": "test_run_001"' in json_str
        assert '"data_asset": "customers.csv"' in json_str
        assert '"status": "failure"' in json_str

    def test_render_with_issues(self, sample_result: ValidationResult) -> None:
        """Test JSON includes issues."""
        reporter = JSONReporter()
        json_str = reporter.render(sample_result)

        assert '"issues"' in json_str
        assert '"null_values"' in json_str
        assert '"issue_count": 2' in json_str

    def test_render_compact(self, sample_result: ValidationResult) -> None:
        """Test compact JSON rendering."""
        reporter = JSONReporter()
        compact = reporter.render_compact(sample_result)

        # No newlines in compact output
        assert "\n" not in compact

    def test_render_without_null_values(self, sample_result: ValidationResult) -> None:
        """Test rendering without null values."""
        reporter = JSONReporter(include_null_values=False)
        json_str = reporter.render(sample_result)

        # Should not contain null values
        assert ": null" not in json_str

    def test_write_to_file(self, sample_result: ValidationResult) -> None:
        """Test writing JSON to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            reporter = JSONReporter()

            reporter.write(sample_result, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "test_run_001" in content

    def test_render_lines(self, sample_result: ValidationResult) -> None:
        """Test JSON Lines rendering."""
        reporter = JSONReporter()
        lines = reporter.render_lines(sample_result)

        # Should have multiple lines
        line_list = lines.strip().split("\n")
        assert len(line_list) >= 3  # header, issues, summary

        # First line should be header
        assert '"type": "header"' in line_list[0]

    def test_severity_counts(self, sample_result: ValidationResult) -> None:
        """Test severity counts in output."""
        reporter = JSONReporter()
        json_str = reporter.render(sample_result)

        assert '"issues_by_severity"' in json_str
        assert '"high": 1' in json_str
        assert '"medium": 1' in json_str


# =============================================================================
# ConsoleReporter Tests
# =============================================================================


class TestConsoleReporter:
    """Tests for console reporter."""

    def test_render_basic(self, sample_result: ValidationResult) -> None:
        """Test basic console rendering."""
        reporter = ConsoleReporter(color=False)
        output = reporter.render(sample_result)

        assert "customers.csv" in output
        assert "test_run_001" in output

    def test_render_failure_status(self, sample_result: ValidationResult) -> None:
        """Test failure status is shown."""
        reporter = ConsoleReporter(color=False)
        output = reporter.render(sample_result)

        assert "FAILED" in output or "Failed" in output

    def test_render_success_status(self, success_result: ValidationResult) -> None:
        """Test success status is shown."""
        reporter = ConsoleReporter(color=False)
        output = reporter.render(success_result)

        assert "PASSED" in output or "Passed" in output

    def test_render_compact(self, sample_result: ValidationResult) -> None:
        """Test compact console rendering."""
        reporter = ConsoleReporter(color=False, compact=True)
        output = reporter.render(sample_result)

        # Compact output should be shorter
        assert len(output) < 500

    def test_render_with_issues_table(self, sample_result: ValidationResult) -> None:
        """Test issues table is included."""
        reporter = ConsoleReporter(color=False)
        output = reporter.render(sample_result)

        assert "email" in output
        assert "null_values" in output

    def test_render_without_header(self, sample_result: ValidationResult) -> None:
        """Test rendering without header."""
        reporter = ConsoleReporter(color=False, show_header=False)
        output = reporter.render(sample_result)

        # Should not have full header panel
        assert "Truthound Report" not in output or len(output) < 200


# =============================================================================
# MarkdownReporter Tests
# =============================================================================


class TestMarkdownReporter:
    """Tests for Markdown reporter."""

    def test_render_basic(self, sample_result: ValidationResult) -> None:
        """Test basic Markdown rendering."""
        reporter = MarkdownReporter()
        md = reporter.render(sample_result)

        assert "# " in md  # Has headings
        assert "customers.csv" in md
        assert "test_run_001" in md

    def test_render_with_badges(self, sample_result: ValidationResult) -> None:
        """Test badges are included."""
        reporter = MarkdownReporter(include_badges=True)
        md = reporter.render(sample_result)

        assert "img.shields.io" in md

    def test_render_without_badges(self, sample_result: ValidationResult) -> None:
        """Test rendering without badges."""
        reporter = MarkdownReporter(include_badges=False)
        md = reporter.render(sample_result)

        assert "img.shields.io" not in md

    def test_render_with_toc(self, sample_result: ValidationResult) -> None:
        """Test table of contents is included."""
        reporter = MarkdownReporter(include_toc=True)
        md = reporter.render(sample_result)

        assert "Table of Contents" in md
        assert "[Overview]" in md

    def test_render_without_toc(self, sample_result: ValidationResult) -> None:
        """Test rendering without TOC."""
        reporter = MarkdownReporter(include_toc=False)
        md = reporter.render(sample_result)

        assert "Table of Contents" not in md

    def test_render_issues_table(self, sample_result: ValidationResult) -> None:
        """Test issues table is rendered."""
        reporter = MarkdownReporter()
        md = reporter.render(sample_result)

        # Should have table headers
        assert "| Column |" in md
        assert "| Issue Type |" in md

    def test_render_statistics(self, sample_result: ValidationResult) -> None:
        """Test statistics are rendered."""
        reporter = MarkdownReporter()
        md = reporter.render(sample_result)

        assert "1,000" in md or "1000" in md  # Total rows
        assert "Statistics" in md

    def test_write_to_file(self, sample_result: ValidationResult) -> None:
        """Test writing Markdown to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            reporter = MarkdownReporter()

            reporter.write(sample_result, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "customers.csv" in content


# =============================================================================
# HTMLReporter Tests
# =============================================================================


# Check if jinja2 is available at module level
try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


@pytest.mark.skipif(not HAS_JINJA2, reason="jinja2 not installed")
class TestHTMLReporter:
    """Tests for HTML reporter."""

    def test_render_basic(self, sample_result: ValidationResult) -> None:
        """Test basic HTML rendering."""
        from truthound.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        html = reporter.render(sample_result)

        assert "<!DOCTYPE html>" in html
        assert "customers.csv" in html
        assert "test_run_001" in html

    def test_render_includes_styles(self, sample_result: ValidationResult) -> None:
        """Test styles are included."""
        from truthound.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        html = reporter.render(sample_result)

        assert "<style>" in html
        assert "font-family" in html

    def test_render_failure_status(self, sample_result: ValidationResult) -> None:
        """Test failure status styling."""
        from truthound.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        html = reporter.render(sample_result)

        assert "failure" in html.lower()

    def test_render_success_status(self, success_result: ValidationResult) -> None:
        """Test success status styling."""
        from truthound.reporters.html_reporter import HTMLReporter

        reporter = HTMLReporter()
        html = reporter.render(success_result)

        assert "success" in html.lower() or "passed" in html.lower()

    def test_write_to_file(self, sample_result: ValidationResult) -> None:
        """Test writing HTML to file."""
        from truthound.reporters.html_reporter import HTMLReporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            reporter = HTMLReporter()

            reporter.write(sample_result, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content


# =============================================================================
# Factory Tests
# =============================================================================


class TestReporterFactory:
    """Tests for reporter factory."""

    def test_get_json_reporter(self) -> None:
        """Test getting JSON reporter."""
        reporter = get_reporter("json")
        assert isinstance(reporter, JSONReporter)

    def test_get_console_reporter(self) -> None:
        """Test getting console reporter."""
        reporter = get_reporter("console")
        assert isinstance(reporter, ConsoleReporter)

    def test_get_markdown_reporter(self) -> None:
        """Test getting Markdown reporter."""
        reporter = get_reporter("markdown")
        assert isinstance(reporter, MarkdownReporter)

    def test_unknown_format_error(self) -> None:
        """Test error for unknown format."""
        with pytest.raises(ReporterError) as exc_info:
            get_reporter("unknown_format")

        assert "Unknown reporter format" in str(exc_info.value)

    def test_factory_with_kwargs(self) -> None:
        """Test factory passes kwargs to reporter."""
        reporter = get_reporter("json", indent=4)
        assert reporter.config.indent == 4

    def test_case_insensitive(self) -> None:
        """Test format names are case-insensitive."""
        reporter1 = get_reporter("JSON")
        reporter2 = get_reporter("json")
        reporter3 = get_reporter("Json")

        assert type(reporter1) == type(reporter2) == type(reporter3)


# =============================================================================
# Integration Tests
# =============================================================================


class TestReporterIntegration:
    """Integration tests for reporters."""

    def test_all_formats_render(self, sample_result: ValidationResult) -> None:
        """Test all formats can render same result."""
        formats = ["json", "console", "markdown"]

        for fmt in formats:
            reporter = get_reporter(fmt)
            output = reporter.render(sample_result)
            assert len(output) > 0

    def test_generate_filename(self, sample_result: ValidationResult) -> None:
        """Test filename generation."""
        reporter = JSONReporter()
        filename = reporter.generate_filename(sample_result)

        assert filename.endswith(".json")
        assert "json_" in filename

    def test_report_method(self, sample_result: ValidationResult) -> None:
        """Test report convenience method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            reporter = JSONReporter()

            content = reporter.report(sample_result, output_path)

            assert output_path.exists()
            assert len(content) > 0

    def test_write_error_no_path(self, sample_result: ValidationResult) -> None:
        """Test WriteError when no path specified."""
        from truthound.reporters.base import WriteError

        reporter = JSONReporter()

        with pytest.raises(WriteError):
            reporter.write(sample_result)

    def test_config_output_path(self, sample_result: ValidationResult) -> None:
        """Test using config output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "configured_report.json"
            reporter = JSONReporter(output_path=str(output_path))

            reporter.write(sample_result)

            assert output_path.exists()
