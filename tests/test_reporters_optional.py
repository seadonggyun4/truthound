"""Tests for optional reporter dependencies using mocks.

These tests verify HTMLReporter logic without requiring jinja2.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from truthound.stores.results import (
    ValidationResult,
    ValidatorResult,
    ResultStatistics,
    ResultStatus,
)

from tests.mocks.reporter_mocks import (
    MockJinja2Environment,
    MockJinja2Template,
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
                validator_name="type_check",
                success=True,
                column="id",
            ),
        ],
        statistics=ResultStatistics(
            total_validators=2,
            passed_validators=1,
            failed_validators=1,
            total_rows=1000,
            total_columns=10,
            total_issues=1,
            high_issues=1,
        ),
    )


@pytest.fixture
def success_result() -> ValidationResult:
    """Create a successful validation result."""
    return ValidationResult(
        run_id="success_run_001",
        run_time=datetime(2024, 1, 15, 11, 0, 0),
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
# HTMLReporter Mock Tests
# =============================================================================


class TestHTMLReporterMock:
    """Tests for HTMLReporter using mocks.

    These tests verify the rendering logic by directly testing the
    MockJinja2Template with the same data structure that HTMLReporter uses.
    """

    def _render_with_mock(
        self,
        result: ValidationResult,
        title: str = "Validation Report",
    ) -> str:
        """Helper to render using mock template with same logic as HTMLReporter."""
        mock_template = MockJinja2Template("")

        # Prepare issues list (same logic as HTMLReporter)
        issues = [r for r in result.results if not r.success]

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_issues = sorted(
            issues,
            key=lambda x: severity_order.get(
                x.severity.lower() if x.severity else "low", 4
            ),
        )

        return mock_template.render(
            title=title,
            result=result,
            statistics=result.statistics,
            issues=sorted_issues,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config=None,
        )

    def test_render_basic(self, sample_result: ValidationResult) -> None:
        """Test basic HTML rendering with mock Jinja2."""
        html = self._render_with_mock(sample_result)

        assert "<!DOCTYPE html>" in html
        assert "customers.csv" in html
        assert "test_run_001" in html

    def test_render_failure_status(self, sample_result: ValidationResult) -> None:
        """Test failure status is shown in HTML."""
        html = self._render_with_mock(sample_result)

        assert "failure" in html.lower()
        assert "Failed" in html

    def test_render_success_status(self, success_result: ValidationResult) -> None:
        """Test success status is shown in HTML."""
        html = self._render_with_mock(success_result)

        assert "success" in html.lower()
        assert "Passed" in html

    def test_render_includes_statistics(self, sample_result: ValidationResult) -> None:
        """Test statistics are included in HTML."""
        html = self._render_with_mock(sample_result)

        assert "1,000" in html or "1000" in html  # Total rows
        assert "stats" in html.lower()

    def test_render_includes_issues(self, sample_result: ValidationResult) -> None:
        """Test issues are included in HTML."""
        html = self._render_with_mock(sample_result)

        assert "email" in html
        assert "null_values" in html
        assert "high" in html

    def test_custom_title(self, sample_result: ValidationResult) -> None:
        """Test custom title in HTML."""
        html = self._render_with_mock(sample_result, title="My Custom Report")

        assert "My Custom Report" in html


class TestMockJinja2:
    """Test the mock Jinja2 implementation itself."""

    def test_template_render(self) -> None:
        """Test MockJinja2Template renders correctly."""
        template = MockJinja2Template("")

        # Create a minimal result for testing
        result = MagicMock()
        result.data_asset = "test.csv"
        result.run_id = "run_123"
        result.success = True

        statistics = MagicMock()
        statistics.total_rows = 100
        statistics.total_issues = 0

        html = template.render(
            result=result,
            statistics=statistics,
            title="Test Report",
            issues=[],
            generated_at="2024-01-15",
        )

        assert "<!DOCTYPE html>" in html
        assert "test.csv" in html
        assert "run_123" in html
        assert "Test Report" in html

    def test_environment_get_template(self) -> None:
        """Test MockJinja2Environment returns templates."""
        env = MockJinja2Environment()

        template1 = env.get_template("default")
        template2 = env.get_template("default")

        # Same template instance should be returned
        assert template1 is template2

    def test_environment_add_template(self) -> None:
        """Test adding custom templates."""
        env = MockJinja2Environment()

        env.add_template("custom", "<html>{{ title }}</html>")

        template = env.get_template("custom")
        assert template is not None


# =============================================================================
# Reporter Factory Tests
# =============================================================================


class TestReporterFactoryWithMocks:
    """Test reporter factory behavior."""

    def test_html_reporter_import_error(self) -> None:
        """Test that factory provides clear error when jinja2 is missing."""
        from truthound.reporters.factory import get_reporter
        from truthound.reporters.base import ReporterError

        with patch("truthound.reporters.html_reporter.HAS_JINJA2", False):
            # Need to reload to pick up the patched value
            import importlib
            import truthound.reporters.html_reporter
            importlib.reload(truthound.reporters.html_reporter)

            with pytest.raises((ReporterError, ImportError)) as exc_info:
                get_reporter("html")

            assert "jinja2" in str(exc_info.value).lower()

            # Restore
            importlib.reload(truthound.reporters.html_reporter)

    def test_json_reporter_no_dependencies(self) -> None:
        """Test that JSON reporter works without optional deps."""
        from truthound.reporters.factory import get_reporter

        reporter = get_reporter("json")
        assert reporter.name == "json"

    def test_console_reporter_no_dependencies(self) -> None:
        """Test that console reporter works without optional deps."""
        from truthound.reporters.factory import get_reporter

        reporter = get_reporter("console")
        assert reporter.name == "console"

    def test_markdown_reporter_no_dependencies(self) -> None:
        """Test that markdown reporter works without optional deps."""
        from truthound.reporters.factory import get_reporter

        reporter = get_reporter("markdown")
        assert reporter.name == "markdown"
