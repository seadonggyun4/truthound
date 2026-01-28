"""Tests for Quality CLI commands."""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
import typer

from truthound.cli_modules.advanced.quality import app


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_scores_data():
    """Sample quality scores data."""
    return {
        "scores": [
            {
                "rule_name": "email_format",
                "rule_type": "pattern",
                "column": "email",
                "metrics": {
                    "precision": 0.96,
                    "recall": 0.95,
                    "f1_score": 0.955,
                    "accuracy": 0.97,
                    "confidence": 0.9,
                    "quality_level": "excellent",
                },
                "recommendation": "Excellent quality. Safe to use.",
                "should_use": True,
            },
            {
                "rule_name": "age_range",
                "rule_type": "range",
                "column": "age",
                "metrics": {
                    "precision": 0.88,
                    "recall": 0.86,
                    "f1_score": 0.87,
                    "accuracy": 0.89,
                    "confidence": 0.85,
                    "quality_level": "good",
                },
                "recommendation": "Good quality. Recommended for use.",
                "should_use": True,
            },
            {
                "rule_name": "phone_pattern",
                "rule_type": "pattern",
                "column": "phone",
                "metrics": {
                    "precision": 0.55,
                    "recall": 0.52,
                    "f1_score": 0.534,
                    "accuracy": 0.6,
                    "confidence": 0.5,
                    "quality_level": "poor",
                },
                "recommendation": "Poor quality. Consider improvements.",
                "should_use": False,
            },
        ],
        "count": 3,
    }


@pytest.fixture
def scores_file(tmp_path, sample_scores_data):
    """Create a temporary scores file."""
    file_path = tmp_path / "scores.json"
    file_path.write_text(json.dumps(sample_scores_data, indent=2))
    return file_path


# =============================================================================
# Test Report Command
# =============================================================================


class TestReportCommand:
    """Tests for the report command."""

    def test_report_console(self, runner, scores_file):
        """Test console report generation."""
        result = runner.invoke(app, ["report", str(scores_file)])

        assert result.exit_code == 0
        assert "email_format" in result.output or "Quality" in result.output

    def test_report_json(self, runner, scores_file, tmp_path):
        """Test JSON report generation."""
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "json",
            "-o", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert "scores" in data

    def test_report_html(self, runner, scores_file, tmp_path):
        """Test HTML report generation."""
        output_file = tmp_path / "report.html"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "html",
            "-o", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_report_markdown(self, runner, scores_file, tmp_path):
        """Test Markdown report generation."""
        output_file = tmp_path / "report.md"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "markdown",
            "-o", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()

    def test_report_with_min_level(self, runner, scores_file, tmp_path):
        """Test report with minimum level filter."""
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "json",
            "-o", str(output_file),
            "--min-level", "good",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 2

    def test_report_with_min_f1(self, runner, scores_file, tmp_path):
        """Test report with minimum F1 filter."""
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "json",
            "-o", str(output_file),
            "--min-f1", "0.8",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 2

    def test_report_should_use_only(self, runner, scores_file, tmp_path):
        """Test report with should_use_only filter."""
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "json",
            "-o", str(output_file),
            "--should-use-only",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 2
        assert all(s["should_use"] for s in data["scores"])

    def test_report_with_max_scores(self, runner, scores_file, tmp_path):
        """Test report with max scores limit."""
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "json",
            "-o", str(output_file),
            "--max", "1",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 1

    def test_report_with_title(self, runner, scores_file, tmp_path):
        """Test report with custom title."""
        output_file = tmp_path / "report.html"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "html",
            "-o", str(output_file),
            "--title", "Custom Report Title",
        ])

        assert result.exit_code == 0

        content = output_file.read_text()
        assert "Custom Report Title" in content

    def test_report_file_not_found(self, runner, tmp_path):
        """Test report with nonexistent file."""
        result = runner.invoke(app, [
            "report", str(tmp_path / "nonexistent.json"),
        ])

        # Should fail gracefully
        assert result.exit_code != 0 or "not found" in result.output.lower() or "error" in result.output.lower()


# =============================================================================
# Test Filter Command
# =============================================================================


class TestFilterCommand:
    """Tests for the filter command."""

    def test_filter_by_min_level(self, runner, scores_file, tmp_path):
        """Test filtering by minimum level."""
        output_file = tmp_path / "filtered.json"
        result = runner.invoke(app, [
            "filter", str(scores_file),
            "-o", str(output_file),
            "--min-level", "good",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 2

    def test_filter_by_max_level(self, runner, scores_file, tmp_path):
        """Test filtering by maximum level."""
        output_file = tmp_path / "filtered.json"
        result = runner.invoke(app, [
            "filter", str(scores_file),
            "-o", str(output_file),
            "--max-level", "good",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 2

    def test_filter_by_f1_range(self, runner, scores_file, tmp_path):
        """Test filtering by F1 score range."""
        output_file = tmp_path / "filtered.json"
        result = runner.invoke(app, [
            "filter", str(scores_file),
            "-o", str(output_file),
            "--min-f1", "0.5",
            "--max-f1", "0.9",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        for score in data["scores"]:
            f1 = score["metrics"]["f1_score"]
            assert 0.5 <= f1 <= 0.9

    def test_filter_by_columns(self, runner, scores_file, tmp_path):
        """Test filtering by columns."""
        output_file = tmp_path / "filtered.json"
        result = runner.invoke(app, [
            "filter", str(scores_file),
            "-o", str(output_file),
            "--columns", "email,age",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 2
        columns = [s["column"] for s in data["scores"]]
        assert all(c in ["email", "age"] for c in columns)

    def test_filter_by_rule_types(self, runner, scores_file, tmp_path):
        """Test filtering by rule types."""
        output_file = tmp_path / "filtered.json"
        result = runner.invoke(app, [
            "filter", str(scores_file),
            "-o", str(output_file),
            "--rule-types", "pattern",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        assert len(data["scores"]) == 2
        assert all(s["rule_type"] == "pattern" for s in data["scores"])

    def test_filter_inverted(self, runner, scores_file, tmp_path):
        """Test inverted filter."""
        output_file = tmp_path / "filtered.json"
        result = runner.invoke(app, [
            "filter", str(scores_file),
            "-o", str(output_file),
            "--min-level", "good",
            "--invert",
        ])

        assert result.exit_code == 0

        data = json.loads(output_file.read_text())
        # Only poor rule should remain
        assert len(data["scores"]) == 1
        assert data["scores"][0]["metrics"]["quality_level"] == "poor"

    def test_filter_display_only(self, runner, scores_file):
        """Test filter with display only (no output file)."""
        result = runner.invoke(app, [
            "filter", str(scores_file),
            "--min-level", "good",
        ])

        assert result.exit_code == 0
        assert "Filtered" in result.output


# =============================================================================
# Test Compare Command
# =============================================================================


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_default(self, runner, scores_file):
        """Test compare with default settings."""
        result = runner.invoke(app, ["compare", str(scores_file)])

        assert result.exit_code == 0
        assert "email_format" in result.output

    def test_compare_by_precision(self, runner, scores_file):
        """Test compare sorted by precision."""
        result = runner.invoke(app, [
            "compare", str(scores_file),
            "--sort-by", "precision",
        ])

        assert result.exit_code == 0

    def test_compare_ascending(self, runner, scores_file):
        """Test compare with ascending order."""
        result = runner.invoke(app, [
            "compare", str(scores_file),
            "--asc",
        ])

        assert result.exit_code == 0

    def test_compare_with_max(self, runner, scores_file):
        """Test compare with max limit."""
        result = runner.invoke(app, [
            "compare", str(scores_file),
            "--max", "1",
        ])

        assert result.exit_code == 0

    def test_compare_group_by_column(self, runner, scores_file):
        """Test compare grouped by column."""
        result = runner.invoke(app, [
            "compare", str(scores_file),
            "--group-by", "column",
        ])

        assert result.exit_code == 0
        assert "Column" in result.output or "column" in result.output.lower()

    def test_compare_group_by_level(self, runner, scores_file):
        """Test compare grouped by level."""
        result = runner.invoke(app, [
            "compare", str(scores_file),
            "--group-by", "level",
        ])

        assert result.exit_code == 0

    def test_compare_with_output(self, runner, scores_file, tmp_path):
        """Test compare with output file."""
        output_file = tmp_path / "comparison.json"
        result = runner.invoke(app, [
            "compare", str(scores_file),
            "-o", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()


# =============================================================================
# Test Summary Command
# =============================================================================


class TestSummaryCommand:
    """Tests for the summary command."""

    def test_summary_basic(self, runner, scores_file):
        """Test basic summary."""
        result = runner.invoke(app, ["summary", str(scores_file)])

        assert result.exit_code == 0
        assert "Total" in result.output or "total" in result.output.lower()

    def test_summary_shows_levels(self, runner, scores_file):
        """Test summary shows quality levels."""
        result = runner.invoke(app, ["summary", str(scores_file)])

        assert result.exit_code == 0
        # Should show level distribution
        assert any(level in result.output.lower() for level in ["excellent", "good", "poor"])

    def test_summary_shows_metrics(self, runner, scores_file):
        """Test summary shows metric averages."""
        result = runner.invoke(app, ["summary", str(scores_file)])

        assert result.exit_code == 0
        # Should show metric averages
        assert "F1" in result.output or "f1" in result.output.lower() or "Metric" in result.output


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_scores_file(self, runner, tmp_path):
        """Test with empty scores file."""
        empty_file = tmp_path / "empty.json"
        empty_file.write_text(json.dumps({"scores": [], "count": 0}))

        result = runner.invoke(app, ["report", str(empty_file)])

        assert result.exit_code == 0
        assert "No" in result.output or "no" in result.output.lower()

    def test_invalid_json_file(self, runner, tmp_path):
        """Test with invalid JSON file."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {")

        result = runner.invoke(app, ["report", str(invalid_file)])

        # Should fail gracefully
        assert result.exit_code != 0 or "Invalid" in result.output or "error" in result.output.lower()

    def test_single_score(self, runner, tmp_path):
        """Test with single score."""
        single_score = {
            "scores": [
                {
                    "rule_name": "test_rule",
                    "rule_type": "pattern",
                    "column": "test",
                    "metrics": {
                        "precision": 0.9,
                        "recall": 0.9,
                        "f1_score": 0.9,
                        "accuracy": 0.9,
                        "confidence": 0.9,
                        "quality_level": "good",
                    },
                    "recommendation": "Test",
                    "should_use": True,
                },
            ],
        }
        file_path = tmp_path / "single.json"
        file_path.write_text(json.dumps(single_score))

        result = runner.invoke(app, ["report", str(file_path)])

        assert result.exit_code == 0

    def test_report_creates_parent_dirs(self, runner, scores_file, tmp_path):
        """Test that report creates parent directories."""
        output_file = tmp_path / "nested" / "dir" / "report.json"
        result = runner.invoke(app, [
            "report", str(scores_file),
            "-f", "json",
            "-o", str(output_file),
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.parent.exists()
