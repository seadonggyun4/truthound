"""Tests for Quality Reporters."""

import json
import pytest
from pathlib import Path

from truthound.profiler.quality import (
    QualityLevel,
    QualityMetrics,
    RuleType,
    RuleQualityScore,
)
from truthound.reporters.quality.reporters import (
    ConsoleQualityReporter,
    JsonQualityReporter,
    MarkdownQualityReporter,
    HtmlQualityReporter,
    JUnitQualityReporter,
)
from truthound.reporters.quality.config import (
    QualityReporterConfig,
    QualityDisplayMode,
    ReportSortOrder,
)
from truthound.reporters.quality.base import (
    QualityStatistics,
    QualityReportResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metrics_excellent() -> QualityMetrics:
    """Create excellent quality metrics."""
    return QualityMetrics(
        precision=0.96,
        recall=0.95,
        f1_score=0.955,
        accuracy=0.97,
        confidence=0.9,
        quality_level=QualityLevel.EXCELLENT,
        precision_ci=(0.93, 0.98),
        recall_ci=(0.92, 0.97),
    )


@pytest.fixture
def sample_metrics_good() -> QualityMetrics:
    """Create good quality metrics."""
    return QualityMetrics(
        precision=0.88,
        recall=0.86,
        f1_score=0.87,
        accuracy=0.89,
        confidence=0.85,
        quality_level=QualityLevel.GOOD,
    )


@pytest.fixture
def sample_metrics_poor() -> QualityMetrics:
    """Create poor quality metrics."""
    return QualityMetrics(
        precision=0.55,
        recall=0.52,
        f1_score=0.534,
        accuracy=0.6,
        confidence=0.5,
        quality_level=QualityLevel.POOR,
    )


@pytest.fixture
def sample_scores(
    sample_metrics_excellent,
    sample_metrics_good,
    sample_metrics_poor,
) -> list[RuleQualityScore]:
    """Create sample quality scores."""
    return [
        RuleQualityScore(
            rule_name="email_format",
            rule_type=RuleType.PATTERN,
            column="email",
            metrics=sample_metrics_excellent,
            recommendation="Excellent quality. Safe to use.",
            should_use=True,
        ),
        RuleQualityScore(
            rule_name="age_range",
            rule_type=RuleType.RANGE,
            column="age",
            metrics=sample_metrics_good,
            recommendation="Good quality. Recommended for use.",
            should_use=True,
        ),
        RuleQualityScore(
            rule_name="phone_pattern",
            rule_type=RuleType.PATTERN,
            column="phone",
            metrics=sample_metrics_poor,
            recommendation="Poor quality. Consider improvements.",
            should_use=False,
        ),
    ]


@pytest.fixture
def single_score(sample_metrics_excellent) -> RuleQualityScore:
    """Create a single quality score."""
    return RuleQualityScore(
        rule_name="test_rule",
        rule_type=RuleType.PATTERN,
        column="test_col",
        metrics=sample_metrics_excellent,
        recommendation="Test recommendation.",
        should_use=True,
    )


# =============================================================================
# Test Console Reporter
# =============================================================================


class TestConsoleQualityReporter:
    """Tests for ConsoleQualityReporter."""

    def test_render_single_score(self, single_score):
        """Test rendering a single score."""
        reporter = ConsoleQualityReporter()
        output = reporter.render(single_score)

        assert "test_rule" in output
        assert "excellent" in output.lower()

    def test_render_multiple_scores(self, sample_scores):
        """Test rendering multiple scores."""
        reporter = ConsoleQualityReporter()
        output = reporter.render(sample_scores)

        assert "email_format" in output
        assert "age_range" in output
        assert "phone_pattern" in output

    def test_render_with_summary(self, sample_scores):
        """Test rendering with summary."""
        config = QualityReporterConfig(include_summary=True)
        reporter = ConsoleQualityReporter(config=config)
        output = reporter.render(sample_scores)

        assert "Summary" in output or "summary" in output.lower()

    def test_render_compact_mode(self, sample_scores):
        """Test rendering in compact mode."""
        config = QualityReporterConfig(display_mode=QualityDisplayMode.COMPACT)
        reporter = ConsoleQualityReporter(config=config)
        output = reporter.render(sample_scores)

        assert "email_format" in output

    def test_render_detailed_mode(self, sample_scores):
        """Test rendering in detailed mode."""
        config = QualityReporterConfig(display_mode=QualityDisplayMode.DETAILED)
        reporter = ConsoleQualityReporter(config=config)
        output = reporter.render(sample_scores)

        assert "Detailed" in output or "detailed" in output.lower()

    def test_empty_scores(self):
        """Test rendering empty scores list."""
        reporter = ConsoleQualityReporter()
        output = reporter.render([])

        assert "No" in output or "no" in output.lower()


# =============================================================================
# Test JSON Reporter
# =============================================================================


class TestJsonQualityReporter:
    """Tests for JsonQualityReporter."""

    def test_render_single_score(self, single_score):
        """Test rendering a single score as JSON."""
        reporter = JsonQualityReporter()
        output = reporter.render(single_score)

        data = json.loads(output)
        assert "scores" in data
        assert len(data["scores"]) == 1
        assert data["scores"][0]["rule_name"] == "test_rule"

    def test_render_multiple_scores(self, sample_scores):
        """Test rendering multiple scores as JSON."""
        reporter = JsonQualityReporter()
        output = reporter.render(sample_scores)

        data = json.loads(output)
        assert len(data["scores"]) == 3

    def test_render_with_indent(self, sample_scores):
        """Test rendering with custom indent."""
        reporter = JsonQualityReporter(indent=4)
        output = reporter.render(sample_scores)

        # Check indentation
        assert "    " in output

    def test_render_includes_statistics(self, sample_scores):
        """Test rendering includes statistics."""
        config = QualityReporterConfig(include_statistics=True)
        reporter = JsonQualityReporter(config=config)
        output = reporter.render(sample_scores)

        data = json.loads(output)
        assert "statistics" in data

    def test_render_includes_count(self, sample_scores):
        """Test rendering includes count."""
        reporter = JsonQualityReporter()
        output = reporter.render(sample_scores)

        data = json.loads(output)
        assert data["count"] == 3

    def test_render_includes_generated_at(self, sample_scores):
        """Test rendering includes generated_at timestamp."""
        reporter = JsonQualityReporter()
        output = reporter.render(sample_scores)

        data = json.loads(output)
        assert "generated_at" in data

    def test_metrics_in_output(self, single_score):
        """Test that metrics are correctly serialized."""
        reporter = JsonQualityReporter()
        output = reporter.render(single_score)

        data = json.loads(output)
        metrics = data["scores"][0]["metrics"]
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics


# =============================================================================
# Test Markdown Reporter
# =============================================================================


class TestMarkdownQualityReporter:
    """Tests for MarkdownQualityReporter."""

    def test_render_single_score(self, single_score):
        """Test rendering a single score as Markdown."""
        reporter = MarkdownQualityReporter()
        output = reporter.render(single_score)

        assert "test_rule" in output
        assert "| Metric |" in output or "F1" in output

    def test_render_multiple_scores(self, sample_scores):
        """Test rendering multiple scores as Markdown."""
        reporter = MarkdownQualityReporter()
        output = reporter.render(sample_scores)

        assert "email_format" in output
        assert "age_range" in output
        assert "phone_pattern" in output

    def test_render_includes_table(self, sample_scores):
        """Test rendering includes Markdown table."""
        reporter = MarkdownQualityReporter()
        output = reporter.render(sample_scores)

        assert "|" in output
        assert "---" in output

    def test_render_includes_title(self, sample_scores):
        """Test rendering includes title."""
        config = QualityReporterConfig(title="My Quality Report")
        reporter = MarkdownQualityReporter(config=config)
        output = reporter.render(sample_scores)

        assert "My Quality Report" in output

    def test_render_includes_emojis(self, sample_scores):
        """Test rendering includes quality level emojis."""
        reporter = MarkdownQualityReporter()
        output = reporter.render(sample_scores)

        # Check for level indicators (emoji or text)
        assert any(emoji in output for emoji in ["🟢", "🔵", "🟡", "🟠", "🔴", "Excellent", "Good"])


# =============================================================================
# Test HTML Reporter
# =============================================================================


class TestHtmlQualityReporter:
    """Tests for HtmlQualityReporter."""

    def test_render_single_score(self, single_score):
        """Test rendering a single score as HTML."""
        reporter = HtmlQualityReporter()
        output = reporter.render(single_score)

        assert "<!DOCTYPE html>" in output
        assert "test_rule" in output

    def test_render_multiple_scores(self, sample_scores):
        """Test rendering multiple scores as HTML."""
        reporter = HtmlQualityReporter()
        output = reporter.render(sample_scores)

        assert "email_format" in output
        assert "age_range" in output
        assert "phone_pattern" in output

    def test_render_includes_css(self, sample_scores):
        """Test rendering includes CSS."""
        reporter = HtmlQualityReporter()
        output = reporter.render(sample_scores)

        assert "<style>" in output
        assert "</style>" in output

    def test_render_includes_title(self, sample_scores):
        """Test rendering includes title."""
        config = QualityReporterConfig(title="HTML Quality Report")
        reporter = HtmlQualityReporter(config=config)
        output = reporter.render(sample_scores)

        assert "HTML Quality Report" in output
        assert "<title>" in output

    def test_render_with_charts(self, sample_scores):
        """Test rendering with charts enabled."""
        config = QualityReporterConfig(include_charts=True)
        reporter = HtmlQualityReporter(config=config)
        output = reporter.render(sample_scores)

        assert "apexcharts" in output.lower() or "chart" in output.lower()

    def test_render_without_charts(self, sample_scores):
        """Test rendering with charts disabled."""
        config = QualityReporterConfig(include_charts=False)
        reporter = HtmlQualityReporter(config=config)
        output = reporter.render(sample_scores)

        # Should still have content
        assert "email_format" in output

    def test_render_includes_summary_cards(self, sample_scores):
        """Test rendering includes summary cards."""
        reporter = HtmlQualityReporter()
        output = reporter.render(sample_scores)

        assert "summary-card" in output or "Summary" in output

    def test_render_valid_html(self, sample_scores):
        """Test rendering produces valid HTML structure."""
        reporter = HtmlQualityReporter()
        output = reporter.render(sample_scores)

        assert "<!DOCTYPE html>" in output
        assert "<html" in output
        assert "</html>" in output
        assert "<head>" in output
        assert "</head>" in output
        assert "<body>" in output
        assert "</body>" in output


# =============================================================================
# Test JUnit Reporter
# =============================================================================


class TestJUnitQualityReporter:
    """Tests for JUnitQualityReporter."""

    def test_render_single_score(self, single_score):
        """Test rendering a single score as JUnit XML."""
        reporter = JUnitQualityReporter()
        output = reporter.render(single_score)

        assert "<testsuite" in output
        assert "<testcase" in output
        assert "test_rule" in output

    def test_render_multiple_scores(self, sample_scores):
        """Test rendering multiple scores as JUnit XML."""
        reporter = JUnitQualityReporter()
        output = reporter.render(sample_scores)

        assert output.count("<testcase") == 3

    def test_render_failure_for_low_f1(self, sample_scores):
        """Test that low F1 scores are marked as failures."""
        reporter = JUnitQualityReporter(min_f1=0.7)
        output = reporter.render(sample_scores)

        assert "<failure" in output
        assert "phone_pattern" in output

    def test_render_all_pass_with_low_threshold(self, sample_scores):
        """Test all pass with low threshold."""
        reporter = JUnitQualityReporter(min_f1=0.1)
        output = reporter.render(sample_scores)

        assert "<failure" not in output

    def test_render_valid_xml(self, sample_scores):
        """Test rendering produces valid XML structure."""
        import xml.etree.ElementTree as ET

        reporter = JUnitQualityReporter()
        output = reporter.render(sample_scores)

        # Should parse without error
        root = ET.fromstring(output)
        assert root.tag == "testsuite"

    def test_render_includes_test_counts(self, sample_scores):
        """Test rendering includes test counts."""
        reporter = JUnitQualityReporter()
        output = reporter.render(sample_scores)

        assert 'tests="3"' in output


# =============================================================================
# Test Base Reporter Features
# =============================================================================


class TestBaseReporterFeatures:
    """Tests for base reporter features."""

    def test_write_to_file(self, sample_scores, tmp_path):
        """Test writing report to file."""
        output_path = tmp_path / "report.json"
        reporter = JsonQualityReporter()
        written_path = reporter.write(sample_scores, output_path)

        assert written_path.exists()
        content = written_path.read_text()
        data = json.loads(content)
        assert len(data["scores"]) == 3

    def test_write_creates_directories(self, sample_scores, tmp_path):
        """Test write creates parent directories."""
        output_path = tmp_path / "subdir" / "nested" / "report.json"
        reporter = JsonQualityReporter()
        written_path = reporter.write(sample_scores, output_path)

        assert written_path.exists()
        assert written_path.parent.exists()

    def test_report_method(self, sample_scores, tmp_path):
        """Test report method returns result."""
        output_path = tmp_path / "report.json"
        reporter = JsonQualityReporter()
        result = reporter.report(sample_scores, output_path)

        assert isinstance(result, QualityReportResult)
        assert result.scores_count == 3
        assert output_path.exists()

    def test_generate_filename(self, sample_scores):
        """Test filename generation."""
        reporter = JsonQualityReporter()
        filename = reporter.generate_filename(sample_scores)

        assert filename.endswith(".json")
        assert "quality" in filename.lower() or "json" in filename.lower()

    def test_render_to_bytes(self, sample_scores):
        """Test render_to_bytes method."""
        reporter = JsonQualityReporter()
        content = reporter.render_to_bytes(sample_scores)

        assert isinstance(content, bytes)
        data = json.loads(content.decode("utf-8"))
        assert len(data["scores"]) == 3


# =============================================================================
# Test Quality Statistics
# =============================================================================


class TestQualityStatistics:
    """Tests for QualityStatistics."""

    def test_from_scores(self, sample_scores):
        """Test creating statistics from scores."""
        stats = QualityStatistics.from_scores(sample_scores)

        assert stats.total_count == 3
        assert stats.excellent_count == 1
        assert stats.good_count == 1
        assert stats.poor_count == 1
        assert stats.should_use_count == 2
        assert stats.should_not_use_count == 1

    def test_from_empty_scores(self):
        """Test creating statistics from empty list."""
        stats = QualityStatistics.from_scores([])

        assert stats.total_count == 0
        assert stats.avg_f1 == 0.0

    def test_metric_aggregates(self, sample_scores):
        """Test metric aggregates are calculated."""
        stats = QualityStatistics.from_scores(sample_scores)

        assert stats.avg_f1 > 0
        assert stats.min_f1 > 0
        assert stats.max_f1 > 0
        assert stats.avg_f1 >= stats.min_f1
        assert stats.avg_f1 <= stats.max_f1

    def test_to_dict(self, sample_scores):
        """Test to_dict method."""
        stats = QualityStatistics.from_scores(sample_scores)
        data = stats.to_dict()

        assert "total_count" in data
        assert "by_level" in data
        assert "metrics" in data
        assert data["total_count"] == 3


# =============================================================================
# Test Sorting and Filtering
# =============================================================================


class TestSortingAndFiltering:
    """Tests for sorting and filtering in reporters."""

    def test_sort_by_f1_desc(self, sample_scores):
        """Test sorting by F1 score descending."""
        config = QualityReporterConfig(sort_order=ReportSortOrder.F1_DESC)
        reporter = JsonQualityReporter(config=config)
        output = reporter.render(sample_scores)

        data = json.loads(output)
        scores = data["scores"]
        f1_scores = [s["metrics"]["f1_score"] for s in scores]
        assert f1_scores == sorted(f1_scores, reverse=True)

    def test_sort_by_f1_asc(self, sample_scores):
        """Test sorting by F1 score ascending."""
        config = QualityReporterConfig(sort_order=ReportSortOrder.F1_ASC)
        reporter = JsonQualityReporter(config=config)
        output = reporter.render(sample_scores)

        data = json.loads(output)
        scores = data["scores"]
        f1_scores = [s["metrics"]["f1_score"] for s in scores]
        assert f1_scores == sorted(f1_scores)

    def test_max_scores_limit(self, sample_scores):
        """Test limiting number of scores."""
        config = QualityReporterConfig(max_scores=2)
        reporter = JsonQualityReporter(config=config)
        output = reporter.render(sample_scores)

        data = json.loads(output)
        assert len(data["scores"]) == 2
