"""Tests for datadocs builder module."""

import json
import tempfile
from pathlib import Path

import pytest

from truthound.datadocs.builder import (
    ProfileDataConverter,
    HTMLReportBuilder,
    generate_html_report,
    generate_report_from_file,
)
from truthound.datadocs.base import (
    ReportTheme,
    ChartType,
    ReportConfig,
)


@pytest.fixture
def sample_profile():
    """Create a sample profile for testing."""
    return {
        "name": "test_data",
        "row_count": 1000,
        "column_count": 5,
        "estimated_memory_bytes": 50000,
        "duplicate_row_count": 10,
        "duplicate_row_ratio": 0.01,
        "source": "test.csv",
        "profiled_at": "2024-01-01T00:00:00",
        "columns": [
            {
                "name": "id",
                "physical_type": "Int64",
                "inferred_type": "integer",
                "row_count": 1000,
                "null_count": 0,
                "null_ratio": 0.0,
                "distinct_count": 1000,
                "unique_ratio": 1.0,
                "is_unique": True,
                "is_constant": False,
            },
            {
                "name": "name",
                "physical_type": "String",
                "inferred_type": "string",
                "row_count": 1000,
                "null_count": 50,
                "null_ratio": 0.05,
                "distinct_count": 800,
                "unique_ratio": 0.8,
                "is_unique": False,
                "is_constant": False,
                "min_length": 3,
                "max_length": 50,
            },
            {
                "name": "email",
                "physical_type": "String",
                "inferred_type": "email",
                "row_count": 1000,
                "null_count": 100,
                "null_ratio": 0.1,
                "distinct_count": 900,
                "unique_ratio": 0.9,
                "is_unique": False,
                "is_constant": False,
                "detected_patterns": [
                    {
                        "pattern": "email",
                        "regex": r".+@.+\..+",
                        "match_ratio": 0.95,
                        "sample_matches": ["test@example.com", "user@domain.com"],
                    }
                ],
            },
            {
                "name": "value",
                "physical_type": "Float64",
                "inferred_type": "float",
                "row_count": 1000,
                "null_count": 0,
                "null_ratio": 0.0,
                "distinct_count": 500,
                "unique_ratio": 0.5,
                "is_unique": False,
                "is_constant": False,
                "distribution": {
                    "mean": 50.5,
                    "std": 15.2,
                    "min": 0.0,
                    "max": 100.0,
                    "median": 51.0,
                },
            },
            {
                "name": "status",
                "physical_type": "String",
                "inferred_type": "categorical",
                "row_count": 1000,
                "null_count": 0,
                "null_ratio": 0.0,
                "distinct_count": 3,
                "unique_ratio": 0.003,
                "is_unique": False,
                "is_constant": False,
                "top_values": [
                    {"value": "active", "count": 600, "ratio": 0.6},
                    {"value": "inactive", "count": 300, "ratio": 0.3},
                    {"value": "pending", "count": 100, "ratio": 0.1},
                ],
            },
        ],
        "correlations": [],
    }


class TestProfileDataConverter:
    """Test ProfileDataConverter class."""

    def test_init_with_dict(self, sample_profile):
        """Test initialization with dict."""
        converter = ProfileDataConverter(sample_profile)
        assert converter.data == sample_profile

    def test_get_overview_metrics(self, sample_profile):
        """Test extracting overview metrics."""
        converter = ProfileDataConverter(sample_profile)
        metrics = converter.get_overview_metrics()

        assert metrics["row_count"] == 1000
        assert metrics["column_count"] == 5
        assert metrics["memory_bytes"] == 50000
        assert "quality_score" in metrics

    def test_quality_score_calculation(self, sample_profile):
        """Test quality score calculation."""
        converter = ProfileDataConverter(sample_profile)
        metrics = converter.get_overview_metrics()

        # Quality score should be high for this profile (low null ratios)
        assert metrics["quality_score"] >= 80

    def test_get_column_data(self, sample_profile):
        """Test getting column data."""
        converter = ProfileDataConverter(sample_profile)
        columns = converter.get_column_data()

        assert len(columns) == 5
        assert columns[0]["name"] == "id"

    def test_get_type_distribution(self, sample_profile):
        """Test getting type distribution chart."""
        converter = ProfileDataConverter(sample_profile)
        chart = converter.get_type_distribution()

        assert chart.chart_type == ChartType.DONUT
        assert "integer" in chart.labels
        assert sum(chart.values) == 5

    def test_get_null_distribution(self, sample_profile):
        """Test getting null distribution chart."""
        converter = ProfileDataConverter(sample_profile)
        chart = converter.get_null_distribution()

        assert chart.chart_type == ChartType.HORIZONTAL_BAR
        assert len(chart.labels) <= 10

    def test_get_patterns(self, sample_profile):
        """Test getting detected patterns."""
        converter = ProfileDataConverter(sample_profile)
        patterns = converter.get_patterns()

        assert len(patterns) == 1
        assert patterns[0]["column"] == "email"
        assert patterns[0]["pattern"] == "email"

    def test_get_alerts(self, sample_profile):
        """Test generating alerts."""
        converter = ProfileDataConverter(sample_profile)
        alerts = converter.get_alerts()

        # Should have alert for low cardinality column
        assert any("status" in a.column or "status" in str(a.title) for a in alerts if a.column)

    def test_get_recommendations(self, sample_profile):
        """Test generating recommendations."""
        converter = ProfileDataConverter(sample_profile)
        recommendations = converter.get_recommendations()

        assert isinstance(recommendations, list)


class TestHTMLReportBuilder:
    """Test HTMLReportBuilder class."""

    def test_init_default(self):
        """Test default initialization."""
        builder = HTMLReportBuilder()
        assert builder.config.theme == ReportTheme.PROFESSIONAL
        # Chart library is automatically selected (ApexCharts for HTML)

    def test_init_with_theme(self):
        """Test initialization with custom theme."""
        builder = HTMLReportBuilder(theme=ReportTheme.DARK)
        assert builder.config.theme == ReportTheme.DARK

    def test_init_with_string_theme(self):
        """Test initialization with string theme."""
        builder = HTMLReportBuilder(theme="light")
        assert builder.config.theme == ReportTheme.LIGHT

    def test_init_with_config(self):
        """Test initialization with full config."""
        config = ReportConfig(
            theme=ReportTheme.MINIMAL,
        )
        builder = HTMLReportBuilder(config=config)
        assert builder.config.theme == ReportTheme.MINIMAL

    def test_build_returns_html(self, sample_profile):
        """Test build returns valid HTML."""
        builder = HTMLReportBuilder()
        html = builder.build(sample_profile, title="Test Report")

        assert html.startswith("<!DOCTYPE html>")
        assert "<html" in html
        assert "</html>" in html
        assert "Test Report" in html

    def test_build_includes_sections(self, sample_profile):
        """Test built HTML includes sections."""
        builder = HTMLReportBuilder()
        html = builder.build(sample_profile)

        assert "section-overview" in html or "Overview" in html
        assert "section-columns" in html or "Column" in html

    def test_build_includes_charts(self, sample_profile):
        """Test built HTML includes chart scripts."""
        builder = HTMLReportBuilder()
        html = builder.build(sample_profile)

        # Should include ApexCharts CDN
        assert "apexcharts" in html.lower()

    def test_build_with_svg_charts(self, sample_profile):
        """Test building with SVG charts (no JS) - used internally for PDF export."""
        # SVG rendering is now automatic for PDF export
        # Direct SVG usage is internal (_use_svg=True)
        builder = HTMLReportBuilder(_use_svg=True)
        html = builder.build(sample_profile)

        # Should have SVG elements, not ApexCharts
        assert "<svg" in html
        assert "apexcharts" not in html.lower()

    def test_save_report(self, sample_profile):
        """Test saving report to file."""
        builder = HTMLReportBuilder()
        html = builder.build(sample_profile)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            saved_path = builder.save(html, path)

            assert saved_path.exists()
            assert saved_path.read_text() == html


class TestGenerateHtmlReport:
    """Test generate_html_report function."""

    def test_basic_generation(self, sample_profile):
        """Test basic report generation."""
        html = generate_html_report(sample_profile)

        assert "<!DOCTYPE html>" in html
        assert "Data Profile Report" in html

    def test_custom_title(self, sample_profile):
        """Test custom title."""
        html = generate_html_report(
            sample_profile,
            title="Custom Title",
            subtitle="Custom Subtitle",
        )

        assert "Custom Title" in html
        assert "Custom Subtitle" in html

    def test_with_output_path(self, sample_profile):
        """Test generating with output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.html"
            html = generate_html_report(
                sample_profile,
                output_path=path,
            )

            assert path.exists()
            assert path.read_text() == html

    def test_different_themes(self, sample_profile):
        """Test generating with different themes."""
        themes = ["light", "dark", "professional", "minimal", "modern"]

        for theme in themes:
            html = generate_html_report(sample_profile, theme=theme)
            assert "<!DOCTYPE html>" in html

    def test_chart_rendering_automatic(self, sample_profile):
        """Test that chart rendering works automatically (ApexCharts for HTML)."""
        # Chart library selection is now automatic:
        # - ApexCharts for HTML (default)
        # - SVG for PDF export (automatic via export_to_pdf)
        html = generate_html_report(sample_profile)
        assert "<!DOCTYPE html>" in html
        # Should use ApexCharts by default
        assert "apexcharts" in html.lower()


class TestGenerateReportFromFile:
    """Test generate_report_from_file function."""

    def test_generate_from_json_file(self, sample_profile):
        """Test generating report from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write profile to file
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(sample_profile))

            # Generate report
            html = generate_report_from_file(profile_path)

            # Check output file exists
            output_path = profile_path.with_suffix(".html")
            assert output_path.exists()
            assert "<!DOCTYPE html>" in html

    def test_custom_output_path(self, sample_profile):
        """Test custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.json"
            profile_path.write_text(json.dumps(sample_profile))

            output_path = Path(tmpdir) / "custom_report.html"
            html = generate_report_from_file(
                profile_path,
                output_path=output_path,
            )

            assert output_path.exists()


class TestHTMLContent:
    """Test the content of generated HTML."""

    def test_html_structure(self, sample_profile):
        """Test HTML has proper structure."""
        html = generate_html_report(sample_profile)

        # Basic structure
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_includes_css(self, sample_profile):
        """Test HTML includes CSS."""
        html = generate_html_report(sample_profile)
        assert "<style>" in html

    def test_includes_meta_tags(self, sample_profile):
        """Test HTML includes meta tags."""
        html = generate_html_report(sample_profile)
        assert '<meta charset="UTF-8">' in html
        assert "viewport" in html

    def test_metrics_displayed(self, sample_profile):
        """Test metrics are displayed."""
        html = generate_html_report(sample_profile)

        # Row count should be displayed
        assert "1,000" in html or "1000" in html

    def test_column_names_displayed(self, sample_profile):
        """Test column names are displayed."""
        html = generate_html_report(sample_profile)

        for col in sample_profile["columns"]:
            assert col["name"] in html

    def test_responsive_design(self, sample_profile):
        """Test includes responsive design elements."""
        html = generate_html_report(sample_profile)

        # Should have media queries
        assert "@media" in html

    def test_print_styles(self, sample_profile):
        """Test includes print styles."""
        html = generate_html_report(sample_profile)

        assert "@media print" in html
