"""Tests for section renderers."""

import pytest

from truthound.datadocs.sections import (
    OverviewSection,
    ColumnsSection,
    QualitySection,
    PatternsSection,
    DistributionSection,
    CorrelationsSection,
    RecommendationsSection,
    AlertsSection,
    CustomSection,
    get_section_renderer,
)
from truthound.datadocs.base import (
    SectionType,
    ChartType,
    ChartSpec,
    SectionSpec,
    AlertSpec,
    SeverityLevel,
)
from truthound.datadocs.charts import ApexChartsRenderer, SVGChartRenderer
from truthound.datadocs.themes import LIGHT_THEME, DARK_THEME


@pytest.fixture
def chart_renderer():
    """Create a chart renderer for testing."""
    return SVGChartRenderer()


@pytest.fixture
def sample_overview_spec():
    """Create a sample overview section spec."""
    return SectionSpec(
        section_type=SectionType.OVERVIEW,
        title="Dataset Overview",
        subtitle="Summary statistics for the dataset",
        metrics={
            "row_count": 10000,
            "column_count": 15,
            "memory_bytes": 5242880,  # 5 MB
            "duplicate_rows": 150,
            "null_cells": 500,
            "quality_score": 92.5,
        },
        charts=[
            ChartSpec(
                chart_type=ChartType.DONUT,
                title="Data Types",
                labels=["String", "Integer", "Float", "Date"],
                values=[6, 4, 3, 2],
                height=250,
            )
        ],
        alerts=[
            AlertSpec(
                severity=SeverityLevel.WARNING,
                title="High null ratio",
                message="Column 'email' has 15% null values",
                column="email",
            )
        ],
    )


@pytest.fixture
def sample_columns_spec():
    """Create a sample columns section spec."""
    return SectionSpec(
        section_type=SectionType.COLUMNS,
        title="Column Details",
        subtitle="Detailed information for each column",
        tables=[
            {
                "title": "Column Summary",
                "headers": ["Column", "Type", "Null %", "Unique %"],
                "rows": [
                    ["id", "integer", "0%", "100%"],
                    ["name", "string", "5%", "95%"],
                    ["email", "email", "15%", "98%"],
                ],
            }
        ],
        metadata={
            "columns": [
                {
                    "name": "id",
                    "physical_type": "Int64",
                    "inferred_type": "integer",
                    "null_ratio": 0.0,
                    "unique_ratio": 1.0,
                    "distinct_count": 10000,
                },
                {
                    "name": "status",
                    "physical_type": "String",
                    "inferred_type": "categorical",
                    "null_ratio": 0.0,
                    "unique_ratio": 0.003,
                    "distinct_count": 3,
                    "top_values": [
                        {"value": "active", "count": 6000},
                        {"value": "inactive", "count": 3000},
                        {"value": "pending", "count": 1000},
                    ],
                },
                {
                    "name": "value",
                    "physical_type": "Float64",
                    "inferred_type": "float",
                    "null_ratio": 0.02,
                    "unique_ratio": 0.5,
                    "distinct_count": 5000,
                    "distribution": {
                        "min": 0.0,
                        "max": 1000.0,
                        "mean": 450.5,
                        "std": 120.3,
                    },
                },
                {
                    "name": "email",
                    "physical_type": "String",
                    "inferred_type": "email",
                    "null_ratio": 0.15,
                    "unique_ratio": 0.98,
                    "distinct_count": 9800,
                    "detected_patterns": [
                        {"pattern": "email", "match_ratio": 0.95},
                    ],
                },
            ]
        },
    )


@pytest.fixture
def sample_quality_spec():
    """Create a sample quality section spec."""
    return SectionSpec(
        section_type=SectionType.QUALITY,
        title="Data Quality",
        subtitle="Quality metrics and assessment",
        metrics={
            "completeness": 95.5,
            "uniqueness": 87.2,
            "validity": 92.0,
            "consistency": 98.5,
        },
        charts=[
            ChartSpec(
                chart_type=ChartType.HORIZONTAL_BAR,
                title="Null Ratio by Column",
                labels=["email", "phone", "address", "name", "id"],
                values=[15, 12, 8, 5, 0],
                height=200,
            )
        ],
    )


@pytest.fixture
def sample_patterns_spec():
    """Create a sample patterns section spec."""
    return SectionSpec(
        section_type=SectionType.PATTERNS,
        title="Detected Patterns",
        subtitle="Automatically detected data patterns",
        metadata={
            "patterns": [
                {
                    "column": "email",
                    "pattern": "email",
                    "match_ratio": 0.95,
                    "sample_matches": ["user@example.com", "test@domain.org"],
                },
                {
                    "column": "phone",
                    "pattern": "phone_us",
                    "match_ratio": 0.82,
                    "sample_matches": ["555-123-4567", "555-987-6543"],
                },
            ]
        },
    )


@pytest.fixture
def sample_correlations_spec():
    """Create a sample correlations section spec."""
    return SectionSpec(
        section_type=SectionType.CORRELATIONS,
        title="Correlations",
        subtitle="Significant column correlations",
        metadata={
            "correlations": [
                ("age", "income", 0.85),
                ("experience", "salary", 0.92),
                ("rating", "reviews", -0.45),
            ]
        },
    )


@pytest.fixture
def sample_recommendations_spec():
    """Create a sample recommendations section spec."""
    return SectionSpec(
        section_type=SectionType.RECOMMENDATIONS,
        title="Recommendations",
        subtitle="Data quality improvement suggestions",
        text_blocks=[
            "Consider adding NOT NULL constraints to the 'id' column",
            "The 'status' column could benefit from an enum/check constraint",
            "Add email format validation for the 'email' column",
        ],
        metadata={
            "validators": [
                {"column": "email", "type": "EmailValidator", "params": {}},
                {"column": "status", "type": "InListValidator", "params": {"values": ["active", "inactive", "pending"]}},
            ]
        },
    )


class TestOverviewSection:
    """Test OverviewSection renderer."""

    def test_render_basic(self, sample_overview_spec, chart_renderer):
        """Test basic overview rendering."""
        section = OverviewSection()
        html = section.render(sample_overview_spec, chart_renderer, LIGHT_THEME)

        assert "section-overview" in html
        assert "Dataset Overview" in html
        assert "Summary statistics" in html

    def test_render_metrics(self, sample_overview_spec, chart_renderer):
        """Test metrics are rendered."""
        section = OverviewSection()
        html = section.render(sample_overview_spec, chart_renderer, LIGHT_THEME)

        assert "10,000" in html  # row_count formatted
        assert "15" in html  # column_count
        assert "92.5%" in html  # quality_score

    def test_render_memory_formatted(self, sample_overview_spec, chart_renderer):
        """Test memory is formatted properly."""
        section = OverviewSection()
        html = section.render(sample_overview_spec, chart_renderer, LIGHT_THEME)

        assert "MB" in html or "5.0" in html  # Memory formatted

    def test_render_charts(self, sample_overview_spec, chart_renderer):
        """Test charts are rendered."""
        section = OverviewSection()
        html = section.render(sample_overview_spec, chart_renderer, LIGHT_THEME)

        assert "Data Types" in html

    def test_render_alerts(self, sample_overview_spec, chart_renderer):
        """Test alerts are rendered."""
        section = OverviewSection()
        html = section.render(sample_overview_spec, chart_renderer, LIGHT_THEME)

        assert "High null ratio" in html
        assert "email" in html


class TestColumnsSection:
    """Test ColumnsSection renderer."""

    def test_render_basic(self, sample_columns_spec, chart_renderer):
        """Test basic columns rendering."""
        section = ColumnsSection()
        html = section.render(sample_columns_spec, chart_renderer, LIGHT_THEME)

        assert "section-columns" in html
        assert "Column Details" in html

    def test_render_column_table(self, sample_columns_spec, chart_renderer):
        """Test column table is rendered."""
        section = ColumnsSection()
        html = section.render(sample_columns_spec, chart_renderer, LIGHT_THEME)

        assert "Column Summary" in html
        assert "<table" in html
        assert "id" in html
        assert "name" in html
        assert "email" in html

    def test_render_column_cards(self, sample_columns_spec, chart_renderer):
        """Test column cards are rendered."""
        section = ColumnsSection()
        html = section.render(sample_columns_spec, chart_renderer, LIGHT_THEME)

        # Check column names appear
        assert "status" in html
        assert "value" in html

    def test_render_with_statistics(self, sample_columns_spec, chart_renderer):
        """Test statistics are rendered for numeric columns."""
        section = ColumnsSection()
        html = section.render(sample_columns_spec, chart_renderer, LIGHT_THEME)

        # Check statistics labels appear
        assert "Min" in html or "min" in html.lower()
        assert "Max" in html or "max" in html.lower()

    def test_render_with_patterns(self, sample_columns_spec, chart_renderer):
        """Test patterns are shown in column cards."""
        section = ColumnsSection()
        html = section.render(sample_columns_spec, chart_renderer, LIGHT_THEME)

        # Pattern tag should appear
        assert "email" in html

    def test_render_top_values_chart(self, sample_columns_spec, chart_renderer):
        """Test top values chart is rendered."""
        section = ColumnsSection()
        html = section.render(sample_columns_spec, chart_renderer, LIGHT_THEME)

        # Top values should be shown
        assert "active" in html
        assert "inactive" in html


class TestQualitySection:
    """Test QualitySection renderer."""

    def test_render_basic(self, sample_quality_spec, chart_renderer):
        """Test basic quality rendering."""
        section = QualitySection()
        html = section.render(sample_quality_spec, chart_renderer, LIGHT_THEME)

        assert "section-quality" in html
        assert "Data Quality" in html

    def test_render_quality_scores(self, sample_quality_spec, chart_renderer):
        """Test quality scores are rendered."""
        section = QualitySection()
        html = section.render(sample_quality_spec, chart_renderer, LIGHT_THEME)

        assert "Completeness" in html
        assert "Uniqueness" in html
        assert "Validity" in html
        assert "Consistency" in html

    def test_render_score_rings(self, sample_quality_spec, chart_renderer):
        """Test score rings (SVG) are rendered."""
        section = QualitySection()
        html = section.render(sample_quality_spec, chart_renderer, LIGHT_THEME)

        assert "circular-chart" in html or "score-ring" in html

    def test_render_charts(self, sample_quality_spec, chart_renderer):
        """Test quality charts are rendered."""
        section = QualitySection()
        html = section.render(sample_quality_spec, chart_renderer, LIGHT_THEME)

        assert "Null Ratio" in html


class TestPatternsSection:
    """Test PatternsSection renderer."""

    def test_render_basic(self, sample_patterns_spec, chart_renderer):
        """Test basic patterns rendering."""
        section = PatternsSection()
        html = section.render(sample_patterns_spec, chart_renderer, LIGHT_THEME)

        assert "section-patterns" in html
        assert "Detected Patterns" in html

    def test_render_patterns_list(self, sample_patterns_spec, chart_renderer):
        """Test patterns list is rendered."""
        section = PatternsSection()
        html = section.render(sample_patterns_spec, chart_renderer, LIGHT_THEME)

        assert "email" in html
        assert "phone" in html
        assert "95" in html or "0.95" in html  # match ratio

    def test_render_sample_matches(self, sample_patterns_spec, chart_renderer):
        """Test sample matches are shown."""
        section = PatternsSection()
        html = section.render(sample_patterns_spec, chart_renderer, LIGHT_THEME)

        assert "user@example.com" in html
        assert "555-123-4567" in html

    def test_render_empty_patterns(self, chart_renderer):
        """Test rendering with no patterns."""
        spec = SectionSpec(
            section_type=SectionType.PATTERNS,
            title="Patterns",
            metadata={"patterns": []},
        )
        section = PatternsSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "No patterns" in html


class TestDistributionSection:
    """Test DistributionSection renderer."""

    def test_render_basic(self, chart_renderer):
        """Test basic distribution rendering."""
        spec = SectionSpec(
            section_type=SectionType.DISTRIBUTION,
            title="Data Distribution",
            charts=[
                ChartSpec(
                    chart_type=ChartType.HISTOGRAM,
                    title="Value Distribution",
                    labels=["0-10", "10-20", "20-30", "30-40"],
                    values=[100, 250, 300, 150],
                    height=250,
                )
            ],
        )
        section = DistributionSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "section-distribution" in html
        assert "Data Distribution" in html

    def test_render_multiple_charts(self, chart_renderer):
        """Test rendering multiple distribution charts."""
        spec = SectionSpec(
            section_type=SectionType.DISTRIBUTION,
            title="Distributions",
            charts=[
                ChartSpec(
                    chart_type=ChartType.HISTOGRAM,
                    title="Age Distribution",
                    labels=["18-25", "25-35", "35-45", "45+"],
                    values=[200, 400, 300, 100],
                    height=200,
                ),
                ChartSpec(
                    chart_type=ChartType.HISTOGRAM,
                    title="Income Distribution",
                    labels=["<50k", "50-100k", "100-150k", "150k+"],
                    values=[300, 400, 200, 100],
                    height=200,
                ),
            ],
        )
        section = DistributionSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "Age Distribution" in html
        assert "Income Distribution" in html


class TestCorrelationsSection:
    """Test CorrelationsSection renderer."""

    def test_render_basic(self, sample_correlations_spec, chart_renderer):
        """Test basic correlations rendering."""
        section = CorrelationsSection()
        html = section.render(sample_correlations_spec, chart_renderer, LIGHT_THEME)

        assert "section-correlations" in html
        assert "Correlations" in html

    def test_render_correlations_list(self, sample_correlations_spec, chart_renderer):
        """Test correlations list is rendered."""
        section = CorrelationsSection()
        html = section.render(sample_correlations_spec, chart_renderer, LIGHT_THEME)

        assert "age" in html
        assert "income" in html
        assert "0.85" in html or "+0.85" in html

    def test_render_negative_correlation(self, sample_correlations_spec, chart_renderer):
        """Test negative correlations are shown correctly."""
        section = CorrelationsSection()
        html = section.render(sample_correlations_spec, chart_renderer, LIGHT_THEME)

        # Negative correlation
        assert "rating" in html
        assert "reviews" in html
        assert "-0.45" in html

    def test_render_empty_correlations(self, chart_renderer):
        """Test rendering with no correlations."""
        spec = SectionSpec(
            section_type=SectionType.CORRELATIONS,
            title="Correlations",
            metadata={"correlations": []},
        )
        section = CorrelationsSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "No significant correlations" in html


class TestRecommendationsSection:
    """Test RecommendationsSection renderer."""

    def test_render_basic(self, sample_recommendations_spec, chart_renderer):
        """Test basic recommendations rendering."""
        section = RecommendationsSection()
        html = section.render(sample_recommendations_spec, chart_renderer, LIGHT_THEME)

        assert "section-recommendations" in html
        assert "Recommendations" in html

    def test_render_recommendations_list(self, sample_recommendations_spec, chart_renderer):
        """Test recommendations are rendered."""
        section = RecommendationsSection()
        html = section.render(sample_recommendations_spec, chart_renderer, LIGHT_THEME)

        assert "NOT NULL" in html
        assert "enum" in html or "status" in html
        assert "email" in html

    def test_render_suggested_validators(self, sample_recommendations_spec, chart_renderer):
        """Test suggested validators are rendered."""
        section = RecommendationsSection()
        html = section.render(sample_recommendations_spec, chart_renderer, LIGHT_THEME)

        assert "EmailValidator" in html
        assert "InListValidator" in html

    def test_render_empty_recommendations(self, chart_renderer):
        """Test rendering with no recommendations."""
        spec = SectionSpec(
            section_type=SectionType.RECOMMENDATIONS,
            title="Recommendations",
            text_blocks=[],
        )
        section = RecommendationsSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "No specific recommendations" in html


class TestAlertsSection:
    """Test AlertsSection renderer."""

    def test_render_basic(self, chart_renderer):
        """Test basic alerts rendering."""
        spec = SectionSpec(
            section_type=SectionType.ALERTS,
            title="Data Quality Alerts",
            alerts=[
                AlertSpec(
                    severity=SeverityLevel.ERROR,
                    title="Critical issue",
                    message="ID column has duplicates",
                ),
                AlertSpec(
                    severity=SeverityLevel.WARNING,
                    title="Warning",
                    message="High null ratio in email column",
                ),
            ],
        )
        section = AlertsSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "section-alerts" in html
        assert "Critical issue" in html
        assert "Warning" in html

    def test_render_empty_alerts(self, chart_renderer):
        """Test rendering with no alerts returns empty."""
        spec = SectionSpec(
            section_type=SectionType.ALERTS,
            title="Alerts",
            alerts=[],
        )
        section = AlertsSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        # Should return empty when no alerts
        assert html == "" or "Alerts" not in html

    def test_render_severity_classes(self, chart_renderer):
        """Test severity classes are applied."""
        spec = SectionSpec(
            section_type=SectionType.ALERTS,
            title="Alerts",
            alerts=[
                AlertSpec(severity=SeverityLevel.ERROR, title="Error", message="msg"),
                AlertSpec(severity=SeverityLevel.WARNING, title="Warning", message="msg"),
                AlertSpec(severity=SeverityLevel.INFO, title="Info", message="msg"),
            ],
        )
        section = AlertsSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "error" in html.lower()
        assert "warning" in html.lower()
        assert "info" in html.lower()


class TestCustomSection:
    """Test CustomSection renderer."""

    def test_render_basic(self, chart_renderer):
        """Test basic custom section rendering."""
        spec = SectionSpec(
            section_type=SectionType.CUSTOM,
            title="Custom Analysis",
            subtitle="Additional insights",
            text_blocks=[
                "This is a custom section with analysis.",
                "It can contain multiple paragraphs.",
            ],
        )
        section = CustomSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "section-custom" in html
        assert "Custom Analysis" in html
        assert "Additional insights" in html

    def test_render_custom_html(self, chart_renderer):
        """Test custom HTML is rendered."""
        spec = SectionSpec(
            section_type=SectionType.CUSTOM,
            title="Custom Section",
            custom_html='<div class="custom-content">Custom HTML content</div>',
        )
        section = CustomSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "custom-content" in html
        assert "Custom HTML content" in html

    def test_render_with_charts(self, chart_renderer):
        """Test custom section with charts."""
        spec = SectionSpec(
            section_type=SectionType.CUSTOM,
            title="Custom Charts",
            charts=[
                ChartSpec(
                    chart_type=ChartType.BAR,
                    title="Custom Chart",
                    labels=["A", "B", "C"],
                    values=[10, 20, 30],
                    height=200,
                )
            ],
        )
        section = CustomSection()
        html = section.render(spec, chart_renderer, LIGHT_THEME)

        assert "Custom Chart" in html


class TestGetSectionRenderer:
    """Test factory function."""

    def test_get_overview(self):
        """Test getting overview renderer."""
        renderer = get_section_renderer(SectionType.OVERVIEW)
        assert isinstance(renderer, OverviewSection)

    def test_get_columns(self):
        """Test getting columns renderer."""
        renderer = get_section_renderer(SectionType.COLUMNS)
        assert isinstance(renderer, ColumnsSection)

    def test_get_quality(self):
        """Test getting quality renderer."""
        renderer = get_section_renderer(SectionType.QUALITY)
        assert isinstance(renderer, QualitySection)

    def test_get_patterns(self):
        """Test getting patterns renderer."""
        renderer = get_section_renderer(SectionType.PATTERNS)
        assert isinstance(renderer, PatternsSection)

    def test_get_distribution(self):
        """Test getting distribution renderer."""
        renderer = get_section_renderer(SectionType.DISTRIBUTION)
        assert isinstance(renderer, DistributionSection)

    def test_get_correlations(self):
        """Test getting correlations renderer."""
        renderer = get_section_renderer(SectionType.CORRELATIONS)
        assert isinstance(renderer, CorrelationsSection)

    def test_get_recommendations(self):
        """Test getting recommendations renderer."""
        renderer = get_section_renderer(SectionType.RECOMMENDATIONS)
        assert isinstance(renderer, RecommendationsSection)

    def test_get_alerts(self):
        """Test getting alerts renderer."""
        renderer = get_section_renderer(SectionType.ALERTS)
        assert isinstance(renderer, AlertsSection)

    def test_get_custom(self):
        """Test getting custom renderer."""
        renderer = get_section_renderer(SectionType.CUSTOM)
        assert isinstance(renderer, CustomSection)

    def test_get_by_string(self):
        """Test getting renderer by string name."""
        renderer = get_section_renderer("overview")
        assert isinstance(renderer, OverviewSection)

        renderer = get_section_renderer("columns")
        assert isinstance(renderer, ColumnsSection)


class TestSectionWithDifferentThemes:
    """Test sections render correctly with different themes."""

    def test_overview_with_dark_theme(self, sample_overview_spec, chart_renderer):
        """Test overview with dark theme."""
        section = OverviewSection()
        html = section.render(sample_overview_spec, chart_renderer, DARK_THEME)

        # Should still render correctly
        assert "section-overview" in html
        assert "Dataset Overview" in html

    def test_quality_with_dark_theme(self, sample_quality_spec, chart_renderer):
        """Test quality with dark theme."""
        section = QualitySection()
        html = section.render(sample_quality_spec, chart_renderer, DARK_THEME)

        assert "section-quality" in html
