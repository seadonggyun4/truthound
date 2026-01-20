"""Tests for chart renderers.

Truthound uses automatic chart rendering:
- ApexCharts for HTML reports (interactive, feature-rich)
- SVG for PDF export (no JavaScript dependency)
"""

import pytest

from truthound.datadocs.charts import (
    ApexChartsRenderer,
    SVGChartRenderer,
    get_chart_renderer,
    CDN_URLS,
)
from truthound.datadocs.base import (
    ChartLibrary,
    ChartType,
    ChartSpec,
)


@pytest.fixture
def sample_bar_spec():
    """Create a sample bar chart spec."""
    return ChartSpec(
        chart_type=ChartType.BAR,
        title="Sales by Month",
        labels=["Jan", "Feb", "Mar", "Apr"],
        values=[100, 150, 120, 180],
        height=300,
    )


@pytest.fixture
def sample_pie_spec():
    """Create a sample pie chart spec."""
    return ChartSpec(
        chart_type=ChartType.PIE,
        title="Market Share",
        labels=["Product A", "Product B", "Product C"],
        values=[45, 30, 25],
        height=300,
    )


@pytest.fixture
def sample_donut_spec():
    """Create a sample donut chart spec."""
    return ChartSpec(
        chart_type=ChartType.DONUT,
        title="Category Distribution",
        labels=["Cat1", "Cat2", "Cat3", "Cat4"],
        values=[40, 25, 20, 15],
        height=300,
    )


@pytest.fixture
def sample_line_spec():
    """Create a sample line chart spec."""
    return ChartSpec(
        chart_type=ChartType.LINE,
        title="Trend Over Time",
        labels=["Q1", "Q2", "Q3", "Q4"],
        values=[10, 25, 18, 32],
        height=300,
    )


@pytest.fixture
def sample_horizontal_bar_spec():
    """Create a sample horizontal bar chart spec."""
    return ChartSpec(
        chart_type=ChartType.HORIZONTAL_BAR,
        title="Top Categories",
        labels=["Category A", "Category B", "Category C"],
        values=[500, 350, 200],
        height=250,
    )


class TestCDNUrls:
    """Test CDN URL definitions."""

    def test_cdn_urls_defined(self):
        """Test chart libraries have CDN URLs defined."""
        assert ChartLibrary.APEXCHARTS in CDN_URLS
        assert ChartLibrary.SVG in CDN_URLS

    def test_svg_has_no_dependencies(self):
        """Test SVG library has no CDN dependencies."""
        assert CDN_URLS[ChartLibrary.SVG] == []

    def test_apexcharts_has_urls(self):
        """Test ApexCharts has CDN URLs."""
        assert len(CDN_URLS[ChartLibrary.APEXCHARTS]) > 0


class TestApexChartsRenderer:
    """Test ApexCharts renderer."""

    def test_get_dependencies(self):
        """Test getting CDN dependencies."""
        renderer = ApexChartsRenderer()
        deps = renderer.get_dependencies()
        assert len(deps) > 0
        assert "apexcharts" in deps[0].lower()

    def test_render_bar_chart(self, sample_bar_spec):
        """Test rendering a bar chart."""
        renderer = ApexChartsRenderer()
        html = renderer.render(sample_bar_spec)

        assert "chart-container" in html
        assert "ApexCharts" in html
        assert "Sales by Month" in html
        assert "script" in html

    def test_render_pie_chart(self, sample_pie_spec):
        """Test rendering a pie chart."""
        renderer = ApexChartsRenderer()
        html = renderer.render(sample_pie_spec)

        assert "Market Share" in html
        assert "ApexCharts" in html

    def test_render_donut_chart(self, sample_donut_spec):
        """Test rendering a donut chart."""
        renderer = ApexChartsRenderer()
        html = renderer.render(sample_donut_spec)

        assert "donut" in html.lower() or "Category Distribution" in html

    def test_render_with_custom_colors(self, sample_bar_spec):
        """Test rendering with custom colors."""
        sample_bar_spec.colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        renderer = ApexChartsRenderer()
        html = renderer.render(sample_bar_spec)

        assert "#ff0000" in html

    def test_render_horizontal_bar(self, sample_horizontal_bar_spec):
        """Test rendering horizontal bar chart."""
        renderer = ApexChartsRenderer()
        html = renderer.render(sample_horizontal_bar_spec)

        assert "horizontal" in html.lower()

    def test_render_line_chart(self, sample_line_spec):
        """Test rendering line chart."""
        renderer = ApexChartsRenderer()
        html = renderer.render(sample_line_spec)

        assert "Trend Over Time" in html

    def test_render_with_series(self):
        """Test rendering with multiple series."""
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            labels=["Jan", "Feb", "Mar"],
            values=[],
            series=[
                {"name": "Series A", "data": [10, 20, 30]},
                {"name": "Series B", "data": [15, 25, 35]},
            ],
            height=300,
        )
        renderer = ApexChartsRenderer()
        html = renderer.render(spec)

        assert "Series A" in html
        assert "Series B" in html


class TestSVGChartRenderer:
    """Test SVG chart renderer (used for PDF export)."""

    def test_get_dependencies(self):
        """Test SVG has no dependencies."""
        renderer = SVGChartRenderer()
        deps = renderer.get_dependencies()
        assert deps == []

    def test_render_bar_chart(self, sample_bar_spec):
        """Test rendering a bar chart as SVG."""
        renderer = SVGChartRenderer()
        html = renderer.render(sample_bar_spec)

        assert "<svg" in html
        assert "<rect" in html
        assert "Sales by Month" in html
        # No JavaScript
        assert "script" not in html.lower() or "<script>" not in html

    def test_render_horizontal_bar(self, sample_horizontal_bar_spec):
        """Test rendering horizontal bar as SVG."""
        renderer = SVGChartRenderer()
        html = renderer.render(sample_horizontal_bar_spec)

        assert "<svg" in html
        assert "<rect" in html

    def test_render_pie_chart(self, sample_pie_spec):
        """Test rendering a pie chart as SVG."""
        renderer = SVGChartRenderer()
        html = renderer.render(sample_pie_spec)

        assert "<svg" in html
        assert "<path" in html
        assert "Market Share" in html

    def test_render_donut_chart(self, sample_donut_spec):
        """Test rendering a donut chart as SVG."""
        renderer = SVGChartRenderer()
        html = renderer.render(sample_donut_spec)

        assert "<svg" in html
        assert "<path" in html

    def test_render_line_chart(self, sample_line_spec):
        """Test rendering a line chart as SVG."""
        renderer = SVGChartRenderer()
        html = renderer.render(sample_line_spec)

        assert "<svg" in html
        assert "<path" in html or "<circle" in html
        assert "Trend Over Time" in html

    def test_render_empty_data(self):
        """Test rendering with empty data."""
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            labels=[],
            values=[],
            height=300,
        )
        renderer = SVGChartRenderer()
        html = renderer.render(spec)

        assert "<svg" in html
        assert "No data" in html

    def test_pie_with_zero_total(self):
        """Test pie chart with zero total."""
        spec = ChartSpec(
            chart_type=ChartType.PIE,
            labels=["A", "B"],
            values=[0, 0],
            height=300,
        )
        renderer = SVGChartRenderer()
        html = renderer.render(spec)

        assert "<svg" in html


class TestGetChartRenderer:
    """Test factory function."""

    def test_get_apexcharts(self):
        """Test getting ApexCharts renderer."""
        renderer = get_chart_renderer(ChartLibrary.APEXCHARTS)
        assert isinstance(renderer, ApexChartsRenderer)

    def test_get_svg(self):
        """Test getting SVG renderer."""
        renderer = get_chart_renderer(ChartLibrary.SVG)
        assert isinstance(renderer, SVGChartRenderer)

    def test_get_by_string(self):
        """Test getting renderer by string name."""
        renderer = get_chart_renderer("apexcharts")
        assert isinstance(renderer, ApexChartsRenderer)

        renderer = get_chart_renderer("svg")
        assert isinstance(renderer, SVGChartRenderer)

    def test_default_renderer(self):
        """Test default renderer is ApexCharts."""
        renderer = get_chart_renderer()
        assert isinstance(renderer, ApexChartsRenderer)


class TestChartSpecOptions:
    """Test ChartSpec configuration options."""

    def test_show_legend_option(self):
        """Test show_legend option."""
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            labels=["A", "B"],
            values=[10, 20],
            height=300,
            show_legend=False,
        )
        renderer = ApexChartsRenderer()
        html = renderer.render(spec)

        # Legend should be disabled in config
        assert '"show": false' in html or '"show":false' in html

    def test_show_grid_option(self):
        """Test show_grid option."""
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            labels=["A", "B"],
            values=[10, 20],
            height=300,
            show_grid=False,
        )
        renderer = ApexChartsRenderer()
        html = renderer.render(spec)
        # Grid config should be present
        assert "grid" in html.lower()

    def test_animation_option(self):
        """Test animation option."""
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            labels=["A", "B"],
            values=[10, 20],
            height=300,
            animation=False,
        )
        renderer = ApexChartsRenderer()
        html = renderer.render(spec)
        # Animation config should be present
        assert "animations" in html.lower()

    def test_custom_width(self):
        """Test custom width option."""
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            labels=["A", "B"],
            values=[10, 20],
            height=300,
            width=500,
        )
        renderer = ApexChartsRenderer()
        html = renderer.render(spec)
        assert "500px" in html

    def test_subtitle_option(self):
        """Test subtitle option."""
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            title="Main Title",
            subtitle="Sub Title",
            labels=["A", "B"],
            values=[10, 20],
            height=300,
        )
        renderer = ApexChartsRenderer()
        html = renderer.render(spec)
        assert "Sub Title" in html


class TestChartTypeMapping:
    """Test chart type mappings work correctly."""

    def test_all_chart_types_render_apexcharts(self):
        """Test all chart types can be rendered with ApexCharts."""
        chart_types = [
            ChartType.BAR,
            ChartType.HORIZONTAL_BAR,
            ChartType.LINE,
            ChartType.PIE,
            ChartType.DONUT,
            ChartType.HISTOGRAM,
        ]

        renderer = ApexChartsRenderer()
        for chart_type in chart_types:
            spec = ChartSpec(
                chart_type=chart_type,
                labels=["A", "B", "C"],
                values=[10, 20, 30],
                height=300,
            )
            html = renderer.render(spec)
            assert "<" in html  # Basic HTML check

    def test_all_chart_types_render_svg(self):
        """Test all chart types can be rendered with SVG."""
        chart_types = [
            ChartType.BAR,
            ChartType.HORIZONTAL_BAR,
            ChartType.LINE,
            ChartType.PIE,
            ChartType.DONUT,
            ChartType.HISTOGRAM,
        ]

        renderer = SVGChartRenderer()
        for chart_type in chart_types:
            spec = ChartSpec(
                chart_type=chart_type,
                labels=["A", "B", "C"],
                values=[10, 20, 30],
                height=300,
            )
            html = renderer.render(spec)
            assert "<svg" in html
