"""Tests for datadocs base module."""

import pytest

from truthound.datadocs.base import (
    ReportTheme,
    ChartLibrary,
    ChartType,
    SectionType,
    ExportFormat,
    SeverityLevel,
    ThemeColors,
    ThemeTypography,
    ThemeSpacing,
    ThemeConfig,
    ReportMetadata,
    ChartSpec,
    SectionSpec,
    AlertSpec,
    ReportConfig,
    ReportSpec,
    RendererRegistry,
    renderer_registry,
)


class TestEnums:
    """Test enum types."""

    def test_report_theme_values(self):
        """Test ReportTheme enum values."""
        assert ReportTheme.LIGHT.value == "light"
        assert ReportTheme.DARK.value == "dark"
        assert ReportTheme.PROFESSIONAL.value == "professional"
        assert ReportTheme.MINIMAL.value == "minimal"
        assert ReportTheme.MODERN.value == "modern"

    def test_chart_library_values(self):
        """Test ChartLibrary enum values.

        Chart library selection is now automatic:
        - ApexCharts for HTML reports (interactive)
        - SVG for PDF export (no JavaScript dependency)
        """
        assert ChartLibrary.APEXCHARTS.value == "apexcharts"
        assert ChartLibrary.SVG.value == "svg"

    def test_chart_type_values(self):
        """Test ChartType enum values."""
        assert ChartType.BAR.value == "bar"
        assert ChartType.PIE.value == "pie"
        assert ChartType.LINE.value == "line"
        assert ChartType.HEATMAP.value == "heatmap"

    def test_section_type_values(self):
        """Test SectionType enum values."""
        assert SectionType.OVERVIEW.value == "overview"
        assert SectionType.COLUMNS.value == "columns"
        assert SectionType.QUALITY.value == "quality"

    def test_severity_level_values(self):
        """Test SeverityLevel enum values."""
        assert SeverityLevel.INFO.value == "info"
        assert SeverityLevel.WARNING.value == "warning"
        assert SeverityLevel.ERROR.value == "error"
        assert SeverityLevel.CRITICAL.value == "critical"


class TestThemeColors:
    """Test ThemeColors dataclass."""

    def test_default_values(self):
        """Test default color values."""
        colors = ThemeColors()
        assert colors.background == "#ffffff"
        assert colors.text_primary == "#1a1a2e"
        assert colors.primary == "#4361ee"
        assert colors.success == "#10b981"
        assert len(colors.chart_palette) == 10

    def test_custom_values(self):
        """Test custom color values."""
        colors = ThemeColors(background="#000000", primary="#ff0000")
        assert colors.background == "#000000"
        assert colors.primary == "#ff0000"

    def test_chart_palette(self):
        """Test chart palette is a tuple."""
        colors = ThemeColors()
        assert isinstance(colors.chart_palette, tuple)
        assert all(c.startswith("#") for c in colors.chart_palette)


class TestThemeConfig:
    """Test ThemeConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = ThemeConfig(name="test")
        assert config.name == "test"
        assert isinstance(config.colors, ThemeColors)
        assert isinstance(config.typography, ThemeTypography)
        assert isinstance(config.spacing, ThemeSpacing)

    def test_to_css_vars(self):
        """Test CSS variables generation."""
        config = ThemeConfig(name="test")
        css = config.to_css_vars()

        assert ":root {" in css
        assert "--color-background:" in css
        assert "--color-primary:" in css
        assert "--font-family:" in css
        assert "--border-radius-md:" in css


class TestReportMetadata:
    """Test ReportMetadata dataclass."""

    def test_default_values(self):
        """Test default metadata values."""
        meta = ReportMetadata()
        assert meta.title == "Data Profile Report"
        assert meta.subtitle == ""
        assert meta.version == "1.0.0"

    def test_custom_values(self):
        """Test custom metadata values."""
        meta = ReportMetadata(
            title="Custom Report",
            subtitle="Test Data",
            author="Test Author",
        )
        assert meta.title == "Custom Report"
        assert meta.subtitle == "Test Data"
        assert meta.author == "Test Author"


class TestChartSpec:
    """Test ChartSpec dataclass."""

    def test_default_values(self):
        """Test default chart spec values."""
        spec = ChartSpec(chart_type=ChartType.BAR)
        assert spec.chart_type == ChartType.BAR
        assert spec.title == ""
        assert spec.labels == []
        assert spec.values == []
        assert spec.height == 300

    def test_with_data(self):
        """Test chart spec with data."""
        spec = ChartSpec(
            chart_type=ChartType.PIE,
            title="Distribution",
            labels=["A", "B", "C"],
            values=[10, 20, 30],
        )
        assert spec.title == "Distribution"
        assert len(spec.labels) == 3
        assert sum(spec.values) == 60


class TestSectionSpec:
    """Test SectionSpec dataclass."""

    def test_default_values(self):
        """Test default section spec values."""
        spec = SectionSpec(
            section_type=SectionType.OVERVIEW,
            title="Overview",
        )
        assert spec.section_type == SectionType.OVERVIEW
        assert spec.title == "Overview"
        assert spec.charts == []
        assert spec.visible is True

    def test_with_content(self):
        """Test section spec with content."""
        chart = ChartSpec(chart_type=ChartType.BAR)
        spec = SectionSpec(
            section_type=SectionType.QUALITY,
            title="Quality",
            charts=[chart],
            metrics={"score": 95.0},
        )
        assert len(spec.charts) == 1
        assert spec.metrics["score"] == 95.0


class TestAlertSpec:
    """Test AlertSpec dataclass."""

    def test_default_severity(self):
        """Test default alert severity."""
        alert = AlertSpec(
            title="Test Alert",
            message="This is a test",
        )
        assert alert.severity == SeverityLevel.INFO

    def test_custom_severity(self):
        """Test custom alert severity."""
        alert = AlertSpec(
            title="Critical Issue",
            message="Something is wrong",
            severity=SeverityLevel.CRITICAL,
            column="test_column",
        )
        assert alert.severity == SeverityLevel.CRITICAL
        assert alert.column == "test_column"


class TestReportConfig:
    """Test ReportConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = ReportConfig()
        assert config.theme == ReportTheme.PROFESSIONAL
        assert config.chart_library == ChartLibrary.APEXCHARTS
        assert config.include_toc is True
        assert len(config.sections) > 0

    def test_custom_sections(self):
        """Test custom sections config."""
        config = ReportConfig(
            sections=[SectionType.OVERVIEW, SectionType.COLUMNS],
        )
        assert len(config.sections) == 2


class TestRendererRegistry:
    """Test RendererRegistry."""

    def test_list_chart_renderers(self):
        """Test listing chart renderers."""
        renderers = renderer_registry.list_chart_renderers()
        assert isinstance(renderers, list)

    def test_list_section_renderers(self):
        """Test listing section renderers."""
        renderers = renderer_registry.list_section_renderers()
        assert isinstance(renderers, list)

    def test_get_nonexistent_chart_renderer(self):
        """Test getting non-existent renderer raises error."""
        registry = RendererRegistry()
        with pytest.raises(KeyError):
            registry.get_chart_renderer(ChartLibrary.APEXCHARTS)

    def test_get_nonexistent_section_renderer(self):
        """Test getting non-existent section renderer raises error."""
        registry = RendererRegistry()
        with pytest.raises(KeyError):
            registry.get_section_renderer(SectionType.OVERVIEW)
