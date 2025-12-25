"""Visualization module for Truthound data profiling.

This package provides comprehensive HTML report generation with
interactive visualizations using multiple rendering backends.

Rendering Backends:
- SVG: Pure SVG charts (no dependencies, always available)
- Plotly: Interactive charts with zoom, pan, hover (recommended)
- ECharts: Apache ECharts for advanced visualizations

Example:
    from truthound.profiler.visualization import (
        HTMLReportGenerator,
        generate_report,
        ReportConfig,
        PlotlyChartRenderer,
    )

    # Generate report with Plotly
    generator = HTMLReportGenerator(chart_renderer=PlotlyChartRenderer())
    html = generator.generate(profile_data)

    # Or use convenience function
    html = generate_report(profile_data, renderer="plotly", theme="dark")
"""

from truthound.profiler.visualization.base import (
    ChartType,
    ColorScheme,
    ReportTheme,
    SectionType,
    ChartData,
    ChartConfig,
    ThemeConfig,
    SectionContent,
    ReportConfig,
    ProfileData,
    COLOR_PALETTES,
    THEME_CONFIGS,
)
from truthound.profiler.visualization.renderers import (
    ChartRenderer,
    ChartRendererProtocol,
    SVGChartRenderer,
    ChartRendererRegistry,
    chart_renderer_registry,
)
from truthound.profiler.visualization.plotly_renderer import (
    PlotlyChartRenderer,
    EChartsChartRenderer,
)
from truthound.profiler.visualization.sections import (
    SectionRenderer,
    SectionRegistry,
    section_registry,
    OverviewSectionRenderer,
    DataQualitySectionRenderer,
    ColumnDetailsSectionRenderer,
    PatternsSectionRenderer,
    RecommendationsSectionRenderer,
    CustomSectionRenderer,
    BaseSectionRenderer,
)
from truthound.profiler.visualization.generator import (
    HTMLReportGenerator,
    ReportExporter,
    generate_report,
    compare_profiles,
    ThemeRegistry,
    theme_registry,
    ProfileDataConverter,
    ReportTemplate,
)

__all__ = [
    # Types
    "ChartType",
    "ColorScheme",
    "ReportTheme",
    "SectionType",
    "ChartData",
    "ChartConfig",
    "ThemeConfig",
    "SectionContent",
    "ReportConfig",
    "ProfileData",
    "COLOR_PALETTES",
    "THEME_CONFIGS",
    # Chart Renderers
    "ChartRenderer",
    "ChartRendererProtocol",
    "SVGChartRenderer",
    "PlotlyChartRenderer",
    "EChartsChartRenderer",
    "ChartRendererRegistry",
    "chart_renderer_registry",
    # Section Renderers
    "SectionRenderer",
    "BaseSectionRenderer",
    "SectionRegistry",
    "section_registry",
    "OverviewSectionRenderer",
    "DataQualitySectionRenderer",
    "ColumnDetailsSectionRenderer",
    "PatternsSectionRenderer",
    "RecommendationsSectionRenderer",
    "CustomSectionRenderer",
    # Theme
    "ThemeRegistry",
    "theme_registry",
    # Generator
    "HTMLReportGenerator",
    "ReportExporter",
    "generate_report",
    "compare_profiles",
    "ProfileDataConverter",
    "ReportTemplate",
]
