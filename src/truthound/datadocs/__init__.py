"""Data Docs module for Truthound (Phase 8).

This module provides comprehensive data documentation and reporting:

**Stage 1: Static HTML Report** (Minimal dependencies - Jinja2 optional)
- Generate beautiful, self-contained HTML reports from profile data
- CI/CD pipeline compatible (artifacts)
- Shareable via email/Slack
- CDN-based charts (no build step required)

**Stage 2: Interactive Dashboard** (Optional - truthound[dashboard])
- Real-time interactive dashboard built with Reflex
- Advanced filtering, drill-down, and exploration
- Live data profiling

Usage:
    # Stage 1: Static HTML Report (default)
    truthound docs generate profile.json -o report.html

    # Stage 2: Interactive Dashboard (requires extras)
    pip install truthound[dashboard]
    truthound dashboard --profile profile.json --port 8080

Example:
    from truthound.datadocs import (
        HTMLReportBuilder,
        generate_html_report,
        ReportConfig,
        ReportTheme,
    )

    # Generate static HTML report
    from truthound.profiler import load_profile
    profile = load_profile("profile.json")
    html = generate_html_report(profile, title="My Data Report")

    # Or use the builder for more control
    builder = HTMLReportBuilder(theme=ReportTheme.PROFESSIONAL)
    html = builder.build(profile)
    builder.save("report.html", html)

    # Launch dashboard (requires truthound[dashboard])
    from truthound.datadocs import launch_dashboard
    launch_dashboard(profile_path="profile.json", port=8080)
"""

from truthound.datadocs.base import (
    # Enums
    ReportTheme,
    ChartLibrary,
    ChartType,
    SectionType,
    ExportFormat,
    SeverityLevel,
    # Data structures
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
    # Protocols
    ChartRenderer,
    SectionRenderer,
    TemplateRenderer,
    ReportRenderer,
    # Base classes
    BaseChartRenderer,
    BaseSectionRenderer,
    BaseReportRenderer,
    # Registry
    RendererRegistry,
    renderer_registry,
    register_chart_renderer,
    register_section_renderer,
)

from truthound.datadocs.themes import (
    LIGHT_THEME,
    DARK_THEME,
    PROFESSIONAL_THEME,
    MINIMAL_THEME,
    MODERN_THEME,
    THEMES,
    get_theme,
    get_available_themes,
)

from truthound.datadocs.charts import (
    ApexChartsRenderer,
    SVGChartRenderer,
    get_chart_renderer,
    CDN_URLS,
)

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

from truthound.datadocs.styles import (
    BASE_CSS,
    LAYOUT_CSS,
    COMPONENTS_CSS,
    RESPONSIVE_CSS,
    PRINT_CSS,
    DARK_MODE_OVERRIDES,
    get_complete_stylesheet,
)

from truthound.datadocs.builder import (
    ProfileDataConverter,
    HTMLReportBuilder,
    generate_html_report,
    generate_report_from_file,
    export_report,
    export_to_pdf,
)


# Dashboard imports (lazy - only if installed)
def launch_dashboard(*args, **kwargs):
    """Launch the interactive dashboard.

    Requires: pip install truthound[dashboard]
    """
    try:
        from truthound.datadocs.dashboard import launch_dashboard as _launch
        return _launch(*args, **kwargs)
    except ImportError as e:
        raise ImportError(
            "Dashboard requires additional dependencies. "
            "Install with: pip install truthound[dashboard]"
        ) from e


def create_dashboard_app(*args, **kwargs):
    """Create a dashboard application instance.

    Requires: pip install truthound[dashboard]
    """
    try:
        from truthound.datadocs.dashboard import create_app as _create
        return _create(*args, **kwargs)
    except ImportError as e:
        raise ImportError(
            "Dashboard requires additional dependencies. "
            "Install with: pip install truthound[dashboard]"
        ) from e


def get_dashboard_config():
    """Get dashboard configuration class.

    Requires: pip install truthound[dashboard]
    """
    try:
        from truthound.datadocs.dashboard import DashboardConfig
        return DashboardConfig
    except ImportError as e:
        raise ImportError(
            "Dashboard requires additional dependencies. "
            "Install with: pip install truthound[dashboard]"
        ) from e


__all__ = [
    # === Enums ===
    "ReportTheme",
    "ChartLibrary",
    "ChartType",
    "SectionType",
    "ExportFormat",
    "SeverityLevel",
    # === Data Structures ===
    "ThemeColors",
    "ThemeTypography",
    "ThemeSpacing",
    "ThemeConfig",
    "ReportMetadata",
    "ChartSpec",
    "SectionSpec",
    "AlertSpec",
    "ReportConfig",
    "ReportSpec",
    # === Protocols ===
    "ChartRenderer",
    "SectionRenderer",
    "TemplateRenderer",
    "ReportRenderer",
    # === Base Classes ===
    "BaseChartRenderer",
    "BaseSectionRenderer",
    "BaseReportRenderer",
    # === Registry ===
    "RendererRegistry",
    "renderer_registry",
    "register_chart_renderer",
    "register_section_renderer",
    # === Themes ===
    "LIGHT_THEME",
    "DARK_THEME",
    "PROFESSIONAL_THEME",
    "MINIMAL_THEME",
    "MODERN_THEME",
    "THEMES",
    "get_theme",
    "get_available_themes",
    # === Chart Renderers ===
    "ApexChartsRenderer",
    "SVGChartRenderer",
    "get_chart_renderer",
    "CDN_URLS",
    # === Section Renderers ===
    "OverviewSection",
    "ColumnsSection",
    "QualitySection",
    "PatternsSection",
    "DistributionSection",
    "CorrelationsSection",
    "RecommendationsSection",
    "AlertsSection",
    "CustomSection",
    "get_section_renderer",
    # === Styles ===
    "BASE_CSS",
    "LAYOUT_CSS",
    "COMPONENTS_CSS",
    "RESPONSIVE_CSS",
    "PRINT_CSS",
    "DARK_MODE_OVERRIDES",
    "get_complete_stylesheet",
    # === Builder ===
    "ProfileDataConverter",
    "HTMLReportBuilder",
    "generate_html_report",
    "generate_report_from_file",
    "export_report",
    "export_to_pdf",
    # === Dashboard (optional) ===
    "launch_dashboard",
    "create_dashboard_app",
    "get_dashboard_config",
]
