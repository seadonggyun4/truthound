"""Base types and abstractions for Data Docs.

This module provides the foundational data structures and protocols
for the Data Docs reporting system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


# =============================================================================
# Enums
# =============================================================================


class ReportTheme(str, Enum):
    """Available report themes."""

    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    MODERN = "modern"


class ChartLibrary(str, Enum):
    """Supported chart libraries (CDN-based).

    ApexCharts is the default for HTML reports (interactive, feature-rich).
    SVG is used for PDF export (no JavaScript dependency).
    """

    APEXCHARTS = "apexcharts"   # ApexCharts - modern, interactive (default)
    SVG = "svg"                 # Pure SVG - for PDF export


class SectionType(str, Enum):
    """Types of report sections."""

    OVERVIEW = "overview"
    COLUMNS = "columns"
    QUALITY = "quality"
    PATTERNS = "patterns"
    DISTRIBUTION = "distribution"
    CORRELATIONS = "correlations"
    RECOMMENDATIONS = "recommendations"
    ALERTS = "alerts"
    CUSTOM = "custom"


class ChartType(str, Enum):
    """Supported chart types."""

    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    LINE = "line"
    PIE = "pie"
    DONUT = "donut"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    BOX = "box"
    GAUGE = "gauge"
    RADAR = "radar"
    TABLE = "table"


class ExportFormat(str, Enum):
    """Report export formats."""

    HTML = "html"
    PDF = "pdf"
    PNG = "png"


class SeverityLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class ThemeColors:
    """Color palette for a theme."""

    # Main colors
    background: str = "#ffffff"
    surface: str = "#f8f9fa"
    text_primary: str = "#1a1a2e"
    text_secondary: str = "#6c757d"

    # Brand colors
    primary: str = "#4361ee"
    secondary: str = "#7209b7"
    accent: str = "#f72585"

    # Semantic colors
    success: str = "#10b981"
    warning: str = "#f59e0b"
    error: str = "#ef4444"
    info: str = "#3b82f6"

    # Border and shadow
    border: str = "#e5e7eb"
    shadow: str = "rgba(0, 0, 0, 0.05)"

    # Chart colors
    chart_palette: tuple[str, ...] = (
        "#4361ee", "#7209b7", "#f72585", "#4cc9f0",
        "#06d6a0", "#ffd166", "#ef476f", "#118ab2",
        "#073b4c", "#ff6b6b"
    )


@dataclass(frozen=True)
class ThemeTypography:
    """Typography settings for a theme."""

    font_family: str = "'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    font_family_mono: str = "'JetBrains Mono', 'SF Mono', 'Consolas', monospace"

    font_size_xs: str = "0.75rem"    # 12px
    font_size_sm: str = "0.875rem"   # 14px
    font_size_base: str = "1rem"     # 16px
    font_size_lg: str = "1.125rem"   # 18px
    font_size_xl: str = "1.25rem"    # 20px
    font_size_2xl: str = "1.5rem"    # 24px
    font_size_3xl: str = "2rem"      # 32px

    font_weight_normal: int = 400
    font_weight_medium: int = 500
    font_weight_semibold: int = 600
    font_weight_bold: int = 700

    line_height_tight: float = 1.25
    line_height_normal: float = 1.5
    line_height_relaxed: float = 1.75


@dataclass(frozen=True)
class ThemeSpacing:
    """Spacing settings for a theme."""

    border_radius_sm: str = "4px"
    border_radius_md: str = "8px"
    border_radius_lg: str = "12px"
    border_radius_xl: str = "16px"

    spacing_xs: str = "0.25rem"   # 4px
    spacing_sm: str = "0.5rem"    # 8px
    spacing_md: str = "1rem"      # 16px
    spacing_lg: str = "1.5rem"    # 24px
    spacing_xl: str = "2rem"      # 32px
    spacing_2xl: str = "3rem"     # 48px

    shadow_sm: str = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    shadow_md: str = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
    shadow_lg: str = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
    shadow_xl: str = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"


@dataclass(frozen=True)
class ThemeConfig:
    """Complete theme configuration."""

    name: str
    colors: ThemeColors = field(default_factory=ThemeColors)
    typography: ThemeTypography = field(default_factory=ThemeTypography)
    spacing: ThemeSpacing = field(default_factory=ThemeSpacing)

    def to_css_vars(self) -> str:
        """Generate CSS custom properties from theme."""
        lines = [":root {"]

        # Colors
        for key, value in vars(self.colors).items():
            if key == "chart_palette":
                for i, color in enumerate(value):
                    lines.append(f"    --chart-color-{i}: {color};")
            else:
                css_key = key.replace("_", "-")
                lines.append(f"    --color-{css_key}: {value};")

        # Typography
        for key, value in vars(self.typography).items():
            css_key = key.replace("_", "-")
            lines.append(f"    --{css_key}: {value};")

        # Spacing
        for key, value in vars(self.spacing).items():
            css_key = key.replace("_", "-")
            lines.append(f"    --{css_key}: {value};")

        lines.append("}")
        return "\n".join(lines)


@dataclass
class ReportMetadata:
    """Metadata for a generated report."""

    title: str = "Data Profile Report"
    subtitle: str = ""
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    data_source: str = ""
    version: str = "1.0.0"
    custom_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChartSpec:
    """Specification for a chart to render."""

    chart_type: ChartType
    title: str = ""
    subtitle: str = ""

    # Data
    labels: list[str] = field(default_factory=list)
    values: list[float | int] = field(default_factory=list)
    series: list[dict[str, Any]] | None = None  # For multi-series

    # Styling
    colors: list[str] | None = None
    height: int = 300
    width: int | None = None  # None = responsive

    # Options
    show_legend: bool = True
    show_labels: bool = True
    show_grid: bool = True
    animation: bool = True

    # Additional options
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertSpec:
    """Specification for an alert/warning."""

    title: str
    message: str
    severity: SeverityLevel = SeverityLevel.INFO
    column: str | None = None
    metric: str | None = None
    value: Any = None
    threshold: Any = None
    suggestion: str | None = None


@dataclass
class SectionSpec:
    """Specification for a report section."""

    section_type: SectionType
    title: str
    subtitle: str = ""

    # Content
    charts: list[ChartSpec] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    alerts: list[AlertSpec] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    custom_html: str = ""

    # Display options
    collapsible: bool = False
    collapsed_default: bool = False
    priority: int = 0  # Higher = shown first
    visible: bool = True

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    # Theme
    theme: ReportTheme = ReportTheme.PROFESSIONAL
    custom_theme: ThemeConfig | None = None

    # Chart library
    chart_library: ChartLibrary = ChartLibrary.APEXCHARTS

    # Sections to include
    sections: list[SectionType] = field(default_factory=lambda: [
        SectionType.OVERVIEW,
        SectionType.QUALITY,
        SectionType.COLUMNS,
        SectionType.PATTERNS,
        SectionType.DISTRIBUTION,
        SectionType.CORRELATIONS,
        SectionType.RECOMMENDATIONS,
        SectionType.ALERTS,
    ])

    # Layout
    include_toc: bool = True
    include_header: bool = True
    include_footer: bool = True
    include_timestamp: bool = True
    include_download_button: bool = True

    # Export options
    embed_resources: bool = True  # Embed CSS/JS for offline viewing
    minify_html: bool = False

    # Custom content
    custom_css: str = ""
    custom_js: str = ""
    logo_url: str | None = None
    logo_base64: str | None = None
    footer_text: str = "Generated by Truthound"

    # Localization
    language: str = "en"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    number_format: str = ",.2f"


@dataclass
class ReportSpec:
    """Complete specification for a report."""

    metadata: ReportMetadata
    config: ReportConfig
    sections: list[SectionSpec] = field(default_factory=list)

    # Raw profile data for reference
    profile_data: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class ChartRenderer(Protocol):
    """Protocol for chart rendering backends."""

    library: ChartLibrary

    def render(self, spec: ChartSpec) -> str:
        """Render a chart specification to HTML/JS."""
        ...

    def get_dependencies(self) -> list[str]:
        """Get CDN URLs for required JS/CSS dependencies."""
        ...


@runtime_checkable
class SectionRenderer(Protocol):
    """Protocol for section rendering."""

    section_type: SectionType

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render a section to HTML."""
        ...


@runtime_checkable
class TemplateRenderer(Protocol):
    """Protocol for template rendering engines."""

    def render(self, template: str, context: dict[str, Any]) -> str:
        """Render a template with context."""
        ...


@runtime_checkable
class ReportRenderer(Protocol):
    """Protocol for complete report rendering."""

    def render(self, spec: ReportSpec) -> str:
        """Render a complete report specification to HTML."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================


class BaseChartRenderer(ABC):
    """Base class for chart renderers."""

    library: ChartLibrary

    @abstractmethod
    def render(self, spec: ChartSpec) -> str:
        """Render a chart specification to HTML/JS."""
        pass

    @abstractmethod
    def get_dependencies(self) -> list[str]:
        """Get CDN URLs for required JS/CSS dependencies."""
        pass

    def _generate_chart_id(self) -> str:
        """Generate a unique chart ID."""
        import uuid
        return f"chart_{uuid.uuid4().hex[:8]}"

    def _format_data_for_js(self, data: Any) -> str:
        """Format Python data for JavaScript."""
        import json
        return json.dumps(data, default=str)


class BaseSectionRenderer(ABC):
    """Base class for section renderers."""

    section_type: SectionType

    @abstractmethod
    def render(
        self,
        spec: SectionSpec,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render a section to HTML."""
        pass

    def _render_charts(
        self,
        charts: list[ChartSpec],
        chart_renderer: ChartRenderer,
    ) -> str:
        """Render multiple charts."""
        return "\n".join(chart_renderer.render(chart) for chart in charts)

    def _render_alerts(self, alerts: list[AlertSpec], theme: ThemeConfig) -> str:
        """Render alerts with appropriate styling."""
        if not alerts:
            return ""

        severity_classes = {
            SeverityLevel.INFO: "alert-info",
            SeverityLevel.WARNING: "alert-warning",
            SeverityLevel.ERROR: "alert-error",
            SeverityLevel.CRITICAL: "alert-critical",
        }

        html_parts = ['<div class="alerts-container">']
        for alert in alerts:
            css_class = severity_classes.get(alert.severity, "alert-info")
            html_parts.append(f'''
                <div class="alert {css_class}">
                    <div class="alert-header">
                        <span class="alert-icon"></span>
                        <span class="alert-title">{alert.title}</span>
                    </div>
                    <div class="alert-message">{alert.message}</div>
                    {f'<div class="alert-suggestion">{alert.suggestion}</div>' if alert.suggestion else ''}
                </div>
            ''')
        html_parts.append('</div>')
        return "\n".join(html_parts)


class BaseReportRenderer(ABC):
    """Base class for report renderers."""

    @abstractmethod
    def render(self, spec: ReportSpec) -> str:
        """Render a complete report specification to HTML."""
        pass


# =============================================================================
# Registry
# =============================================================================


class RendererRegistry:
    """Registry for renderer implementations."""

    def __init__(self) -> None:
        self._chart_renderers: dict[ChartLibrary, type[BaseChartRenderer]] = {}
        self._section_renderers: dict[SectionType, type[BaseSectionRenderer]] = {}

    def register_chart_renderer(
        self,
        library: ChartLibrary,
        renderer_class: type[BaseChartRenderer],
    ) -> None:
        """Register a chart renderer."""
        self._chart_renderers[library] = renderer_class

    def register_section_renderer(
        self,
        section_type: SectionType,
        renderer_class: type[BaseSectionRenderer],
    ) -> None:
        """Register a section renderer."""
        self._section_renderers[section_type] = renderer_class

    def get_chart_renderer(self, library: ChartLibrary) -> type[BaseChartRenderer]:
        """Get a chart renderer class."""
        if library not in self._chart_renderers:
            raise KeyError(f"Chart renderer for {library.value} not registered")
        return self._chart_renderers[library]

    def get_section_renderer(self, section_type: SectionType) -> type[BaseSectionRenderer]:
        """Get a section renderer class."""
        if section_type not in self._section_renderers:
            raise KeyError(f"Section renderer for {section_type.value} not registered")
        return self._section_renderers[section_type]

    def list_chart_renderers(self) -> list[ChartLibrary]:
        """List available chart renderers."""
        return list(self._chart_renderers.keys())

    def list_section_renderers(self) -> list[SectionType]:
        """List available section renderers."""
        return list(self._section_renderers.keys())


# Global registry instance
renderer_registry = RendererRegistry()


# =============================================================================
# Decorator for registration
# =============================================================================


def register_chart_renderer(
    library: ChartLibrary,
) -> Callable[[type[BaseChartRenderer]], type[BaseChartRenderer]]:
    """Decorator to register a chart renderer."""
    def decorator(cls: type[BaseChartRenderer]) -> type[BaseChartRenderer]:
        renderer_registry.register_chart_renderer(library, cls)
        return cls
    return decorator


def register_section_renderer(
    section_type: SectionType,
) -> Callable[[type[BaseSectionRenderer]], type[BaseSectionRenderer]]:
    """Decorator to register a section renderer."""
    def decorator(cls: type[BaseSectionRenderer]) -> type[BaseSectionRenderer]:
        renderer_registry.register_section_renderer(section_type, cls)
        return cls
    return decorator
