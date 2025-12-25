"""Base types and data structures for visualization.

This module contains enums, dataclasses, and constants used across
the visualization system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# =============================================================================
# Enums
# =============================================================================


class ChartType(Enum):
    """Supported chart types."""

    BAR = "bar"
    HORIZONTAL_BAR = "horizontal_bar"
    PIE = "pie"
    DONUT = "donut"
    LINE = "line"
    AREA = "area"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TABLE = "table"
    GAUGE = "gauge"
    SPARKLINE = "sparkline"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    BOX = "box"
    VIOLIN = "violin"
    RADAR = "radar"
    FUNNEL = "funnel"


class ColorScheme(Enum):
    """Predefined color schemes."""

    DEFAULT = "default"
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"
    QUALITATIVE = "qualitative"
    TRAFFIC_LIGHT = "traffic_light"
    PASTEL = "pastel"
    VIBRANT = "vibrant"


class ReportTheme(Enum):
    """Report theme options."""

    LIGHT = "light"
    DARK = "dark"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    COLORFUL = "colorful"


class SectionType(Enum):
    """Types of report sections."""

    OVERVIEW = "overview"
    COLUMN_DETAILS = "column_details"
    DATA_QUALITY = "data_quality"
    STATISTICS = "statistics"
    PATTERNS = "patterns"
    ALERTS = "alerts"
    RECOMMENDATIONS = "recommendations"
    CORRELATIONS = "correlations"
    DISTRIBUTION = "distribution"
    CUSTOM = "custom"


# =============================================================================
# Color Palettes
# =============================================================================


COLOR_PALETTES: Dict[ColorScheme, List[str]] = {
    ColorScheme.DEFAULT: [
        "#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"
    ],
    ColorScheme.CATEGORICAL: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ],
    ColorScheme.SEQUENTIAL: [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b"
    ],
    ColorScheme.DIVERGING: [
        "#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffbf",
        "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"
    ],
    ColorScheme.QUALITATIVE: [
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd"
    ],
    ColorScheme.TRAFFIC_LIGHT: [
        "#2ecc71", "#f1c40f", "#e74c3c"  # green, yellow, red
    ],
    ColorScheme.PASTEL: [
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
    ],
    ColorScheme.VIBRANT: [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
        "#ffff33", "#a65628", "#f781bf", "#999999"
    ],
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChartData:
    """Data container for chart rendering."""

    labels: List[str] = field(default_factory=list)
    values: List[Union[int, float]] = field(default_factory=list)
    series: Optional[List[Dict[str, Any]]] = None  # For multi-series charts
    colors: Optional[List[str]] = None
    title: str = ""
    subtitle: str = ""
    x_label: str = ""
    y_label: str = ""
    show_legend: bool = True
    show_values: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "values": self.values,
            "series": self.series,
            "colors": self.colors,
            "title": self.title,
            "subtitle": self.subtitle,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "show_legend": self.show_legend,
            "show_values": self.show_values,
            "metadata": self.metadata,
        }


@dataclass
class ChartConfig:
    """Configuration for chart rendering."""

    chart_type: ChartType = ChartType.BAR
    width: int = 600
    height: int = 400
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    animation: bool = True
    interactive: bool = True
    responsive: bool = True
    custom_colors: Optional[List[str]] = None
    show_toolbar: bool = True
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chart_type": self.chart_type.value,
            "width": self.width,
            "height": self.height,
            "color_scheme": self.color_scheme.value,
            "animation": self.animation,
            "interactive": self.interactive,
            "responsive": self.responsive,
            "custom_colors": self.custom_colors,
            "show_toolbar": self.show_toolbar,
            "extra_options": self.extra_options,
        }


@dataclass
class ThemeConfig:
    """Theme configuration for reports."""

    name: str
    background_color: str = "#ffffff"
    text_color: str = "#333333"
    primary_color: str = "#4e79a7"
    secondary_color: str = "#f28e2c"
    accent_color: str = "#e15759"
    border_color: str = "#e0e0e0"
    font_family: str = "system-ui, -apple-system, sans-serif"
    header_bg: str = "#f8f9fa"
    card_bg: str = "#ffffff"
    shadow: str = "0 2px 4px rgba(0,0,0,0.1)"
    border_radius: str = "8px"

    def to_css_vars(self) -> str:
        """Generate CSS custom properties."""
        return f"""
        :root {{
            --bg-color: {self.background_color};
            --text-color: {self.text_color};
            --primary-color: {self.primary_color};
            --secondary-color: {self.secondary_color};
            --accent-color: {self.accent_color};
            --border-color: {self.border_color};
            --header-bg: {self.header_bg};
            --card-bg: {self.card_bg};
            --shadow: {self.shadow};
            --border-radius: {self.border_radius};
            --font-family: {self.font_family};
        }}
        """


@dataclass
class SectionContent:
    """Content for a report section."""

    section_type: SectionType
    title: str
    charts: List[Tuple[ChartData, ChartConfig]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    text_blocks: List[str] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    collapsible: bool = False
    priority: int = 0  # Higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Data Profile Report"
    subtitle: str = ""
    theme: ReportTheme = ReportTheme.LIGHT
    logo_path: Optional[str] = None
    logo_base64: Optional[str] = None
    include_toc: bool = True
    include_timestamp: bool = True
    include_summary: bool = True
    sections: List[SectionType] = field(default_factory=lambda: [
        SectionType.OVERVIEW,
        SectionType.DATA_QUALITY,
        SectionType.COLUMN_DETAILS,
        SectionType.PATTERNS,
        SectionType.RECOMMENDATIONS,
    ])
    custom_css: Optional[str] = None
    custom_js: Optional[str] = None
    embed_resources: bool = True
    language: str = "en"
    renderer: str = "auto"  # auto, svg, plotly, echarts


@dataclass
class ProfileData:
    """Container for profile data to be visualized."""

    table_name: str
    row_count: int
    column_count: int
    columns: List[Dict[str, Any]]  # Column profiles
    quality_scores: Optional[Dict[str, float]] = None
    patterns_found: Optional[List[Dict[str, Any]]] = None
    alerts: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[str]] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Theme Definitions
# =============================================================================


THEME_CONFIGS: Dict[ReportTheme, ThemeConfig] = {
    ReportTheme.LIGHT: ThemeConfig(
        name="light",
        background_color="#ffffff",
        text_color="#333333",
        primary_color="#4e79a7",
        secondary_color="#f28e2c",
        accent_color="#e15759",
        border_color="#e0e0e0",
        header_bg="#f8f9fa",
        card_bg="#ffffff",
    ),
    ReportTheme.DARK: ThemeConfig(
        name="dark",
        background_color="#1a1a2e",
        text_color="#eaeaea",
        primary_color="#64b5f6",
        secondary_color="#ffb74d",
        accent_color="#ef5350",
        border_color="#333355",
        header_bg="#16213e",
        card_bg="#0f3460",
        shadow="0 2px 4px rgba(0,0,0,0.3)",
    ),
    ReportTheme.PROFESSIONAL: ThemeConfig(
        name="professional",
        background_color="#f5f5f5",
        text_color="#2c3e50",
        primary_color="#2c3e50",
        secondary_color="#3498db",
        accent_color="#e74c3c",
        border_color="#bdc3c7",
        header_bg="#ecf0f1",
        card_bg="#ffffff",
        font_family="'Segoe UI', Tahoma, Geneva, sans-serif",
        border_radius="4px",
    ),
    ReportTheme.MINIMAL: ThemeConfig(
        name="minimal",
        background_color="#ffffff",
        text_color="#000000",
        primary_color="#000000",
        secondary_color="#666666",
        accent_color="#333333",
        border_color="#cccccc",
        header_bg="#ffffff",
        card_bg="#ffffff",
        shadow="none",
        border_radius="0",
    ),
    ReportTheme.COLORFUL: ThemeConfig(
        name="colorful",
        background_color="#fafafa",
        text_color="#333333",
        primary_color="#6c5ce7",
        secondary_color="#00b894",
        accent_color="#fd79a8",
        border_color="#dfe6e9",
        header_bg="#ffffff",
        card_bg="#ffffff",
        border_radius="12px",
    ),
}
