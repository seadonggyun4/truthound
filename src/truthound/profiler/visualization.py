"""
Profile Visualization Module - HTML Reports for Truthound.

This module provides comprehensive HTML report generation for data profiling results,
supporting multiple visualization types, themes, and export formats.

Architecture:
- ChartRenderer: Abstract base for chart rendering (supports multiple backends)
- ReportSection: Composable report sections
- ReportTemplate: Template engine for HTML generation
- HTMLReportGenerator: Main interface for generating reports
- ReportExporter: Export to various formats (HTML, PDF, JSON)

Extensibility:
- Custom chart renderers via ChartRendererRegistry
- Custom report sections via SectionRegistry
- Custom themes via ThemeRegistry
- Plugin system for additional visualizations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
import base64
import html
import io
import json
import os
import threading
from datetime import datetime
from pathlib import Path


# =============================================================================
# Enums and Constants
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


class ColorScheme(Enum):
    """Predefined color schemes."""
    DEFAULT = "default"
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"
    QUALITATIVE = "qualitative"
    TRAFFIC_LIGHT = "traffic_light"


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


@dataclass
class ChartConfig:
    """Configuration for chart rendering."""
    chart_type: ChartType = ChartType.BAR
    width: int = 600
    height: int = 400
    color_scheme: ColorScheme = ColorScheme.DEFAULT
    animation: bool = False
    interactive: bool = True
    responsive: bool = True
    custom_colors: Optional[List[str]] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)


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


# =============================================================================
# Abstract Base Classes and Protocols
# =============================================================================

@runtime_checkable
class ChartRendererProtocol(Protocol):
    """Protocol for chart renderers."""

    def render(self, data: ChartData, config: ChartConfig) -> str:
        """Render chart to HTML/SVG string."""
        ...

    def supports_chart_type(self, chart_type: ChartType) -> bool:
        """Check if renderer supports the chart type."""
        ...


class ChartRenderer(ABC):
    """Abstract base class for chart rendering."""

    @abstractmethod
    def render(self, data: ChartData, config: ChartConfig) -> str:
        """Render chart to HTML/SVG string."""
        pass

    @abstractmethod
    def supports_chart_type(self, chart_type: ChartType) -> bool:
        """Check if renderer supports the chart type."""
        pass

    def get_colors(self, config: ChartConfig, count: int) -> List[str]:
        """Get colors for the chart."""
        if config.custom_colors:
            return config.custom_colors[:count]
        palette = COLOR_PALETTES.get(config.color_scheme, COLOR_PALETTES[ColorScheme.DEFAULT])
        # Cycle through palette if needed
        return [palette[i % len(palette)] for i in range(count)]


class SectionRenderer(ABC):
    """Abstract base class for section rendering."""

    @abstractmethod
    def render(
        self,
        content: SectionContent,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render section to HTML string."""
        pass

    @abstractmethod
    def get_section_type(self) -> SectionType:
        """Get the section type this renderer handles."""
        pass


# =============================================================================
# Registries
# =============================================================================

class ChartRendererRegistry:
    """Registry for chart renderers."""

    _instance: Optional["ChartRendererRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ChartRendererRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._renderers: Dict[str, ChartRenderer] = {}
                    cls._instance._default: Optional[str] = None
        return cls._instance

    def register(self, name: str, renderer: ChartRenderer, default: bool = False) -> None:
        """Register a chart renderer."""
        self._renderers[name] = renderer
        if default or self._default is None:
            self._default = name

    def get(self, name: Optional[str] = None) -> Optional[ChartRenderer]:
        """Get a chart renderer by name or default."""
        if name is None:
            name = self._default
        return self._renderers.get(name) if name else None

    def list_renderers(self) -> List[str]:
        """List all registered renderer names."""
        return list(self._renderers.keys())

    def clear(self) -> None:
        """Clear all registered renderers."""
        self._renderers.clear()
        self._default = None


class SectionRegistry:
    """Registry for section renderers."""

    _instance: Optional["SectionRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SectionRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._renderers: Dict[SectionType, SectionRenderer] = {}
        return cls._instance

    def register(self, renderer: SectionRenderer) -> None:
        """Register a section renderer."""
        self._renderers[renderer.get_section_type()] = renderer

    def get(self, section_type: SectionType) -> Optional[SectionRenderer]:
        """Get a section renderer by type."""
        return self._renderers.get(section_type)

    def list_sections(self) -> List[SectionType]:
        """List all registered section types."""
        return list(self._renderers.keys())

    def clear(self) -> None:
        """Clear all registered section renderers."""
        self._renderers.clear()


class ThemeRegistry:
    """Registry for custom themes."""

    _instance: Optional["ThemeRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ThemeRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._themes: Dict[str, ThemeConfig] = {}
                    # Register built-in themes
                    for theme_enum, config in THEME_CONFIGS.items():
                        cls._instance._themes[theme_enum.value] = config
        return cls._instance

    def register(self, name: str, theme: ThemeConfig) -> None:
        """Register a custom theme."""
        self._themes[name] = theme

    def get(self, name: Union[str, ReportTheme]) -> ThemeConfig:
        """Get a theme by name."""
        if isinstance(name, ReportTheme):
            name = name.value
        return self._themes.get(name, THEME_CONFIGS[ReportTheme.LIGHT])

    def list_themes(self) -> List[str]:
        """List all registered theme names."""
        return list(self._themes.keys())


# Singleton instances
chart_renderer_registry = ChartRendererRegistry()
section_registry = SectionRegistry()
theme_registry = ThemeRegistry()


# =============================================================================
# Built-in Chart Renderers
# =============================================================================

class SVGChartRenderer(ChartRenderer):
    """Pure SVG chart renderer - no external dependencies."""

    SUPPORTED_TYPES = {
        ChartType.BAR,
        ChartType.HORIZONTAL_BAR,
        ChartType.PIE,
        ChartType.DONUT,
        ChartType.LINE,
        ChartType.HISTOGRAM,
        ChartType.GAUGE,
        ChartType.SPARKLINE,
        ChartType.TABLE,
    }

    def supports_chart_type(self, chart_type: ChartType) -> bool:
        return chart_type in self.SUPPORTED_TYPES

    def render(self, data: ChartData, config: ChartConfig) -> str:
        """Render chart to SVG string."""
        if config.chart_type == ChartType.BAR:
            return self._render_bar(data, config)
        elif config.chart_type == ChartType.HORIZONTAL_BAR:
            return self._render_horizontal_bar(data, config)
        elif config.chart_type == ChartType.PIE:
            return self._render_pie(data, config, donut=False)
        elif config.chart_type == ChartType.DONUT:
            return self._render_pie(data, config, donut=True)
        elif config.chart_type == ChartType.LINE:
            return self._render_line(data, config)
        elif config.chart_type == ChartType.HISTOGRAM:
            return self._render_histogram(data, config)
        elif config.chart_type == ChartType.GAUGE:
            return self._render_gauge(data, config)
        elif config.chart_type == ChartType.SPARKLINE:
            return self._render_sparkline(data, config)
        elif config.chart_type == ChartType.TABLE:
            return self._render_table(data, config)
        else:
            return f"<p>Chart type {config.chart_type.value} not supported by SVG renderer</p>"

    def _svg_header(self, config: ChartConfig) -> str:
        """Generate SVG header with proper attributes."""
        responsive = 'viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet"'.format(
            w=config.width, h=config.height
        ) if config.responsive else f'width="{config.width}" height="{config.height}"'

        return f'''<svg xmlns="http://www.w3.org/2000/svg" {responsive}
            style="max-width: 100%; height: auto;">'''

    def _render_bar(self, data: ChartData, config: ChartConfig) -> str:
        """Render vertical bar chart."""
        if not data.values:
            return "<p>No data available</p>"

        w, h = config.width, config.height
        margin = {"top": 40, "right": 20, "bottom": 60, "left": 60}
        chart_w = w - margin["left"] - margin["right"]
        chart_h = h - margin["top"] - margin["bottom"]

        colors = self.get_colors(config, len(data.values))
        max_val = max(data.values) if data.values else 1
        bar_width = chart_w / len(data.values) * 0.8
        bar_gap = chart_w / len(data.values) * 0.2

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                      f'font-size="16" font-weight="bold">{html.escape(data.title)}</text>')

        # Y-axis
        svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
                  f'x2="{margin["left"]}" y2="{h - margin["bottom"]}" '
                  f'stroke="#ccc" stroke-width="1"/>')

        # X-axis
        svg.append(f'<line x1="{margin["left"]}" y1="{h - margin["bottom"]}" '
                  f'x2="{w - margin["right"]}" y2="{h - margin["bottom"]}" '
                  f'stroke="#ccc" stroke-width="1"/>')

        # Bars
        for i, (label, value) in enumerate(zip(data.labels, data.values)):
            x = margin["left"] + i * (bar_width + bar_gap) + bar_gap / 2
            bar_h = (value / max_val) * chart_h if max_val > 0 else 0
            y = h - margin["bottom"] - bar_h

            # Bar
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
                      f'fill="{colors[i]}" rx="2">'
                      f'<title>{html.escape(label)}: {value}</title></rect>')

            # Value label
            if data.show_values:
                svg.append(f'<text x="{x + bar_width/2}" y="{y - 5}" '
                          f'text-anchor="middle" font-size="11">{value:.1f}</text>')

            # X-axis label
            label_y = h - margin["bottom"] + 15
            svg.append(f'<text x="{x + bar_width/2}" y="{label_y}" '
                      f'text-anchor="middle" font-size="10" '
                      f'transform="rotate(-45 {x + bar_width/2} {label_y})">'
                      f'{html.escape(str(label)[:15])}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_horizontal_bar(self, data: ChartData, config: ChartConfig) -> str:
        """Render horizontal bar chart."""
        if not data.values:
            return "<p>No data available</p>"

        w, h = config.width, config.height
        margin = {"top": 40, "right": 60, "bottom": 30, "left": 120}
        chart_w = w - margin["left"] - margin["right"]
        chart_h = h - margin["top"] - margin["bottom"]

        colors = self.get_colors(config, len(data.values))
        max_val = max(data.values) if data.values else 1
        bar_height = chart_h / len(data.values) * 0.8
        bar_gap = chart_h / len(data.values) * 0.2

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                      f'font-size="16" font-weight="bold">{html.escape(data.title)}</text>')

        # Bars
        for i, (label, value) in enumerate(zip(data.labels, data.values)):
            y = margin["top"] + i * (bar_height + bar_gap) + bar_gap / 2
            bar_w = (value / max_val) * chart_w if max_val > 0 else 0

            # Label
            svg.append(f'<text x="{margin["left"] - 5}" y="{y + bar_height/2 + 4}" '
                      f'text-anchor="end" font-size="11">{html.escape(str(label)[:20])}</text>')

            # Bar
            svg.append(f'<rect x="{margin["left"]}" y="{y}" width="{bar_w}" height="{bar_height}" '
                      f'fill="{colors[i]}" rx="2">'
                      f'<title>{html.escape(label)}: {value}</title></rect>')

            # Value
            if data.show_values:
                svg.append(f'<text x="{margin["left"] + bar_w + 5}" y="{y + bar_height/2 + 4}" '
                          f'font-size="11">{value:.1f}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_pie(self, data: ChartData, config: ChartConfig, donut: bool = False) -> str:
        """Render pie or donut chart."""
        if not data.values:
            return "<p>No data available</p>"

        import math

        w, h = config.width, config.height
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 40
        inner_radius = radius * 0.6 if donut else 0

        colors = self.get_colors(config, len(data.values))
        total = sum(data.values) if data.values else 1

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                      f'font-size="16" font-weight="bold">{html.escape(data.title)}</text>')

        # Slices
        start_angle = -math.pi / 2
        for i, (label, value) in enumerate(zip(data.labels, data.values)):
            if value <= 0:
                continue

            angle = (value / total) * 2 * math.pi
            end_angle = start_angle + angle

            # Arc path
            large_arc = 1 if angle > math.pi else 0

            # Outer arc
            x1 = cx + radius * math.cos(start_angle)
            y1 = cy + radius * math.sin(start_angle)
            x2 = cx + radius * math.cos(end_angle)
            y2 = cy + radius * math.sin(end_angle)

            if donut:
                # Inner arc
                ix1 = cx + inner_radius * math.cos(start_angle)
                iy1 = cy + inner_radius * math.sin(start_angle)
                ix2 = cx + inner_radius * math.cos(end_angle)
                iy2 = cy + inner_radius * math.sin(end_angle)

                path = (f'M {ix1} {iy1} L {x1} {y1} '
                       f'A {radius} {radius} 0 {large_arc} 1 {x2} {y2} '
                       f'L {ix2} {iy2} '
                       f'A {inner_radius} {inner_radius} 0 {large_arc} 0 {ix1} {iy1} Z')
            else:
                path = (f'M {cx} {cy} L {x1} {y1} '
                       f'A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z')

            pct = (value / total) * 100
            svg.append(f'<path d="{path}" fill="{colors[i]}">'
                      f'<title>{html.escape(label)}: {value} ({pct:.1f}%)</title></path>')

            # Label
            if pct > 5:
                label_angle = start_angle + angle / 2
                label_radius = radius * 0.7 if not donut else (radius + inner_radius) / 2
                lx = cx + label_radius * math.cos(label_angle)
                ly = cy + label_radius * math.sin(label_angle)
                svg.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" '
                          f'font-size="10" fill="white">{pct:.1f}%</text>')

            start_angle = end_angle

        # Legend
        if data.show_legend:
            legend_x = w - 100
            for i, label in enumerate(data.labels[:8]):  # Limit legend items
                ly = 50 + i * 18
                svg.append(f'<rect x="{legend_x}" y="{ly}" width="12" height="12" fill="{colors[i]}"/>')
                svg.append(f'<text x="{legend_x + 16}" y="{ly + 10}" font-size="10">'
                          f'{html.escape(str(label)[:12])}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_line(self, data: ChartData, config: ChartConfig) -> str:
        """Render line chart."""
        if not data.values:
            return "<p>No data available</p>"

        w, h = config.width, config.height
        margin = {"top": 40, "right": 20, "bottom": 60, "left": 60}
        chart_w = w - margin["left"] - margin["right"]
        chart_h = h - margin["top"] - margin["bottom"]

        colors = self.get_colors(config, 1)
        max_val = max(data.values) if data.values else 1
        min_val = min(data.values) if data.values else 0
        val_range = max_val - min_val if max_val != min_val else 1

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                      f'font-size="16" font-weight="bold">{html.escape(data.title)}</text>')

        # Grid and axes
        svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
                  f'x2="{margin["left"]}" y2="{h - margin["bottom"]}" stroke="#ccc"/>')
        svg.append(f'<line x1="{margin["left"]}" y1="{h - margin["bottom"]}" '
                  f'x2="{w - margin["right"]}" y2="{h - margin["bottom"]}" stroke="#ccc"/>')

        # Build path
        points = []
        step = chart_w / (len(data.values) - 1) if len(data.values) > 1 else 0

        for i, value in enumerate(data.values):
            x = margin["left"] + i * step
            y = h - margin["bottom"] - ((value - min_val) / val_range) * chart_h
            points.append(f'{x},{y}')

        # Line
        svg.append(f'<polyline points="{" ".join(points)}" '
                  f'fill="none" stroke="{colors[0]}" stroke-width="2"/>')

        # Points
        for i, (value, point) in enumerate(zip(data.values, points)):
            x, y = point.split(',')
            svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{colors[0]}">'
                      f'<title>{data.labels[i] if i < len(data.labels) else i}: {value}</title></circle>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_histogram(self, data: ChartData, config: ChartConfig) -> str:
        """Render histogram (bar chart with no gaps)."""
        if not data.values:
            return "<p>No data available</p>"

        w, h = config.width, config.height
        margin = {"top": 40, "right": 20, "bottom": 60, "left": 60}
        chart_w = w - margin["left"] - margin["right"]
        chart_h = h - margin["top"] - margin["bottom"]

        colors = self.get_colors(config, 1)
        max_val = max(data.values) if data.values else 1
        bar_width = chart_w / len(data.values)

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                      f'font-size="16" font-weight="bold">{html.escape(data.title)}</text>')

        # Axes
        svg.append(f'<line x1="{margin["left"]}" y1="{h - margin["bottom"]}" '
                  f'x2="{w - margin["right"]}" y2="{h - margin["bottom"]}" stroke="#ccc"/>')

        # Bars
        for i, value in enumerate(data.values):
            x = margin["left"] + i * bar_width
            bar_h = (value / max_val) * chart_h if max_val > 0 else 0
            y = h - margin["bottom"] - bar_h

            label = data.labels[i] if i < len(data.labels) else str(i)
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
                      f'fill="{colors[0]}" stroke="white" stroke-width="0.5">'
                      f'<title>{html.escape(label)}: {value}</title></rect>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_gauge(self, data: ChartData, config: ChartConfig) -> str:
        """Render gauge chart."""
        import math

        if not data.values:
            return "<p>No data available</p>"

        value = data.values[0]
        max_val = data.metadata.get("max", 100)
        min_val = data.metadata.get("min", 0)

        w, h = config.width, min(config.height, config.width * 0.6)
        cx, cy = w / 2, h - 30
        radius = min(w, h * 1.5) / 2 - 40

        # Calculate angle (180 degrees for gauge)
        ratio = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
        ratio = max(0, min(1, ratio))
        angle = math.pi * (1 - ratio)

        # Color based on value
        if ratio < 0.33:
            color = COLOR_PALETTES[ColorScheme.TRAFFIC_LIGHT][2]  # Red
        elif ratio < 0.67:
            color = COLOR_PALETTES[ColorScheme.TRAFFIC_LIGHT][1]  # Yellow
        else:
            color = COLOR_PALETTES[ColorScheme.TRAFFIC_LIGHT][0]  # Green

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                      f'font-size="16" font-weight="bold">{html.escape(data.title)}</text>')

        # Background arc
        svg.append(f'<path d="M {cx - radius} {cy} A {radius} {radius} 0 0 1 {cx + radius} {cy}" '
                  f'fill="none" stroke="#e0e0e0" stroke-width="20" stroke-linecap="round"/>')

        # Value arc
        end_x = cx + radius * math.cos(angle)
        end_y = cy - radius * math.sin(angle)
        large_arc = 1 if ratio > 0.5 else 0

        svg.append(f'<path d="M {cx - radius} {cy} A {radius} {radius} 0 {large_arc} 1 {end_x} {end_y}" '
                  f'fill="none" stroke="{color}" stroke-width="20" stroke-linecap="round"/>')

        # Needle
        needle_x = cx + (radius - 30) * math.cos(angle)
        needle_y = cy - (radius - 30) * math.sin(angle)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{needle_x}" y2="{needle_y}" '
                  f'stroke="#333" stroke-width="3" stroke-linecap="round"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="#333"/>')

        # Value text
        svg.append(f'<text x="{cx}" y="{cy - 20}" text-anchor="middle" '
                  f'font-size="24" font-weight="bold">{value:.1f}</text>')

        # Min/Max labels
        svg.append(f'<text x="{cx - radius}" y="{cy + 20}" text-anchor="middle" '
                  f'font-size="11">{min_val}</text>')
        svg.append(f'<text x="{cx + radius}" y="{cy + 20}" text-anchor="middle" '
                  f'font-size="11">{max_val}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_sparkline(self, data: ChartData, config: ChartConfig) -> str:
        """Render sparkline (mini line chart)."""
        if not data.values:
            return ""

        w = config.width
        h = config.height

        colors = self.get_colors(config, 1)
        max_val = max(data.values)
        min_val = min(data.values)
        val_range = max_val - min_val if max_val != min_val else 1

        step = w / (len(data.values) - 1) if len(data.values) > 1 else 0

        points = []
        for i, value in enumerate(data.values):
            x = i * step
            y = h - ((value - min_val) / val_range) * h
            points.append(f'{x},{y}')

        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
            <polyline points="{" ".join(points)}" fill="none" stroke="{colors[0]}" stroke-width="1.5"/>
        </svg>'''

    def _render_table(self, data: ChartData, config: ChartConfig) -> str:
        """Render data as HTML table."""
        if not data.labels and not data.values:
            return "<p>No data available</p>"

        html_parts = ['<table class="data-table">']

        # Header
        if data.metadata.get("headers"):
            html_parts.append('<thead><tr>')
            for header in data.metadata["headers"]:
                html_parts.append(f'<th>{html.escape(str(header))}</th>')
            html_parts.append('</tr></thead>')

        # Body
        html_parts.append('<tbody>')
        rows = data.metadata.get("rows", [])
        if rows:
            for row in rows:
                html_parts.append('<tr>')
                for cell in row:
                    html_parts.append(f'<td>{html.escape(str(cell))}</td>')
                html_parts.append('</tr>')
        else:
            # Use labels/values as simple two-column table
            for label, value in zip(data.labels, data.values):
                html_parts.append(f'<tr><td>{html.escape(str(label))}</td>'
                                 f'<td>{value}</td></tr>')

        html_parts.append('</tbody></table>')
        return '\n'.join(html_parts)


# =============================================================================
# Section Renderers
# =============================================================================

class BaseSectionRenderer(SectionRenderer):
    """Base class for section renderers with common functionality."""

    def render_charts(
        self,
        charts: List[Tuple[ChartData, ChartConfig]],
        chart_renderer: ChartRenderer,
    ) -> str:
        """Render all charts in the section."""
        html_parts = []
        for data, config in charts:
            if chart_renderer.supports_chart_type(config.chart_type):
                chart_html = chart_renderer.render(data, config)
                html_parts.append(f'<div class="chart-container">{chart_html}</div>')
        return '\n'.join(html_parts)

    def render_tables(self, tables: List[Dict[str, Any]]) -> str:
        """Render all tables in the section."""
        html_parts = []
        for table in tables:
            html_parts.append('<table class="data-table">')

            # Headers
            if "headers" in table:
                html_parts.append('<thead><tr>')
                for header in table["headers"]:
                    html_parts.append(f'<th>{html.escape(str(header))}</th>')
                html_parts.append('</tr></thead>')

            # Rows
            html_parts.append('<tbody>')
            for row in table.get("rows", []):
                html_parts.append('<tr>')
                for cell in row:
                    html_parts.append(f'<td>{html.escape(str(cell))}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody></table>')

        return '\n'.join(html_parts)

    def render_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """Render alert boxes."""
        html_parts = []
        for alert in alerts:
            level = alert.get("level", "info")
            message = alert.get("message", "")
            html_parts.append(f'<div class="alert alert-{level}">{html.escape(message)}</div>')
        return '\n'.join(html_parts)


class OverviewSectionRenderer(BaseSectionRenderer):
    """Renderer for overview section."""

    def get_section_type(self) -> SectionType:
        return SectionType.OVERVIEW

    def render(
        self,
        content: SectionContent,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        html_parts = [
            f'<section class="report-section section-overview" id="section-overview">',
            f'<h2>{html.escape(content.title)}</h2>',
            '<div class="section-content">',
        ]

        # Text blocks (summary info)
        for text in content.text_blocks:
            html_parts.append(f'<p>{text}</p>')

        # Charts (usually pie/donut for data types, gauges for quality)
        html_parts.append('<div class="charts-grid">')
        html_parts.append(self.render_charts(content.charts, chart_renderer))
        html_parts.append('</div>')

        # Tables (summary statistics)
        html_parts.append(self.render_tables(content.tables))

        html_parts.extend(['</div>', '</section>'])
        return '\n'.join(html_parts)


class DataQualitySectionRenderer(BaseSectionRenderer):
    """Renderer for data quality section."""

    def get_section_type(self) -> SectionType:
        return SectionType.DATA_QUALITY

    def render(
        self,
        content: SectionContent,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        html_parts = [
            f'<section class="report-section section-quality" id="section-quality">',
            f'<h2>{html.escape(content.title)}</h2>',
            '<div class="section-content">',
        ]

        # Alerts first (important issues)
        if content.alerts:
            html_parts.append('<div class="alerts-container">')
            html_parts.append(self.render_alerts(content.alerts))
            html_parts.append('</div>')

        # Quality gauges
        html_parts.append('<div class="quality-gauges">')
        html_parts.append(self.render_charts(content.charts, chart_renderer))
        html_parts.append('</div>')

        # Detailed tables
        html_parts.append(self.render_tables(content.tables))

        html_parts.extend(['</div>', '</section>'])
        return '\n'.join(html_parts)


class ColumnDetailsSectionRenderer(BaseSectionRenderer):
    """Renderer for column details section."""

    def get_section_type(self) -> SectionType:
        return SectionType.COLUMN_DETAILS

    def render(
        self,
        content: SectionContent,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        collapsible_attr = 'class="collapsible"' if content.collapsible else ''
        html_parts = [
            f'<section class="report-section section-columns" id="section-columns" {collapsible_attr}>',
            f'<h2>{html.escape(content.title)}</h2>',
            '<div class="section-content">',
        ]

        # Column cards
        for i, (data, config) in enumerate(content.charts):
            column_name = data.title or f"Column {i + 1}"
            html_parts.append(f'''
                <div class="column-card">
                    <h3>{html.escape(column_name)}</h3>
                    <div class="column-chart">
                        {chart_renderer.render(data, config) if chart_renderer.supports_chart_type(config.chart_type) else ''}
                    </div>
                </div>
            ''')

        # Summary table
        html_parts.append(self.render_tables(content.tables))

        html_parts.extend(['</div>', '</section>'])
        return '\n'.join(html_parts)


class PatternsSectionRenderer(BaseSectionRenderer):
    """Renderer for patterns section."""

    def get_section_type(self) -> SectionType:
        return SectionType.PATTERNS

    def render(
        self,
        content: SectionContent,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        html_parts = [
            f'<section class="report-section section-patterns" id="section-patterns">',
            f'<h2>{html.escape(content.title)}</h2>',
            '<div class="section-content">',
        ]

        # Pattern charts
        html_parts.append('<div class="patterns-charts">')
        html_parts.append(self.render_charts(content.charts, chart_renderer))
        html_parts.append('</div>')

        # Pattern tables
        html_parts.append(self.render_tables(content.tables))

        html_parts.extend(['</div>', '</section>'])
        return '\n'.join(html_parts)


class RecommendationsSectionRenderer(BaseSectionRenderer):
    """Renderer for recommendations section."""

    def get_section_type(self) -> SectionType:
        return SectionType.RECOMMENDATIONS

    def render(
        self,
        content: SectionContent,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        html_parts = [
            f'<section class="report-section section-recommendations" id="section-recommendations">',
            f'<h2>{html.escape(content.title)}</h2>',
            '<div class="section-content">',
        ]

        # Recommendation list
        if content.text_blocks:
            html_parts.append('<ul class="recommendations-list">')
            for rec in content.text_blocks:
                html_parts.append(f'<li>{html.escape(rec)}</li>')
            html_parts.append('</ul>')

        # Additional info
        html_parts.append(self.render_tables(content.tables))

        html_parts.extend(['</div>', '</section>'])
        return '\n'.join(html_parts)


class CustomSectionRenderer(BaseSectionRenderer):
    """Renderer for custom sections."""

    def get_section_type(self) -> SectionType:
        return SectionType.CUSTOM

    def render(
        self,
        content: SectionContent,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        html_parts = [
            f'<section class="report-section section-custom" id="section-{content.metadata.get("id", "custom")}">',
            f'<h2>{html.escape(content.title)}</h2>',
            '<div class="section-content">',
        ]

        # Render in order: text, alerts, charts, tables
        for text in content.text_blocks:
            html_parts.append(f'<p>{text}</p>')

        html_parts.append(self.render_alerts(content.alerts))
        html_parts.append(self.render_charts(content.charts, chart_renderer))
        html_parts.append(self.render_tables(content.tables))

        html_parts.extend(['</div>', '</section>'])
        return '\n'.join(html_parts)


# =============================================================================
# Template Engine
# =============================================================================

class ReportTemplate:
    """Template engine for HTML report generation."""

    def __init__(self, theme: ThemeConfig):
        self.theme = theme

    def get_css(self, custom_css: Optional[str] = None) -> str:
        """Generate CSS styles for the report."""
        t = self.theme
        css = f'''
        :root {{
            --bg-color: {t.background_color};
            --text-color: {t.text_color};
            --primary-color: {t.primary_color};
            --secondary-color: {t.secondary_color};
            --accent-color: {t.accent_color};
            --border-color: {t.border_color};
            --header-bg: {t.header_bg};
            --card-bg: {t.card_bg};
            --shadow: {t.shadow};
            --border-radius: {t.border_radius};
            --font-family: {t.font_family};
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: var(--font-family);
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}

        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .report-header {{
            background: var(--header-bg);
            padding: 30px;
            border-radius: var(--border-radius);
            margin-bottom: 30px;
            box-shadow: var(--shadow);
        }}

        .report-header h1 {{
            color: var(--primary-color);
            font-size: 2rem;
            margin-bottom: 10px;
        }}

        .report-header .subtitle {{
            color: var(--text-color);
            opacity: 0.8;
            font-size: 1.1rem;
        }}

        .report-header .timestamp {{
            color: var(--text-color);
            opacity: 0.6;
            font-size: 0.9rem;
            margin-top: 10px;
        }}

        .report-logo {{
            max-height: 60px;
            margin-bottom: 15px;
        }}

        .toc {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: var(--border-radius);
            margin-bottom: 30px;
            box-shadow: var(--shadow);
        }}

        .toc h3 {{
            margin-bottom: 15px;
            color: var(--primary-color);
        }}

        .toc ul {{
            list-style: none;
        }}

        .toc li {{
            padding: 5px 0;
        }}

        .toc a {{
            color: var(--secondary-color);
            text-decoration: none;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        .report-section {{
            background: var(--card-bg);
            padding: 25px;
            border-radius: var(--border-radius);
            margin-bottom: 25px;
            box-shadow: var(--shadow);
        }}

        .report-section h2 {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .chart-container {{
            background: var(--bg-color);
            padding: 15px;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
        }}

        .column-card {{
            background: var(--bg-color);
            padding: 20px;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            margin-bottom: 15px;
        }}

        .column-card h3 {{
            color: var(--secondary-color);
            margin-bottom: 15px;
            font-size: 1.1rem;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}

        .data-table th,
        .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        .data-table th {{
            background: var(--header-bg);
            font-weight: 600;
            color: var(--primary-color);
        }}

        .data-table tr:hover {{
            background: var(--header-bg);
        }}

        .alert {{
            padding: 15px;
            border-radius: var(--border-radius);
            margin: 10px 0;
        }}

        .alert-info {{
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            color: #1565c0;
        }}

        .alert-warning {{
            background: #fff3e0;
            border-left: 4px solid #ff9800;
            color: #e65100;
        }}

        .alert-error {{
            background: #ffebee;
            border-left: 4px solid #f44336;
            color: #c62828;
        }}

        .alert-success {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            color: #2e7d32;
        }}

        .quality-gauges {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin: 20px 0;
        }}

        .recommendations-list {{
            list-style: disc;
            padding-left: 25px;
        }}

        .recommendations-list li {{
            padding: 8px 0;
        }}

        .report-footer {{
            text-align: center;
            padding: 20px;
            color: var(--text-color);
            opacity: 0.6;
            font-size: 0.9rem;
        }}

        @media print {{
            body {{
                padding: 0;
            }}
            .report-section {{
                break-inside: avoid;
                box-shadow: none;
                border: 1px solid var(--border-color);
            }}
        }}
        '''

        if custom_css:
            css += f'\n/* Custom CSS */\n{custom_css}'

        return css

    def get_js(self, custom_js: Optional[str] = None) -> str:
        """Generate JavaScript for the report."""
        js = '''
        document.addEventListener('DOMContentLoaded', function() {
            // Smooth scroll for TOC links
            document.querySelectorAll('.toc a').forEach(function(link) {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    var targetId = this.getAttribute('href').substring(1);
                    var target = document.getElementById(targetId);
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });

            // Collapsible sections
            document.querySelectorAll('.collapsible h2').forEach(function(header) {
                header.style.cursor = 'pointer';
                header.addEventListener('click', function() {
                    var content = this.nextElementSibling;
                    if (content.style.display === 'none') {
                        content.style.display = 'block';
                        this.classList.remove('collapsed');
                    } else {
                        content.style.display = 'none';
                        this.classList.add('collapsed');
                    }
                });
            });
        });
        '''

        if custom_js:
            js += f'\n// Custom JS\n{custom_js}'

        return js

    def render_header(self, config: ReportConfig) -> str:
        """Render report header."""
        logo_html = ""
        if config.logo_base64:
            logo_html = f'<img src="data:image/png;base64,{config.logo_base64}" class="report-logo" alt="Logo">'
        elif config.logo_path and os.path.exists(config.logo_path):
            with open(config.logo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = Path(config.logo_path).suffix.lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif", "svg": "image/svg+xml"}.get(ext[1:], "image/png")
            logo_html = f'<img src="data:{mime};base64,{b64}" class="report-logo" alt="Logo">'

        timestamp_html = ""
        if config.include_timestamp:
            timestamp_html = f'<div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>'

        subtitle_html = ""
        if config.subtitle:
            subtitle_html = f'<div class="subtitle">{html.escape(config.subtitle)}</div>'

        return f'''
        <header class="report-header">
            {logo_html}
            <h1>{html.escape(config.title)}</h1>
            {subtitle_html}
            {timestamp_html}
        </header>
        '''

    def render_toc(self, sections: List[SectionContent]) -> str:
        """Render table of contents."""
        toc_items = []
        for section in sections:
            section_id = f"section-{section.section_type.value}"
            toc_items.append(f'<li><a href="#{section_id}">{html.escape(section.title)}</a></li>')

        return f'''
        <nav class="toc">
            <h3>Table of Contents</h3>
            <ul>
                {''.join(toc_items)}
            </ul>
        </nav>
        '''

    def render_footer(self) -> str:
        """Render report footer."""
        return '''
        <footer class="report-footer">
            <p>Generated by Truthound Data Profiler</p>
        </footer>
        '''

    def render_document(
        self,
        config: ReportConfig,
        sections_html: str,
        toc_html: str = "",
    ) -> str:
        """Render complete HTML document."""
        css = self.get_css(config.custom_css)
        js = self.get_js(config.custom_js)
        header = self.render_header(config)
        footer = self.render_footer()

        return f'''<!DOCTYPE html>
<html lang="{config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(config.title)}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="report-container">
        {header}
        {toc_html}
        <main class="report-content">
            {sections_html}
        </main>
        {footer}
    </div>
    <script>
{js}
    </script>
</body>
</html>'''


# =============================================================================
# Profile Data Converter
# =============================================================================

class ProfileDataConverter:
    """Converts ProfileData to SectionContent for rendering."""

    def __init__(self, profile: ProfileData):
        self.profile = profile

    def create_overview_section(self) -> SectionContent:
        """Create overview section from profile data."""
        # Summary text
        text_blocks = [
            f"Table: <strong>{html.escape(self.profile.table_name)}</strong>",
            f"Rows: <strong>{self.profile.row_count:,}</strong> | "
            f"Columns: <strong>{self.profile.column_count}</strong>",
        ]

        if self.profile.timestamp:
            text_blocks.append(f"Profiled at: {self.profile.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # Data type distribution chart
        type_counts: Dict[str, int] = {}
        for col in self.profile.columns:
            dtype = col.get("inferred_type", col.get("data_type", "unknown"))
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        charts = []
        if type_counts:
            charts.append((
                ChartData(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    title="Data Type Distribution",
                    show_legend=True,
                ),
                ChartConfig(chart_type=ChartType.DONUT, width=350, height=300)
            ))

        # Summary statistics table
        tables = []
        if self.profile.columns:
            tables.append({
                "headers": ["Metric", "Value"],
                "rows": [
                    ["Total Columns", self.profile.column_count],
                    ["Total Rows", f"{self.profile.row_count:,}"],
                    ["Numeric Columns", sum(1 for c in self.profile.columns if c.get("is_numeric", False))],
                    ["Text Columns", sum(1 for c in self.profile.columns if c.get("data_type") == "string")],
                    ["Columns with Nulls", sum(1 for c in self.profile.columns if c.get("null_count", 0) > 0)],
                ]
            })

        return SectionContent(
            section_type=SectionType.OVERVIEW,
            title="Overview",
            text_blocks=text_blocks,
            charts=charts,
            tables=tables,
            priority=100,
        )

    def create_quality_section(self) -> SectionContent:
        """Create data quality section from profile data."""
        charts = []
        alerts = []
        tables = []

        # Quality score gauge
        if self.profile.quality_scores:
            overall = self.profile.quality_scores.get("overall", 0)
            charts.append((
                ChartData(
                    values=[overall * 100],
                    title="Overall Quality Score",
                    metadata={"min": 0, "max": 100},
                ),
                ChartConfig(chart_type=ChartType.GAUGE, width=250, height=180)
            ))

        # Completeness chart
        completeness_data = []
        for col in self.profile.columns[:10]:  # Top 10
            null_pct = (col.get("null_count", 0) / self.profile.row_count * 100) if self.profile.row_count > 0 else 0
            completeness_data.append({
                "name": col.get("name", "unknown"),
                "completeness": 100 - null_pct,
            })

        if completeness_data:
            charts.append((
                ChartData(
                    labels=[d["name"] for d in completeness_data],
                    values=[d["completeness"] for d in completeness_data],
                    title="Column Completeness (%)",
                ),
                ChartConfig(chart_type=ChartType.HORIZONTAL_BAR, width=400, height=300)
            ))

        # Alerts from profile
        if self.profile.alerts:
            for alert in self.profile.alerts:
                alerts.append({
                    "level": alert.get("severity", "info"),
                    "message": alert.get("message", str(alert)),
                })

        # Quality metrics table
        quality_rows = []
        for col in self.profile.columns:
            null_pct = (col.get("null_count", 0) / self.profile.row_count * 100) if self.profile.row_count > 0 else 0
            unique_pct = (col.get("unique_count", 0) / self.profile.row_count * 100) if self.profile.row_count > 0 else 0
            quality_rows.append([
                col.get("name", "unknown"),
                f"{100 - null_pct:.1f}%",
                f"{unique_pct:.1f}%",
                col.get("inferred_type", "unknown"),
            ])

        if quality_rows:
            tables.append({
                "headers": ["Column", "Completeness", "Uniqueness", "Type"],
                "rows": quality_rows,
            })

        return SectionContent(
            section_type=SectionType.DATA_QUALITY,
            title="Data Quality",
            charts=charts,
            alerts=alerts,
            tables=tables,
            priority=90,
        )

    def create_columns_section(self) -> SectionContent:
        """Create column details section from profile data."""
        charts = []

        for col in self.profile.columns:
            col_name = col.get("name", "unknown")

            # Value distribution if available
            value_counts = col.get("value_counts", {})
            if value_counts and len(value_counts) <= 20:
                sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                charts.append((
                    ChartData(
                        labels=[str(k) for k, v in sorted_counts],
                        values=[v for k, v in sorted_counts],
                        title=col_name,
                    ),
                    ChartConfig(chart_type=ChartType.BAR, width=350, height=250)
                ))
            elif col.get("histogram"):
                # Histogram data
                hist = col["histogram"]
                charts.append((
                    ChartData(
                        labels=hist.get("bins", []),
                        values=hist.get("counts", []),
                        title=col_name,
                    ),
                    ChartConfig(chart_type=ChartType.HISTOGRAM, width=350, height=250)
                ))

        # Column statistics table
        tables = []
        col_rows = []
        for col in self.profile.columns:
            stats = col.get("statistics", {})
            col_rows.append([
                col.get("name", "unknown"),
                col.get("data_type", "unknown"),
                col.get("unique_count", "N/A"),
                col.get("null_count", 0),
                stats.get("mean", "N/A") if stats else "N/A",
                stats.get("min", "N/A") if stats else "N/A",
                stats.get("max", "N/A") if stats else "N/A",
            ])

        if col_rows:
            tables.append({
                "headers": ["Name", "Type", "Unique", "Nulls", "Mean", "Min", "Max"],
                "rows": col_rows,
            })

        return SectionContent(
            section_type=SectionType.COLUMN_DETAILS,
            title="Column Details",
            charts=charts,
            tables=tables,
            collapsible=True,
            priority=70,
        )

    def create_patterns_section(self) -> SectionContent:
        """Create patterns section from profile data."""
        charts = []
        tables = []

        if self.profile.patterns_found:
            # Pattern distribution
            pattern_types: Dict[str, int] = {}
            for p in self.profile.patterns_found:
                ptype = p.get("pattern_type", "unknown")
                pattern_types[ptype] = pattern_types.get(ptype, 0) + 1

            if pattern_types:
                charts.append((
                    ChartData(
                        labels=list(pattern_types.keys()),
                        values=list(pattern_types.values()),
                        title="Pattern Types Found",
                    ),
                    ChartConfig(chart_type=ChartType.PIE, width=350, height=300)
                ))

            # Pattern details table
            pattern_rows = []
            for p in self.profile.patterns_found[:20]:  # Limit
                pattern_rows.append([
                    p.get("column", "N/A"),
                    p.get("pattern_type", "unknown"),
                    p.get("pattern", "N/A"),
                    f"{p.get('confidence', 0) * 100:.1f}%",
                    p.get("sample_count", "N/A"),
                ])

            if pattern_rows:
                tables.append({
                    "headers": ["Column", "Type", "Pattern", "Confidence", "Samples"],
                    "rows": pattern_rows,
                })

        return SectionContent(
            section_type=SectionType.PATTERNS,
            title="Detected Patterns",
            charts=charts,
            tables=tables,
            priority=60,
        )

    def create_recommendations_section(self) -> SectionContent:
        """Create recommendations section from profile data."""
        text_blocks = []

        if self.profile.recommendations:
            text_blocks = self.profile.recommendations
        else:
            # Generate basic recommendations
            for col in self.profile.columns:
                null_pct = (col.get("null_count", 0) / self.profile.row_count * 100) if self.profile.row_count > 0 else 0
                if null_pct > 20:
                    text_blocks.append(
                        f"Column '{col.get('name')}' has {null_pct:.1f}% null values. "
                        f"Consider handling missing data."
                    )

                unique_pct = (col.get("unique_count", 0) / self.profile.row_count * 100) if self.profile.row_count > 0 else 0
                if unique_pct == 100 and self.profile.row_count > 1:
                    text_blocks.append(
                        f"Column '{col.get('name')}' appears to be a unique identifier."
                    )

        if not text_blocks:
            text_blocks = ["No specific recommendations. Data quality appears good."]

        return SectionContent(
            section_type=SectionType.RECOMMENDATIONS,
            title="Recommendations",
            text_blocks=text_blocks,
            priority=50,
        )


# =============================================================================
# Main Report Generator
# =============================================================================

class HTMLReportGenerator:
    """Main interface for generating HTML reports from profile data."""

    def __init__(
        self,
        chart_renderer: Optional[ChartRenderer] = None,
        theme: Optional[ThemeConfig] = None,
    ):
        self.chart_renderer = chart_renderer or SVGChartRenderer()
        self._theme = theme
        self._section_renderers: Dict[SectionType, SectionRenderer] = {}
        self._register_default_renderers()

    def _register_default_renderers(self) -> None:
        """Register default section renderers."""
        self._section_renderers[SectionType.OVERVIEW] = OverviewSectionRenderer()
        self._section_renderers[SectionType.DATA_QUALITY] = DataQualitySectionRenderer()
        self._section_renderers[SectionType.COLUMN_DETAILS] = ColumnDetailsSectionRenderer()
        self._section_renderers[SectionType.PATTERNS] = PatternsSectionRenderer()
        self._section_renderers[SectionType.RECOMMENDATIONS] = RecommendationsSectionRenderer()
        self._section_renderers[SectionType.CUSTOM] = CustomSectionRenderer()

    def register_section_renderer(self, renderer: SectionRenderer) -> None:
        """Register a custom section renderer."""
        self._section_renderers[renderer.get_section_type()] = renderer

    def generate(
        self,
        profile: ProfileData,
        config: Optional[ReportConfig] = None,
    ) -> str:
        """Generate HTML report from profile data."""
        config = config or ReportConfig()
        theme = self._theme or theme_registry.get(config.theme)
        template = ReportTemplate(theme)

        # Convert profile to sections
        converter = ProfileDataConverter(profile)
        sections: List[SectionContent] = []

        for section_type in config.sections:
            if section_type == SectionType.OVERVIEW:
                sections.append(converter.create_overview_section())
            elif section_type == SectionType.DATA_QUALITY:
                sections.append(converter.create_quality_section())
            elif section_type == SectionType.COLUMN_DETAILS:
                sections.append(converter.create_columns_section())
            elif section_type == SectionType.PATTERNS:
                sections.append(converter.create_patterns_section())
            elif section_type == SectionType.RECOMMENDATIONS:
                sections.append(converter.create_recommendations_section())

        # Sort by priority
        sections.sort(key=lambda s: s.priority, reverse=True)

        # Render sections
        sections_html = []
        for section in sections:
            renderer = self._section_renderers.get(section.section_type)
            if renderer:
                sections_html.append(renderer.render(section, self.chart_renderer, theme))

        # Generate TOC if requested
        toc_html = template.render_toc(sections) if config.include_toc else ""

        # Render document
        return template.render_document(config, '\n'.join(sections_html), toc_html)

    def generate_from_dict(
        self,
        profile_dict: Dict[str, Any],
        config: Optional[ReportConfig] = None,
    ) -> str:
        """Generate HTML report from profile dictionary."""
        profile = ProfileData(
            table_name=profile_dict.get("table_name", "Unknown"),
            row_count=profile_dict.get("row_count", 0),
            column_count=profile_dict.get("column_count", len(profile_dict.get("columns", []))),
            columns=profile_dict.get("columns", []),
            quality_scores=profile_dict.get("quality_scores"),
            patterns_found=profile_dict.get("patterns"),
            alerts=profile_dict.get("alerts"),
            recommendations=profile_dict.get("recommendations"),
            timestamp=datetime.fromisoformat(profile_dict["timestamp"]) if profile_dict.get("timestamp") else None,
            metadata=profile_dict.get("metadata", {}),
        )
        return self.generate(profile, config)

    def save(
        self,
        profile: ProfileData,
        output_path: Union[str, Path],
        config: Optional[ReportConfig] = None,
    ) -> Path:
        """Generate and save HTML report to file."""
        html_content = self.generate(profile, config)
        output_path = Path(output_path)
        output_path.write_text(html_content, encoding="utf-8")
        return output_path


# =============================================================================
# Report Exporter
# =============================================================================

class ReportExporter:
    """Export reports to various formats."""

    def __init__(self, generator: Optional[HTMLReportGenerator] = None):
        self.generator = generator or HTMLReportGenerator()

    def to_html(
        self,
        profile: ProfileData,
        config: Optional[ReportConfig] = None,
    ) -> str:
        """Export to HTML string."""
        return self.generator.generate(profile, config)

    def to_json(self, profile: ProfileData) -> str:
        """Export profile data to JSON."""
        data = {
            "table_name": profile.table_name,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "columns": profile.columns,
            "quality_scores": profile.quality_scores,
            "patterns_found": profile.patterns_found,
            "alerts": profile.alerts,
            "recommendations": profile.recommendations,
            "timestamp": profile.timestamp.isoformat() if profile.timestamp else None,
            "metadata": profile.metadata,
        }
        return json.dumps(data, indent=2, default=str)

    def to_file(
        self,
        profile: ProfileData,
        output_path: Union[str, Path],
        format: str = "html",
        config: Optional[ReportConfig] = None,
    ) -> Path:
        """Export to file with specified format."""
        output_path = Path(output_path)

        if format.lower() == "html":
            content = self.to_html(profile, config)
            output_path.write_text(content, encoding="utf-8")
        elif format.lower() == "json":
            content = self.to_json(profile)
            output_path.write_text(content, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_report(
    profile: Union[ProfileData, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[ReportConfig] = None,
    theme: Optional[ReportTheme] = None,
) -> Union[str, Path]:
    """
    Convenience function to generate an HTML report.

    Args:
        profile: Profile data or dictionary
        output_path: Optional path to save report
        config: Report configuration
        theme: Report theme

    Returns:
        HTML string if no output_path, Path if saved to file
    """
    if config is None:
        config = ReportConfig()

    if theme:
        config.theme = theme

    generator = HTMLReportGenerator()

    if isinstance(profile, dict):
        html_content = generator.generate_from_dict(profile, config)
    else:
        html_content = generator.generate(profile, config)

    if output_path:
        path = Path(output_path)
        path.write_text(html_content, encoding="utf-8")
        return path

    return html_content


def compare_profiles(
    profiles: List[ProfileData],
    labels: Optional[List[str]] = None,
    config: Optional[ReportConfig] = None,
) -> str:
    """
    Generate a comparison report for multiple profiles.

    Args:
        profiles: List of profile data to compare
        labels: Optional labels for each profile
        config: Report configuration

    Returns:
        HTML report string
    """
    config = config or ReportConfig(title="Profile Comparison Report")
    labels = labels or [f"Profile {i+1}" for i in range(len(profiles))]

    # Create comparison sections
    sections: List[SectionContent] = []

    # Overview comparison
    overview_rows = []
    for label, profile in zip(labels, profiles):
        overview_rows.append([
            label,
            profile.table_name,
            f"{profile.row_count:,}",
            profile.column_count,
        ])

    sections.append(SectionContent(
        section_type=SectionType.OVERVIEW,
        title="Profile Comparison",
        tables=[{
            "headers": ["Label", "Table", "Rows", "Columns"],
            "rows": overview_rows,
        }],
        priority=100,
    ))

    # Quality comparison chart
    if all(p.quality_scores for p in profiles):
        quality_data = ChartData(
            labels=labels,
            values=[p.quality_scores.get("overall", 0) * 100 for p in profiles],
            title="Quality Score Comparison",
        )
        sections.append(SectionContent(
            section_type=SectionType.DATA_QUALITY,
            title="Quality Comparison",
            charts=[(quality_data, ChartConfig(chart_type=ChartType.BAR, width=500, height=300))],
            priority=90,
        ))

    # Generate report
    theme = theme_registry.get(config.theme)
    template = ReportTemplate(theme)
    generator = HTMLReportGenerator()

    sections_html = []
    for section in sections:
        renderer = generator._section_renderers.get(section.section_type)
        if renderer:
            sections_html.append(renderer.render(section, generator.chart_renderer, theme))

    return template.render_document(
        config,
        '\n'.join(sections_html),
        template.render_toc(sections) if config.include_toc else "",
    )


# =============================================================================
# Register Default Renderers
# =============================================================================

# Register SVG renderer as default
chart_renderer_registry.register("svg", SVGChartRenderer(), default=True)

# Register section renderers
section_registry.register(OverviewSectionRenderer())
section_registry.register(DataQualitySectionRenderer())
section_registry.register(ColumnDetailsSectionRenderer())
section_registry.register(PatternsSectionRenderer())
section_registry.register(RecommendationsSectionRenderer())
section_registry.register(CustomSectionRenderer())
