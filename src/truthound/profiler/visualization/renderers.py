"""Chart renderer base classes and SVG implementation.

This module provides the abstract ChartRenderer interface and a pure SVG
implementation that works without any external dependencies.
"""

from __future__ import annotations

import html
import math
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from truthound.profiler.visualization.base import (
    ChartConfig,
    ChartData,
    ChartType,
    ColorScheme,
    COLOR_PALETTES,
)


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

    name: str = "base"
    requires_js: bool = False

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

    def get_js_dependencies(self) -> List[str]:
        """Get JavaScript dependencies for this renderer."""
        return []

    def get_css_dependencies(self) -> List[str]:
        """Get CSS dependencies for this renderer."""
        return []


# =============================================================================
# Chart Renderer Registry
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

    def get_available_renderers(self) -> List[str]:
        """Get renderers that have available dependencies."""
        available = []
        for name, renderer in self._renderers.items():
            # SVG is always available
            if name == "svg":
                available.append(name)
            # Check if dependencies can be imported
            elif name == "plotly":
                try:
                    import plotly
                    available.append(name)
                except ImportError:
                    pass
        return available

    def clear(self) -> None:
        """Clear all registered renderers."""
        self._renderers.clear()
        self._default = None


# Global registry
chart_renderer_registry = ChartRendererRegistry()


# =============================================================================
# SVG Chart Renderer
# =============================================================================


class SVGChartRenderer(ChartRenderer):
    """Pure SVG chart renderer - no external dependencies.

    Supports basic chart types with hover tooltips using native SVG.
    Suitable for static reports or environments where JS is not available.
    """

    name = "svg"
    requires_js = False

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
        ChartType.AREA,
    }

    def supports_chart_type(self, chart_type: ChartType) -> bool:
        return chart_type in self.SUPPORTED_TYPES

    def render(self, data: ChartData, config: ChartConfig) -> str:
        """Render chart to SVG string."""
        renderers = {
            ChartType.BAR: self._render_bar,
            ChartType.HORIZONTAL_BAR: self._render_horizontal_bar,
            ChartType.PIE: lambda d, c: self._render_pie(d, c, donut=False),
            ChartType.DONUT: lambda d, c: self._render_pie(d, c, donut=True),
            ChartType.LINE: self._render_line,
            ChartType.AREA: self._render_area,
            ChartType.HISTOGRAM: self._render_histogram,
            ChartType.GAUGE: self._render_gauge,
            ChartType.SPARKLINE: self._render_sparkline,
            ChartType.TABLE: self._render_table,
        }

        renderer = renderers.get(config.chart_type)
        if renderer:
            return renderer(data, config)
        return f'<p class="unsupported">Chart type {config.chart_type.value} not supported by SVG renderer</p>'

    def _svg_header(self, config: ChartConfig, extra_attrs: str = "") -> str:
        """Generate SVG header with proper attributes."""
        if config.responsive:
            viewport = f'viewBox="0 0 {config.width} {config.height}" preserveAspectRatio="xMidYMid meet"'
            style = 'style="max-width: 100%; height: auto;"'
        else:
            viewport = f'width="{config.width}" height="{config.height}"'
            style = ""

        return f'<svg xmlns="http://www.w3.org/2000/svg" {viewport} {style} {extra_attrs}>'

    def _render_bar(self, data: ChartData, config: ChartConfig) -> str:
        """Render vertical bar chart."""
        if not data.values:
            return '<p class="no-data">No data available</p>'

        w, h = config.width, config.height
        margin = {"top": 50, "right": 30, "bottom": 70, "left": 60}
        chart_w = w - margin["left"] - margin["right"]
        chart_h = h - margin["top"] - margin["bottom"]

        colors = self.get_colors(config, len(data.values))
        max_val = max(data.values) if data.values else 1
        bar_width = chart_w / len(data.values) * 0.75
        bar_gap = chart_w / len(data.values) * 0.25

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                       f'font-size="16" font-weight="bold" fill="currentColor">{html.escape(data.title)}</text>')

        # Grid lines
        for i in range(5):
            y = margin["top"] + (chart_h / 4) * i
            val = max_val * (1 - i / 4)
            svg.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{w - margin["right"]}" y2="{y}" '
                       f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4"/>')
            svg.append(f'<text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" '
                       f'font-size="10" fill="#666">{val:.0f}</text>')

        # Axes
        svg.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
                   f'x2="{margin["left"]}" y2="{h - margin["bottom"]}" stroke="#999" stroke-width="1"/>')
        svg.append(f'<line x1="{margin["left"]}" y1="{h - margin["bottom"]}" '
                   f'x2="{w - margin["right"]}" y2="{h - margin["bottom"]}" stroke="#999" stroke-width="1"/>')

        # Bars
        for i, (label, value) in enumerate(zip(data.labels, data.values)):
            x = margin["left"] + i * (bar_width + bar_gap) + bar_gap / 2
            bar_h = (value / max_val) * chart_h if max_val > 0 else 0
            y = h - margin["bottom"] - bar_h

            # Bar with hover effect
            svg.append(f'''<g class="bar-group">
                <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}"
                      fill="{colors[i]}" rx="3" class="bar">
                    <title>{html.escape(str(label))}: {value:,.2f}</title>
                </rect>''')

            # Value label on top
            if data.show_values and bar_h > 20:
                svg.append(f'<text x="{x + bar_width/2}" y="{y - 5}" '
                           f'text-anchor="middle" font-size="11" fill="#333">{value:,.1f}</text>')

            svg.append('</g>')

            # X-axis label
            label_y = h - margin["bottom"] + 15
            svg.append(f'<text x="{x + bar_width/2}" y="{label_y}" '
                       f'text-anchor="end" font-size="10" fill="#666" '
                       f'transform="rotate(-45 {x + bar_width/2} {label_y})">'
                       f'{html.escape(str(label)[:18])}</text>')

        # Y-axis label
        if data.y_label:
            svg.append(f'<text x="15" y="{h/2}" text-anchor="middle" font-size="11" fill="#666" '
                       f'transform="rotate(-90 15 {h/2})">{html.escape(data.y_label)}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_horizontal_bar(self, data: ChartData, config: ChartConfig) -> str:
        """Render horizontal bar chart."""
        if not data.values:
            return '<p class="no-data">No data available</p>'

        w, h = config.width, config.height
        margin = {"top": 50, "right": 80, "bottom": 30, "left": 140}
        chart_w = w - margin["left"] - margin["right"]
        chart_h = h - margin["top"] - margin["bottom"]

        colors = self.get_colors(config, len(data.values))
        max_val = max(data.values) if data.values else 1
        bar_height = chart_h / len(data.values) * 0.75
        bar_gap = chart_h / len(data.values) * 0.25

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                       f'font-size="16" font-weight="bold" fill="currentColor">{html.escape(data.title)}</text>')

        # Bars
        for i, (label, value) in enumerate(zip(data.labels, data.values)):
            y = margin["top"] + i * (bar_height + bar_gap) + bar_gap / 2
            bar_w = (value / max_val) * chart_w if max_val > 0 else 0

            # Label
            svg.append(f'<text x="{margin["left"] - 10}" y="{y + bar_height/2 + 4}" '
                       f'text-anchor="end" font-size="11" fill="#666">{html.escape(str(label)[:25])}</text>')

            # Bar
            svg.append(f'<rect x="{margin["left"]}" y="{y}" width="{bar_w}" height="{bar_height}" '
                       f'fill="{colors[i]}" rx="3">'
                       f'<title>{html.escape(str(label))}: {value:,.2f}</title></rect>')

            # Value
            if data.show_values:
                svg.append(f'<text x="{margin["left"] + bar_w + 5}" y="{y + bar_height/2 + 4}" '
                           f'font-size="11" fill="#333">{value:,.1f}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_pie(self, data: ChartData, config: ChartConfig, donut: bool = False) -> str:
        """Render pie or donut chart."""
        if not data.values:
            return '<p class="no-data">No data available</p>'

        w, h = config.width, config.height
        cx, cy = w / 2, h / 2 + 10
        radius = min(w, h) / 2 - 50
        inner_radius = radius * 0.55 if donut else 0

        colors = self.get_colors(config, len(data.values))
        total = sum(data.values) if data.values else 1

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                       f'font-size="16" font-weight="bold" fill="currentColor">{html.escape(data.title)}</text>')

        # Center text for donut
        if donut:
            svg.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" font-size="24" font-weight="bold" fill="currentColor">'
                       f'{total:,.0f}</text>')
            svg.append(f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" font-size="12" fill="#666">Total</text>')

        # Slices
        start_angle = -math.pi / 2
        for i, (label, value) in enumerate(zip(data.labels, data.values)):
            if value <= 0:
                continue

            angle = (value / total) * 2 * math.pi
            end_angle = start_angle + angle

            large_arc = 1 if angle > math.pi else 0

            # Outer arc points
            x1 = cx + radius * math.cos(start_angle)
            y1 = cy + radius * math.sin(start_angle)
            x2 = cx + radius * math.cos(end_angle)
            y2 = cy + radius * math.sin(end_angle)

            if donut:
                # Inner arc points
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
            svg.append(f'<path d="{path}" fill="{colors[i]}" stroke="white" stroke-width="2" class="pie-slice">'
                       f'<title>{html.escape(str(label))}: {value:,.2f} ({pct:.1f}%)</title></path>')

            # Label for larger slices
            if pct > 5:
                label_angle = start_angle + angle / 2
                label_radius = radius * 0.7 if not donut else (radius + inner_radius) / 2
                lx = cx + label_radius * math.cos(label_angle)
                ly = cy + label_radius * math.sin(label_angle)
                svg.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" '
                           f'font-size="11" fill="white" font-weight="bold">{pct:.1f}%</text>')

            start_angle = end_angle

        # Legend
        if data.show_legend and len(data.labels) <= 10:
            legend_x = w - 100
            for i, label in enumerate(data.labels[:10]):
                ly = 50 + i * 20
                svg.append(f'<rect x="{legend_x}" y="{ly}" width="14" height="14" rx="2" fill="{colors[i]}"/>')
                svg.append(f'<text x="{legend_x + 20}" y="{ly + 11}" font-size="11" fill="#666">'
                           f'{html.escape(str(label)[:15])}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_line(self, data: ChartData, config: ChartConfig) -> str:
        """Render line chart."""
        if not data.values:
            return '<p class="no-data">No data available</p>'

        w, h = config.width, config.height
        margin = {"top": 50, "right": 30, "bottom": 60, "left": 60}
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
                       f'font-size="16" font-weight="bold" fill="currentColor">{html.escape(data.title)}</text>')

        # Grid and axes
        for i in range(5):
            y = margin["top"] + (chart_h / 4) * i
            val = max_val - (val_range / 4) * i
            svg.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{w - margin["right"]}" y2="{y}" '
                       f'stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4"/>')
            svg.append(f'<text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" '
                       f'font-size="10" fill="#666">{val:.1f}</text>')

        svg.append(f'<line x1="{margin["left"]}" y1="{h - margin["bottom"]}" '
                   f'x2="{w - margin["right"]}" y2="{h - margin["bottom"]}" stroke="#999"/>')

        # Build path
        points = []
        step = chart_w / (len(data.values) - 1) if len(data.values) > 1 else 0

        for i, value in enumerate(data.values):
            x = margin["left"] + i * step
            y = h - margin["bottom"] - ((value - min_val) / val_range) * chart_h
            points.append(f'{x},{y}')

        # Line
        svg.append(f'<polyline points="{" ".join(points)}" '
                   f'fill="none" stroke="{colors[0]}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')

        # Points
        for i, (value, point) in enumerate(zip(data.values, points)):
            x, y = point.split(',')
            label = data.labels[i] if i < len(data.labels) else str(i)
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[0]}" stroke="white" stroke-width="2">'
                       f'<title>{html.escape(str(label))}: {value:,.2f}</title></circle>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_area(self, data: ChartData, config: ChartConfig) -> str:
        """Render area chart."""
        if not data.values:
            return '<p class="no-data">No data available</p>'

        w, h = config.width, config.height
        margin = {"top": 50, "right": 30, "bottom": 60, "left": 60}
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
                       f'font-size="16" font-weight="bold" fill="currentColor">{html.escape(data.title)}</text>')

        # Build area path
        step = chart_w / (len(data.values) - 1) if len(data.values) > 1 else 0
        baseline_y = h - margin["bottom"]

        area_points = [f'{margin["left"]},{baseline_y}']
        line_points = []

        for i, value in enumerate(data.values):
            x = margin["left"] + i * step
            y = h - margin["bottom"] - ((value - min_val) / val_range) * chart_h
            area_points.append(f'{x},{y}')
            line_points.append(f'{x},{y}')

        area_points.append(f'{margin["left"] + (len(data.values) - 1) * step},{baseline_y}')

        # Area fill
        svg.append(f'<polygon points="{" ".join(area_points)}" '
                   f'fill="{colors[0]}" fill-opacity="0.3"/>')

        # Line
        svg.append(f'<polyline points="{" ".join(line_points)}" '
                   f'fill="none" stroke="{colors[0]}" stroke-width="2"/>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_histogram(self, data: ChartData, config: ChartConfig) -> str:
        """Render histogram (bar chart with no gaps)."""
        if not data.values:
            return '<p class="no-data">No data available</p>'

        w, h = config.width, config.height
        margin = {"top": 50, "right": 30, "bottom": 60, "left": 60}
        chart_w = w - margin["left"] - margin["right"]
        chart_h = h - margin["top"] - margin["bottom"]

        colors = self.get_colors(config, 1)
        max_val = max(data.values) if data.values else 1
        bar_width = chart_w / len(data.values)

        svg = [self._svg_header(config)]

        # Title
        if data.title:
            svg.append(f'<text x="{w/2}" y="25" text-anchor="middle" '
                       f'font-size="16" font-weight="bold" fill="currentColor">{html.escape(data.title)}</text>')

        # Axes
        svg.append(f'<line x1="{margin["left"]}" y1="{h - margin["bottom"]}" '
                   f'x2="{w - margin["right"]}" y2="{h - margin["bottom"]}" stroke="#999"/>')

        # Bars
        for i, value in enumerate(data.values):
            x = margin["left"] + i * bar_width
            bar_h = (value / max_val) * chart_h if max_val > 0 else 0
            y = h - margin["bottom"] - bar_h

            label = data.labels[i] if i < len(data.labels) else str(i)
            svg.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" '
                       f'fill="{colors[0]}" stroke="white" stroke-width="0.5">'
                       f'<title>{html.escape(str(label))}: {value:,.2f}</title></rect>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def _render_gauge(self, data: ChartData, config: ChartConfig) -> str:
        """Render gauge chart."""
        if not data.values:
            return '<p class="no-data">No data available</p>'

        value = data.values[0]
        max_val = data.metadata.get("max", 100)
        min_val = data.metadata.get("min", 0)

        w, h = config.width, min(config.height, config.width * 0.65)
        cx, cy = w / 2, h - 35
        radius = min(w, h * 1.5) / 2 - 45

        # Calculate angle
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
                       f'font-size="14" font-weight="bold" fill="currentColor">{html.escape(data.title)}</text>')

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
        needle_x = cx + (radius - 35) * math.cos(angle)
        needle_y = cy - (radius - 35) * math.sin(angle)
        svg.append(f'<line x1="{cx}" y1="{cy}" x2="{needle_x}" y2="{needle_y}" '
                   f'stroke="#333" stroke-width="3" stroke-linecap="round"/>')
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="#333"/>')

        # Value text
        svg.append(f'<text x="{cx}" y="{cy - 25}" text-anchor="middle" '
                   f'font-size="28" font-weight="bold" fill="currentColor">{value:.1f}</text>')

        # Min/Max labels
        svg.append(f'<text x="{cx - radius + 10}" y="{cy + 22}" text-anchor="start" '
                   f'font-size="11" fill="#666">{min_val}</text>')
        svg.append(f'<text x="{cx + radius - 10}" y="{cy + 22}" text-anchor="end" '
                   f'font-size="11" fill="#666">{max_val}</text>')

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
            y = h - ((value - min_val) / val_range) * h * 0.9 - h * 0.05
            points.append(f'{x},{y}')

        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
            <polyline points="{" ".join(points)}" fill="none" stroke="{colors[0]}" stroke-width="1.5"/>
            <circle cx="{points[-1].split(",")[0]}" cy="{points[-1].split(",")[1]}" r="3" fill="{colors[0]}"/>
        </svg>'''

    def _render_table(self, data: ChartData, config: ChartConfig) -> str:
        """Render data as HTML table."""
        if not data.labels and not data.values:
            return '<p class="no-data">No data available</p>'

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
                                  f'<td>{value:,.2f}</td></tr>')

        html_parts.append('</tbody></table>')
        return '\n'.join(html_parts)


# Register SVG renderer as default
chart_renderer_registry.register("svg", SVGChartRenderer(), default=True)
