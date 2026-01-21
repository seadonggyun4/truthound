"""Chart rendering with CDN-based libraries.

This module provides chart renderers that use CDN-hosted JavaScript libraries
for zero-dependency chart generation. No npm/node required.

Supported libraries:
- ApexCharts: Modern, interactive charts (default for HTML reports)
- SVG: Pure SVG rendering (used for PDF export, no JS dependency)
"""

from __future__ import annotations

import json
from typing import Any

from truthound.datadocs.base import (
    ChartLibrary,
    ChartType,
    ChartSpec,
    BaseChartRenderer,
    register_chart_renderer,
)


# =============================================================================
# CDN URLs
# =============================================================================

CDN_URLS = {
    ChartLibrary.APEXCHARTS: [
        "https://cdn.jsdelivr.net/npm/apexcharts@3.45.1/dist/apexcharts.min.js",
    ],
    ChartLibrary.SVG: [],  # No dependencies
}


# =============================================================================
# ApexCharts Renderer (Default)
# =============================================================================


@register_chart_renderer(ChartLibrary.APEXCHARTS)
class ApexChartsRenderer(BaseChartRenderer):
    """Chart renderer using ApexCharts.

    ApexCharts provides modern, responsive, and interactive charts
    with a clean API and beautiful defaults. This is the default
    renderer for HTML reports.

    Supports all chart types:
    - Bar, Horizontal Bar, Line, Pie, Donut
    - Histogram, Heatmap, Scatter, Box
    - Gauge, Radar
    """

    library = ChartLibrary.APEXCHARTS

    def get_dependencies(self) -> list[str]:
        return CDN_URLS[ChartLibrary.APEXCHARTS]

    def render(self, spec: ChartSpec) -> str:
        chart_id = self._generate_chart_id()
        options = self._build_options(spec)
        options_json = json.dumps(options, indent=2, default=str)

        height_style = f"height: {spec.height}px;"
        width_style = f"width: {spec.width}px;" if spec.width else "width: 100%;"

        # JavaScript to resolve CSS variables at runtime
        # ApexCharts doesn't natively support CSS variables, so we resolve them manually
        return f'''
<div class="chart-container">
    {f'<h4 class="chart-title">{spec.title}</h4>' if spec.title else ''}
    {f'<p class="chart-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    <div id="{chart_id}" style="{height_style} {width_style}"></div>
</div>
<script>
(function() {{
    // Helper to resolve CSS variables
    function getCSSVar(varName) {{
        return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    }}

    // Resolve CSS variable references in options
    function resolveVars(obj) {{
        if (typeof obj === 'string' && obj.startsWith('var(--')) {{
            var varName = obj.slice(4, -1);
            return getCSSVar(varName) || obj;
        }}
        if (Array.isArray(obj)) {{
            return obj.map(resolveVars);
        }}
        if (obj && typeof obj === 'object') {{
            var resolved = {{}};
            for (var key in obj) {{
                resolved[key] = resolveVars(obj[key]);
            }}
            return resolved;
        }}
        return obj;
    }}

    var options = {options_json};
    options = resolveVars(options);

    var chart = new ApexCharts(document.querySelector("#{chart_id}"), options);
    chart.render();
}})();
</script>
'''

    def _build_options(self, spec: ChartSpec) -> dict[str, Any]:
        """Build ApexCharts options from ChartSpec."""
        chart_type = self._map_chart_type(spec.chart_type)

        # Use CSS variables for theme-aware text colors
        # These will be resolved at runtime via JavaScript
        options: dict[str, Any] = {
            "chart": {
                "type": chart_type,
                "height": spec.height,
                "toolbar": {
                    "show": True,
                    "tools": {
                        "download": True,
                        "selection": True,
                        "zoom": True,
                        "zoomin": True,
                        "zoomout": True,
                        "pan": True,
                        "reset": True,
                    },
                },
                "animations": {"enabled": spec.animation},
                "fontFamily": "inherit",
                "foreColor": "var(--color-text-primary)",  # Main chart text color
            },
            "grid": {
                "show": spec.show_grid,
                "borderColor": "var(--color-border)",
            },
            "legend": {
                "show": spec.show_legend,
                "position": "bottom",
                "labels": {
                    "colors": "var(--color-text-primary)",  # Legend text color
                },
            },
            "dataLabels": {
                "enabled": spec.show_labels,
                "style": {
                    "colors": ["var(--color-text-primary)"],  # Data label text color
                },
            },
            "tooltip": {
                "theme": "false",  # Disable built-in theme, use custom CSS
                "style": {
                    "fontSize": "12px",
                },
            },
            "xaxis": {
                "labels": {
                    "style": {
                        "colors": "var(--color-text-secondary)",  # X-axis label color
                    },
                },
                "axisBorder": {
                    "color": "var(--color-border)",
                },
                "axisTicks": {
                    "color": "var(--color-border)",
                },
            },
            "yaxis": {
                "labels": {
                    "style": {
                        "colors": "var(--color-text-secondary)",  # Y-axis label color
                    },
                },
            },
            "responsive": [{
                "breakpoint": 600,
                "options": {
                    "chart": {"height": 300},
                    "legend": {"position": "bottom"},
                }
            }],
        }

        # Add colors
        if spec.colors:
            options["colors"] = spec.colors
        else:
            options["colors"] = [
                "var(--chart-color-0)", "var(--chart-color-1)",
                "var(--chart-color-2)", "var(--chart-color-3)",
                "var(--chart-color-4)", "var(--chart-color-5)",
                "var(--chart-color-6)", "var(--chart-color-7)",
                "var(--chart-color-8)", "var(--chart-color-9)",
            ]

        # Add data based on chart type
        if spec.chart_type in (ChartType.PIE, ChartType.DONUT):
            options["series"] = spec.values
            options["labels"] = spec.labels
        elif spec.chart_type == ChartType.RADAR:
            options["series"] = spec.series or [{"name": "Value", "data": spec.values}]
            options["xaxis"] = {"categories": spec.labels}
        elif spec.chart_type == ChartType.HEATMAP:
            options["series"] = spec.series or []
        elif spec.chart_type == ChartType.HISTOGRAM:
            options["series"] = [{"name": "Count", "data": spec.values}]
            options["xaxis"] = {"categories": spec.labels}
        else:
            # Bar, Line, Area, etc.
            if spec.series:
                options["series"] = spec.series
            else:
                options["series"] = [{"name": "Value", "data": spec.values}]
            options["xaxis"] = {"categories": spec.labels}

        # Chart-specific options
        if spec.chart_type == ChartType.HORIZONTAL_BAR:
            options["plotOptions"] = {"bar": {"horizontal": True}}
        elif spec.chart_type == ChartType.DONUT:
            options["plotOptions"] = {"pie": {"donut": {"size": "55%"}}}
        elif spec.chart_type == ChartType.GAUGE:
            options["chart"]["type"] = "radialBar"
            options["plotOptions"] = {
                "radialBar": {
                    "startAngle": -135,
                    "endAngle": 135,
                    "hollow": {"size": "60%"},
                    "track": {"background": "var(--color-border)"},
                    "dataLabels": {
                        "name": {"show": True},
                        "value": {"show": True, "fontSize": "24px"},
                    },
                }
            }
            options["series"] = spec.values
            options["labels"] = spec.labels

        # Merge extra options
        if spec.options:
            options = self._deep_merge(options, spec.options)

        return options

    def _map_chart_type(self, chart_type: ChartType) -> str:
        """Map ChartType to ApexCharts type string."""
        mapping = {
            ChartType.BAR: "bar",
            ChartType.HORIZONTAL_BAR: "bar",
            ChartType.LINE: "line",
            ChartType.PIE: "pie",
            ChartType.DONUT: "donut",
            ChartType.HISTOGRAM: "bar",
            ChartType.HEATMAP: "heatmap",
            ChartType.SCATTER: "scatter",
            ChartType.BOX: "boxPlot",
            ChartType.GAUGE: "radialBar",
            ChartType.RADAR: "radar",
        }
        return mapping.get(chart_type, "bar")

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


# =============================================================================
# SVG Renderer (For PDF Export)
# =============================================================================


@register_chart_renderer(ChartLibrary.SVG)
class SVGChartRenderer(BaseChartRenderer):
    """Pure SVG chart renderer with no JavaScript dependencies.

    Used for PDF export where JavaScript cannot be executed.
    Provides basic chart rendering for static output.

    Supports:
    - Bar, Horizontal Bar, Line
    - Pie, Donut
    """

    library = ChartLibrary.SVG

    def get_dependencies(self) -> list[str]:
        return []  # No dependencies

    def render(self, spec: ChartSpec) -> str:
        width = spec.width or 600
        height = spec.height

        # Default colors
        colors = spec.colors or [
            "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4",
            "#22c55e", "#f59e0b", "#ef4444", "#6366f1",
        ]

        svg_content = ""

        if spec.chart_type in (ChartType.PIE, ChartType.DONUT):
            svg_content = self._render_pie(spec, width, height, colors)
        elif spec.chart_type == ChartType.BAR:
            svg_content = self._render_bar(spec, width, height, colors)
        elif spec.chart_type == ChartType.HORIZONTAL_BAR:
            svg_content = self._render_horizontal_bar(spec, width, height, colors)
        elif spec.chart_type == ChartType.LINE:
            svg_content = self._render_line(spec, width, height, colors)
        else:
            # Fallback to bar chart for unsupported types
            svg_content = self._render_bar(spec, width, height, colors)

        return f'''
<div class="chart-container">
    {f'<h4 class="chart-title">{spec.title}</h4>' if spec.title else ''}
    {f'<p class="chart-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    {svg_content}
</div>
'''

    def _render_bar(
        self,
        spec: ChartSpec,
        width: int,
        height: int,
        colors: list[str],
    ) -> str:
        """Render a vertical bar chart as SVG."""
        if not spec.values:
            return self._render_empty(width, height)

        margin = {"top": 20, "right": 20, "bottom": 60, "left": 60}
        chart_width = width - margin["left"] - margin["right"]
        chart_height = height - margin["top"] - margin["bottom"]

        max_val = max(spec.values) if spec.values else 1
        bar_width = chart_width / len(spec.values) * 0.8
        bar_gap = chart_width / len(spec.values) * 0.2

        bars = []
        labels = []

        for i, (label, value) in enumerate(zip(spec.labels, spec.values)):
            bar_height = (value / max_val) * chart_height
            x = margin["left"] + i * (bar_width + bar_gap) + bar_gap / 2
            y = margin["top"] + chart_height - bar_height
            color = colors[i % len(colors)]

            bars.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                f'fill="{color}" rx="4">'
                f'<title>{label}: {value}</title></rect>'
            )

            # X-axis label
            label_x = x + bar_width / 2
            label_y = height - margin["bottom"] + 20
            labels.append(
                f'<text x="{label_x}" y="{label_y}" text-anchor="middle" '
                f'class="chart-label" transform="rotate(-45 {label_x} {label_y})">{label[:12]}</text>'
            )

        return f'''
<svg width="{width}" height="{height}" class="svg-chart">
    {"".join(bars)}
    {"".join(labels)}
</svg>
'''

    def _render_horizontal_bar(
        self,
        spec: ChartSpec,
        width: int,
        height: int,
        colors: list[str],
    ) -> str:
        """Render a horizontal bar chart as SVG."""
        if not spec.values:
            return self._render_empty(width, height)

        margin = {"top": 20, "right": 20, "bottom": 30, "left": 100}
        chart_width = width - margin["left"] - margin["right"]
        chart_height = height - margin["top"] - margin["bottom"]

        max_val = max(spec.values) if spec.values else 1
        bar_height = chart_height / len(spec.values) * 0.8
        bar_gap = chart_height / len(spec.values) * 0.2

        bars = []
        labels = []

        for i, (label, value) in enumerate(zip(spec.labels, spec.values)):
            bar_width = (value / max_val) * chart_width
            x = margin["left"]
            y = margin["top"] + i * (bar_height + bar_gap) + bar_gap / 2
            color = colors[i % len(colors)]

            bars.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" '
                f'fill="{color}" rx="4">'
                f'<title>{label}: {value}</title></rect>'
            )

            # Y-axis label
            label_x = margin["left"] - 10
            label_y = y + bar_height / 2
            labels.append(
                f'<text x="{label_x}" y="{label_y}" text-anchor="end" '
                f'dominant-baseline="middle" class="chart-label">{label[:15]}</text>'
            )

        return f'''
<svg width="{width}" height="{height}" class="svg-chart">
    {"".join(bars)}
    {"".join(labels)}
</svg>
'''

    def _render_pie(
        self,
        spec: ChartSpec,
        width: int,
        height: int,
        colors: list[str],
    ) -> str:
        """Render a pie chart as SVG."""
        if not spec.values:
            return self._render_empty(width, height)

        cx = width / 2
        cy = height / 2
        radius = min(width, height) / 2 - 40
        inner_radius = radius * 0.5 if spec.chart_type == ChartType.DONUT else 0

        total = sum(spec.values)
        if total == 0:
            return self._render_empty(width, height)

        slices = []
        current_angle = -90  # Start at top

        for i, (label, value) in enumerate(zip(spec.labels, spec.values)):
            if value == 0:
                continue

            angle = (value / total) * 360
            start_angle = current_angle
            end_angle = current_angle + angle

            # Calculate arc path
            start_rad = start_angle * 3.14159 / 180
            end_rad = end_angle * 3.14159 / 180

            x1 = cx + radius * __import__("math").cos(start_rad)
            y1 = cy + radius * __import__("math").sin(start_rad)
            x2 = cx + radius * __import__("math").cos(end_rad)
            y2 = cy + radius * __import__("math").sin(end_rad)

            large_arc = 1 if angle > 180 else 0
            color = colors[i % len(colors)]

            if inner_radius > 0:
                # Donut
                ix1 = cx + inner_radius * __import__("math").cos(start_rad)
                iy1 = cy + inner_radius * __import__("math").sin(start_rad)
                ix2 = cx + inner_radius * __import__("math").cos(end_rad)
                iy2 = cy + inner_radius * __import__("math").sin(end_rad)

                path = (
                    f"M {x1} {y1} "
                    f"A {radius} {radius} 0 {large_arc} 1 {x2} {y2} "
                    f"L {ix2} {iy2} "
                    f"A {inner_radius} {inner_radius} 0 {large_arc} 0 {ix1} {iy1} Z"
                )
            else:
                # Pie
                path = (
                    f"M {cx} {cy} "
                    f"L {x1} {y1} "
                    f"A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z"
                )

            slices.append(
                f'<path d="{path}" fill="{color}" stroke="white" stroke-width="2">'
                f'<title>{label}: {value} ({value/total*100:.1f}%)</title></path>'
            )

            current_angle = end_angle

        return f'''
<svg width="{width}" height="{height}" class="svg-chart">
    {"".join(slices)}
</svg>
'''

    def _render_line(
        self,
        spec: ChartSpec,
        width: int,
        height: int,
        colors: list[str],
    ) -> str:
        """Render a line chart as SVG."""
        if not spec.values:
            return self._render_empty(width, height)

        margin = {"top": 20, "right": 20, "bottom": 40, "left": 60}
        chart_width = width - margin["left"] - margin["right"]
        chart_height = height - margin["top"] - margin["bottom"]

        max_val = max(spec.values) if spec.values else 1
        min_val = min(spec.values) if spec.values else 0
        range_val = max_val - min_val if max_val != min_val else 1

        points = []
        for i, value in enumerate(spec.values):
            x = margin["left"] + (i / (len(spec.values) - 1)) * chart_width if len(spec.values) > 1 else margin["left"]
            y = margin["top"] + chart_height - ((value - min_val) / range_val) * chart_height
            points.append(f"{x},{y}")

        path = "M " + " L ".join(points)
        color = colors[0]

        # Create dots
        dots = []
        for i, value in enumerate(spec.values):
            x = margin["left"] + (i / (len(spec.values) - 1)) * chart_width if len(spec.values) > 1 else margin["left"]
            y = margin["top"] + chart_height - ((value - min_val) / range_val) * chart_height
            label = spec.labels[i] if i < len(spec.labels) else str(i)
            dots.append(
                f'<circle cx="{x}" cy="{y}" r="4" fill="{color}">'
                f'<title>{label}: {value}</title></circle>'
            )

        return f'''
<svg width="{width}" height="{height}" class="svg-chart">
    <path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>
    {"".join(dots)}
</svg>
'''

    def _render_empty(self, width: int, height: int) -> str:
        """Render an empty state."""
        return f'''
<svg width="{width}" height="{height}" class="svg-chart">
    <text x="{width/2}" y="{height/2}" text-anchor="middle" class="chart-empty">
        No data available
    </text>
</svg>
'''


# =============================================================================
# Factory Function
# =============================================================================


def get_chart_renderer(library: ChartLibrary | str = ChartLibrary.APEXCHARTS) -> BaseChartRenderer:
    """Get a chart renderer instance.

    Args:
        library: Chart library to use (apexcharts or svg)

    Returns:
        Chart renderer instance

    Note:
        ApexCharts is used by default for HTML reports.
        SVG is used automatically for PDF export.
    """
    from truthound.datadocs.base import renderer_registry

    if isinstance(library, str):
        library = ChartLibrary(library)

    renderer_class = renderer_registry.get_chart_renderer(library)
    return renderer_class()
