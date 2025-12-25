"""Chart rendering with CDN-based libraries.

This module provides chart renderers that use CDN-hosted JavaScript libraries
for zero-dependency chart generation. No npm/node required.

Supported libraries:
- ApexCharts: Modern, interactive charts (recommended)
- Chart.js: Lightweight, widely used
- Plotly.js: Scientific visualization
- SVG: Pure SVG fallback (no JS)
"""

from __future__ import annotations

import json
import uuid
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
    ChartLibrary.CHARTJS: [
        "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
    ],
    ChartLibrary.PLOTLY: [
        "https://cdn.plot.ly/plotly-2.29.0.min.js",
    ],
    ChartLibrary.SVG: [],  # No dependencies
}


# =============================================================================
# ApexCharts Renderer (Recommended)
# =============================================================================


@register_chart_renderer(ChartLibrary.APEXCHARTS)
class ApexChartsRenderer(BaseChartRenderer):
    """Chart renderer using ApexCharts.

    ApexCharts provides modern, responsive, and interactive charts
    with a clean API and beautiful defaults.
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

        return f'''
<div class="chart-container">
    {f'<h4 class="chart-title">{spec.title}</h4>' if spec.title else ''}
    {f'<p class="chart-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    <div id="{chart_id}" style="{height_style} {width_style}"></div>
</div>
<script>
(function() {{
    var options = {options_json};
    var chart = new ApexCharts(document.querySelector("#{chart_id}"), options);
    chart.render();
}})();
</script>
'''

    def _build_options(self, spec: ChartSpec) -> dict[str, Any]:
        """Build ApexCharts options from ChartSpec."""
        chart_type = self._map_chart_type(spec.chart_type)

        options: dict[str, Any] = {
            "chart": {
                "type": chart_type,
                "height": spec.height,
                "toolbar": {"show": True},
                "animations": {"enabled": spec.animation},
                "fontFamily": "inherit",
            },
            "grid": {
                "show": spec.show_grid,
                "borderColor": "var(--color-border)",
            },
            "legend": {
                "show": spec.show_legend,
                "position": "bottom",
            },
            "dataLabels": {
                "enabled": spec.show_labels,
            },
            "tooltip": {
                "theme": "light",
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
# Chart.js Renderer
# =============================================================================


@register_chart_renderer(ChartLibrary.CHARTJS)
class ChartJSRenderer(BaseChartRenderer):
    """Chart renderer using Chart.js.

    Chart.js is a lightweight, simple charting library
    with good documentation and wide adoption.
    """

    library = ChartLibrary.CHARTJS

    def get_dependencies(self) -> list[str]:
        return CDN_URLS[ChartLibrary.CHARTJS]

    def render(self, spec: ChartSpec) -> str:
        chart_id = self._generate_chart_id()
        config = self._build_config(spec)
        config_json = json.dumps(config, indent=2, default=str)

        return f'''
<div class="chart-container">
    {f'<h4 class="chart-title">{spec.title}</h4>' if spec.title else ''}
    {f'<p class="chart-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    <canvas id="{chart_id}" style="max-height: {spec.height}px;"></canvas>
</div>
<script>
(function() {{
    var ctx = document.getElementById('{chart_id}').getContext('2d');
    var config = {config_json};
    new Chart(ctx, config);
}})();
</script>
'''

    def _build_config(self, spec: ChartSpec) -> dict[str, Any]:
        """Build Chart.js configuration from ChartSpec."""
        chart_type = self._map_chart_type(spec.chart_type)

        # Default colors
        colors = spec.colors or [
            "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4",
            "#22c55e", "#f59e0b", "#ef4444", "#6366f1",
            "#14b8a6", "#f97316"
        ]

        config: dict[str, Any] = {
            "type": chart_type,
            "data": {
                "labels": spec.labels,
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {
                        "display": spec.show_legend,
                        "position": "bottom",
                    },
                },
            },
        }

        # Add data
        if spec.chart_type in (ChartType.PIE, ChartType.DONUT):
            config["data"]["datasets"] = [{
                "data": spec.values,
                "backgroundColor": colors[:len(spec.values)],
            }]
        elif spec.series:
            config["data"]["datasets"] = [
                {
                    "label": s.get("name", f"Series {i}"),
                    "data": s.get("data", []),
                    "backgroundColor": colors[i % len(colors)],
                    "borderColor": colors[i % len(colors)],
                }
                for i, s in enumerate(spec.series)
            ]
        else:
            config["data"]["datasets"] = [{
                "label": "Value",
                "data": spec.values,
                "backgroundColor": colors[0],
                "borderColor": colors[0],
            }]

        # Chart-specific options
        if spec.chart_type == ChartType.HORIZONTAL_BAR:
            config["options"]["indexAxis"] = "y"
        elif spec.chart_type == ChartType.DONUT:
            config["options"]["cutout"] = "50%"

        return config

    def _map_chart_type(self, chart_type: ChartType) -> str:
        """Map ChartType to Chart.js type string."""
        mapping = {
            ChartType.BAR: "bar",
            ChartType.HORIZONTAL_BAR: "bar",
            ChartType.LINE: "line",
            ChartType.PIE: "pie",
            ChartType.DONUT: "doughnut",
            ChartType.HISTOGRAM: "bar",
            ChartType.SCATTER: "scatter",
            ChartType.RADAR: "radar",
        }
        return mapping.get(chart_type, "bar")


# =============================================================================
# Plotly.js Renderer
# =============================================================================


@register_chart_renderer(ChartLibrary.PLOTLY)
class PlotlyJSRenderer(BaseChartRenderer):
    """Chart renderer using Plotly.js.

    Plotly.js provides scientific-grade, interactive visualizations
    with extensive customization options.
    """

    library = ChartLibrary.PLOTLY

    def get_dependencies(self) -> list[str]:
        return CDN_URLS[ChartLibrary.PLOTLY]

    def render(self, spec: ChartSpec) -> str:
        chart_id = self._generate_chart_id()
        data, layout = self._build_figure(spec)
        data_json = json.dumps(data, indent=2, default=str)
        layout_json = json.dumps(layout, indent=2, default=str)

        return f'''
<div class="chart-container">
    {f'<h4 class="chart-title">{spec.title}</h4>' if spec.title else ''}
    {f'<p class="chart-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    <div id="{chart_id}" style="height: {spec.height}px; width: 100%;"></div>
</div>
<script>
(function() {{
    var data = {data_json};
    var layout = {layout_json};
    var config = {{responsive: true, displayModeBar: true}};
    Plotly.newPlot('{chart_id}', data, layout, config);
}})();
</script>
'''

    def _build_figure(self, spec: ChartSpec) -> tuple[list[dict], dict]:
        """Build Plotly figure data and layout."""
        colors = spec.colors or [
            "#3b82f6", "#8b5cf6", "#ec4899", "#06b6d4",
            "#22c55e", "#f59e0b", "#ef4444", "#6366f1",
        ]

        data = []
        layout: dict[str, Any] = {
            "showlegend": spec.show_legend,
            "legend": {"orientation": "h", "y": -0.15},
            "margin": {"t": 30, "r": 30, "b": 50, "l": 50},
            "paper_bgcolor": "transparent",
            "plot_bgcolor": "transparent",
        }

        if spec.chart_type in (ChartType.PIE, ChartType.DONUT):
            trace: dict[str, Any] = {
                "type": "pie",
                "labels": spec.labels,
                "values": spec.values,
                "marker": {"colors": colors[:len(spec.values)]},
            }
            if spec.chart_type == ChartType.DONUT:
                trace["hole"] = 0.4
            data.append(trace)

        elif spec.chart_type == ChartType.HEATMAP:
            data.append({
                "type": "heatmap",
                "z": spec.series[0].get("data") if spec.series else [spec.values],
                "colorscale": "Blues",
            })

        elif spec.chart_type == ChartType.BOX:
            for i, s in enumerate(spec.series or [{"data": spec.values}]):
                data.append({
                    "type": "box",
                    "y": s.get("data", []),
                    "name": s.get("name", f"Series {i}"),
                    "marker": {"color": colors[i % len(colors)]},
                })

        elif spec.chart_type == ChartType.SCATTER:
            for i, s in enumerate(spec.series or [{"data": spec.values}]):
                data.append({
                    "type": "scatter",
                    "mode": "markers",
                    "x": list(range(len(s.get("data", [])))),
                    "y": s.get("data", []),
                    "name": s.get("name", f"Series {i}"),
                    "marker": {"color": colors[i % len(colors)]},
                })

        else:
            # Bar, Line, Area, Histogram
            trace_type = self._map_chart_type(spec.chart_type)

            if spec.series:
                for i, s in enumerate(spec.series):
                    trace = {
                        "type": trace_type,
                        "x": spec.labels,
                        "y": s.get("data", []),
                        "name": s.get("name", f"Series {i}"),
                        "marker": {"color": colors[i % len(colors)]},
                    }
                    if spec.chart_type == ChartType.HORIZONTAL_BAR:
                        trace["orientation"] = "h"
                        trace["x"], trace["y"] = trace["y"], trace["x"]
                    data.append(trace)
            else:
                trace = {
                    "type": trace_type,
                    "x": spec.labels,
                    "y": spec.values,
                    "marker": {"color": colors[0]},
                }
                if spec.chart_type == ChartType.HORIZONTAL_BAR:
                    trace["orientation"] = "h"
                    trace["x"], trace["y"] = trace["y"], trace["x"]
                data.append(trace)

        return data, layout

    def _map_chart_type(self, chart_type: ChartType) -> str:
        """Map ChartType to Plotly type string."""
        mapping = {
            ChartType.BAR: "bar",
            ChartType.HORIZONTAL_BAR: "bar",
            ChartType.LINE: "scatter",
            ChartType.HISTOGRAM: "bar",
            ChartType.SCATTER: "scatter",
            ChartType.HEATMAP: "heatmap",
            ChartType.BOX: "box",
        }
        return mapping.get(chart_type, "bar")


# =============================================================================
# SVG Fallback Renderer (No JS)
# =============================================================================


@register_chart_renderer(ChartLibrary.SVG)
class SVGChartRenderer(BaseChartRenderer):
    """Pure SVG chart renderer with no JavaScript dependencies.

    Provides basic chart rendering when JavaScript is not available
    or for environments that require static output.
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
            # Fallback to bar chart
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
        library: Chart library to use

    Returns:
        Chart renderer instance
    """
    from truthound.datadocs.base import renderer_registry

    if isinstance(library, str):
        library = ChartLibrary(library)

    renderer_class = renderer_registry.get_chart_renderer(library)
    return renderer_class()
