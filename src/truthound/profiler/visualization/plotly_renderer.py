"""Plotly and ECharts chart renderers for interactive visualizations.

This module provides interactive chart renderers using:
- Plotly: Modern interactive charts with excellent Python integration
- ECharts: Apache ECharts for rich visualization options

Both renderers produce HTML that includes the necessary JavaScript for interactivity.
"""

from __future__ import annotations

import html
import json
import uuid
from typing import Any, Dict, List, Optional

from truthound.profiler.visualization.base import (
    ChartConfig,
    ChartData,
    ChartType,
    ColorScheme,
    COLOR_PALETTES,
)
from truthound.profiler.visualization.renderers import (
    ChartRenderer,
    chart_renderer_registry,
)


class PlotlyChartRenderer(ChartRenderer):
    """Plotly-based interactive chart renderer.

    Supports all major chart types with full interactivity:
    - Zoom, pan, hover
    - Export to PNG/SVG
    - Responsive sizing
    - Animations

    Requires: pip install plotly

    Example:
        renderer = PlotlyChartRenderer()
        html = renderer.render(data, config)
    """

    name = "plotly"
    requires_js = True

    # Plotly CDN
    PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

    SUPPORTED_TYPES = {
        ChartType.BAR,
        ChartType.HORIZONTAL_BAR,
        ChartType.PIE,
        ChartType.DONUT,
        ChartType.LINE,
        ChartType.AREA,
        ChartType.SCATTER,
        ChartType.HISTOGRAM,
        ChartType.HEATMAP,
        ChartType.BOX,
        ChartType.VIOLIN,
        ChartType.GAUGE,
        ChartType.FUNNEL,
        ChartType.TABLE,
    }

    def __init__(self, use_cdn: bool = True):
        """Initialize Plotly renderer.

        Args:
            use_cdn: If True, use CDN for Plotly JS. If False, embed local.
        """
        self.use_cdn = use_cdn
        self._plotly_available = self._check_plotly()

    def _check_plotly(self) -> bool:
        """Check if Plotly is available."""
        try:
            import plotly
            return True
        except ImportError:
            return False

    def supports_chart_type(self, chart_type: ChartType) -> bool:
        return chart_type in self.SUPPORTED_TYPES

    def get_js_dependencies(self) -> List[str]:
        return [self.PLOTLY_CDN] if self.use_cdn else []

    def render(self, data: ChartData, config: ChartConfig) -> str:
        """Render chart using Plotly."""
        if not self._plotly_available:
            return '<p class="error">Plotly not installed. Install with: pip install plotly</p>'

        # Generate unique div ID
        div_id = f"plotly-chart-{uuid.uuid4().hex[:8]}"

        # Get trace and layout based on chart type
        trace, layout = self._build_chart(data, config)

        # Create Plotly config
        plotly_config = {
            "responsive": config.responsive,
            "displayModeBar": config.show_toolbar,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }

        # Build HTML with embedded Plotly call
        chart_html = f'''
        <div id="{div_id}" style="width: 100%; height: {config.height}px;"></div>
        <script>
            (function() {{
                var data = {json.dumps([trace])};
                var layout = {json.dumps(layout)};
                var config = {json.dumps(plotly_config)};
                Plotly.newPlot('{div_id}', data, layout, config);
            }})();
        </script>
        '''

        return chart_html

    def _build_chart(
        self,
        data: ChartData,
        config: ChartConfig,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build Plotly trace and layout."""
        colors = self.get_colors(config, len(data.values) if data.values else 10)

        builders = {
            ChartType.BAR: self._build_bar,
            ChartType.HORIZONTAL_BAR: self._build_hbar,
            ChartType.PIE: lambda d, c, cols: self._build_pie(d, c, cols, donut=False),
            ChartType.DONUT: lambda d, c, cols: self._build_pie(d, c, cols, donut=True),
            ChartType.LINE: self._build_line,
            ChartType.AREA: self._build_area,
            ChartType.SCATTER: self._build_scatter,
            ChartType.HISTOGRAM: self._build_histogram,
            ChartType.HEATMAP: self._build_heatmap,
            ChartType.BOX: self._build_box,
            ChartType.GAUGE: self._build_gauge,
            ChartType.TABLE: self._build_table,
        }

        builder = builders.get(config.chart_type, self._build_bar)
        return builder(data, config, colors)

    def _base_layout(self, data: ChartData, config: ChartConfig) -> Dict[str, Any]:
        """Create base layout configuration."""
        return {
            "title": {
                "text": data.title,
                "font": {"size": 16},
            },
            "showlegend": data.show_legend,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 60, "r": 30, "t": 50, "b": 60},
            "font": {"family": "system-ui, -apple-system, sans-serif"},
            "hovermode": "closest",
        }

    def _build_bar(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build vertical bar chart."""
        trace = {
            "type": "bar",
            "x": data.labels,
            "y": data.values,
            "marker": {"color": colors},
            "text": [f"{v:,.2f}" for v in data.values] if data.show_values else None,
            "textposition": "outside",
            "hovertemplate": "%{x}: %{y:,.2f}<extra></extra>",
        }

        layout = self._base_layout(data, config)
        layout["xaxis"] = {"title": data.x_label, "tickangle": -45}
        layout["yaxis"] = {"title": data.y_label, "gridcolor": "#e0e0e0"}
        layout["bargap"] = 0.2

        return trace, layout

    def _build_hbar(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build horizontal bar chart."""
        trace = {
            "type": "bar",
            "y": data.labels,
            "x": data.values,
            "orientation": "h",
            "marker": {"color": colors},
            "text": [f"{v:,.2f}" for v in data.values] if data.show_values else None,
            "textposition": "outside",
            "hovertemplate": "%{y}: %{x:,.2f}<extra></extra>",
        }

        layout = self._base_layout(data, config)
        layout["xaxis"] = {"title": data.x_label, "gridcolor": "#e0e0e0"}
        layout["yaxis"] = {"title": data.y_label}
        layout["margin"]["l"] = 150
        layout["bargap"] = 0.2

        return trace, layout

    def _build_pie(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
        donut: bool = False,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build pie or donut chart."""
        trace = {
            "type": "pie",
            "labels": data.labels,
            "values": data.values,
            "marker": {"colors": colors},
            "textinfo": "percent+label" if data.show_values else "percent",
            "textposition": "auto",
            "hovertemplate": "%{label}: %{value:,.2f} (%{percent})<extra></extra>",
            "hole": 0.5 if donut else 0,
        }

        layout = self._base_layout(data, config)
        layout["showlegend"] = data.show_legend and len(data.labels) <= 10

        return trace, layout

    def _build_line(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build line chart."""
        trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "x": data.labels if data.labels else list(range(len(data.values))),
            "y": data.values,
            "line": {"color": colors[0], "width": 2.5},
            "marker": {"size": 8, "color": colors[0]},
            "hovertemplate": "%{x}: %{y:,.2f}<extra></extra>",
        }

        layout = self._base_layout(data, config)
        layout["xaxis"] = {"title": data.x_label, "gridcolor": "#e0e0e0"}
        layout["yaxis"] = {"title": data.y_label, "gridcolor": "#e0e0e0"}

        return trace, layout

    def _build_area(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build area chart."""
        trace = {
            "type": "scatter",
            "mode": "lines",
            "x": data.labels if data.labels else list(range(len(data.values))),
            "y": data.values,
            "fill": "tozeroy",
            "fillcolor": colors[0] + "40",  # Add transparency
            "line": {"color": colors[0], "width": 2},
            "hovertemplate": "%{x}: %{y:,.2f}<extra></extra>",
        }

        layout = self._base_layout(data, config)
        layout["xaxis"] = {"title": data.x_label, "gridcolor": "#e0e0e0"}
        layout["yaxis"] = {"title": data.y_label, "gridcolor": "#e0e0e0"}

        return trace, layout

    def _build_scatter(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build scatter plot."""
        # Extract x, y from series if available
        if data.series:
            x_vals = [s.get("x", i) for i, s in enumerate(data.series)]
            y_vals = [s.get("y", 0) for s in data.series]
            sizes = [s.get("size", 10) for s in data.series]
        else:
            x_vals = list(range(len(data.values)))
            y_vals = data.values
            sizes = [10] * len(data.values)

        trace = {
            "type": "scatter",
            "mode": "markers",
            "x": x_vals,
            "y": y_vals,
            "marker": {
                "color": colors[0],
                "size": sizes,
                "opacity": 0.7,
            },
            "hovertemplate": "(%{x}, %{y})<extra></extra>",
        }

        layout = self._base_layout(data, config)
        layout["xaxis"] = {"title": data.x_label, "gridcolor": "#e0e0e0"}
        layout["yaxis"] = {"title": data.y_label, "gridcolor": "#e0e0e0"}

        return trace, layout

    def _build_histogram(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build histogram."""
        trace = {
            "type": "histogram",
            "x": data.values,
            "marker": {"color": colors[0]},
            "nbinsx": data.metadata.get("bins", 20),
            "hovertemplate": "Range: %{x}<br>Count: %{y}<extra></extra>",
        }

        layout = self._base_layout(data, config)
        layout["xaxis"] = {"title": data.x_label or "Value", "gridcolor": "#e0e0e0"}
        layout["yaxis"] = {"title": data.y_label or "Frequency", "gridcolor": "#e0e0e0"}
        layout["bargap"] = 0.05

        return trace, layout

    def _build_heatmap(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build heatmap."""
        z_values = data.metadata.get("z", [data.values])
        x_labels = data.labels or list(range(len(z_values[0]) if z_values else 0))
        y_labels = data.metadata.get("y_labels", list(range(len(z_values))))

        trace = {
            "type": "heatmap",
            "z": z_values,
            "x": x_labels,
            "y": y_labels,
            "colorscale": "Viridis",
            "hoverongaps": False,
            "hovertemplate": "%{x}, %{y}: %{z:.2f}<extra></extra>",
        }

        layout = self._base_layout(data, config)
        layout["xaxis"] = {"title": data.x_label}
        layout["yaxis"] = {"title": data.y_label}

        return trace, layout

    def _build_box(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build box plot."""
        trace = {
            "type": "box",
            "y": data.values,
            "name": data.title or "Distribution",
            "marker": {"color": colors[0]},
            "boxpoints": "outliers",
        }

        layout = self._base_layout(data, config)
        layout["yaxis"] = {"title": data.y_label, "gridcolor": "#e0e0e0"}

        return trace, layout

    def _build_gauge(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build gauge chart."""
        value = data.values[0] if data.values else 0
        max_val = data.metadata.get("max", 100)
        min_val = data.metadata.get("min", 0)

        # Determine color based on value
        ratio = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
        if ratio < 0.33:
            bar_color = COLOR_PALETTES[ColorScheme.TRAFFIC_LIGHT][2]
        elif ratio < 0.67:
            bar_color = COLOR_PALETTES[ColorScheme.TRAFFIC_LIGHT][1]
        else:
            bar_color = COLOR_PALETTES[ColorScheme.TRAFFIC_LIGHT][0]

        trace = {
            "type": "indicator",
            "mode": "gauge+number",
            "value": value,
            "title": {"text": data.title, "font": {"size": 14}},
            "gauge": {
                "axis": {"range": [min_val, max_val], "tickwidth": 1},
                "bar": {"color": bar_color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e0e0e0",
                "steps": [
                    {"range": [min_val, max_val * 0.33], "color": "#ffebee"},
                    {"range": [max_val * 0.33, max_val * 0.67], "color": "#fff8e1"},
                    {"range": [max_val * 0.67, max_val], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": value,
                },
            },
        }

        layout = {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "system-ui"},
            "margin": {"l": 30, "r": 30, "t": 60, "b": 30},
        }

        return trace, layout

    def _build_table(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build table."""
        headers = data.metadata.get("headers", ["Label", "Value"])
        rows = data.metadata.get("rows", list(zip(data.labels, data.values)))

        # Transpose rows for Plotly table format
        values = [list(col) for col in zip(*rows)] if rows else [[], []]

        trace = {
            "type": "table",
            "header": {
                "values": headers,
                "fill": {"color": colors[0]},
                "font": {"color": "white", "size": 12},
                "align": "left",
            },
            "cells": {
                "values": values,
                "fill": {"color": ["#f8f9fa", "#ffffff"]},
                "font": {"color": "#333", "size": 11},
                "align": "left",
            },
        }

        layout = {
            "title": {"text": data.title},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        }

        return trace, layout


class EChartsChartRenderer(ChartRenderer):
    """Apache ECharts-based interactive chart renderer.

    Provides rich visualization options with excellent performance.
    Good for complex visualizations like treemaps, sankey diagrams, etc.

    Example:
        renderer = EChartsChartRenderer()
        html = renderer.render(data, config)
    """

    name = "echarts"
    requires_js = True

    # ECharts CDN
    ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"

    SUPPORTED_TYPES = {
        ChartType.BAR,
        ChartType.HORIZONTAL_BAR,
        ChartType.PIE,
        ChartType.DONUT,
        ChartType.LINE,
        ChartType.AREA,
        ChartType.SCATTER,
        ChartType.HEATMAP,
        ChartType.TREEMAP,
        ChartType.SANKEY,
        ChartType.RADAR,
        ChartType.GAUGE,
        ChartType.FUNNEL,
    }

    def __init__(self, use_cdn: bool = True):
        self.use_cdn = use_cdn

    def supports_chart_type(self, chart_type: ChartType) -> bool:
        return chart_type in self.SUPPORTED_TYPES

    def get_js_dependencies(self) -> List[str]:
        return [self.ECHARTS_CDN] if self.use_cdn else []

    def render(self, data: ChartData, config: ChartConfig) -> str:
        """Render chart using ECharts."""
        div_id = f"echarts-chart-{uuid.uuid4().hex[:8]}"
        options = self._build_options(data, config)

        chart_html = f'''
        <div id="{div_id}" style="width: 100%; height: {config.height}px;"></div>
        <script>
            (function() {{
                var chart = echarts.init(document.getElementById('{div_id}'));
                var option = {json.dumps(options)};
                chart.setOption(option);
                window.addEventListener('resize', function() {{
                    chart.resize();
                }});
            }})();
        </script>
        '''

        return chart_html

    def _build_options(self, data: ChartData, config: ChartConfig) -> Dict[str, Any]:
        """Build ECharts options."""
        colors = self.get_colors(config, len(data.values) if data.values else 10)

        builders = {
            ChartType.BAR: self._build_bar,
            ChartType.HORIZONTAL_BAR: self._build_hbar,
            ChartType.PIE: lambda d, c, cols: self._build_pie(d, c, cols, False),
            ChartType.DONUT: lambda d, c, cols: self._build_pie(d, c, cols, True),
            ChartType.LINE: self._build_line,
            ChartType.GAUGE: self._build_gauge,
            ChartType.RADAR: self._build_radar,
            ChartType.TREEMAP: self._build_treemap,
            ChartType.FUNNEL: self._build_funnel,
        }

        builder = builders.get(config.chart_type, self._build_bar)
        return builder(data, config, colors)

    def _base_options(self, data: ChartData, config: ChartConfig, colors: List[str]) -> Dict[str, Any]:
        """Create base options."""
        return {
            "color": colors,
            "title": {
                "text": data.title,
                "left": "center",
                "textStyle": {"fontSize": 16},
            },
            "tooltip": {"trigger": "item"},
            "animation": config.animation,
        }

    def _build_bar(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> Dict[str, Any]:
        """Build bar chart options."""
        options = self._base_options(data, config, colors)
        options.update({
            "xAxis": {
                "type": "category",
                "data": data.labels,
                "axisLabel": {"rotate": 45},
            },
            "yAxis": {"type": "value"},
            "series": [{
                "type": "bar",
                "data": data.values,
                "itemStyle": {"borderRadius": [4, 4, 0, 0]},
                "label": {"show": data.show_values, "position": "top"},
            }],
            "grid": {"bottom": 80},
        })
        return options

    def _build_hbar(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> Dict[str, Any]:
        """Build horizontal bar chart options."""
        options = self._base_options(data, config, colors)
        options.update({
            "xAxis": {"type": "value"},
            "yAxis": {
                "type": "category",
                "data": data.labels,
            },
            "series": [{
                "type": "bar",
                "data": data.values,
                "itemStyle": {"borderRadius": [0, 4, 4, 0]},
                "label": {"show": data.show_values, "position": "right"},
            }],
            "grid": {"left": 120},
        })
        return options

    def _build_pie(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
        donut: bool = False,
    ) -> Dict[str, Any]:
        """Build pie chart options."""
        options = self._base_options(data, config, colors)
        options.update({
            "legend": {
                "orient": "vertical",
                "right": 10,
                "top": "center",
            },
            "series": [{
                "type": "pie",
                "radius": ["45%", "70%"] if donut else ["0%", "70%"],
                "center": ["40%", "50%"],
                "data": [
                    {"name": label, "value": value}
                    for label, value in zip(data.labels, data.values)
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)",
                    },
                },
                "label": {
                    "show": data.show_values,
                    "formatter": "{b}: {d}%",
                },
            }],
        })
        return options

    def _build_line(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> Dict[str, Any]:
        """Build line chart options."""
        options = self._base_options(data, config, colors)
        options.update({
            "xAxis": {
                "type": "category",
                "data": data.labels if data.labels else list(range(len(data.values))),
            },
            "yAxis": {"type": "value"},
            "series": [{
                "type": "line",
                "data": data.values,
                "smooth": True,
                "symbol": "circle",
                "symbolSize": 8,
                "lineStyle": {"width": 2.5},
            }],
        })
        return options

    def _build_gauge(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> Dict[str, Any]:
        """Build gauge chart options."""
        value = data.values[0] if data.values else 0
        max_val = data.metadata.get("max", 100)

        options = self._base_options(data, config, colors)
        options.update({
            "series": [{
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "min": 0,
                "max": max_val,
                "progress": {"show": True, "width": 18},
                "axisLine": {"lineStyle": {"width": 18}},
                "axisTick": {"show": False},
                "splitLine": {"length": 10, "lineStyle": {"width": 2}},
                "axisLabel": {"distance": 25, "fontSize": 10},
                "pointer": {"length": "60%", "width": 6},
                "detail": {
                    "valueAnimation": True,
                    "formatter": "{value}",
                    "fontSize": 24,
                    "offsetCenter": [0, "30%"],
                },
                "data": [{"value": value, "name": data.title}],
            }],
        })
        return options

    def _build_radar(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> Dict[str, Any]:
        """Build radar chart options."""
        max_val = max(data.values) if data.values else 100

        options = self._base_options(data, config, colors)
        options.update({
            "radar": {
                "indicator": [
                    {"name": label, "max": max_val}
                    for label in data.labels
                ],
            },
            "series": [{
                "type": "radar",
                "data": [{
                    "value": data.values,
                    "name": data.title or "Value",
                }],
                "areaStyle": {"opacity": 0.3},
            }],
        })
        return options

    def _build_treemap(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> Dict[str, Any]:
        """Build treemap options."""
        tree_data = data.metadata.get("tree_data", [
            {"name": label, "value": value}
            for label, value in zip(data.labels, data.values)
        ])

        options = self._base_options(data, config, colors)
        options.update({
            "series": [{
                "type": "treemap",
                "data": tree_data,
                "levels": [
                    {"itemStyle": {"borderWidth": 0, "gapWidth": 2}},
                ],
                "label": {"show": True},
            }],
        })
        return options

    def _build_funnel(
        self,
        data: ChartData,
        config: ChartConfig,
        colors: List[str],
    ) -> Dict[str, Any]:
        """Build funnel chart options."""
        options = self._base_options(data, config, colors)
        options.update({
            "legend": {"left": "left"},
            "series": [{
                "type": "funnel",
                "left": "10%",
                "width": "80%",
                "sort": "descending",
                "gap": 2,
                "label": {"show": True, "position": "inside"},
                "data": [
                    {"name": label, "value": value}
                    for label, value in zip(data.labels, data.values)
                ],
            }],
        })
        return options


# Register renderers
try:
    import plotly
    chart_renderer_registry.register("plotly", PlotlyChartRenderer())
except ImportError:
    pass

chart_renderer_registry.register("echarts", EChartsChartRenderer())
