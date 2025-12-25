"""Report section renderers.

This module provides section rendering for different parts of the profile report,
such as overview, column details, data quality, etc.
"""

from __future__ import annotations

import html
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from truthound.profiler.visualization.base import (
    ChartConfig,
    ChartData,
    ChartType,
    ProfileData,
    SectionContent,
    SectionType,
    ThemeConfig,
    COLOR_PALETTES,
    ColorScheme,
)
from truthound.profiler.visualization.renderers import ChartRenderer


class SectionRenderer(ABC):
    """Abstract base for section renderers."""

    section_type: SectionType = SectionType.CUSTOM
    priority: int = 0

    @abstractmethod
    def render(
        self,
        data: ProfileData,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render the section to HTML."""
        pass

    def render_card(
        self,
        title: str,
        content: str,
        icon: str = "",
        collapsible: bool = False,
    ) -> str:
        """Render a card container."""
        collapse_attr = 'class="collapsible"' if collapsible else ""
        icon_html = f'<span class="card-icon">{icon}</span>' if icon else ""

        return f'''
        <div class="card" {collapse_attr}>
            <div class="card-header">
                {icon_html}
                <h3>{html.escape(title)}</h3>
            </div>
            <div class="card-body">
                {content}
            </div>
        </div>
        '''


class OverviewSectionRenderer(SectionRenderer):
    """Render the overview section with key metrics."""

    section_type = SectionType.OVERVIEW
    priority = 100

    def render(
        self,
        data: ProfileData,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render overview section."""
        parts = ['<section id="overview" class="section">']
        parts.append('<h2>Overview</h2>')

        # Key metrics cards
        parts.append('<div class="metrics-grid">')

        metrics = [
            ("Rows", f"{data.row_count:,}", "rows-icon"),
            ("Columns", f"{data.column_count}", "cols-icon"),
            ("Table", data.table_name, "table-icon"),
        ]

        if data.quality_scores:
            overall = data.quality_scores.get("overall", 0)
            metrics.append(("Quality Score", f"{overall:.1%}", "quality-icon"))

        for label, value, icon_class in metrics:
            parts.append(f'''
            <div class="metric-card">
                <div class="metric-icon {icon_class}"></div>
                <div class="metric-value">{html.escape(str(value))}</div>
                <div class="metric-label">{html.escape(label)}</div>
            </div>
            ''')

        parts.append('</div>')

        # Data type distribution chart
        if data.columns:
            type_counts: Dict[str, int] = {}
            for col in data.columns:
                dtype = col.get("inferred_type", col.get("physical_type", "unknown"))
                type_counts[dtype] = type_counts.get(dtype, 0) + 1

            chart_data = ChartData(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                title="Data Type Distribution",
            )
            chart_config = ChartConfig(
                chart_type=ChartType.PIE,
                width=400,
                height=300,
            )

            parts.append('<div class="chart-container">')
            parts.append(chart_renderer.render(chart_data, chart_config))
            parts.append('</div>')

        parts.append('</section>')
        return '\n'.join(parts)


class DataQualitySectionRenderer(SectionRenderer):
    """Render data quality section with metrics and alerts."""

    section_type = SectionType.DATA_QUALITY
    priority = 90

    def render(
        self,
        data: ProfileData,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render data quality section."""
        parts = ['<section id="data-quality" class="section">']
        parts.append('<h2>Data Quality</h2>')

        # Quality score gauge
        if data.quality_scores:
            overall = data.quality_scores.get("overall", 0) * 100
            gauge_data = ChartData(
                values=[overall],
                title="Overall Quality Score",
                metadata={"min": 0, "max": 100},
            )
            gauge_config = ChartConfig(
                chart_type=ChartType.GAUGE,
                width=300,
                height=220,
            )

            parts.append('<div class="chart-container center">')
            parts.append(chart_renderer.render(gauge_data, gauge_config))
            parts.append('</div>')

            # Dimension scores
            dimensions = ["completeness", "validity", "uniqueness", "consistency"]
            dim_scores = []
            dim_labels = []

            for dim in dimensions:
                if dim in data.quality_scores:
                    dim_labels.append(dim.capitalize())
                    dim_scores.append(data.quality_scores[dim] * 100)

            if dim_scores:
                bar_data = ChartData(
                    labels=dim_labels,
                    values=dim_scores,
                    title="Quality by Dimension",
                    y_label="Score (%)",
                )
                bar_config = ChartConfig(
                    chart_type=ChartType.HORIZONTAL_BAR,
                    width=500,
                    height=200,
                )

                parts.append('<div class="chart-container">')
                parts.append(chart_renderer.render(bar_data, bar_config))
                parts.append('</div>')

        # Null ratio per column
        if data.columns:
            null_data = ChartData(
                labels=[col.get("name", "")[:20] for col in data.columns],
                values=[col.get("null_ratio", 0) * 100 for col in data.columns],
                title="Null Ratio by Column (%)",
                y_label="Null %",
            )
            null_config = ChartConfig(
                chart_type=ChartType.BAR,
                width=600,
                height=300,
                color_scheme=ColorScheme.DIVERGING,
            )

            parts.append('<div class="chart-container">')
            parts.append(chart_renderer.render(null_data, null_config))
            parts.append('</div>')

        # Alerts
        if data.alerts:
            parts.append('<div class="alerts-container">')
            parts.append('<h3>Data Quality Alerts</h3>')

            for alert in data.alerts[:10]:  # Limit to 10
                severity = alert.get("severity", "info")
                message = alert.get("message", "")
                column = alert.get("column", "")

                parts.append(f'''
                <div class="alert alert-{severity}">
                    <strong>{html.escape(column)}</strong>: {html.escape(message)}
                </div>
                ''')

            parts.append('</div>')

        parts.append('</section>')
        return '\n'.join(parts)


class ColumnDetailsSectionRenderer(SectionRenderer):
    """Render detailed column information."""

    section_type = SectionType.COLUMN_DETAILS
    priority = 80

    def render(
        self,
        data: ProfileData,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render column details section."""
        parts = ['<section id="column-details" class="section">']
        parts.append('<h2>Column Details</h2>')

        # Summary table
        parts.append('<div class="table-responsive">')
        parts.append('<table class="data-table">')
        parts.append('''
        <thead>
            <tr>
                <th>Column</th>
                <th>Type</th>
                <th>Non-Null</th>
                <th>Unique</th>
                <th>Min</th>
                <th>Max</th>
                <th>Mean</th>
            </tr>
        </thead>
        <tbody>
        ''')

        for col in data.columns:
            name = col.get("name", "")
            dtype = col.get("inferred_type", col.get("physical_type", ""))
            null_ratio = col.get("null_ratio", 0)
            non_null_pct = (1 - null_ratio) * 100
            unique_ratio = col.get("unique_ratio", 0) * 100

            dist = col.get("distribution", {})
            min_val = dist.get("min", "")
            max_val = dist.get("max", "")
            mean_val = dist.get("mean", "")

            # Format numeric values
            min_str = f"{min_val:,.2f}" if isinstance(min_val, (int, float)) else str(min_val)
            max_str = f"{max_val:,.2f}" if isinstance(max_val, (int, float)) else str(max_val)
            mean_str = f"{mean_val:,.2f}" if isinstance(mean_val, (int, float)) else str(mean_val)

            parts.append(f'''
            <tr>
                <td><strong>{html.escape(name)}</strong></td>
                <td><code>{html.escape(str(dtype))}</code></td>
                <td>{non_null_pct:.1f}%</td>
                <td>{unique_ratio:.1f}%</td>
                <td>{html.escape(min_str[:20])}</td>
                <td>{html.escape(max_str[:20])}</td>
                <td>{html.escape(mean_str[:20])}</td>
            </tr>
            ''')

        parts.append('</tbody></table></div>')

        # Per-column details with histograms
        parts.append('<div class="column-details-grid">')

        for col in data.columns[:12]:  # Limit to first 12 columns
            parts.append(self._render_column_card(col, chart_renderer))

        parts.append('</div>')
        parts.append('</section>')

        return '\n'.join(parts)

    def _render_column_card(
        self,
        col: Dict[str, Any],
        chart_renderer: ChartRenderer,
    ) -> str:
        """Render a single column detail card."""
        name = col.get("name", "Unknown")
        dtype = col.get("inferred_type", col.get("physical_type", "unknown"))

        parts = [f'<div class="column-card">']
        parts.append(f'<h4>{html.escape(name)}</h4>')
        parts.append(f'<p class="column-type">{html.escape(str(dtype))}</p>')

        # Key stats
        null_ratio = col.get("null_ratio", 0)
        unique_ratio = col.get("unique_ratio", 0)

        parts.append(f'''
        <div class="column-stats">
            <span>Null: {null_ratio:.1%}</span>
            <span>Unique: {unique_ratio:.1%}</span>
        </div>
        ''')

        # Histogram for numeric columns
        dist = col.get("distribution", {})
        if dist.get("histogram_values"):
            hist_data = ChartData(
                values=dist["histogram_values"],
                labels=dist.get("histogram_bins", []),
                title="",
            )
            hist_config = ChartConfig(
                chart_type=ChartType.HISTOGRAM,
                width=200,
                height=100,
            )
            parts.append(chart_renderer.render(hist_data, hist_config))

        # Top values for categorical
        top_values = col.get("top_values", [])
        if top_values:
            parts.append('<div class="top-values">')
            parts.append('<strong>Top values:</strong>')
            parts.append('<ul>')
            for val, count in top_values[:5]:
                parts.append(f'<li>{html.escape(str(val)[:30])}: {count}</li>')
            parts.append('</ul></div>')

        parts.append('</div>')
        return '\n'.join(parts)


class PatternsSectionRenderer(SectionRenderer):
    """Render detected patterns section."""

    section_type = SectionType.PATTERNS
    priority = 70

    def render(
        self,
        data: ProfileData,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render patterns section."""
        if not data.patterns_found:
            return ""

        parts = ['<section id="patterns" class="section">']
        parts.append('<h2>Detected Patterns</h2>')

        parts.append('<div class="patterns-grid">')

        for pattern in data.patterns_found[:20]:
            column = pattern.get("column", "")
            pattern_name = pattern.get("name", "")
            match_ratio = pattern.get("match_ratio", 0)
            regex = pattern.get("regex", "")

            # Color based on match ratio
            if match_ratio >= 0.9:
                color_class = "high"
            elif match_ratio >= 0.7:
                color_class = "medium"
            else:
                color_class = "low"

            parts.append(f'''
            <div class="pattern-card {color_class}">
                <div class="pattern-header">
                    <strong>{html.escape(column)}</strong>
                    <span class="match-ratio">{match_ratio:.1%}</span>
                </div>
                <div class="pattern-name">{html.escape(pattern_name)}</div>
                <code class="pattern-regex">{html.escape(regex[:50])}</code>
            </div>
            ''')

        parts.append('</div>')
        parts.append('</section>')

        return '\n'.join(parts)


class RecommendationsSectionRenderer(SectionRenderer):
    """Render recommendations section."""

    section_type = SectionType.RECOMMENDATIONS
    priority = 60

    def render(
        self,
        data: ProfileData,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render recommendations section."""
        if not data.recommendations:
            return ""

        parts = ['<section id="recommendations" class="section">']
        parts.append('<h2>Recommendations</h2>')

        parts.append('<div class="recommendations-list">')

        for i, rec in enumerate(data.recommendations, 1):
            parts.append(f'''
            <div class="recommendation-item">
                <span class="rec-number">{i}</span>
                <span class="rec-text">{html.escape(rec)}</span>
            </div>
            ''')

        parts.append('</div>')
        parts.append('</section>')

        return '\n'.join(parts)


# =============================================================================
# Section Registry
# =============================================================================


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
        self._renderers[renderer.section_type] = renderer

    def get(self, section_type: SectionType) -> Optional[SectionRenderer]:
        """Get a section renderer by type."""
        return self._renderers.get(section_type)

    def get_all(self) -> List[SectionRenderer]:
        """Get all registered renderers sorted by priority."""
        return sorted(
            self._renderers.values(),
            key=lambda r: r.priority,
            reverse=True,
        )

    def list_sections(self) -> List[SectionType]:
        """List all registered section types."""
        return list(self._renderers.keys())


# Global registry
section_registry = SectionRegistry()

# Register default sections
section_registry.register(OverviewSectionRenderer())
section_registry.register(DataQualitySectionRenderer())
section_registry.register(ColumnDetailsSectionRenderer())
section_registry.register(PatternsSectionRenderer())
section_registry.register(RecommendationsSectionRenderer())


class CustomSectionRenderer(SectionRenderer):
    """Custom section renderer for user-defined sections."""

    section_type = SectionType.CUSTOM
    priority = 50

    def __init__(self, title: str = "Custom", content_func: Optional[callable] = None):
        """Initialize custom section renderer.

        Args:
            title: Section title
            content_func: Function that takes (data, chart_renderer, theme) and returns HTML
        """
        self.title = title
        self.content_func = content_func

    def render(
        self,
        data: ProfileData,
        chart_renderer: ChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        """Render custom section."""
        if self.content_func:
            content = self.content_func(data, chart_renderer, theme)
        else:
            content = "<p>No custom content defined.</p>"

        return f'''
        <section id="custom" class="section">
            <h2>{html.escape(self.title)}</h2>
            {content}
        </section>
        '''


# Alias for backward compatibility
BaseSectionRenderer = SectionRenderer
