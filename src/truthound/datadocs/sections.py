"""Section renderers for Data Docs reports.

This module provides renderers for each section type in the report.
Each section is designed to present specific aspects of the data profile.
"""

from __future__ import annotations

from html import escape
from typing import Any

from truthound.datadocs.base import (
    SectionType,
    ChartType,
    ChartSpec,
    SectionSpec,
    AlertSpec,
    SeverityLevel,
    BaseSectionRenderer,
    BaseChartRenderer,
    ThemeConfig,
    register_section_renderer,
)


# =============================================================================
# Overview Section
# =============================================================================


@register_section_renderer(SectionType.OVERVIEW)
class OverviewSection(BaseSectionRenderer):
    """Renders the overview section with key dataset metrics."""

    section_type = SectionType.OVERVIEW

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        metrics_html = self._render_metric_cards(spec.metrics, spec.metadata.get("labels", {}))
        charts_html = self._render_charts(spec.charts, chart_renderer)
        alerts_html = self._render_alerts(spec.alerts, theme)

        return f'''
<section class="report-section section-overview" id="section-overview">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {metrics_html}
        {charts_html}
        {alerts_html}
    </div>
</section>
'''

    def _render_metric_cards(self, metrics: dict[str, Any], labels: dict[str, str] | None = None) -> str:
        """Render metric cards for overview."""
        if not metrics:
            return ""

        labels = labels or {}
        cards = []
        card_definitions = [
            ("row_count", labels.get("overview.row_count", "Rows"), labels.get("overview.row_count.desc", "Total number of rows"), "icon-rows"),
            ("column_count", labels.get("overview.column_count", "Columns"), labels.get("overview.column_count.desc", "Total number of columns"), "icon-columns"),
            ("memory_bytes", labels.get("overview.memory_bytes", "Memory"), labels.get("overview.memory_bytes.desc", "Estimated memory size"), "icon-memory"),
            ("duplicate_rows", labels.get("overview.duplicate_rows", "Duplicates"), labels.get("overview.duplicate_rows.desc", "Duplicate row count"), "icon-duplicates"),
            ("null_cells", labels.get("overview.null_cells", "Missing"), labels.get("overview.null_cells.desc", "Total null cells"), "icon-missing"),
            ("quality_score", labels.get("overview.quality_score", "Quality"), labels.get("overview.quality_score.desc", "Overall data quality"), "icon-quality"),
        ]

        for key, label, desc, icon_class in card_definitions:
            if key in metrics:
                value = metrics[key]
                formatted_value = self._format_metric_value(key, value)
                cards.append(f'''
                    <div class="metric-card">
                        <div class="metric-icon {icon_class}"></div>
                        <div class="metric-content">
                            <span class="metric-value">{formatted_value}</span>
                            <span class="metric-label">{label}</span>
                        </div>
                    </div>
                ''')

        # Handle any additional metrics
        for key, value in metrics.items():
            if key not in [d[0] for d in card_definitions]:
                formatted_value = self._format_metric_value(key, value)
                label = labels.get(f"overview.{key}", key.replace("_", " ").title())
                cards.append(f'''
                    <div class="metric-card">
                        <div class="metric-content">
                            <span class="metric-value">{formatted_value}</span>
                            <span class="metric-label">{label}</span>
                        </div>
                    </div>
                ''')

        return f'<div class="metrics-grid">{"".join(cards)}</div>'

    def _format_metric_value(self, key: str, value: Any) -> str:
        """Format a metric value for display."""
        if key == "memory_bytes":
            # Convert bytes to human readable
            for unit in ["B", "KB", "MB", "GB"]:
                if abs(value) < 1024:
                    return f"{value:.1f} {unit}"
                value /= 1024
            return f"{value:.1f} TB"
        elif key == "quality_score":
            return f"{value:.1f}%"
        elif isinstance(value, float):
            if value < 1:
                return f"{value:.2%}"
            return f"{value:,.2f}"
        elif isinstance(value, int):
            return f"{value:,}"
        return escape(str(value))


# =============================================================================
# Columns Section
# =============================================================================


@register_section_renderer(SectionType.COLUMNS)
class ColumnsSection(BaseSectionRenderer):
    """Renders detailed column information."""

    section_type = SectionType.COLUMNS

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        tables_html = self._render_column_tables(spec.tables)
        labels = spec.metadata.get("labels", {})
        column_cards = self._render_column_cards(
            spec.metadata.get("columns", []),
            chart_renderer,
            labels,
        )

        return f'''
<section class="report-section section-columns" id="section-columns">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {tables_html}
        {column_cards}
    </div>
</section>
'''

    def _render_column_tables(self, tables: list[dict[str, Any]]) -> str:
        """Render column summary tables."""
        if not tables:
            return ""

        html_parts = []
        for table in tables:
            title = escape(str(table.get("title", "Column Summary")))
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            if not rows:
                continue

            header_html = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
            rows_html = ""
            for row in rows:
                cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
                rows_html += f"<tr>{cells}</tr>"

            html_parts.append(f'''
                <div class="table-container">
                    <h4 class="table-title">{title}</h4>
                    <table class="data-table">
                        <thead><tr>{header_html}</tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            ''')

        return "".join(html_parts)

    def _render_column_cards(
        self,
        columns: list[dict[str, Any]],
        chart_renderer: BaseChartRenderer,
        labels: dict[str, str] | None = None,
    ) -> str:
        """Render detailed column cards."""
        if not columns:
            return ""

        labels = labels or {}
        cards = []
        for col in columns:
            name = col.get("name", labels.get("column.unknown", "Unknown"))
            dtype = col.get("physical_type", col.get("dtype", "unknown"))
            inferred = col.get("inferred_type", "")
            null_ratio = col.get("null_ratio", 0)
            unique_ratio = col.get("unique_ratio", 0)
            distinct = col.get("distinct_count", 0)

            # Type badge
            type_class = self._get_type_class(inferred or dtype)

            # Quality indicator
            quality_class = "quality-good" if null_ratio < 0.05 else "quality-warning" if null_ratio < 0.2 else "quality-bad"

            # Distribution chart if available
            distribution_chart = ""
            if "top_values" in col and col["top_values"]:
                top_values = col["top_values"][:5]
                chart_spec = ChartSpec(
                    chart_type=ChartType.HORIZONTAL_BAR,
                    labels=[str(v.get("value", ""))[:20] for v in top_values],
                    values=[v.get("count", 0) for v in top_values],
                    height=150,
                    show_legend=False,
                    show_labels=False,
                    options={"value_unit": "count"},
                )
                distribution_chart = chart_renderer.render(chart_spec)

            # Statistics if numeric
            stats_html = ""
            if "distribution" in col and col["distribution"]:
                dist = col["distribution"]
                stats_html = f'''
                    <div class="column-stats">
                        <div class="stat"><span class="stat-label">{labels.get("column.min", "Min")}</span><span class="stat-value">{dist.get("min", "-")}</span></div>
                        <div class="stat"><span class="stat-label">{labels.get("column.max", "Max")}</span><span class="stat-value">{dist.get("max", "-")}</span></div>
                        <div class="stat"><span class="stat-label">{labels.get("column.mean", "Mean")}</span><span class="stat-value">{self._format_number(dist.get("mean"))}</span></div>
                        <div class="stat"><span class="stat-label">{labels.get("column.std", "Std")}</span><span class="stat-value">{self._format_number(dist.get("std"))}</span></div>
                    </div>
                '''

            # Patterns if detected
            patterns_html = ""
            if "detected_patterns" in col and col["detected_patterns"]:
                patterns = col["detected_patterns"][:3]
                pattern_items = "".join(
                    f'<span class="pattern-tag">{escape(str(p.get("pattern", "")))}</span>'
                    for p in patterns
                )
                patterns_html = f'<div class="column-patterns">{pattern_items}</div>'

            cards.append(f'''
                <div class="column-card">
                    <div class="column-header">
                        <h4 class="column-name">{escape(str(name))}</h4>
                        <span class="column-type {type_class}">{escape(str(inferred or dtype))}</span>
                    </div>
                    <div class="column-metrics">
                        <div class="metric-mini {quality_class}">
                            <span class="metric-label">{labels.get("column.null", "Null")}</span>
                            <span class="metric-value">{null_ratio:.1%}</span>
                        </div>
                        <div class="metric-mini">
                            <span class="metric-label">{labels.get("column.unique", "Unique")}</span>
                            <span class="metric-value">{unique_ratio:.1%}</span>
                        </div>
                        <div class="metric-mini">
                            <span class="metric-label">{labels.get("column.distinct", "Distinct")}</span>
                            <span class="metric-value">{distinct:,}</span>
                        </div>
                    </div>
                    {stats_html}
                    {patterns_html}
                    {distribution_chart}
                </div>
            ''')

        return f'<div class="columns-grid">{"".join(cards)}</div>'

    def _get_type_class(self, dtype: str) -> str:
        """Get CSS class for data type."""
        dtype_lower = dtype.lower()
        if "int" in dtype_lower or "float" in dtype_lower or "numeric" in dtype_lower:
            return "type-numeric"
        elif "str" in dtype_lower or "text" in dtype_lower or "char" in dtype_lower:
            return "type-string"
        elif "date" in dtype_lower or "time" in dtype_lower:
            return "type-datetime"
        elif "bool" in dtype_lower:
            return "type-boolean"
        elif "email" in dtype_lower:
            return "type-email"
        elif "url" in dtype_lower:
            return "type-url"
        elif "phone" in dtype_lower:
            return "type-phone"
        return "type-other"

    def _format_number(self, value: Any) -> str:
        """Format a number for display."""
        if value is None:
            return "-"
        if isinstance(value, float):
            if abs(value) >= 1000000:
                return f"{value:.2e}"
            elif abs(value) >= 1:
                return f"{value:,.2f}"
            else:
                return f"{value:.4f}"
        return str(value)


# =============================================================================
# Quality Section
# =============================================================================


@register_section_renderer(SectionType.QUALITY)
class QualitySection(BaseSectionRenderer):
    """Renders data quality metrics and assessments."""

    section_type = SectionType.QUALITY

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        charts_html = self._render_charts(spec.charts, chart_renderer)
        metrics_html = self._render_quality_metrics(spec.metrics, spec.metadata.get("labels", {}), text_color=theme.colors.text_primary)
        alerts_html = self._render_alerts(spec.alerts, theme)

        return f'''
<section class="report-section section-quality" id="section-quality">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {metrics_html}
        {charts_html}
        {alerts_html}
    </div>
</section>
'''

    def _render_quality_metrics(self, metrics: dict[str, Any], labels: dict[str, str] | None = None, *, text_color: str = "#1f2937") -> str:
        """Render quality score gauges and metrics."""
        if not metrics:
            return ""

        labels = labels or {}
        scores = []
        quality_dimensions = [
            ("completeness", labels.get("quality.completeness", "Completeness"), labels.get("quality.completeness.desc", "Measures data completeness")),
            ("uniqueness", labels.get("quality.uniqueness", "Uniqueness"), labels.get("quality.uniqueness.desc", "Measures unique value ratio")),
            ("validity", labels.get("quality.validity", "Validity"), labels.get("quality.validity.desc", "Measures data format validity")),
            ("consistency", labels.get("quality.consistency", "Consistency"), labels.get("quality.consistency.desc", "Measures data consistency")),
        ]

        for key, label, desc in quality_dimensions:
            if key in metrics:
                score = metrics[key]
                color_class = "score-good" if score >= 80 else "score-warning" if score >= 60 else "score-bad"
                # Use inline styles for PDF compatibility
                stroke_color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
                scores.append(f'''
                    <div class="quality-score-card {color_class}">
                        <div class="score-ring">
                            <svg viewBox="0 0 36 36" class="circular-chart">
                                <path class="circle-bg" fill="none" stroke="#e5e7eb" stroke-width="3.8" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                                <path class="circle" fill="none" stroke="{stroke_color}" stroke-width="2.8" stroke-linecap="round" stroke-dasharray="{score}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                                <text x="18" y="20.35" class="percentage" fill="{text_color}" font-size="0.5em" font-weight="bold" text-anchor="middle">{score:.0f}%</text>
                            </svg>
                        </div>
                        <div class="score-label">{label}</div>
                        <div class="score-desc">{desc}</div>
                    </div>
                ''')

        return f'<div class="quality-scores-grid">{"".join(scores)}</div>' if scores else ""


# =============================================================================
# Patterns Section
# =============================================================================


@register_section_renderer(SectionType.PATTERNS)
class PatternsSection(BaseSectionRenderer):
    """Renders detected patterns in the data."""

    section_type = SectionType.PATTERNS

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        labels = spec.metadata.get("labels", {})
        patterns_html = self._render_patterns_list(spec.metadata.get("patterns", []), labels)
        tables_html = self._render_patterns_table(spec.tables, labels)

        return f'''
<section class="report-section section-patterns" id="section-patterns">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {patterns_html}
        {tables_html}
    </div>
</section>
'''

    def _render_patterns_list(self, patterns: list[dict[str, Any]], labels: dict[str, str] | None = None) -> str:
        """Render patterns as a visual list."""
        labels = labels or {}
        if not patterns:
            return f'<p class="no-data">{labels.get("patterns.none", "No patterns detected")}</p>'

        items = []
        for p in patterns:
            column = p.get("column", "Unknown")
            pattern = p.get("pattern", "")
            match_ratio = p.get("match_ratio", 0)
            samples = p.get("sample_matches", [])[:3]

            match_class = "match-high" if match_ratio >= 0.9 else "match-medium" if match_ratio >= 0.7 else "match-low"

            samples_html = ""
            if samples:
                sample_items = ", ".join(f'<code>{s}</code>' for s in samples)
                samples_html = f'<div class="pattern-samples">{labels.get("patterns.examples", "Examples")}: {sample_items}</div>'

            items.append(f'''
                <div class="pattern-item">
                    <div class="pattern-header">
                        <span class="pattern-column">{escape(str(column))}</span>
                        <span class="pattern-name">{escape(str(pattern))}</span>
                        <span class="pattern-match {match_class}">{match_ratio:.1%}</span>
                    </div>
                    {samples_html}
                </div>
            ''')

        return f'<div class="patterns-list">{"".join(items)}</div>'

    def _render_patterns_table(self, tables: list[dict[str, Any]], labels: dict[str, str] | None = None) -> str:
        """Render patterns as a table."""
        labels = labels or {}
        if not tables:
            return ""

        html_parts = []
        for table in tables:
            title = escape(str(table.get("title", "")))
            headers = table.get("headers", [
                labels.get("patterns.column", "Column"),
                labels.get("patterns.pattern", "Pattern"),
                labels.get("patterns.match_rate", "Match Rate"),
                labels.get("patterns.samples", "Samples"),
            ])
            rows = table.get("rows", [])

            if not rows:
                continue

            header_html = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
            rows_html = ""
            for row in rows:
                cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
                rows_html += f"<tr>{cells}</tr>"

            html_parts.append(f'''
                <div class="table-container">
                    {f'<h4 class="table-title">{title}</h4>' if title else ''}
                    <table class="data-table">
                        <thead><tr>{header_html}</tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            ''')

        return "".join(html_parts)


# =============================================================================
# Distribution Section
# =============================================================================


@register_section_renderer(SectionType.DISTRIBUTION)
class DistributionSection(BaseSectionRenderer):
    """Renders data distribution visualizations."""

    section_type = SectionType.DISTRIBUTION

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        charts_html = self._render_charts(spec.charts, chart_renderer)

        return f'''
<section class="report-section section-distribution" id="section-distribution">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        <div class="charts-grid">
            {charts_html}
        </div>
    </div>
</section>
'''


# =============================================================================
# Correlations Section
# =============================================================================


@register_section_renderer(SectionType.CORRELATIONS)
class CorrelationsSection(BaseSectionRenderer):
    """Renders column correlations."""

    section_type = SectionType.CORRELATIONS

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        charts_html = self._render_charts(spec.charts, chart_renderer)
        correlations_html = self._render_correlations_list(
            spec.metadata.get("correlations", []),
            spec.metadata.get("labels", {}),
        )

        return f'''
<section class="report-section section-correlations" id="section-correlations">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {charts_html}
        {correlations_html}
    </div>
</section>
'''

    def _render_correlations_list(
        self,
        correlations: list[tuple[str, str, float]],
        labels: dict[str, str] | None = None,
    ) -> str:
        """Render significant correlations as a list."""
        labels = labels or {}
        if not correlations:
            return f'<p class="no-data">{labels.get("correlations.none", "No significant correlations found")}</p>'

        items = []
        for col1, col2, corr in correlations:
            corr_class = "corr-strong" if abs(corr) >= 0.8 else "corr-moderate" if abs(corr) >= 0.5 else "corr-weak"
            direction = "positive" if corr > 0 else "negative"

            items.append(f'''
                <div class="correlation-item {corr_class}">
                    <span class="corr-col">{col1}</span>
                    <span class="corr-arrow">↔</span>
                    <span class="corr-col">{col2}</span>
                    <span class="corr-value {direction}">{corr:+.3f}</span>
                </div>
            ''')

        return f'<div class="correlations-list">{"".join(items)}</div>'


# =============================================================================
# Recommendations Section
# =============================================================================


@register_section_renderer(SectionType.RECOMMENDATIONS)
class RecommendationsSection(BaseSectionRenderer):
    """Renders data quality recommendations."""

    section_type = SectionType.RECOMMENDATIONS

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        labels = spec.metadata.get("labels", {})
        recommendations_html = self._render_recommendations(spec.text_blocks, labels)
        validators_html = self._render_suggested_validators(spec.metadata.get("validators", []), labels)

        return f'''
<section class="report-section section-recommendations" id="section-recommendations">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {recommendations_html}
        {validators_html}
    </div>
</section>
'''

    def _render_recommendations(
        self,
        recommendations: list[str],
        labels: dict[str, str] | None = None,
    ) -> str:
        """Render recommendations list."""
        labels = labels or {}
        if not recommendations:
            return f'<p class="no-data">{labels.get("recommendations.none", "No specific recommendations at this time")}</p>'

        items = "".join(f'<li class="recommendation-item">{escape(str(r), quote=False)}</li>' for r in recommendations)
        return f'<ul class="recommendations-list">{items}</ul>'

    def _render_suggested_validators(
        self,
        validators: list[dict[str, Any]],
        labels: dict[str, str] | None = None,
    ) -> str:
        """Render suggested validators."""
        labels = labels or {}
        if not validators:
            return ""

        items = []
        for v in validators:
            column = v.get("column", "")
            validator_type = v.get("type", "")
            params = v.get("params", {})

            params_str = ", ".join(f"{k}={v}" for k, v in params.items())

            items.append(f'''
                <div class="validator-suggestion">
                    <span class="validator-column">{escape(str(column))}</span>
                    <code class="validator-code">{escape(str(validator_type))}({escape(params_str)})</code>
                </div>
            ''')

        return f'''
            <div class="validators-section">
                <h4>{labels.get("recommendations.validators", "Suggested Validators")}</h4>
                <div class="validators-list">{"".join(items)}</div>
            </div>
        '''


# =============================================================================
# Alerts Section
# =============================================================================


@register_section_renderer(SectionType.ALERTS)
class AlertsSection(BaseSectionRenderer):
    """Renders data quality alerts and warnings."""

    section_type = SectionType.ALERTS

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        if not spec.alerts:
            return ""

        alerts_html = self._render_alerts(spec.alerts, theme)

        return f'''
<section class="report-section section-alerts" id="section-alerts">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {alerts_html}
    </div>
</section>
'''


# =============================================================================
# Custom Section
# =============================================================================


@register_section_renderer(SectionType.CUSTOM)
class CustomSection(BaseSectionRenderer):
    """Renders custom content sections."""

    section_type = SectionType.CUSTOM

    def render(
        self,
        spec: SectionSpec,
        chart_renderer: BaseChartRenderer,
        theme: ThemeConfig,
    ) -> str:
        charts_html = self._render_charts(spec.charts, chart_renderer)
        text_html = "".join(f'<p>{escape(str(t))}</p>' for t in spec.text_blocks)

        return f'''
<section class="report-section section-custom" id="section-{spec.title.lower().replace(" ", "-")}">
    <div class="section-header">
        <h2 class="section-title">{spec.title}</h2>
        {f'<p class="section-subtitle">{spec.subtitle}</p>' if spec.subtitle else ''}
    </div>

    <div class="section-content">
        {text_html}
        {spec.custom_html}
        {charts_html}
    </div>
</section>
'''


# =============================================================================
# Factory Function
# =============================================================================


def get_section_renderer(section_type: SectionType | str) -> BaseSectionRenderer:
    """Get a section renderer instance.

    Args:
        section_type: Section type to render

    Returns:
        Section renderer instance
    """
    from truthound.datadocs.base import renderer_registry

    if isinstance(section_type, str):
        section_type = SectionType(section_type)

    renderer_class = renderer_registry.get_section_renderer(section_type)
    return renderer_class()
