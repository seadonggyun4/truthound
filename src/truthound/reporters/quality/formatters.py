"""Formatters for quality score data.

This module provides formatters that transform quality scores into
various string representations. Formatters are used by reporters
to render the actual content.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Sequence

from truthound.reporters.quality.protocols import (
    QualityFormatterProtocol,
    QualityReportable,
)
from truthound.reporters.quality.config import (
    QualityReporterConfig,
    QualityDisplayMode,
)
from truthound.reporters.quality.base import QualityStatistics

if TYPE_CHECKING:
    from truthound.profiler.quality import RuleQualityScore


# =============================================================================
# Base Formatter
# =============================================================================


class QualityFormatter(ABC):
    """Abstract base class for quality formatters."""

    name: str = "base"

    def __init__(self, config: QualityReporterConfig | None = None) -> None:
        self._config = config or QualityReporterConfig()

    @property
    def config(self) -> QualityReporterConfig:
        """Get formatter configuration."""
        return self._config

    def format_metric(self, value: float, as_percentage: bool = True) -> str:
        """Format a metric value.

        Args:
            value: Metric value to format.
            as_percentage: Whether to format as percentage.

        Returns:
            Formatted string.
        """
        precision = self._config.metric_precision
        if as_percentage and self._config.percentage_format:
            return f"{value * 100:.{precision-2}f}%"
        return f"{value:.{precision}f}"

    @abstractmethod
    def format_score(self, score: QualityReportable) -> str:
        """Format a single quality score."""
        pass

    @abstractmethod
    def format_scores(self, scores: Sequence[QualityReportable]) -> str:
        """Format multiple quality scores."""
        pass

    @abstractmethod
    def format_summary(
        self,
        scores: Sequence[QualityReportable],
        include_statistics: bool = True,
    ) -> str:
        """Format a summary of scores."""
        pass


# =============================================================================
# Console Formatter
# =============================================================================


class ConsoleFormatter(QualityFormatter):
    """Formatter for console/terminal output using Rich."""

    name = "console"

    # Color mapping for quality levels
    LEVEL_COLORS = {
        "excellent": "green",
        "good": "blue",
        "acceptable": "yellow",
        "poor": "dark_orange",
        "unacceptable": "red",
    }

    def format_score(self, score: QualityReportable) -> str:
        """Format a single score for console."""
        lines = []
        metrics = score.metrics

        # Header
        level = metrics.quality_level.value.lower()
        level_color = self.LEVEL_COLORS.get(level, "white")
        lines.append(f"[bold]{score.rule_name}[/bold] [{level_color}]({level})[/{level_color}]")

        # Core metrics
        f1 = self.format_metric(metrics.f1_score)
        precision = self.format_metric(metrics.precision)
        recall = self.format_metric(metrics.recall)
        lines.append(f"  F1: {f1} | Precision: {precision} | Recall: {recall}")

        if self._config.display_mode in (QualityDisplayMode.DETAILED, QualityDisplayMode.FULL):
            # Additional metrics
            accuracy = self.format_metric(metrics.accuracy)
            confidence = self.format_metric(metrics.confidence)
            lines.append(f"  Accuracy: {accuracy} | Confidence: {confidence}")

            # Confidence intervals
            if self._config.include_confidence_intervals:
                p_ci = metrics.precision_ci
                r_ci = metrics.recall_ci
                lines.append(
                    f"  CI (95%): Precision [{self.format_metric(p_ci[0])}, {self.format_metric(p_ci[1])}] | "
                    f"Recall [{self.format_metric(r_ci[0])}, {self.format_metric(r_ci[1])}]"
                )

        # Recommendation
        if self._config.include_recommendations:
            use_status = "[green]✓[/green]" if score.should_use else "[red]✗[/red]"
            lines.append(f"  {use_status} {score.recommendation}")

        return "\n".join(lines)

    def format_scores(self, scores: Sequence[QualityReportable]) -> str:
        """Format multiple scores as a table for console."""
        if not scores:
            return "[dim]No quality scores to display[/dim]"

        # Build Rich table markup
        lines = []
        lines.append("[bold]Quality Score Report[/bold]")
        lines.append("")

        # Table header
        header = "| Rule Name | Level | F1 | Precision | Recall | Use? |"
        separator = "|-----------|-------|-----|-----------|--------|------|"
        lines.extend([header, separator])

        # Table rows
        for score in scores:
            metrics = score.metrics
            level = metrics.quality_level.value.lower()
            level_color = self.LEVEL_COLORS.get(level, "white")

            row = (
                f"| {score.rule_name[:20]} | "
                f"[{level_color}]{level}[/{level_color}] | "
                f"{self.format_metric(metrics.f1_score)} | "
                f"{self.format_metric(metrics.precision)} | "
                f"{self.format_metric(metrics.recall)} | "
                f"{'✓' if score.should_use else '✗'} |"
            )
            lines.append(row)

        return "\n".join(lines)

    def format_summary(
        self,
        scores: Sequence[QualityReportable],
        include_statistics: bool = True,
    ) -> str:
        """Format summary for console."""
        lines = []
        lines.append("[bold]Quality Summary[/bold]")
        lines.append("")

        if not scores:
            lines.append("[dim]No scores to summarize[/dim]")
            return "\n".join(lines)

        stats = QualityStatistics.from_scores(scores)

        lines.append(f"Total Rules: {stats.total_count}")
        lines.append(
            f"Recommended for Use: [green]{stats.should_use_count}[/green] | "
            f"Not Recommended: [red]{stats.should_not_use_count}[/red]"
        )

        if include_statistics:
            lines.append("")
            lines.append("By Quality Level:")
            lines.append(f"  [green]Excellent[/green]: {stats.excellent_count}")
            lines.append(f"  [blue]Good[/blue]: {stats.good_count}")
            lines.append(f"  [yellow]Acceptable[/yellow]: {stats.acceptable_count}")
            lines.append(f"  [dark_orange]Poor[/dark_orange]: {stats.poor_count}")
            lines.append(f"  [red]Unacceptable[/red]: {stats.unacceptable_count}")

            lines.append("")
            lines.append("Metric Averages:")
            lines.append(f"  F1 Score: {self.format_metric(stats.avg_f1)}")
            lines.append(f"  Precision: {self.format_metric(stats.avg_precision)}")
            lines.append(f"  Recall: {self.format_metric(stats.avg_recall)}")
            lines.append(f"  Confidence: {self.format_metric(stats.avg_confidence)}")

        return "\n".join(lines)


# =============================================================================
# JSON Formatter
# =============================================================================


class JsonFormatter(QualityFormatter):
    """Formatter for JSON output."""

    name = "json"

    def __init__(
        self,
        config: QualityReporterConfig | None = None,
        indent: int = 2,
        sort_keys: bool = False,
    ) -> None:
        super().__init__(config)
        self._indent = indent
        self._sort_keys = sort_keys

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for special types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "value"):  # Enum
            return obj.value
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def format_score(self, score: QualityReportable) -> str:
        """Format a single score as JSON."""
        data = score.to_dict()
        return json.dumps(
            data,
            indent=self._indent,
            sort_keys=self._sort_keys,
            default=self._json_serializer,
        )

    def format_scores(self, scores: Sequence[QualityReportable]) -> str:
        """Format multiple scores as JSON array."""
        data = {
            "scores": [s.to_dict() for s in scores],
            "count": len(scores),
            "generated_at": datetime.now().isoformat(),
        }

        if self._config.include_statistics:
            stats = QualityStatistics.from_scores(scores)
            data["statistics"] = stats.to_dict()

        return json.dumps(
            data,
            indent=self._indent,
            sort_keys=self._sort_keys,
            default=self._json_serializer,
        )

    def format_summary(
        self,
        scores: Sequence[QualityReportable],
        include_statistics: bool = True,
    ) -> str:
        """Format summary as JSON."""
        stats = QualityStatistics.from_scores(scores)
        data = {
            "summary": {
                "total_count": stats.total_count,
                "should_use_count": stats.should_use_count,
                "should_not_use_count": stats.should_not_use_count,
            },
            "generated_at": datetime.now().isoformat(),
        }

        if include_statistics:
            data["statistics"] = stats.to_dict()

        return json.dumps(
            data,
            indent=self._indent,
            sort_keys=self._sort_keys,
            default=self._json_serializer,
        )


# =============================================================================
# Markdown Formatter
# =============================================================================


class MarkdownFormatter(QualityFormatter):
    """Formatter for Markdown output."""

    name = "markdown"

    def format_score(self, score: QualityReportable) -> str:
        """Format a single score as Markdown."""
        lines = []
        metrics = score.metrics
        level = metrics.quality_level.value.lower()

        # Header
        level_emoji = {
            "excellent": "🟢",
            "good": "🔵",
            "acceptable": "🟡",
            "poor": "🟠",
            "unacceptable": "🔴",
        }.get(level, "⚪")

        lines.append(f"### {score.rule_name} {level_emoji}")
        lines.append("")

        # Metrics table
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| F1 Score | {self.format_metric(metrics.f1_score)} |")
        lines.append(f"| Precision | {self.format_metric(metrics.precision)} |")
        lines.append(f"| Recall | {self.format_metric(metrics.recall)} |")
        lines.append(f"| Accuracy | {self.format_metric(metrics.accuracy)} |")
        lines.append(f"| Confidence | {self.format_metric(metrics.confidence)} |")
        lines.append(f"| Quality Level | {level.title()} |")
        lines.append("")

        # Recommendation
        if self._config.include_recommendations:
            status = "✅ Recommended" if score.should_use else "❌ Not Recommended"
            lines.append(f"**Status:** {status}")
            lines.append("")
            lines.append(f"> {score.recommendation}")

        return "\n".join(lines)

    def format_scores(self, scores: Sequence[QualityReportable]) -> str:
        """Format multiple scores as Markdown."""
        lines = []
        lines.append(f"# {self._config.title}")
        lines.append("")
        lines.append(f"*Generated: {datetime.now().strftime(self._config.timestamp_format)}*")
        lines.append("")

        if self._config.description:
            lines.append(self._config.description)
            lines.append("")

        # Summary table
        lines.append("## Overview")
        lines.append("")
        lines.append("| Rule | Level | F1 | Precision | Recall | Recommended |")
        lines.append("|------|-------|-----|-----------|--------|-------------|")

        for score in scores:
            metrics = score.metrics
            level = metrics.quality_level.value.lower()
            rec = "✅" if score.should_use else "❌"
            lines.append(
                f"| {score.rule_name} | {level.title()} | "
                f"{self.format_metric(metrics.f1_score)} | "
                f"{self.format_metric(metrics.precision)} | "
                f"{self.format_metric(metrics.recall)} | {rec} |"
            )

        lines.append("")

        # Detailed sections if configured
        if self._config.display_mode in (QualityDisplayMode.DETAILED, QualityDisplayMode.FULL):
            lines.append("## Details")
            lines.append("")
            for score in scores:
                lines.append(self.format_score(score))
                lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def format_summary(
        self,
        scores: Sequence[QualityReportable],
        include_statistics: bool = True,
    ) -> str:
        """Format summary as Markdown."""
        lines = []
        lines.append("## Quality Summary")
        lines.append("")

        stats = QualityStatistics.from_scores(scores)

        lines.append(f"- **Total Rules:** {stats.total_count}")
        lines.append(f"- **Recommended:** {stats.should_use_count}")
        lines.append(f"- **Not Recommended:** {stats.should_not_use_count}")
        lines.append("")

        if include_statistics:
            lines.append("### Quality Distribution")
            lines.append("")
            lines.append("| Level | Count |")
            lines.append("|-------|-------|")
            lines.append(f"| 🟢 Excellent | {stats.excellent_count} |")
            lines.append(f"| 🔵 Good | {stats.good_count} |")
            lines.append(f"| 🟡 Acceptable | {stats.acceptable_count} |")
            lines.append(f"| 🟠 Poor | {stats.poor_count} |")
            lines.append(f"| 🔴 Unacceptable | {stats.unacceptable_count} |")
            lines.append("")

            lines.append("### Metric Averages")
            lines.append("")
            lines.append(f"- **F1 Score:** {self.format_metric(stats.avg_f1)}")
            lines.append(f"- **Precision:** {self.format_metric(stats.avg_precision)}")
            lines.append(f"- **Recall:** {self.format_metric(stats.avg_recall)}")
            lines.append(f"- **Confidence:** {self.format_metric(stats.avg_confidence)}")

        return "\n".join(lines)


# =============================================================================
# HTML Formatter
# =============================================================================


class HtmlFormatter(QualityFormatter):
    """Formatter for HTML output."""

    name = "html"

    # CSS for inline styling
    DEFAULT_CSS = """
    <style>
        :root {
            --color-excellent: #22c55e;
            --color-good: #3b82f6;
            --color-acceptable: #eab308;
            --color-poor: #f97316;
            --color-unacceptable: #ef4444;
            --bg-color: #ffffff;
            --text-color: #1f2937;
            --border-color: #e5e7eb;
        }
        body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 2rem; background: var(--bg-color); color: var(--text-color); }
        h1, h2, h3 { margin-top: 1.5rem; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }
        .summary-card { background: #f9fafb; border: 1px solid var(--border-color); border-radius: 0.5rem; padding: 1rem; }
        .summary-card h3 { margin: 0 0 0.5rem 0; font-size: 0.875rem; color: #6b7280; }
        .summary-card .value { font-size: 1.5rem; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid var(--border-color); }
        th { background: #f9fafb; font-weight: 600; }
        .level-excellent { color: var(--color-excellent); }
        .level-good { color: var(--color-good); }
        .level-acceptable { color: var(--color-acceptable); }
        .level-poor { color: var(--color-poor); }
        .level-unacceptable { color: var(--color-unacceptable); }
        .badge { display: inline-block; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500; }
        .badge-success { background: #dcfce7; color: #166534; }
        .badge-danger { background: #fee2e2; color: #991b1b; }
        .recommendation { margin-top: 0.5rem; padding: 0.75rem; background: #f9fafb; border-radius: 0.25rem; font-size: 0.875rem; }
        .metric-bar { width: 100%; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
        .metric-bar-fill { height: 100%; border-radius: 4px; }
        .chart-container { margin: 1rem 0; min-height: 300px; }
        footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border-color); font-size: 0.75rem; color: #6b7280; }
    </style>
    """

    def _get_level_class(self, level: str) -> str:
        """Get CSS class for quality level."""
        return f"level-{level.lower()}"

    def _render_metric_bar(self, value: float, color: str = "#3b82f6") -> str:
        """Render a metric as a progress bar."""
        width = max(0, min(100, value * 100))
        return f'<div class="metric-bar"><div class="metric-bar-fill" style="width: {width}%; background: {color};"></div></div>'

    def format_score(self, score: QualityReportable) -> str:
        """Format a single score as HTML."""
        metrics = score.metrics
        level = metrics.quality_level.value.lower()
        level_class = self._get_level_class(level)

        status_badge = (
            '<span class="badge badge-success">Recommended</span>'
            if score.should_use
            else '<span class="badge badge-danger">Not Recommended</span>'
        )

        html = f"""
        <div class="score-card">
            <h3>{score.rule_name} <span class="{level_class}">({level.title()})</span></h3>
            {status_badge}
            <table>
                <tr><td>F1 Score</td><td>{self.format_metric(metrics.f1_score)}</td><td>{self._render_metric_bar(metrics.f1_score)}</td></tr>
                <tr><td>Precision</td><td>{self.format_metric(metrics.precision)}</td><td>{self._render_metric_bar(metrics.precision)}</td></tr>
                <tr><td>Recall</td><td>{self.format_metric(metrics.recall)}</td><td>{self._render_metric_bar(metrics.recall)}</td></tr>
                <tr><td>Accuracy</td><td>{self.format_metric(metrics.accuracy)}</td><td>{self._render_metric_bar(metrics.accuracy)}</td></tr>
                <tr><td>Confidence</td><td>{self.format_metric(metrics.confidence)}</td><td>{self._render_metric_bar(metrics.confidence)}</td></tr>
            </table>
            <div class="recommendation">{score.recommendation}</div>
        </div>
        """
        return html

    def format_scores(self, scores: Sequence[QualityReportable]) -> str:
        """Format multiple scores as full HTML document."""
        stats = QualityStatistics.from_scores(scores)
        timestamp = datetime.now().strftime(self._config.timestamp_format)

        # Build table rows
        table_rows = []
        for score in scores:
            metrics = score.metrics
            level = metrics.quality_level.value.lower()
            level_class = self._get_level_class(level)
            status = "✓" if score.should_use else "✗"
            status_class = "badge-success" if score.should_use else "badge-danger"

            row = f"""
            <tr>
                <td>{score.rule_name}</td>
                <td class="{level_class}">{level.title()}</td>
                <td>{self.format_metric(metrics.f1_score)}</td>
                <td>{self.format_metric(metrics.precision)}</td>
                <td>{self.format_metric(metrics.recall)}</td>
                <td>{self.format_metric(metrics.confidence)}</td>
                <td><span class="badge {status_class}">{status}</span></td>
            </tr>
            """
            table_rows.append(row)

        # Build chart data (ApexCharts)
        chart_labels = [s.rule_name[:15] for s in scores]
        f1_data = [round(s.metrics.f1_score * 100, 1) for s in scores]
        precision_data = [round(s.metrics.precision * 100, 1) for s in scores]
        recall_data = [round(s.metrics.recall * 100, 1) for s in scores]

        chart_script = ""
        if self._config.include_charts:
            chart_script = f"""
            <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
            <script>
                var chartOptions = {{
                    series: [
                        {{ name: 'F1 Score', data: {f1_data} }},
                        {{ name: 'Precision', data: {precision_data} }},
                        {{ name: 'Recall', data: {recall_data} }}
                    ],
                    chart: {{ type: 'bar', height: 350 }},
                    plotOptions: {{ bar: {{ horizontal: false, columnWidth: '55%' }} }},
                    dataLabels: {{ enabled: false }},
                    xaxis: {{ categories: {chart_labels} }},
                    yaxis: {{ max: 100, title: {{ text: 'Score (%)' }} }},
                    fill: {{ opacity: 1 }},
                    colors: ['#3b82f6', '#22c55e', '#f59e0b'],
                    legend: {{ position: 'top' }}
                }};
                var chart = new ApexCharts(document.querySelector("#metrics-chart"), chartOptions);
                chart.render();
            </script>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._config.title}</title>
    {self.DEFAULT_CSS}
    {self._config.custom_css or ''}
</head>
<body>
    <h1>{self._config.title}</h1>
    <p>Generated: {timestamp}</p>

    <h2>Summary</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <h3>Total Rules</h3>
            <div class="value">{stats.total_count}</div>
        </div>
        <div class="summary-card">
            <h3>Recommended</h3>
            <div class="value" style="color: var(--color-excellent);">{stats.should_use_count}</div>
        </div>
        <div class="summary-card">
            <h3>Not Recommended</h3>
            <div class="value" style="color: var(--color-unacceptable);">{stats.should_not_use_count}</div>
        </div>
        <div class="summary-card">
            <h3>Avg F1 Score</h3>
            <div class="value">{self.format_metric(stats.avg_f1)}</div>
        </div>
    </div>

    <h2>Quality Distribution</h2>
    <div class="summary-grid">
        <div class="summary-card">
            <h3>🟢 Excellent</h3>
            <div class="value level-excellent">{stats.excellent_count}</div>
        </div>
        <div class="summary-card">
            <h3>🔵 Good</h3>
            <div class="value level-good">{stats.good_count}</div>
        </div>
        <div class="summary-card">
            <h3>🟡 Acceptable</h3>
            <div class="value level-acceptable">{stats.acceptable_count}</div>
        </div>
        <div class="summary-card">
            <h3>🟠 Poor</h3>
            <div class="value level-poor">{stats.poor_count}</div>
        </div>
        <div class="summary-card">
            <h3>🔴 Unacceptable</h3>
            <div class="value level-unacceptable">{stats.unacceptable_count}</div>
        </div>
    </div>

    {'<h2>Metrics Comparison</h2><div id="metrics-chart" class="chart-container"></div>' if self._config.include_charts else ''}

    <h2>Rule Details</h2>
    <table>
        <thead>
            <tr>
                <th>Rule Name</th>
                <th>Level</th>
                <th>F1</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Confidence</th>
                <th>Recommended</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>

    <footer>
        Generated by Truthound Quality Reporter | {timestamp}
    </footer>

    {chart_script}
</body>
</html>
"""
        return html

    def format_summary(
        self,
        scores: Sequence[QualityReportable],
        include_statistics: bool = True,
    ) -> str:
        """Format summary as HTML fragment."""
        stats = QualityStatistics.from_scores(scores)

        html = f"""
        <div class="quality-summary">
            <h2>Quality Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Rules</h3>
                    <div class="value">{stats.total_count}</div>
                </div>
                <div class="summary-card">
                    <h3>Recommended</h3>
                    <div class="value" style="color: var(--color-excellent);">{stats.should_use_count}</div>
                </div>
                <div class="summary-card">
                    <h3>Avg F1</h3>
                    <div class="value">{self.format_metric(stats.avg_f1)}</div>
                </div>
            </div>
        </div>
        """
        return html
