"""HTML Report Generator.

Main entry point for generating HTML profile reports.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from truthound.profiler.visualization.base import (
    ProfileData,
    ReportConfig,
    ReportTheme,
    SectionType,
    ThemeConfig,
    THEME_CONFIGS,
)
from truthound.profiler.visualization.renderers import (
    ChartRenderer,
    chart_renderer_registry,
)
from truthound.profiler.visualization.sections import section_registry


class HTMLReportGenerator:
    """Generate comprehensive HTML reports from profile data.

    Example:
        from truthound.profiler.visualization import (
            HTMLReportGenerator,
            ProfileData,
            ReportConfig,
        )

        data = ProfileData(
            table_name="users",
            row_count=10000,
            column_count=15,
            columns=[...],
        )

        config = ReportConfig(
            title="User Data Profile",
            theme=ReportTheme.DARK,
            renderer="plotly",
        )

        generator = HTMLReportGenerator(config)
        html = generator.generate(data)
    """

    def __init__(
        self,
        config: ReportConfig | None = None,
        chart_renderer: ChartRenderer | None = None,
    ):
        """Initialize generator.

        Args:
            config: Report configuration
            chart_renderer: Chart renderer to use
        """
        self.config = config or ReportConfig()

        # Get chart renderer
        if chart_renderer:
            self._chart_renderer = chart_renderer
        else:
            renderer_name = self.config.renderer
            if renderer_name == "auto":
                # Prefer plotly if available
                available = chart_renderer_registry.get_available_renderers()
                renderer_name = "plotly" if "plotly" in available else "svg"

            self._chart_renderer = chart_renderer_registry.get(renderer_name)
            if not self._chart_renderer:
                from truthound.profiler.visualization.renderers import SVGChartRenderer
                self._chart_renderer = SVGChartRenderer()

        self._theme = THEME_CONFIGS.get(self.config.theme, THEME_CONFIGS[ReportTheme.LIGHT])

    def generate(self, data: ProfileData, config: ReportConfig | None = None) -> str:
        """Generate HTML report from profile data.

        Args:
            data: Profile data to visualize
            config: Optional config override (uses generator's config if not provided)

        Returns:
            Complete HTML document string
        """
        # Use provided config or fall back to generator's config
        if config is not None:
            old_config = self.config
            self.config = config
            self._theme = THEME_CONFIGS.get(config.theme, THEME_CONFIGS[ReportTheme.LIGHT])
        parts = []

        # HTML document start
        parts.append(self._render_head(data))
        parts.append('<body>')

        # Header
        parts.append(self._render_header(data))

        # Navigation / TOC
        if self.config.include_toc:
            parts.append(self._render_toc())

        # Main content
        parts.append('<main class="container">')

        # Summary
        if self.config.include_summary:
            parts.append(self._render_summary(data))

        # Sections
        for section_type in self.config.sections:
            renderer = section_registry.get(section_type)
            if renderer:
                section_html = renderer.render(data, self._chart_renderer, self._theme)
                if section_html:
                    parts.append(section_html)

        parts.append('</main>')

        # Footer
        parts.append(self._render_footer(data))

        # Scripts
        parts.append(self._render_scripts())

        parts.append('</body></html>')

        return '\n'.join(parts)

    def save(self, data: ProfileData, path: str | Path, config: ReportConfig | None = None) -> Path:
        """Generate and save HTML report to file.

        Args:
            data: Profile data to visualize
            path: Output file path
            config: Optional config override

        Returns:
            Path to the saved file
        """
        html_content = self.generate(data, config)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")
        return output_path

    def _render_head(self, data: ProfileData) -> str:
        """Render HTML head section."""
        title = self.config.title or f"Profile: {data.table_name}"

        # JS dependencies
        js_deps = []
        if self._chart_renderer.requires_js:
            js_deps.extend(self._chart_renderer.get_js_dependencies())

        js_tags = '\n'.join(f'<script src="{url}"></script>' for url in js_deps)

        return f'''<!DOCTYPE html>
<html lang="{self.config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    {js_tags}
    <style>
        {self._get_base_css()}
        {self._theme.to_css_vars()}
        {self.config.custom_css or ""}
    </style>
</head>'''

    def _render_header(self, data: ProfileData) -> str:
        """Render page header."""
        title = self.config.title or f"Profile: {data.table_name}"
        subtitle = self.config.subtitle or ""

        timestamp = ""
        if self.config.include_timestamp:
            ts = data.timestamp or datetime.now()
            timestamp = f'<span class="timestamp">Generated: {ts.strftime("%Y-%m-%d %H:%M:%S")}</span>'

        logo = ""
        if self.config.logo_base64:
            logo = f'<img src="data:image/png;base64,{self.config.logo_base64}" class="logo" alt="Logo">'
        elif self.config.logo_path:
            logo = f'<img src="{html.escape(self.config.logo_path)}" class="logo" alt="Logo">'

        return f'''
        <header class="report-header">
            {logo}
            <div class="header-text">
                <h1>{html.escape(title)}</h1>
                {f'<p class="subtitle">{html.escape(subtitle)}</p>' if subtitle else ""}
                {timestamp}
            </div>
        </header>
        '''

    def _render_toc(self) -> str:
        """Render table of contents."""
        toc_items = []

        for section_type in self.config.sections:
            section_id = section_type.value
            section_name = section_type.value.replace("_", " ").title()
            toc_items.append(f'<a href="#{section_id}" class="toc-item">{section_name}</a>')

        return f'''
        <nav class="toc">
            {''.join(toc_items)}
        </nav>
        '''

    def _render_summary(self, data: ProfileData) -> str:
        """Render quick summary."""
        return f'''
        <div class="summary-banner">
            <div class="summary-stat">
                <span class="stat-value">{data.row_count:,}</span>
                <span class="stat-label">Rows</span>
            </div>
            <div class="summary-stat">
                <span class="stat-value">{data.column_count}</span>
                <span class="stat-label">Columns</span>
            </div>
            <div class="summary-stat">
                <span class="stat-value">{data.table_name}</span>
                <span class="stat-label">Table</span>
            </div>
        </div>
        '''

    def _render_footer(self, data: ProfileData) -> str:
        """Render page footer."""
        return f'''
        <footer class="report-footer">
            <p>Generated by <strong>Truthound</strong> Data Profiler</p>
            <p class="footer-meta">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </footer>
        '''

    def _render_scripts(self) -> str:
        """Render JavaScript section."""
        base_js = '''
        <script>
            // Collapsible sections
            document.querySelectorAll('.collapsible .card-header').forEach(header => {
                header.addEventListener('click', () => {
                    header.parentElement.classList.toggle('collapsed');
                });
            });

            // Smooth scroll for TOC
            document.querySelectorAll('.toc-item').forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const target = document.querySelector(link.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });

            // Print-friendly
            window.addEventListener('beforeprint', () => {
                document.querySelectorAll('.collapsible').forEach(el => {
                    el.classList.remove('collapsed');
                });
            });
        </script>
        '''

        custom_js = self.config.custom_js or ""

        return base_js + (f'<script>{custom_js}</script>' if custom_js else "")

    def _get_base_css(self) -> str:
        """Get base CSS styles."""
        return '''
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-family);
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .report-header {
            background: var(--header-bg);
            padding: 24px 40px;
            display: flex;
            align-items: center;
            gap: 24px;
            border-bottom: 1px solid var(--border-color);
        }

        .report-header .logo {
            height: 48px;
        }

        .report-header h1 {
            font-size: 24px;
            font-weight: 600;
            color: var(--text-color);
        }

        .report-header .subtitle {
            color: #666;
            font-size: 14px;
        }

        .report-header .timestamp {
            display: block;
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }

        /* TOC */
        .toc {
            background: var(--card-bg);
            padding: 12px 20px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .toc-item {
            color: var(--primary-color);
            text-decoration: none;
            font-size: 14px;
            padding: 4px 8px;
            border-radius: 4px;
            transition: background 0.2s;
        }

        .toc-item:hover {
            background: var(--header-bg);
        }

        /* Summary Banner */
        .summary-banner {
            display: flex;
            gap: 24px;
            justify-content: center;
            padding: 24px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border-radius: var(--border-radius);
            margin: 20px 0;
        }

        .summary-stat {
            text-align: center;
            color: white;
            padding: 12px 24px;
        }

        .summary-stat .stat-value {
            display: block;
            font-size: 28px;
            font-weight: 700;
        }

        .summary-stat .stat-label {
            font-size: 12px;
            text-transform: uppercase;
            opacity: 0.9;
        }

        /* Sections */
        .section {
            margin: 32px 0;
        }

        .section h2 {
            font-size: 20px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--primary-color);
        }

        /* Cards */
        .card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            box-shadow: var(--shadow);
            margin-bottom: 16px;
            overflow: hidden;
        }

        .card-header {
            background: var(--header-bg);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .card-header h3 {
            font-size: 16px;
            font-weight: 600;
        }

        .card-body {
            padding: 16px;
        }

        .collapsible.collapsed .card-body {
            display: none;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 20px;
            text-align: center;
            box-shadow: var(--shadow);
        }

        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--primary-color);
        }

        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-top: 4px;
        }

        /* Charts */
        .chart-container {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 16px;
            margin: 16px 0;
            box-shadow: var(--shadow);
        }

        .chart-container.center {
            display: flex;
            justify-content: center;
        }

        /* Tables */
        .table-responsive {
            overflow-x: auto;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .data-table th,
        .data-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .data-table th {
            background: var(--header-bg);
            font-weight: 600;
            color: var(--text-color);
        }

        .data-table tr:hover {
            background: var(--header-bg);
        }

        .data-table code {
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
        }

        /* Column Details Grid */
        .column-details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
        }

        .column-card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 16px;
            box-shadow: var(--shadow);
        }

        .column-card h4 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .column-type {
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }

        .column-stats {
            display: flex;
            gap: 16px;
            font-size: 12px;
            color: #888;
        }

        /* Alerts */
        .alerts-container {
            margin-top: 24px;
        }

        .alert {
            padding: 12px 16px;
            border-radius: var(--border-radius);
            margin-bottom: 8px;
            font-size: 14px;
        }

        .alert-error { background: #fee2e2; color: #991b1b; border-left: 4px solid #dc2626; }
        .alert-warning { background: #fef3c7; color: #92400e; border-left: 4px solid #f59e0b; }
        .alert-info { background: #dbeafe; color: #1e40af; border-left: 4px solid #3b82f6; }
        .alert-success { background: #d1fae5; color: #065f46; border-left: 4px solid #10b981; }

        /* Patterns */
        .patterns-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 12px;
        }

        .pattern-card {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            padding: 12px;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--primary-color);
        }

        .pattern-card.high { border-left-color: #10b981; }
        .pattern-card.medium { border-left-color: #f59e0b; }
        .pattern-card.low { border-left-color: #ef4444; }

        .pattern-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }

        .match-ratio {
            font-weight: 600;
            color: var(--primary-color);
        }

        .pattern-name {
            font-size: 13px;
            color: #666;
        }

        .pattern-regex {
            display: block;
            font-size: 11px;
            color: #888;
            margin-top: 4px;
            word-break: break-all;
        }

        /* Recommendations */
        .recommendations-list {
            background: var(--card-bg);
            border-radius: var(--border-radius);
            overflow: hidden;
        }

        .recommendation-item {
            display: flex;
            gap: 12px;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }

        .recommendation-item:last-child {
            border-bottom: none;
        }

        .rec-number {
            background: var(--primary-color);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 600;
            flex-shrink: 0;
        }

        /* Footer */
        .report-footer {
            text-align: center;
            padding: 24px;
            margin-top: 40px;
            border-top: 1px solid var(--border-color);
            color: #666;
            font-size: 13px;
        }

        .footer-meta {
            font-size: 11px;
            color: #888;
            margin-top: 4px;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .summary-banner {
                flex-direction: column;
                gap: 12px;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .toc {
                justify-content: center;
            }
        }

        /* Print */
        @media print {
            .toc { display: none; }
            .card { break-inside: avoid; }
            .section { break-before: page; }
        }
        '''


class ReportExporter:
    """Export reports to various formats.

    Supports both static methods and instance-based API.
    """

    def __init__(self, config: ReportConfig | None = None) -> None:
        """Initialize exporter with optional configuration.

        Args:
            config: Report configuration for exports
        """
        self._config = config
        self._generator = HTMLReportGenerator(config)

    def to_html(self, profile: ProfileData, config: ReportConfig | None = None) -> str:
        """Export profile to HTML string.

        Args:
            profile: Profile data to export
            config: Optional config override

        Returns:
            HTML string
        """
        return self._generator.generate(profile, config or self._config)

    def to_json(self, profile: ProfileData) -> str:
        """Export profile to JSON string.

        Args:
            profile: Profile data to export

        Returns:
            JSON string
        """
        data = ProfileDataConverter.to_dict(profile)
        return json.dumps(data, indent=2, default=str)

    def to_file(
        self,
        profile: ProfileData,
        path: str | Path,
        format: str = "html",
        config: ReportConfig | None = None,
    ) -> None:
        """Export profile to file.

        Args:
            profile: Profile data to export
            path: Output file path
            format: Output format ("html", "json", "pdf")
            config: Optional config override for HTML
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "html":
            content = self.to_html(profile, config)
            path.write_text(content, encoding="utf-8")
        elif format == "json":
            content = self.to_json(profile)
            path.write_text(content, encoding="utf-8")
        elif format == "pdf":
            html_content = self.to_html(profile, config)
            self.to_pdf(html_content, path)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def save_html(html_content: str, path: str | Path) -> None:
        """Save HTML report to file.

        Args:
            html_content: HTML string
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_content, encoding="utf-8")

    @staticmethod
    def to_pdf(html_content: str, path: str | Path) -> None:
        """Convert HTML to PDF (requires weasyprint or pdfkit).

        Args:
            html_content: HTML string
            path: Output file path
        """
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(str(path))
        except ImportError:
            try:
                import pdfkit
                pdfkit.from_string(html_content, str(path))
            except ImportError:
                raise ImportError(
                    "PDF export requires weasyprint or pdfkit. "
                    "Install with: pip install weasyprint or pip install pdfkit"
                )


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_report(
    data: ProfileData | Dict[str, Any],
    title: str = "",
    theme: str | ReportTheme = "light",
    renderer: str = "auto",
    config: ReportConfig | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> str | Path:
    """Generate an HTML report from profile data.

    Args:
        data: Profile data (ProfileData or dict)
        title: Report title (ignored if config is provided)
        theme: Theme name or ReportTheme enum (ignored if config is provided)
        renderer: Chart renderer ("svg", "plotly", "echarts", "auto") (ignored if config is provided)
        config: Pre-built ReportConfig (if provided, title/theme/renderer are ignored)
        output_path: Optional path to save the report (if provided, returns Path instead of string)
        **kwargs: Additional ReportConfig options (ignored if config is provided)

    Returns:
        HTML string if output_path is None, Path to saved file otherwise
    """
    if isinstance(data, dict):
        # Extract table_name from dict, checking both 'table_name' and 'name' keys
        table_name = data.get("table_name", data.get("name", "data"))
        data = ProfileData(
            table_name=table_name,
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", len(data.get("columns", []))),
            columns=data.get("columns", []),
            quality_scores=data.get("quality_scores"),
            patterns_found=data.get("patterns_found"),
            alerts=data.get("alerts"),
            recommendations=data.get("recommendations"),
            timestamp=data.get("timestamp"),
        )

    # Use provided config or build one from parameters
    if config is None:
        if isinstance(theme, str):
            theme = ReportTheme(theme)

        config = ReportConfig(
            title=title or f"Profile: {data.table_name}",
            theme=theme,
            renderer=renderer,
        )

    generator = HTMLReportGenerator(config)
    html_content = generator.generate(data)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")
        return output_path

    return html_content


def compare_profiles(
    profiles: ProfileData | List[ProfileData],
    profile_b: ProfileData | None = None,
    title: str = "Profile Comparison",
    labels: List[str] | None = None,
    **kwargs: Any,
) -> str:
    """Generate a comparison report between profiles.

    Can be called with either:
    - compare_profiles(profile_a, profile_b) - two profile arguments
    - compare_profiles([profile1, profile2], labels=["Before", "After"]) - list of profiles

    Args:
        profiles: First profile or list of profiles to compare
        profile_b: Second profile (optional if profiles is a list)
        title: Report title
        labels: Labels for each profile (e.g., ["Before", "After"])
        **kwargs: Additional options

    Returns:
        HTML comparison report
    """
    # Handle list of profiles
    if isinstance(profiles, list):
        if len(profiles) < 2:
            raise ValueError("Need at least 2 profiles to compare")
        profile_a = profiles[0]
        profile_b = profiles[1]
        if labels and len(labels) >= 2:
            label_a, label_b = labels[0], labels[1]
        else:
            label_a = profile_a.table_name
            label_b = profile_b.table_name
    else:
        profile_a = profiles
        if profile_b is None:
            raise ValueError("profile_b is required when profiles is not a list")
        label_a = profile_a.table_name
        label_b = profile_b.table_name

    # Create comparison data
    comparison_data = ProfileData(
        table_name=f"{label_a} vs {label_b}",
        row_count=profile_a.row_count,
        column_count=profile_a.column_count,
        columns=[],
        metadata={
            "comparison": True,
            "profile_a": label_a,
            "profile_b": label_b,
        },
        recommendations=[
            f"Comparing {label_a} with {label_b}",
        ],
    )

    # Compare columns
    for col_a in profile_a.columns:
        col_name = col_a.get("name", "")
        col_b = next(
            (c for c in profile_b.columns if c.get("name") == col_name),
            None,
        )

        if col_b:
            comparison_data.columns.append({
                "name": col_name,
                "inferred_type": col_a.get("inferred_type"),
                "null_ratio_a": col_a.get("null_ratio", 0),
                "null_ratio_b": col_b.get("null_ratio", 0),
                "unique_ratio_a": col_a.get("unique_ratio", 0),
                "unique_ratio_b": col_b.get("unique_ratio", 0),
                "change": "modified",
            })
        else:
            comparison_data.columns.append({
                **col_a,
                "change": "removed",
            })

    # Find new columns in B
    for col_b in profile_b.columns:
        col_name = col_b.get("name", "")
        if not any(c.get("name") == col_name for c in profile_a.columns):
            comparison_data.columns.append({
                **col_b,
                "change": "added",
            })

    return generate_report(comparison_data, title=title, **kwargs)


# =============================================================================
# Theme Registry
# =============================================================================


import threading


class ThemeRegistry:
    """Registry for report themes.

    Allows custom themes to be registered and retrieved.
    """

    _instance: Optional["ThemeRegistry"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "ThemeRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # Only initialize themes once
        if not ThemeRegistry._initialized:
            self._themes: Dict[str, ThemeConfig] = {}
            # Populate with built-in themes
            for theme_enum, theme_config in THEME_CONFIGS.items():
                self._themes[theme_enum.value] = theme_config
            ThemeRegistry._initialized = True

    def register(self, name: str, theme: ThemeConfig) -> None:
        """Register a custom theme."""
        self._themes[name] = theme

    def get(self, name: str | ReportTheme) -> Optional[ThemeConfig]:
        """Get a theme by name."""
        if isinstance(name, ReportTheme):
            name = name.value
        return self._themes.get(name)

    def list_themes(self) -> List[str]:
        """List all available themes."""
        return list(self._themes.keys())


# Global theme registry
theme_registry = ThemeRegistry()


# =============================================================================
# Profile Data Converter
# =============================================================================


class ProfileDataConverter:
    """Convert between different profile data formats.

    Handles conversion from raw dict data to ProfileData and vice versa.
    Supports both static methods and instance-based API.
    """

    def __init__(self, profile: ProfileData | None = None) -> None:
        """Initialize converter with optional profile data.

        Args:
            profile: ProfileData to convert/process
        """
        self._profile = profile

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ProfileData:
        """Convert dictionary to ProfileData.

        Args:
            data: Raw profile data dictionary

        Returns:
            ProfileData instance
        """
        return ProfileData(
            table_name=data.get("name", data.get("table_name", "data")),
            row_count=data.get("row_count", 0),
            column_count=data.get("column_count", len(data.get("columns", []))),
            columns=data.get("columns", []),
            quality_scores=data.get("quality_scores"),
            patterns_found=data.get("patterns_found"),
            alerts=data.get("alerts"),
            recommendations=data.get("recommendations"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def to_dict(data: ProfileData) -> Dict[str, Any]:
        """Convert ProfileData to dictionary.

        Args:
            data: ProfileData instance

        Returns:
            Dictionary representation
        """
        result = {
            "table_name": data.table_name,
            "row_count": data.row_count,
            "column_count": data.column_count,
            "columns": data.columns,
        }

        if data.quality_scores:
            result["quality_scores"] = data.quality_scores
        if data.patterns_found:
            result["patterns_found"] = data.patterns_found
        if data.alerts:
            result["alerts"] = data.alerts
        if data.recommendations:
            result["recommendations"] = data.recommendations
        if data.timestamp:
            result["timestamp"] = data.timestamp.isoformat() if hasattr(data.timestamp, 'isoformat') else data.timestamp
        if data.metadata:
            result["metadata"] = data.metadata

        return result

    def create_overview_section(self) -> SectionContent:
        """Create overview section from profile data.

        Returns:
            SectionContent for overview section
        """
        if self._profile is None:
            raise ValueError("No profile data set")

        from truthound.profiler.visualization.base import SectionContent

        return SectionContent(
            section_type=SectionType.OVERVIEW,
            title="Overview",
            text_blocks=[
                f"Table: {self._profile.table_name}",
                f"Rows: {self._profile.row_count:,}",
                f"Columns: {self._profile.column_count}",
            ],
        )

    def create_quality_section(self) -> SectionContent:
        """Create data quality section from profile data.

        Returns:
            SectionContent for quality section
        """
        if self._profile is None:
            raise ValueError("No profile data set")

        from truthound.profiler.visualization.base import SectionContent

        text_blocks = []
        if self._profile.quality_scores:
            for key, value in self._profile.quality_scores.items():
                if isinstance(value, float):
                    text_blocks.append(f"{key}: {value:.1%}")
                else:
                    text_blocks.append(f"{key}: {value}")

        return SectionContent(
            section_type=SectionType.DATA_QUALITY,
            title="Data Quality",
            text_blocks=text_blocks,
        )


# =============================================================================
# Report Template
# =============================================================================


class ReportTemplate:
    """Customizable report template.

    Allows for complete customization of the report structure and styling.
    Can be initialized with a ThemeConfig or individual template parameters.
    """

    def __init__(
        self,
        name_or_theme: str | ThemeConfig = "default",
        base_css: str = "",
        header_template: str = "",
        footer_template: str = "",
        section_templates: Optional[Dict[str, str]] = None,
    ):
        """Initialize report template.

        Args:
            name_or_theme: Template name (str) or ThemeConfig instance
            base_css: Base CSS styles (ignored if ThemeConfig passed)
            header_template: HTML template for header
            footer_template: HTML template for footer
            section_templates: HTML templates for each section type
        """
        if isinstance(name_or_theme, ThemeConfig):
            self._theme = name_or_theme
            self.name = name_or_theme.name
            self.base_css = ""
        else:
            self._theme = None
            self.name = name_or_theme
            self.base_css = base_css

        self.header_template = header_template
        self.footer_template = footer_template
        self.section_templates = section_templates or {}

    def get_css(self) -> str:
        """Get CSS styles for this template.

        Returns:
            CSS string with theme variables
        """
        if self._theme:
            return self._theme.to_css_vars()
        return self.base_css

    def get_js(self) -> str:
        """Get JavaScript for this template.

        Returns:
            JavaScript string for interactivity
        """
        return '''
            // Collapsible sections
            document.querySelectorAll('.collapsible .card-header').forEach(header => {
                header.addEventListener('click', () => {
                    header.parentElement.classList.toggle('collapsed');
                });
            });

            // Smooth scroll for TOC
            document.querySelectorAll('.toc-item').forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const target = document.querySelector(link.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
        '''

    def render_header(self, config: ReportConfig) -> str:
        """Render the header section.

        Args:
            config: Report configuration

        Returns:
            HTML string for header
        """
        title = config.title or "Data Profile Report"
        subtitle = config.subtitle or ""

        timestamp = ""
        if config.include_timestamp:
            ts = datetime.now()
            timestamp = f'<span class="timestamp">Generated: {ts.strftime("%Y-%m-%d %H:%M:%S")}</span>'

        return f'''
        <header class="report-header">
            <div class="header-text">
                <h1>{html.escape(title)}</h1>
                {f'<p class="subtitle">{html.escape(subtitle)}</p>' if subtitle else ""}
                {timestamp}
            </div>
        </header>
        '''

    def apply(self, generator: HTMLReportGenerator) -> None:
        """Apply this template to a generator.

        Args:
            generator: HTMLReportGenerator to modify
        """
        css = self.get_css()
        if css:
            if generator.config.custom_css:
                generator.config.custom_css += "\n" + css
            else:
                generator.config.custom_css = css

    @classmethod
    def load_from_file(cls, path: str | Path) -> "ReportTemplate":
        """Load template from a file.

        Args:
            path: Path to template file (JSON)

        Returns:
            ReportTemplate instance
        """
        path = Path(path)
        data = json.loads(path.read_text())

        return cls(
            name_or_theme=data.get("name", path.stem),
            base_css=data.get("base_css", ""),
            header_template=data.get("header_template", ""),
            footer_template=data.get("footer_template", ""),
            section_templates=data.get("section_templates"),
        )
