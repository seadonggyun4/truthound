"""HTML Report Builder for Data Docs.

This module provides the main builder class for generating static HTML reports
from profile data. It orchestrates the rendering of all sections and produces
a complete, self-contained HTML document.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from truthound.datadocs.base import (
    ReportTheme,
    ChartLibrary,
    ChartType,
    SectionType,
    SeverityLevel,
    ReportMetadata,
    ReportConfig,
    ReportSpec,
    ChartSpec,
    SectionSpec,
    AlertSpec,
    ThemeConfig,
    BaseChartRenderer,
)
from truthound.datadocs.themes import get_theme, THEMES
from truthound.datadocs.charts import get_chart_renderer, CDN_URLS
from truthound.datadocs.sections import get_section_renderer
from truthound.datadocs.styles import get_complete_stylesheet


# =============================================================================
# Profile Data Converter
# =============================================================================


class ProfileDataConverter:
    """Converts TableProfile to report-ready data structures."""

    def __init__(self, profile: dict[str, Any] | Any) -> None:
        """Initialize with profile data.

        Args:
            profile: TableProfile dict or object
        """
        if hasattr(profile, "to_dict"):
            self.data = profile.to_dict()
        else:
            self.data = profile

    def get_overview_metrics(self) -> dict[str, Any]:
        """Extract overview metrics from profile."""
        metrics = {
            "row_count": self.data.get("row_count", 0),
            "column_count": self.data.get("column_count", 0),
            "memory_bytes": self.data.get("estimated_memory_bytes", 0),
        }

        # Duplicate rows
        dup_count = self.data.get("duplicate_row_count", 0)
        if dup_count > 0:
            metrics["duplicate_rows"] = dup_count
            metrics["duplicate_ratio"] = self.data.get("duplicate_row_ratio", 0)

        # Calculate total null cells
        total_nulls = sum(
            col.get("null_count", 0)
            for col in self.data.get("columns", [])
        )
        if total_nulls > 0:
            metrics["null_cells"] = total_nulls

        # Calculate overall quality score
        quality_score = self._calculate_quality_score()
        metrics["quality_score"] = quality_score

        return metrics

    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)."""
        columns = self.data.get("columns", [])
        if not columns:
            return 100.0

        # Completeness component (40%)
        avg_null_ratio = sum(c.get("null_ratio", 0) for c in columns) / len(columns)
        completeness_score = (1 - avg_null_ratio) * 100

        # Uniqueness component (30%)
        # Penalize constant columns and low uniqueness
        uniqueness_scores = []
        for col in columns:
            if col.get("is_constant", False):
                uniqueness_scores.append(50)  # Constants are okay but not great
            else:
                uniqueness_scores.append(min(col.get("unique_ratio", 0) * 100, 100))
        uniqueness_score = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 100

        # Validity component (30%)
        # Based on pattern detection and type inference
        validity_score = 100  # Default high if no issues detected
        for col in columns:
            inferred = col.get("inferred_type", "unknown")
            if inferred in ("unknown", "string") and col.get("detected_patterns"):
                validity_score = max(validity_score - 5, 50)

        # Weighted average
        overall = (
            completeness_score * 0.4 +
            uniqueness_score * 0.3 +
            validity_score * 0.3
        )
        return round(overall, 1)

    def get_column_data(self) -> list[dict[str, Any]]:
        """Get formatted column data."""
        return self.data.get("columns", [])

    def get_type_distribution(self) -> ChartSpec:
        """Get chart spec for data type distribution."""
        columns = self.data.get("columns", [])
        type_counts: dict[str, int] = {}

        for col in columns:
            dtype = col.get("inferred_type", col.get("physical_type", "unknown"))
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        return ChartSpec(
            chart_type=ChartType.DONUT,
            title="Data Types Distribution",
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            height=300,
        )

    def get_null_distribution(self) -> ChartSpec:
        """Get chart spec for null value distribution."""
        columns = self.data.get("columns", [])

        # Sort by null ratio descending
        sorted_cols = sorted(
            [(c.get("name", ""), c.get("null_ratio", 0)) for c in columns],
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10

        return ChartSpec(
            chart_type=ChartType.HORIZONTAL_BAR,
            title="Top Columns by Missing Values",
            labels=[c[0] for c in sorted_cols],
            values=[c[1] * 100 for c in sorted_cols],
            height=300,
        )

    def get_uniqueness_distribution(self) -> ChartSpec:
        """Get chart spec for uniqueness distribution."""
        columns = self.data.get("columns", [])

        # Sort by unique ratio
        sorted_cols = sorted(
            [(c.get("name", ""), c.get("unique_ratio", 0)) for c in columns],
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10

        return ChartSpec(
            chart_type=ChartType.HORIZONTAL_BAR,
            title="Top Columns by Uniqueness",
            labels=[c[0] for c in sorted_cols],
            values=[c[1] * 100 for c in sorted_cols],
            height=300,
        )

    def get_patterns(self) -> list[dict[str, Any]]:
        """Get detected patterns from all columns."""
        patterns = []
        for col in self.data.get("columns", []):
            for pattern in col.get("detected_patterns", []):
                patterns.append({
                    "column": col.get("name", ""),
                    "pattern": pattern.get("pattern", ""),
                    "regex": pattern.get("regex", ""),
                    "match_ratio": pattern.get("match_ratio", 0),
                    "sample_matches": pattern.get("sample_matches", []),
                })
        return patterns

    def get_correlations(self) -> list[tuple[str, str, float]]:
        """Get column correlations."""
        corrs = self.data.get("correlations", [])
        if isinstance(corrs, list) and corrs:
            if isinstance(corrs[0], dict):
                return [
                    (c.get("column1", ""), c.get("column2", ""), c.get("correlation", 0))
                    for c in corrs
                ]
            return corrs
        return []

    def get_alerts(self) -> list[AlertSpec]:
        """Generate alerts based on profile data."""
        alerts = []

        for col in self.data.get("columns", []):
            name = col.get("name", "")
            null_ratio = col.get("null_ratio", 0)
            unique_ratio = col.get("unique_ratio", 0)
            is_constant = col.get("is_constant", False)

            # High null ratio alert
            if null_ratio > 0.5:
                alerts.append(AlertSpec(
                    title=f"High Missing Values in '{name}'",
                    message=f"Column has {null_ratio:.1%} missing values",
                    severity=SeverityLevel.WARNING if null_ratio < 0.8 else SeverityLevel.ERROR,
                    column=name,
                    metric="null_ratio",
                    value=null_ratio,
                    threshold=0.5,
                    suggestion="Consider imputation or removal",
                ))

            # Constant column alert
            if is_constant:
                alerts.append(AlertSpec(
                    title=f"Constant Column: '{name}'",
                    message="Column contains only one unique value",
                    severity=SeverityLevel.INFO,
                    column=name,
                    suggestion="Consider removing if not informative",
                ))

            # Very low uniqueness (possible ID column issues)
            if unique_ratio < 0.01 and not is_constant:
                row_count = self.data.get("row_count", 1)
                if row_count > 100:
                    alerts.append(AlertSpec(
                        title=f"Low Cardinality in '{name}'",
                        message=f"Only {col.get('distinct_count', 0)} unique values in {row_count:,} rows",
                        severity=SeverityLevel.INFO,
                        column=name,
                    ))

        # Duplicate rows alert
        dup_ratio = self.data.get("duplicate_row_ratio", 0)
        if dup_ratio > 0.1:
            alerts.append(AlertSpec(
                title="Significant Duplicate Rows",
                message=f"{dup_ratio:.1%} of rows are duplicates",
                severity=SeverityLevel.WARNING,
                metric="duplicate_ratio",
                value=dup_ratio,
                threshold=0.1,
                suggestion="Consider deduplication",
            ))

        return alerts

    def get_recommendations(self) -> list[str]:
        """Generate recommendations based on profile data."""
        recommendations = []

        for col in self.data.get("columns", []):
            validators = col.get("suggested_validators", [])
            for v in validators[:2]:  # Limit per column
                recommendations.append(
                    f"Add {v} validator for column '{col.get('name', '')}'"
                )

        # General recommendations
        dup_ratio = self.data.get("duplicate_row_ratio", 0)
        if dup_ratio > 0.05:
            recommendations.append(
                "Consider implementing duplicate row detection in your data pipeline"
            )

        # Check for high null columns
        high_null_cols = [
            c.get("name", "")
            for c in self.data.get("columns", [])
            if c.get("null_ratio", 0) > 0.3
        ]
        if high_null_cols:
            recommendations.append(
                f"Review data collection for columns with high missing values: {', '.join(high_null_cols[:3])}"
            )

        return recommendations[:10]  # Limit recommendations


# =============================================================================
# HTML Report Builder
# =============================================================================


class HTMLReportBuilder:
    """Builder for generating static HTML reports from profile data."""

    def __init__(
        self,
        theme: ReportTheme | str = ReportTheme.PROFESSIONAL,
        chart_library: ChartLibrary | str = ChartLibrary.APEXCHARTS,
        config: ReportConfig | None = None,
    ) -> None:
        """Initialize the report builder.

        Args:
            theme: Report theme to use
            chart_library: Chart library for visualizations
            config: Optional full configuration
        """
        if config:
            self.config = config
        else:
            self.config = ReportConfig(
                theme=ReportTheme(theme) if isinstance(theme, str) else theme,
                chart_library=ChartLibrary(chart_library) if isinstance(chart_library, str) else chart_library,
            )

        self._theme_config = self.config.custom_theme or get_theme(self.config.theme)
        self._chart_renderer = get_chart_renderer(self.config.chart_library)

    def build(
        self,
        profile: dict[str, Any] | Any,
        title: str = "Data Profile Report",
        subtitle: str = "",
        description: str = "",
    ) -> str:
        """Build a complete HTML report from profile data.

        Args:
            profile: TableProfile dict or object
            title: Report title
            subtitle: Report subtitle
            description: Report description

        Returns:
            Complete HTML document as string
        """
        converter = ProfileDataConverter(profile)

        # Create metadata
        metadata = ReportMetadata(
            title=title,
            subtitle=subtitle,
            description=description,
            data_source=profile.get("source", "") if isinstance(profile, dict) else getattr(profile, "source", ""),
            created_at=datetime.now(),
        )

        # Build sections
        sections = self._build_sections(converter)

        # Create report spec
        spec = ReportSpec(
            metadata=metadata,
            config=self.config,
            sections=sections,
            profile_data=converter.data,
        )

        return self._render_html(spec)

    def _build_sections(self, converter: ProfileDataConverter) -> list[SectionSpec]:
        """Build all section specifications."""
        sections = []

        for section_type in self.config.sections:
            spec = self._build_section(section_type, converter)
            if spec and spec.visible:
                sections.append(spec)

        return sections

    def _build_section(
        self,
        section_type: SectionType,
        converter: ProfileDataConverter,
    ) -> SectionSpec | None:
        """Build a single section specification."""
        if section_type == SectionType.OVERVIEW:
            return SectionSpec(
                section_type=section_type,
                title="Overview",
                subtitle="Dataset summary and key metrics",
                metrics=converter.get_overview_metrics(),
                charts=[
                    converter.get_type_distribution(),
                ],
            )

        elif section_type == SectionType.QUALITY:
            metrics = converter.get_overview_metrics()
            quality_score = metrics.get("quality_score", 100)

            # Calculate dimension scores
            columns = converter.get_column_data()
            completeness = (1 - sum(c.get("null_ratio", 0) for c in columns) / len(columns)) * 100 if columns else 100
            uniqueness = sum(c.get("unique_ratio", 0) for c in columns) / len(columns) * 100 if columns else 100

            return SectionSpec(
                section_type=section_type,
                title="Data Quality",
                subtitle="Quality metrics and assessments",
                metrics={
                    "overall": quality_score,
                    "completeness": completeness,
                    "uniqueness": min(uniqueness, 100),
                },
                charts=[
                    converter.get_null_distribution(),
                ],
                alerts=converter.get_alerts(),
            )

        elif section_type == SectionType.COLUMNS:
            columns = converter.get_column_data()

            # Build summary table
            table = {
                "title": "Column Summary",
                "headers": ["Column", "Type", "Null %", "Unique %", "Distinct"],
                "rows": [
                    [
                        c.get("name", ""),
                        c.get("inferred_type", c.get("physical_type", "")),
                        f"{c.get('null_ratio', 0):.1%}",
                        f"{c.get('unique_ratio', 0):.1%}",
                        f"{c.get('distinct_count', 0):,}",
                    ]
                    for c in columns
                ],
            }

            return SectionSpec(
                section_type=section_type,
                title="Column Details",
                subtitle=f"{len(columns)} columns analyzed",
                tables=[table],
                metadata={"columns": columns},
            )

        elif section_type == SectionType.PATTERNS:
            patterns = converter.get_patterns()
            return SectionSpec(
                section_type=section_type,
                title="Detected Patterns",
                subtitle="Automatically detected data patterns",
                metadata={"patterns": patterns},
                visible=len(patterns) > 0,
            )

        elif section_type == SectionType.DISTRIBUTION:
            return SectionSpec(
                section_type=section_type,
                title="Value Distribution",
                subtitle="Distribution analysis across columns",
                charts=[
                    converter.get_uniqueness_distribution(),
                ],
            )

        elif section_type == SectionType.CORRELATIONS:
            correlations = converter.get_correlations()
            return SectionSpec(
                section_type=section_type,
                title="Correlations",
                subtitle="Column relationships and correlations",
                metadata={"correlations": correlations},
                visible=len(correlations) > 0,
            )

        elif section_type == SectionType.RECOMMENDATIONS:
            recommendations = converter.get_recommendations()
            return SectionSpec(
                section_type=section_type,
                title="Recommendations",
                subtitle="Suggested improvements and validations",
                text_blocks=recommendations,
                visible=len(recommendations) > 0,
            )

        elif section_type == SectionType.ALERTS:
            alerts = converter.get_alerts()
            return SectionSpec(
                section_type=section_type,
                title="Alerts",
                subtitle="Data quality issues and warnings",
                alerts=alerts,
                visible=len(alerts) > 0,
            )

        return None

    def _render_html(self, spec: ReportSpec) -> str:
        """Render the complete HTML document."""
        is_dark = spec.config.theme == ReportTheme.DARK
        css = get_complete_stylesheet(
            self._theme_config.to_css_vars(),
            is_dark=is_dark,
        )

        # Render sections
        sections_html = []
        for section_spec in spec.sections:
            renderer = get_section_renderer(section_spec.section_type)
            section_html = renderer.render(
                section_spec,
                self._chart_renderer,
                self._theme_config,
            )
            sections_html.append(section_html)

        # Build TOC
        toc_html = ""
        if spec.config.include_toc:
            toc_items = []
            for section_spec in spec.sections:
                section_id = f"section-{section_spec.section_type.value}"
                toc_items.append(
                    f'<li class="toc-item"><a href="#{section_id}">{section_spec.title}</a></li>'
                )
            toc_html = f'''
                <nav class="report-toc">
                    <h3 class="toc-title">Contents</h3>
                    <ul class="toc-list">{"".join(toc_items)}</ul>
                </nav>
            '''

        # Build header
        header_html = ""
        if spec.config.include_header:
            logo_html = ""
            if spec.config.logo_base64:
                logo_html = f'<img src="{spec.config.logo_base64}" alt="Logo" class="report-logo">'
            elif spec.config.logo_url:
                logo_html = f'<img src="{spec.config.logo_url}" alt="Logo" class="report-logo">'

            meta_items = []
            if spec.config.include_timestamp:
                meta_items.append(
                    f'<span class="report-meta-item">Generated: {spec.metadata.created_at.strftime(spec.config.date_format)}</span>'
                )
            if spec.metadata.data_source:
                meta_items.append(
                    f'<span class="report-meta-item">Source: {html.escape(spec.metadata.data_source)}</span>'
                )

            header_html = f'''
                <header class="report-header">
                    <div class="report-header-main">
                        <div>
                            <h1 class="report-title">{html.escape(spec.metadata.title)}</h1>
                            {f'<p class="report-subtitle">{html.escape(spec.metadata.subtitle)}</p>' if spec.metadata.subtitle else ''}
                        </div>
                        {logo_html}
                    </div>
                    {f'<div class="report-meta">{"".join(meta_items)}</div>' if meta_items else ''}
                </header>
            '''

        # Build footer
        footer_html = ""
        if spec.config.include_footer:
            footer_html = f'''
                <footer class="report-footer">
                    <p>{html.escape(spec.config.footer_text)}</p>
                </footer>
            '''

        # Get CDN dependencies
        cdn_scripts = self._chart_renderer.get_dependencies()
        scripts_html = "\n".join(
            f'<script src="{url}"></script>' for url in cdn_scripts
        )

        # Build complete HTML
        html_content = f'''<!DOCTYPE html>
<html lang="{spec.config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(spec.metadata.title)}</title>
    <meta name="description" content="{html.escape(spec.metadata.description or 'Data Profile Report')}">
    <meta name="generator" content="Truthound">
    <style>
{css}
{spec.config.custom_css}
    </style>
    {scripts_html}
</head>
<body>
    <div class="report-container">
        {header_html}
        {toc_html}
        <main class="report-main">
            {"".join(sections_html)}
        </main>
        {footer_html}
    </div>
    <script>
{spec.config.custom_js}
    </script>
</body>
</html>'''

        return html_content

    def save(self, html_content: str, path: str | Path) -> Path:
        """Save HTML content to file.

        Args:
            html_content: HTML content to save
            path: Output file path

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html_content, encoding="utf-8")
        return path


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_html_report(
    profile: dict[str, Any] | Any,
    title: str = "Data Profile Report",
    subtitle: str = "",
    theme: ReportTheme | str = ReportTheme.PROFESSIONAL,
    chart_library: ChartLibrary | str = ChartLibrary.APEXCHARTS,
    output_path: str | Path | None = None,
) -> str:
    """Generate an HTML report from profile data.

    Args:
        profile: TableProfile dict or object
        title: Report title
        subtitle: Report subtitle
        theme: Report theme
        chart_library: Chart library for visualizations
        output_path: Optional path to save the report

    Returns:
        HTML content as string
    """
    builder = HTMLReportBuilder(theme=theme, chart_library=chart_library)
    html_content = builder.build(profile, title=title, subtitle=subtitle)

    if output_path:
        builder.save(html_content, output_path)

    return html_content


def generate_report_from_file(
    profile_path: str | Path,
    output_path: str | Path | None = None,
    title: str = "Data Profile Report",
    theme: ReportTheme | str = ReportTheme.PROFESSIONAL,
) -> str:
    """Generate an HTML report from a profile JSON file.

    Args:
        profile_path: Path to profile JSON file
        output_path: Optional path to save the report
        title: Report title
        theme: Report theme

    Returns:
        HTML content as string
    """
    profile_path = Path(profile_path)
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    if not output_path:
        output_path = profile_path.with_suffix(".html")

    return generate_html_report(
        profile=profile,
        title=title,
        theme=theme,
        output_path=output_path,
    )


def export_report(
    profile: dict[str, Any] | Any,
    output_path: str | Path,
    format: str = "html",
    **kwargs: Any,
) -> Path:
    """Export a report to the specified format.

    Args:
        profile: TableProfile dict or object
        output_path: Output file path
        format: Export format (html, pdf, png)
        **kwargs: Additional arguments passed to the builder

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)

    if format == "html":
        html_content = generate_html_report(profile, **kwargs)
        output_path.write_text(html_content, encoding="utf-8")
        return output_path

    elif format == "pdf":
        # PDF export requires additional dependencies
        return export_to_pdf(profile, output_path, **kwargs)

    else:
        raise ValueError(f"Unsupported export format: {format}")


def export_to_pdf(
    profile: dict[str, Any] | Any,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Export report to PDF.

    Requires: pip install weasyprint

    Args:
        profile: TableProfile dict or object
        output_path: Output PDF file path
        **kwargs: Additional arguments

    Returns:
        Path to PDF file
    """
    try:
        from weasyprint import HTML
    except ImportError:
        raise ImportError(
            "PDF export requires weasyprint. "
            "Install with: pip install weasyprint"
        )

    output_path = Path(output_path)
    html_content = generate_html_report(profile, **kwargs)

    # Convert to PDF
    HTML(string=html_content).write_pdf(output_path)

    return output_path
