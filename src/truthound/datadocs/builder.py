"""HTML Report Builder for Data Docs.

This module provides the main builder class for generating static HTML reports
from profile data. It orchestrates the rendering of all sections and produces
a complete, self-contained HTML document.
"""

from __future__ import annotations

import html
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
from truthound.datadocs.i18n import get_catalog
from truthound.datadocs.report_document import ALERT_THRESHOLDS, ResearchReportDocument
from truthound.datadocs.report_renderers import ReportDocumentRenderer
from truthound.datadocs.sections import get_section_renderer
from truthound.datadocs.styles import get_complete_stylesheet
from truthound.datadocs.type_labels import profile_type_label


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

    def get_type_distribution(self, title: str = "Data Types Distribution") -> ChartSpec:
        """Get chart spec for data type distribution."""
        columns = self.data.get("columns", [])
        type_counts: dict[str, int] = {}

        for col in columns:
            dtype = profile_type_label(col)
            type_counts[dtype] = type_counts.get(dtype, 0) + 1

        return ChartSpec(
            chart_type=ChartType.DONUT,
            title=title,
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            height=300,
        )

    def get_null_distribution(self, title: str = "Top Columns by Missing Values") -> ChartSpec:
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
            title=title,
            labels=[c[0] for c in sorted_cols],
            values=[c[1] * 100 for c in sorted_cols],
            height=300,
        )

    def get_uniqueness_distribution(self, title: str = "Top Columns by Uniqueness") -> ChartSpec:
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
            title=title,
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

    def get_alerts(self, *, language: str = "en") -> list[AlertSpec]:
        """Generate alerts based on profile data."""
        alerts = []
        ko = language.startswith("ko")

        for col in self.data.get("columns", []):
            name = col.get("name", "")
            null_ratio = col.get("null_ratio", 0)
            unique_ratio = col.get("unique_ratio", 0)
            is_constant = col.get("is_constant", False)

            # High null ratio alert
            if null_ratio > ALERT_THRESHOLDS.high_missing_warning_threshold:
                title = f"High Missing Values in '{name}'"
                message = f"Column has {null_ratio:.1%} missing values"
                suggestion = "Consider imputation or removal"
                if ko:
                    title = f"'{name}' 컬럼의 결측값 비율이 높습니다"
                    message = f"해당 컬럼의 결측값 비율은 {null_ratio:.1%}로 확인되었습니다"
                    suggestion = "수집 기준, 결측 대체 기준 또는 제외 기준을 검토하십시오"
                alerts.append(AlertSpec(
                    title=title,
                    message=message,
                    severity=SeverityLevel.WARNING
                    if null_ratio < ALERT_THRESHOLDS.high_missing_error_threshold
                    else SeverityLevel.ERROR,
                    column=name,
                    metric="null_ratio",
                    value=null_ratio,
                    threshold=ALERT_THRESHOLDS.high_missing_warning_threshold,
                    suggestion=suggestion,
                ))

            # Constant column alert
            if is_constant:
                title = f"Constant Column: '{name}'"
                message = "Column contains only one unique value"
                suggestion = "Consider removing if not informative"
                if ko:
                    title = f"'{name}' 컬럼이 단일 값으로 구성되어 있습니다"
                    message = "해당 컬럼은 서로 다른 값이 1개로 확인되었습니다"
                    suggestion = "분석 목적상 정보성이 낮은 컬럼인지 검토하십시오"
                alerts.append(AlertSpec(
                    title=title,
                    message=message,
                    severity=SeverityLevel.INFO,
                    column=name,
                    suggestion=suggestion,
                ))

            # Very low uniqueness (possible ID column issues)
            if unique_ratio < ALERT_THRESHOLDS.low_uniqueness_threshold and not is_constant:
                row_count = self.data.get("row_count", 1)
                if row_count > ALERT_THRESHOLDS.low_uniqueness_min_rows:
                    distinct_count = col.get("distinct_count", 0)
                    title = f"Low Cardinality in '{name}'"
                    message = f"Only {distinct_count} unique values in {row_count:,} rows"
                    if ko:
                        title = f"'{name}' 컬럼의 고유값 수가 낮습니다"
                        message = f"전체 {row_count:,}행 중 서로 다른 값이 {distinct_count:,}개로 확인되었습니다"
                    alerts.append(AlertSpec(
                        title=title,
                        message=message,
                        severity=SeverityLevel.INFO,
                        column=name,
                    ))

        # Duplicate rows alert
        dup_ratio = self.data.get("duplicate_row_ratio", 0)
        if dup_ratio > ALERT_THRESHOLDS.duplicate_warning_threshold:
            title = "Significant Duplicate Rows"
            message = f"{dup_ratio:.1%} of rows are duplicates"
            suggestion = "Consider deduplication"
            if ko:
                title = "중복 행 비율이 기준보다 높습니다"
                message = f"전체 행 중 {dup_ratio:.1%}가 중복 행으로 확인되었습니다"
                suggestion = "중복 제거 기준과 원천 데이터 적재 절차를 검토하십시오"
            alerts.append(AlertSpec(
                title=title,
                message=message,
                severity=SeverityLevel.WARNING,
                metric="duplicate_ratio",
                value=dup_ratio,
                threshold=ALERT_THRESHOLDS.duplicate_warning_threshold,
                suggestion=suggestion,
            ))

        return alerts

    def get_recommendations(self, *, language: str = "en") -> list[str]:
        """Generate recommendations based on profile data."""
        recommendations = []
        ko = language.startswith("ko")

        for col in self.data.get("columns", []):
            validators = col.get("suggested_validators", [])
            for v in validators[:2]:  # Limit per column
                name = col.get("name", "")
                if ko:
                    recommendations.append(f"'{name}' 컬럼에 '{v}' 검증 규칙 적용을 검토하십시오")
                else:
                    recommendations.append(f"Add {v} validator for column '{name}'")

        # General recommendations
        dup_ratio = self.data.get("duplicate_row_ratio", 0)
        if dup_ratio > 0.05:
            if ko:
                recommendations.append("데이터 처리 절차에 중복 행 탐지 및 처리 기준을 반영하십시오")
            else:
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
            columns = ", ".join(high_null_cols[:3])
            if ko:
                recommendations.append(f"결측값 비율이 높은 컬럼의 수집 기준과 결측 대체 기준을 점검하십시오: {columns}")
            else:
                recommendations.append(
                    f"Review data collection for columns with high missing values: {columns}"
                )

        return recommendations[:10]  # Limit recommendations


# =============================================================================
# HTML Report Builder
# =============================================================================


class HTMLReportBuilder:
    """Builder for generating static HTML reports from profile data.

    Uses ApexCharts for interactive charts in HTML reports.
    For PDF export, use export_to_pdf() which automatically uses SVG rendering.
    """

    def __init__(
        self,
        theme: ReportTheme | str = ReportTheme.LIGHT,
        config: ReportConfig | None = None,
        *,
        language: str | None = None,
        _use_svg: bool = False,
    ) -> None:
        """Initialize the report builder.

        Args:
            theme: Report theme to use
            config: Optional full configuration
            language: Optional locale override (e.g. "ko")
            _use_svg: Internal flag for PDF export (uses SVG renderer)
        """
        if config:
            self.config = config
            if language:
                self.config.language = language
            self._theme_config = self.config.custom_theme or get_theme(self.config.theme)
            if not self.config.custom_theme:
                self.config.theme = ReportTheme(self._theme_config.name)
        else:
            self._theme_config = get_theme(theme)
            self.config = ReportConfig(
                theme=ReportTheme(self._theme_config.name),
                language=language or "en",
            )

        self._use_svg = _use_svg
        # Use SVG for PDF, ApexCharts for HTML
        chart_lib = ChartLibrary.SVG if _use_svg else ChartLibrary.APEXCHARTS
        self._chart_renderer = get_chart_renderer(chart_lib)
        if _use_svg:
            self._chart_renderer.text_color = self._theme_config.colors.text_primary
            self._chart_renderer.muted_text_color = self._theme_config.colors.text_secondary
        self._report_document = ResearchReportDocument(
            self._label,
            language=self.config.language,
            theme_name=self._theme_config.name,
        )
        self._report_renderer = ReportDocumentRenderer(self._report_document)
        if self.config.language.startswith("ko") and self.config.footer_text == "Generated by Truthound":
            self.config.footer_text = self._label(
                "report.generated_by_framework",
                "Truthound 데이터 품질 프레임워크에서 생성",
            )

    def _label(self, key: str, default: str, **params: Any) -> str:
        catalog = get_catalog(self.config.language)
        return catalog.get(key, default, **params)

    def _chapter_title(self, number: int, key: str, default: str) -> str:
        return self._report_document.captions.chapter(number, key, default)

    def _appendix_title(self, letter: str, key: str, default: str) -> str:
        return self._report_document.captions.appendix(letter, key, default)

    def _table_caption(self, number: int, title: str) -> str:
        return self._report_document.captions.table(number, title)

    def _figure_caption(self, number: int, title: str) -> str:
        return self._report_document.captions.figure(number, title)

    def _labels(self) -> dict[str, str]:
        return {
            "overview.row_count": self._label("stats.rows", "Rows"),
            "overview.column_count": self._label("stats.columns", "Columns"),
            "overview.memory_bytes": self._label("stats.memory", "Memory"),
            "overview.duplicate_rows": self._label("stats.duplicates", "Duplicates"),
            "overview.duplicate_ratio": self._label("stats.duplicate_ratio", "Duplicate Ratio"),
            "overview.null_cells": self._label("stats.missing", "Missing"),
            "overview.quality_score": self._label("stats.quality", "Quality"),
            "overview.row_count.desc": self._label("stats.rows.desc", "Total number of rows"),
            "overview.column_count.desc": self._label("stats.columns.desc", "Total number of columns"),
            "overview.memory_bytes.desc": self._label("stats.memory.desc", "Estimated memory size"),
            "overview.duplicate_rows.desc": self._label("stats.duplicates.desc", "Duplicate row count"),
            "overview.null_cells.desc": self._label("stats.missing.desc", "Total null cells"),
            "overview.quality_score.desc": self._label("stats.quality.desc", "Overall data quality"),
            "column.null": self._label("stats.null", "Null"),
            "column.unique": self._label("stats.unique", "Unique"),
            "column.distinct": self._label("stats.distinct", "Distinct"),
            "column.unknown": self._label("common.unknown", "Unknown"),
            "column.min": self._label("stats.min.short", "Min"),
            "column.max": self._label("stats.max.short", "Max"),
            "column.mean": self._label("stats.mean", "Mean"),
            "column.std": self._label("stats.std.short", "Std"),
            "quality.completeness": self._label("quality.completeness", "Completeness"),
            "quality.uniqueness": self._label("quality.uniqueness", "Uniqueness"),
            "quality.validity": self._label("quality.validity", "Validity"),
            "quality.consistency": self._label("quality.consistency", "Consistency"),
            "quality.completeness.desc": self._label("quality.completeness.desc", "Measures data completeness"),
            "quality.uniqueness.desc": self._label("quality.uniqueness.desc", "Measures unique value ratio"),
            "quality.validity.desc": self._label("quality.validity.desc", "Measures data format validity"),
            "quality.consistency.desc": self._label("quality.consistency.desc", "Measures data consistency"),
            "patterns.none": self._label("patterns.none", "No patterns detected"),
            "patterns.examples": self._label("patterns.examples", "Examples"),
            "patterns.column": self._label("table.column", "Column"),
            "patterns.pattern": self._label("table.pattern", "Pattern"),
            "patterns.match_rate": self._label("table.match_rate", "Match Rate"),
            "patterns.samples": self._label("table.samples", "Samples"),
            "correlations.none": self._label("correlations.none", "No significant correlations found"),
            "recommendations.none": self._label("recommendations.none", "No specific recommendations at this time"),
            "recommendations.validators": self._label("recommendations.validators", "Suggested Validators"),
        }

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

    def build_for_pdf(
        self,
        profile: dict[str, Any] | Any,
        title: str = "Data Profile Report",
        subtitle: str = "",
        description: str = "",
    ) -> str:
        """Build a professional PDF-ready HTML report from profile data.

        This method generates HTML optimized for PDF export with:
        - Professional cover/title page with date
        - Document-style table of contents with numbering
        - Section numbering
        - Professional typography and layout

        Args:
            profile: TableProfile dict or object
            title: Report title
            subtitle: Report subtitle
            description: Report description

        Returns:
            Complete HTML document optimized for PDF export
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

        return self._render_html(spec, for_pdf=True)

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
                title=self._label("section.overview", "Overview"),
                subtitle=self._label("section.overview.subtitle", "Dataset summary and key metrics"),
                metrics=converter.get_overview_metrics(),
                charts=[
                    converter.get_type_distribution(
                        self._figure_caption(
                            1,
                            self._label("chart.data_types", "Data Types Distribution"),
                        )
                    ),
                ],
                metadata={"labels": self._labels()},
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
                title=self._label("section.quality", "Data Quality"),
                subtitle=self._label("section.quality.subtitle", "Quality metrics and assessments"),
                metrics={
                    "overall": quality_score,
                    "completeness": completeness,
                    "uniqueness": min(uniqueness, 100),
                },
                charts=[
                    converter.get_null_distribution(
                        self._figure_caption(
                            2,
                            self._label("chart.missing_values", "Top Columns by Missing Values"),
                        )
                    ),
                ],
                alerts=converter.get_alerts(language=self.config.language),
                metadata={"labels": self._labels()},
            )

        elif section_type == SectionType.COLUMNS:
            columns = converter.get_column_data()

            # Build summary table
            table = {
                "title": self._table_caption(1, self._label("table.column_summary", "Column Summary")),
                "headers": [
                    self._label("table.column", "Column"),
                    self._label("table.type", "Type"),
                    self._label("table.null_percent", "Null %"),
                    self._label("table.unique_percent", "Unique %"),
                    self._label("table.distinct", "Distinct"),
                ],
                "rows": [
                    [
                        c.get("name", ""),
                        profile_type_label(c),
                        f"{c.get('null_ratio', 0):.1%}",
                        f"{c.get('unique_ratio', 0):.1%}",
                        f"{c.get('distinct_count', 0):,}",
                    ]
                    for c in columns
                ],
            }

            return SectionSpec(
                section_type=section_type,
                title=self._label("section.columns", "Column Details"),
                subtitle=self._label("section.columns.subtitle", "{count} columns analyzed", count=len(columns)),
                tables=[table],
                metadata={"columns": columns, "labels": self._labels()},
            )

        elif section_type == SectionType.PATTERNS:
            patterns = converter.get_patterns()
            return SectionSpec(
                section_type=section_type,
                title=self._label("section.patterns", "Detected Patterns"),
                subtitle=self._label("section.patterns.subtitle", "Automatically detected data patterns"),
                metadata={"patterns": patterns, "labels": self._labels()},
                visible=len(patterns) > 0,
            )

        elif section_type == SectionType.DISTRIBUTION:
            return SectionSpec(
                section_type=section_type,
                title=self._label("section.distribution", "Value Distribution"),
                subtitle=self._label("section.distribution.subtitle", "Distribution analysis across columns"),
                charts=[
                    converter.get_uniqueness_distribution(
                        self._figure_caption(
                            3,
                            self._label("chart.uniqueness", "Top Columns by Uniqueness"),
                        )
                    ),
                ],
                metadata={"labels": self._labels()},
            )

        elif section_type == SectionType.CORRELATIONS:
            correlations = converter.get_correlations()
            return SectionSpec(
                section_type=section_type,
                title=self._label("section.correlations", "Correlations"),
                subtitle=self._label("section.correlations.subtitle", "Column relationships and correlations"),
                metadata={"correlations": correlations, "labels": self._labels()},
                visible=len(correlations) > 0,
            )

        elif section_type == SectionType.RECOMMENDATIONS:
            recommendations = converter.get_recommendations(language=self.config.language)
            return SectionSpec(
                section_type=section_type,
                title=self._label("section.recommendations", "Recommendations"),
                subtitle=self._label("section.recommendations.subtitle", "Suggested improvements and validations"),
                text_blocks=recommendations,
                metadata={"labels": self._labels()},
                visible=len(recommendations) > 0,
            )

        elif section_type == SectionType.ALERTS:
            alerts = converter.get_alerts(language=self.config.language)
            return SectionSpec(
                section_type=section_type,
                title=self._label("section.alerts", "Alerts"),
                subtitle=self._label("section.alerts.subtitle", "Data quality issues and warnings"),
                alerts=alerts,
                metadata={"labels": self._labels()},
                visible=len(alerts) > 0,
            )

        return None

    def _render_executive_summary(self, spec: ReportSpec) -> str:
        return self._report_renderer.render_executive_summary(spec)

    def _render_quality_framework(self, spec: ReportSpec) -> str:
        return self._report_renderer.render_quality_framework()

    def _render_appendices(self, spec: ReportSpec) -> str:
        return self._report_renderer.render_appendices(spec)

    def _render_html(self, spec: ReportSpec, for_pdf: bool = False) -> str:
        """Render the complete HTML document.

        Args:
            spec: Report specification
            for_pdf: Whether rendering for PDF export (uses professional document layout)

        Returns:
            Complete HTML document as string
        """
        is_dark = self._theme_config.name == ReportTheme.DARK.value
        css = get_complete_stylesheet(
            self._theme_config.to_css_vars(),
            is_dark=is_dark,
            include_apexcharts=not self._use_svg,
        )

        # Render sections, then group them into report chapters.
        rendered_sections: dict[SectionType, str] = {}
        for section_spec in spec.sections:
            renderer = get_section_renderer(section_spec.section_type)
            section_html = renderer.render(
                section_spec,
                self._chart_renderer,
                self._theme_config,
            )
            rendered_sections[section_spec.section_type] = section_html

        document_chapter_groups = self._report_document.chapter_groups(spec)
        chapters_html = self._report_renderer.render_chapters(
            document_chapter_groups,
            rendered_sections,
        )
        executive_summary_html = self._render_executive_summary(spec)
        quality_framework_html = self._render_quality_framework(spec)
        appendices_html = self._render_appendices(spec)

        # Build TOC - professional style for PDF
        toc_html = ""
        if spec.config.include_toc:
            toc_html = self._report_renderer.render_toc(document_chapter_groups, for_pdf=for_pdf)

        # Build title/cover page for PDF
        title_page_html = ""
        if for_pdf:
            logo_html = ""
            if spec.config.logo_base64:
                logo_html = f'<img src="{spec.config.logo_base64}" alt="Logo" class="cover-logo">'
            elif spec.config.logo_url:
                logo_html = f'<img src="{spec.config.logo_url}" alt="Logo" class="cover-logo">'

            # Get overview metrics for cover page
            overview_data = spec.profile_data
            row_count = overview_data.get("row_count", 0)
            column_count = overview_data.get("column_count", 0)

            title_page_html = f'''
                <section class="cover-page">
                    {logo_html}
                    <div class="cover-content">
                        <h1 class="cover-title">{html.escape(spec.metadata.title)}</h1>
                        {f'<p class="cover-subtitle">{html.escape(spec.metadata.subtitle)}</p>' if spec.metadata.subtitle else ''}
                        <div class="cover-divider"></div>
                        <div class="cover-meta">
                            <div class="cover-meta-item">
                                <span class="cover-meta-label">{self._label("report.date", "Report Date")}</span>
                                <span class="cover-meta-value">{spec.metadata.created_at.strftime(spec.config.date_format)}</span>
                            </div>
                            {f'<div class="cover-meta-item"><span class="cover-meta-label">{self._label("report.data_source", "Data Source")}</span><span class="cover-meta-value">{html.escape(spec.metadata.data_source)}</span></div>' if spec.metadata.data_source else ''}
                            <div class="cover-meta-item">
                                <span class="cover-meta-label">{self._label("report.dataset_size", "Dataset Size")}</span>
                                <span class="cover-meta-value">{self._label("report.dataset_size.value", "{rows:,} rows × {columns} columns", rows=row_count, columns=column_count)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="cover-footer">
                        <p class="cover-generator">{self._label("report.generated_by_framework", "Generated by Truthound Data Quality Framework")}</p>
                        <p class="cover-timestamp">{spec.metadata.created_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
                    </div>
                </section>
            '''

        # Build header (not for PDF - uses cover page instead)
        header_html = ""
        if spec.config.include_header and not for_pdf:
            logo_html = ""
            if spec.config.logo_base64:
                logo_html = f'<img src="{spec.config.logo_base64}" alt="Logo" class="report-logo">'
            elif spec.config.logo_url:
                logo_html = f'<img src="{spec.config.logo_url}" alt="Logo" class="report-logo">'

            meta_items = []
            if spec.config.include_timestamp:
                meta_items.append(
                    f'<span class="report-meta-item">{self._label("report.generated_at", "Generated at")}: {spec.metadata.created_at.strftime(spec.config.date_format)}</span>'
                )
            if spec.metadata.data_source:
                meta_items.append(
                    f'<span class="report-meta-item">{self._label("report.data_source", "Source")}: {html.escape(spec.metadata.data_source)}</span>'
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
            if for_pdf:
                footer_html = f'''
                    <footer class="report-footer-professional">
                        <div class="footer-line"></div>
                        <p class="footer-text">{html.escape(spec.config.footer_text)}</p>
                        <p class="footer-disclaimer">{self._label("report.disclaimer", "This report was automatically generated and should be reviewed for accuracy.")}</p>
                    </footer>
                '''
            else:
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

        # Add PDF-specific CSS
        pdf_css = ""
        if for_pdf:
            pdf_css = self._get_pdf_professional_css()

        # Build complete HTML
        html_content = f'''<!DOCTYPE html>
<html lang="{spec.config.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(spec.metadata.title)}</title>
    <meta name="description" content="{html.escape(spec.metadata.description or self._label("report.profile_description", "Data Profile Report"))}">
    <meta name="generator" content="Truthound">
    <style>
{css}
{pdf_css}
{spec.config.custom_css}
    </style>
    {scripts_html}
</head>
<body{' class="pdf-document"' if for_pdf else ''}>
    <div class="report-container">
        {title_page_html}
        {header_html}
        {toc_html}
        <main class="report-main">
            {executive_summary_html}
            {chapters_html}
            {quality_framework_html}
            {appendices_html}
        </main>
        {footer_html}
    </div>
    <script>
{spec.config.custom_js}
    </script>
</body>
</html>'''

        return html_content

    def _get_pdf_professional_css(self) -> str:
        """Get CSS for Korean public/research PDF document styling."""
        return '''
/* =============================================================================
   Korean Public/Research PDF Document Styling
   ============================================================================= */

/* Cover Page */
.cover-page {
    min-height: 257mm;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 24mm 18mm;
    page-break-after: always;
    background: var(--color-surface);
    border: 1.2pt solid var(--color-border);
}

.cover-logo {
    max-height: 24mm;
    margin-bottom: 18mm;
}

.cover-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.cover-title {
    font-size: 20pt;
    font-weight: 700;
    color: var(--color-primary);
    margin-bottom: 5mm;
    letter-spacing: 0;
}

.cover-subtitle {
    font-size: 11pt;
    color: var(--color-text-secondary);
    margin-bottom: 12mm;
    font-weight: 400;
}

.cover-divider {
    width: 42mm;
    height: 0;
    border-top: 2pt solid var(--color-primary);
    border-bottom: 0.6pt solid var(--color-secondary);
    margin: 12mm 0;
}

.cover-meta {
    display: flex;
    flex-direction: column;
    gap: 4mm;
    margin-top: 8mm;
}

.cover-meta-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.cover-meta-label {
    font-size: 8.5pt;
    color: var(--color-text-secondary);
    font-weight: 600;
}

.cover-meta-value {
    font-size: 10pt;
    color: var(--color-text-primary);
    font-weight: 500;
}

.cover-footer {
    margin-top: auto;
    padding-top: 2rem;
}

.cover-generator {
    font-size: 9pt;
    color: var(--color-primary);
    font-weight: 500;
    margin-bottom: 2mm;
}

.cover-timestamp {
    font-size: 8pt;
    color: var(--color-text-secondary);
}

/* Professional Table of Contents */
.report-toc-professional {
    page-break-after: always;
    padding: 12mm 4mm;
    border: 1pt solid var(--color-border);
}

.toc-title-professional {
    font-size: 14pt;
    font-weight: 700;
    color: var(--color-primary);
    margin-bottom: 8mm;
    padding-bottom: 3mm;
    border-bottom: 1.5pt solid var(--color-primary);
}

.toc-list-professional {
    list-style: none;
    margin: 0;
    padding: 0;
}

.toc-row-professional {
    margin: 0;
    padding: 0;
}

.toc-link-professional {
    display: flex;
    align-items: baseline;
    gap: 2mm;
    min-width: 0;
    padding: 3mm 0;
    color: var(--color-text-primary);
    text-decoration: none;
    font-weight: 500;
    line-height: 1.45;
}

.toc-number {
    flex: 0 0 8mm;
    font-weight: 600;
    color: var(--color-primary);
    white-space: nowrap;
}

.toc-entry {
    flex: 0 1 auto;
    min-width: 0;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: keep-all;
}

.toc-leader {
    flex: 1 1 20mm;
    min-width: 14mm;
    height: 0;
    border-bottom: 0.5pt dotted var(--color-border);
    transform: translateY(-1mm);
}

.toc-page {
    flex: 0 0 9mm;
    text-align: right;
    color: var(--color-text-secondary);
    white-space: nowrap;
}

.toc-page::after {
    content: target-counter(attr(data-target), page);
}

.toc-link-professional:hover .toc-entry {
    color: var(--color-primary);
}

/* Professional Footer */
.report-footer-professional {
    margin-top: 14mm;
    padding-top: 5mm;
}

.footer-line {
    height: 0;
    border-top: 1pt solid var(--color-primary);
    margin-bottom: 4mm;
}

.footer-text {
    font-size: 9pt;
    color: var(--color-text-primary);
    font-weight: 500;
    margin-bottom: 1.5mm;
}

.footer-disclaimer {
    font-size: 8pt;
    color: var(--color-text-secondary);
    font-style: italic;
}

/* PDF Document Body */
body.pdf-document {
    font-size: var(--font-size-base);
    line-height: var(--line-height-normal);
}

body.pdf-document .report-container {
    width: auto;
    max-width: none;
    min-height: auto;
    padding: 0;
    box-shadow: none;
    border: 0;
}

body.pdf-document .section-header {
    margin-top: 2rem;
    margin-bottom: 1.5rem;
}

body.pdf-document .section-title {
    font-size: 13pt;
    color: var(--color-primary);
    border-bottom: 1.5pt solid var(--color-primary);
    padding-bottom: 2mm;
}

body.pdf-document .section-subtitle {
    font-size: 0.9375rem;
    margin-top: 0.5rem;
}

body.pdf-document .executive-summary {
    page-break-after: always;
}

body.pdf-document .executive-summary-grid {
    grid-template-columns: 1fr 1fr;
}

body.pdf-document .report-chapter {
    page-break-before: always;
}

body.pdf-document .chapter-title {
    font-size: 15pt;
}

body.pdf-document .quality-framework {
    page-break-before: always;
}

body.pdf-document .report-appendix {
    page-break-before: always;
}

body.pdf-document .appendix-title {
    font-size: 13pt;
}

body.pdf-document .report-section {
    page-break-inside: auto;
    break-inside: auto;
    margin-bottom: 2rem;
}

body.pdf-document .chart-container {
    page-break-inside: avoid;
    margin: 1rem 0;
}

body.pdf-document .data-table {
    font-size: var(--font-size-sm);
}

body.pdf-document .data-table thead {
    display: table-header-group;
}

body.pdf-document .data-table tr,
body.pdf-document .auditability-block,
body.pdf-document .executive-summary-item {
    page-break-inside: avoid;
    break-inside: avoid;
}

body.pdf-document .data-table th {
    background-color: var(--color-primary);
    color: #ffffff;
    font-weight: 600;
}

body.pdf-document .metric-card {
    box-shadow: none;
    border: 0.5pt solid var(--color-border);
}

body.pdf-document .column-card {
    box-shadow: none;
    border: 0.5pt solid var(--color-border);
    page-break-inside: avoid;
}

/* Ensure charts don't break across pages */
body.pdf-document .svg-chart {
    page-break-inside: avoid;
}

/* Quality score styling for PDF */
body.pdf-document .quality-score-card {
    box-shadow: none;
    border: 0.5pt solid var(--color-border);
}

/* Hide interactive elements in PDF */
body.pdf-document .download-button,
body.pdf-document .no-print {
    display: none !important;
}
'''

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
    theme: ReportTheme | str = ReportTheme.LIGHT,
    output_path: str | Path | None = None,
    language: str = "en",
) -> str:
    """Generate an HTML report from profile data.

    Uses ApexCharts for interactive chart rendering.

    Args:
        profile: TableProfile dict or object
        title: Report title
        subtitle: Report subtitle
        theme: Report theme
        output_path: Optional path to save the report
        language: Report locale

    Returns:
        HTML content as string
    """
    builder = HTMLReportBuilder(theme=theme, language=language)
    html_content = builder.build(profile, title=title, subtitle=subtitle)

    if output_path:
        builder.save(html_content, output_path)

    return html_content


def generate_report_from_file(
    profile_path: str | Path,
    output_path: str | Path | None = None,
    title: str = "Data Profile Report",
    theme: ReportTheme | str = ReportTheme.LIGHT,
    language: str = "en",
) -> str:
    """Generate an HTML report from a profile JSON file.

    Args:
        profile_path: Path to profile JSON file
        output_path: Optional path to save the report
        title: Report title
        theme: Report theme
        language: Report locale

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
        language=language,
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


def _get_weasyprint_install_instructions() -> str:
    """Get OS-specific installation instructions for WeasyPrint dependencies."""
    import platform

    system = platform.system().lower()

    instructions = [
        "PDF export requires WeasyPrint and system dependencies.",
        "",
        "Step 1: Install system dependencies",
    ]

    if system == "darwin":  # macOS
        instructions.extend([
            "  macOS (Homebrew):",
            "    brew install pango cairo gdk-pixbuf libffi",
            "",
        ])
    elif system == "linux":
        instructions.extend([
            "  Ubuntu/Debian:",
            "    sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 \\",
            "      libgdk-pixbuf2.0-0 libffi-dev shared-mime-info",
            "",
            "  Fedora/RHEL:",
            "    sudo dnf install pango gdk-pixbuf2 libffi-devel",
            "",
            "  Alpine:",
            "    apk add pango gdk-pixbuf libffi-dev",
            "",
        ])
    elif system == "windows":
        instructions.extend([
            "  Windows:",
            "    Install GTK3 runtime from:",
            "    https://github.com/nickvidal/weasyprint/releases/download/v62.3/weasyprint-62.3-gtk3-bundled.zip",
            "    Or use: pip install weasyprint[gtk3]",
            "",
        ])
    else:
        instructions.extend([
            "  See WeasyPrint documentation for your OS:",
            "  https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation",
            "",
        ])

    instructions.extend([
        "Step 2: Install Python package",
        "  pip install truthound[pdf]",
        "",
        "For detailed instructions, see:",
        "  https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation",
    ])

    return "\n".join(instructions)


class WeasyPrintDependencyError(ImportError):
    """Raised when WeasyPrint or its system dependencies are missing."""

    def __init__(self, original_error: Exception | None = None):
        self.original_error = original_error
        message = _get_weasyprint_install_instructions()
        if original_error:
            message = f"{original_error}\n\n{message}"
        super().__init__(message)


def export_to_pdf(
    profile: dict[str, Any] | Any,
    output_path: str | Path,
    title: str = "Data Profile Report",
    subtitle: str = "",
    theme: ReportTheme | str = ReportTheme.LIGHT,
    language: str = "en",
) -> Path:
    """Export report to PDF with professional document formatting.

    Uses SVG rendering for charts (compatible with PDF generation).
    Includes:
    - Professional cover/title page with date
    - Document-style table of contents
    - Numbered sections
    - Value labels on all charts
    - Professional typography and layout

    Requires:
        - System dependencies: pango, cairo, gdk-pixbuf (see error message for OS-specific commands)
        - Python package: pip install truthound[pdf]

    Args:
        profile: TableProfile dict or object
        output_path: Output PDF file path
        title: Report title
        subtitle: Report subtitle
        theme: Report theme
        language: Report locale

    Returns:
        Path to PDF file

    Raises:
        WeasyPrintDependencyError: If WeasyPrint or system dependencies are missing
    """
    try:
        from weasyprint import HTML
    except ImportError as e:
        raise WeasyPrintDependencyError(original_error=e)

    output_path = Path(output_path)

    # Use SVG renderer for PDF (no JavaScript)
    builder = HTMLReportBuilder(theme=theme, language=language, _use_svg=True)

    # Build HTML with professional PDF formatting
    html_content = builder.build_for_pdf(profile, title=title, subtitle=subtitle)

    # Convert to PDF - catch system library errors
    try:
        HTML(string=html_content).write_pdf(output_path)
    except OSError as e:
        # Catch errors like "cannot load library 'libpango-1.0-0'"
        if "cannot load library" in str(e) or "libpango" in str(e) or "cairo" in str(e):
            raise WeasyPrintDependencyError(original_error=e)
        raise

    return output_path
