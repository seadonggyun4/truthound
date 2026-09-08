"""Message catalog for report internationalization.

This module provides translation catalogs for report UI elements,
supporting 15+ languages with extensible architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass
class ReportCatalog:
    """Translation catalog for report messages.

    Provides access to translated strings for report UI elements.

    Attributes:
        locale: Locale code (e.g., "en", "ko", "ja").
        messages: Dictionary of message key to translated string.
        metadata: Additional catalog metadata.
    """

    locale: str
    messages: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(
        self,
        key: str,
        default: str | None = None,
        **params,
    ) -> str:
        """Get a translated message.

        Args:
            key: Message key (e.g., "report.title").
            default: Default value if not found.
            **params: Parameters for string formatting.

        Returns:
            Translated message or default.
        """
        template = self.messages.get(key, default or key)

        if params:
            try:
                return template.format(**params)
            except KeyError:
                return template

        return template

    def __getitem__(self, key: str) -> str:
        """Get message by key."""
        return self.messages.get(key, key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.messages

    def __len__(self) -> int:
        """Return message count."""
        return len(self.messages)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self.messages)

    def keys(self) -> list[str]:
        """Return all message keys."""
        return list(self.messages.keys())

    def merge(self, other: ReportCatalog) -> ReportCatalog:
        """Merge with another catalog.

        Args:
            other: Catalog to merge (takes precedence).

        Returns:
            New merged catalog.
        """
        return ReportCatalog(
            locale=self.locale,
            messages={**self.messages, **other.messages},
            metadata={**self.metadata, **other.metadata},
        )

    def extend(self, messages: dict[str, str]) -> ReportCatalog:
        """Extend with additional messages.

        Args:
            messages: Messages to add.

        Returns:
            New extended catalog.
        """
        return ReportCatalog(
            locale=self.locale,
            messages={**self.messages, **messages},
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "locale": self.locale,
            "messages": self.messages.copy(),
            "metadata": self.metadata.copy(),
        }

    def to_json(self, path: Path) -> None:
        """Save to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(
        cls,
        locale: str,
        messages: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> ReportCatalog:
        """Create from dictionary."""
        return cls(
            locale=locale,
            messages=messages.copy(),
            metadata=metadata or {},
        )

    @classmethod
    def from_json(cls, path: Path) -> ReportCatalog:
        """Load from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            locale=data.get("locale", "en"),
            messages=data.get("messages", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def builder(cls, locale: str) -> CatalogBuilder:
        """Create a catalog builder."""
        return CatalogBuilder(locale)


class CatalogBuilder:
    """Fluent builder for ReportCatalog."""

    def __init__(self, locale: str) -> None:
        self._locale = locale
        self._messages: dict[str, str] = {}
        self._metadata: dict[str, Any] = {}

    def add(self, key: str, value: str) -> CatalogBuilder:
        """Add a message."""
        self._messages[key] = value
        return self

    def add_report_section(
        self,
        title: str,
        subtitle: str,
        summary: str,
        details: str,
    ) -> CatalogBuilder:
        """Add report section messages."""
        self._messages["report.title"] = title
        self._messages["report.subtitle"] = subtitle
        self._messages["report.summary"] = summary
        self._messages["report.details"] = details
        return self

    def add_quality_labels(
        self,
        excellent: str,
        good: str,
        fair: str,
        poor: str,
        critical: str,
    ) -> CatalogBuilder:
        """Add quality grade labels."""
        self._messages["quality.excellent"] = excellent
        self._messages["quality.good"] = good
        self._messages["quality.fair"] = fair
        self._messages["quality.poor"] = poor
        self._messages["quality.critical"] = critical
        return self

    def add_section_titles(
        self,
        overview: str,
        columns: str,
        alerts: str,
        recommendations: str,
        statistics: str,
    ) -> CatalogBuilder:
        """Add section title messages."""
        self._messages["section.overview"] = overview
        self._messages["section.columns"] = columns
        self._messages["section.alerts"] = alerts
        self._messages["section.recommendations"] = recommendations
        self._messages["section.statistics"] = statistics
        return self

    def add_alert_labels(
        self,
        critical: str,
        warning: str,
        info: str,
    ) -> CatalogBuilder:
        """Add alert severity labels."""
        self._messages["alert.critical"] = critical
        self._messages["alert.warning"] = warning
        self._messages["alert.info"] = info
        return self

    def add_stats_labels(
        self,
        row_count: str,
        column_count: str,
        null_ratio: str,
        unique_ratio: str,
        duplicate_count: str,
    ) -> CatalogBuilder:
        """Add statistics labels."""
        self._messages["stats.row_count"] = row_count
        self._messages["stats.column_count"] = column_count
        self._messages["stats.null_ratio"] = null_ratio
        self._messages["stats.unique_ratio"] = unique_ratio
        self._messages["stats.duplicate_count"] = duplicate_count
        return self

    def with_metadata(self, **metadata: Any) -> CatalogBuilder:
        """Add metadata."""
        self._metadata.update(metadata)
        return self

    def build(self) -> ReportCatalog:
        """Build the catalog."""
        return ReportCatalog(
            locale=self._locale,
            messages=self._messages.copy(),
            metadata=self._metadata.copy(),
        )


class CatalogRegistry:
    """Registry for managing translation catalogs."""

    def __init__(self, lazy_load: bool = True) -> None:
        self._catalogs: dict[str, ReportCatalog] = {}
        self._fallback_locale = "en"
        self._lazy_load = lazy_load
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of default catalogs."""
        if self._initialized:
            return
        self._initialized = True
        if self._lazy_load:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default catalogs."""
        self._catalogs["en"] = _create_english_catalog()
        self._catalogs["ko"] = _create_korean_catalog()
        self._catalogs["ja"] = _create_japanese_catalog()
        self._catalogs["zh"] = _create_chinese_catalog()
        self._catalogs["de"] = _create_german_catalog()
        self._catalogs["fr"] = _create_french_catalog()
        self._catalogs["es"] = _create_spanish_catalog()
        self._catalogs["pt"] = _create_portuguese_catalog()
        self._catalogs["it"] = _create_italian_catalog()
        self._catalogs["ru"] = _create_russian_catalog()
        self._catalogs["ar"] = _create_arabic_catalog()
        self._catalogs["th"] = _create_thai_catalog()
        self._catalogs["vi"] = _create_vietnamese_catalog()
        self._catalogs["id"] = _create_indonesian_catalog()
        self._catalogs["tr"] = _create_turkish_catalog()

    def get(self, locale: str) -> ReportCatalog:
        """Get catalog for locale.

        Args:
            locale: Locale code.

        Returns:
            Catalog for locale or fallback.
        """
        self._ensure_initialized()

        # Try exact match
        if locale in self._catalogs:
            return self._catalogs[locale]

        # Try language only (e.g., "en_US" -> "en")
        lang = locale.split("_")[0].split("-")[0]
        if lang in self._catalogs:
            return self._catalogs[lang]

        # Fallback
        return self._catalogs[self._fallback_locale]

    def register(self, catalog: ReportCatalog) -> None:
        """Register a catalog."""
        self._ensure_initialized()
        self._catalogs[catalog.locale] = catalog

    def unregister(self, locale: str) -> None:
        """Unregister a catalog."""
        self._ensure_initialized()
        self._catalogs.pop(locale, None)

    def list_locales(self) -> list[str]:
        """List registered locales."""
        self._ensure_initialized()
        return list(self._catalogs.keys())

    def set_fallback(self, locale: str) -> None:
        """Set fallback locale."""
        self._fallback_locale = locale


# Global registry
_registry = CatalogRegistry()


def get_catalog(locale: str) -> ReportCatalog:
    """Get catalog for locale."""
    return _registry.get(locale)


def register_catalog(catalog: ReportCatalog) -> None:
    """Register a catalog."""
    _registry.register(catalog)


def get_supported_locales() -> list[str]:
    """Get supported locale codes."""
    return _registry.list_locales()


def create_catalog_builder(locale: str) -> CatalogBuilder:
    """Create a catalog builder."""
    return CatalogBuilder(locale)


# ============================================
# Built-in Catalogs
# ============================================


def _create_english_catalog() -> ReportCatalog:
    """Create English catalog."""
    return ReportCatalog.from_dict(
        "en",
        {
            # Report
            "report.title": "Data Quality Report",
            "validation.execution_title": "Execution issues detected",
            "validation.execution_message": "{count} execution issue(s) occurred during validation.",
            "validation.quality_title": "Validation issues detected",
            "validation.quality_message": "{count} validation issue(s) require review.",
            "validation.no_data": "No data available.",
            "report.subtitle": "Automated Data Profiling and Validation",
            "report.summary": "Summary",
            "report.details": "Details",
            "report.generated_at": "Generated at",
            "report.generated_by": "Generated by",
            "report.generated_by_framework": "Generated by Truthound Data Quality Framework",
            "report.toc": "Table of Contents",
            "report.contents": "Contents",
            "report.date": "Report Date",
            "report.data_source": "Data Source",
            "report.dataset_size": "Dataset Size",
            "report.dataset_size.value": "{rows:,} rows × {columns} columns",
            "report.disclaimer": "This report was automatically generated and should be reviewed for accuracy.",
            "report.profile_description": "Data Profile Report",
            "report.executive_summary": "Executive Summary",
            "report.quality_framework": "Quality Framework Mapping",
            "report.table_caption": "Table {number}. {title}",
            "report.figure_caption": "Figure {number}. {title}",
            "chapter.analysis_overview": "Analysis Overview",
            "chapter.quality_results": "Data Quality Diagnostic Results",
            "chapter.column_diagnostics": "Column-Level Diagnostics",
            "chapter.risk_factors": "Detected Patterns and Risk Factors",
            "chapter.recommendations": "Recommendations",
            "chapter.analysis_overview.lead": "This chapter summarizes the analyzed data scale and core profile metrics. Column-level structure is summarized in {column_table}, and the observed data type mix is shown in {type_figure}.",
            "chapter.quality_results.lead": "This chapter interprets profile-driven quality signals. Missing-value patterns are shown in {missing_figure}, and the quality dimension mapping in {quality_table} distinguishes measured evidence from dimensions that require additional input.",
            "chapter.column_diagnostics.lead": "This chapter reviews each profiled column while preserving source column names and technical type identifiers. The full audit-oriented column profile is provided in {profile_appendix}.",
            "chapter.risk_factors.lead": "This chapter groups detected patterns, value distribution signals, correlations, and alerts as review evidence. Uniqueness distribution is shown in {uniqueness_figure}; business meaning should be confirmed with domain rules before final decisions.",
            "chapter.recommendations.lead": "This chapter converts profile evidence into priority review actions. Metric definitions and reproducibility metadata are available in {metrics_appendix} and {repro_appendix}.",
            "summary.subtitle": "Purpose, findings, risks, actions, and limitations",
            "summary.purpose": "Purpose",
            "summary.purpose.text": "Assess the dataset profile and summarize actionable data quality risks.",
            "summary.data_overview": "Data Overview",
            "summary.data_overview.text": "The analyzed dataset contains {rows:,} rows and {columns} columns.",
            "summary.key_findings": "Key Findings",
            "summary.key_findings.text": "The overall quality score is {score:.1f}% ({grade}); duplicate rows account for {duplicate_ratio:.2%}.",
            "summary.risks": "Risks",
            "summary.risk.missing": "{count} columns require missing-value review.",
            "summary.risk.duplicates": "Duplicate rows account for {duplicate_ratio:.2%}, which requires source loading and deduplication review.",
            "summary.risk.low_uniqueness": "{count} columns show low uniqueness and may require code-list or constant-field review.",
            "summary.risk.none": "No high-priority structural risk was detected from the available profile.",
            "summary.priority_actions": "Priority Actions",
            "summary.action.missing": "Review collection and imputation rules for high-missing columns.",
            "summary.action.duplicates": "Review upstream keys, ingestion retries, and deduplication rules before downstream use.",
            "summary.action.low_uniqueness": "Confirm whether low-uniqueness columns are intentional categories, constants, or collection defects.",
            "summary.action.validators": "Maintain suggested validation rules for monitored columns.",
            "summary.limitations": "Limitations",
            "summary.limitations.text": "This report is based on the supplied profile metadata and does not prove business accuracy without domain validation.",
            "quality.framework.subtitle": "Interpretation layer based on recognized data quality dimensions",
            "quality.framework.note": "The mapping below explains how available Truthound profile signals relate to common data quality dimensions. It does not change metric calculations.",
            "quality.framework.table": "Quality dimension mapping",
            "quality.framework.caption": "Reference basis: Eurostat ESS/QAF, ISO 8000, Wang and Strong, Batini and Scannapieco, and DAMA-DMBOK quality dimensions.",
            "quality.dimension": "Dimension",
            "quality.metric": "Evidence",
            "quality.measurement_status": "Status",
            "quality.note": "Note",
            "quality.status.measured": "Measured",
            "quality.status.partial": "Partially measured",
            "quality.status.not_measured": "Input required",
            "quality.dimension.completeness": "Completeness",
            "quality.dimension.completeness.metric": "Missing value ratio",
            "quality.dimension.completeness.note": "Estimated from null ratios across profiled columns.",
            "quality.dimension.completeness.limitation": "Completeness describes observed null coverage; it does not prove semantic correctness.",
            "quality.dimension.uniqueness": "Uniqueness",
            "quality.dimension.uniqueness.metric": "Distinct and unique value ratios",
            "quality.dimension.uniqueness.note": "Estimated from distinct counts and unique ratios.",
            "quality.dimension.uniqueness.limitation": "Uniqueness can indicate duplicate risk but does not determine whether repeated values are valid.",
            "quality.dimension.validity": "Validity",
            "quality.dimension.validity.metric": "Detected pattern and inferred type signals",
            "quality.dimension.validity.note": "Requires explicit business rules for complete validation.",
            "quality.dimension.validity.limitation": "Pattern and type signals are evidence, not a substitute for complete domain validation rules.",
            "quality.dimension.accuracy": "Accuracy",
            "quality.dimension.accuracy.metric": "Domain truth comparison",
            "quality.dimension.accuracy.note": "Cannot be proven from a profile alone without reference data.",
            "quality.dimension.accuracy.limitation": "Accuracy requires authoritative reference data or domain review outside this profile report.",
            "quality.dimension.timeliness": "Timeliness",
            "quality.dimension.timeliness.metric": "Freshness and update SLA",
            "quality.dimension.timeliness.note": "Requires ingestion timestamps or freshness policy.",
            "quality.dimension.timeliness.limitation": "Timeliness requires source freshness metadata or an update policy not inferred from values alone.",
            "appendix.metrics": "Metric Definitions and Formulae",
            "appendix.reproducibility": "Execution Environment and Reproducibility",
            "appendix.full_profile": "Full Column Profile",
            "appendix.quality_coverage": "Quality Dimension Coverage and Limitations",
            "appendix.methodology": "Diagnostic Criteria and Thresholds",
            "appendix.metrics.lead": "This appendix defines the metrics used in the body of the report. It documents interpretation boundaries and does not redefine Truthound profile calculations.",
            "appendix.reproducibility.lead": "This appendix records execution metadata needed to reproduce the report artifact without exposing raw input data.",
            "appendix.full_profile.lead": "This appendix provides the complete column profile used by the body analysis while preserving source column names and technical type identifiers.",
            "appendix.quality_coverage.lead": "This appendix supports audit review by separating measured profile evidence from quality dimensions that require business rules, reference data, or freshness metadata.",
            "appendix.methodology.lead": "This appendix records the threshold policy used by generated alerts and recommendations so that reviewers can reproduce the report interpretation.",
            "appendix.metrics.table": "Metric definitions",
            "appendix.full_profile.table": "Full column profile",
            "appendix.quality_coverage.table": "Quality dimension coverage",
            "appendix.methodology.table": "Diagnostic criteria and thresholds",
            "appendix.quality_coverage.note": "This appendix separates profiled evidence from dimensions that require business rules, reference data, or source freshness metadata.",
            "appendix.metric": "Metric",
            "appendix.definition": "Definition",
            "appendix.limitation": "Limitation",
            "methodology.criterion": "Criterion",
            "methodology.threshold": "Threshold",
            "methodology.interpretation": "Interpretation",
            "methodology.missing_review": "Missing-value review",
            "methodology.missing_review.note": "Columns at or above this threshold can drive executive-summary review guidance without changing the underlying profile calculation.",
            "methodology.high_missing": "High missing values",
            "methodology.high_missing.note": "Columns above the warning threshold are flagged for missing-value review; columns at or above the error threshold are treated as higher severity.",
            "methodology.constant_column": "Constant column",
            "methodology.constant_column.threshold": "1 distinct value",
            "methodology.constant_column.note": "Columns marked constant by the supplied profile are flagged as potentially low-information fields.",
            "methodology.low_uniqueness": "Low uniqueness",
            "methodology.low_uniqueness.threshold": "< {uniqueness}; rows > {rows}",
            "methodology.low_uniqueness.note": "Columns below the uniqueness threshold are flagged only when the profiled row count is above the minimum row count.",
            "methodology.duplicate_rows": "Duplicate rows",
            "methodology.duplicate_rows.note": "Datasets above the duplicate-row threshold are flagged for source loading and deduplication review.",
            "methodology.quality_score_limit": "Quality score limitation",
            "methodology.quality_score_limit.threshold": "Profile-derived",
            "methodology.quality_score_limit.note": "The quality score summarizes profile metadata and does not prove business accuracy, timeliness, or fitness for use.",
            "appendix.truthound_version": "Truthound version",
            "appendix.python_version": "Python version",
            "appendix.platform": "Platform",
            "appendix.theme": "Theme",
            "appendix.language": "Language",
            "appendix.generated_at": "Generated at",
            "appendix.source_label": "Source label",
            "appendix.input_fingerprint": "Input fingerprint",
            "appendix.formula.rows": "Total number of profiled rows.",
            "appendix.formula.columns": "Total number of profiled columns.",
            "appendix.formula.missing": "Total null cells observed in profiled columns.",
            "appendix.formula.duplicates": "Rows identified as duplicates by the supplied profile.",
            "appendix.formula.duplicate_ratio": "Duplicate rows divided by total rows.",
            "appendix.formula.quality_score": "Existing Truthound profile quality score; this report does not alter its calculation.",
            # Sections
            "section.overview": "Overview",
            "section.overview.subtitle": "Dataset summary and key metrics",
            "section.quality": "Data Quality",
            "section.quality.subtitle": "Quality metrics and assessments",
            "section.columns": "Column Details",
            "section.columns.subtitle": "{count} columns analyzed",
            "section.alerts": "Alerts",
            "section.alerts.subtitle": "Data quality issues and warnings",
            "section.recommendations": "Recommendations",
            "section.recommendations.subtitle": "Suggested improvements and validations",
            "section.statistics": "Statistics",
            "section.distribution": "Value Distribution",
            "section.distribution.subtitle": "Distribution analysis across columns",
            "section.patterns": "Detected Patterns",
            "section.patterns.subtitle": "Automatically detected data patterns",
            "section.correlations": "Correlations",
            "section.correlations.subtitle": "Column relationships and correlations",
            # Quality
            "quality.score": "Quality Score",
            "quality.completeness": "Completeness",
            "quality.uniqueness": "Uniqueness",
            "quality.validity": "Validity",
            "quality.consistency": "Consistency",
            "quality.completeness.desc": "Measures data completeness",
            "quality.uniqueness.desc": "Measures unique value ratio",
            "quality.validity.desc": "Measures data format validity",
            "quality.consistency.desc": "Measures data consistency",
            "quality.grade": "Grade",
            "quality.excellent": "Excellent",
            "quality.good": "Good",
            "quality.fair": "Fair",
            "quality.poor": "Poor",
            "quality.critical": "Critical",
            # Alerts
            "alert.critical": "Critical",
            "alert.warning": "Warning",
            "alert.info": "Information",
            "alert.count": "{count} alerts found",
            # Stats
            "stats.row_count": "Row Count",
            "stats.rows": "Rows",
            "stats.columns": "Columns",
            "stats.memory": "Memory",
            "stats.duplicates": "Duplicates",
            "stats.duplicate_ratio": "Duplicate Ratio",
            "stats.missing": "Missing",
            "stats.quality": "Quality",
            "stats.rows.desc": "Total number of rows",
            "stats.columns.desc": "Total number of columns",
            "stats.memory.desc": "Estimated memory size",
            "stats.duplicates.desc": "Duplicate row count",
            "stats.missing.desc": "Total null cells",
            "stats.quality.desc": "Overall data quality",
            "stats.null": "Null",
            "stats.unique": "Unique",
            "stats.distinct": "Distinct",
            "stats.column_count": "Column Count",
            "stats.null_ratio": "Null Ratio",
            "stats.unique_ratio": "Unique Ratio",
            "stats.duplicate_count": "Duplicate Count",
            "stats.mean": "Mean",
            "stats.median": "Median",
            "stats.std_dev": "Standard Deviation",
            "stats.std.short": "Std",
            "stats.min": "Minimum",
            "stats.min.short": "Min",
            "stats.max": "Maximum",
            "stats.max.short": "Max",
            # Tables and charts
            "table.column_summary": "Column Summary",
            "table.column": "Column",
            "table.type": "Type",
            "table.null_percent": "Null %",
            "table.unique_percent": "Unique %",
            "table.distinct": "Distinct",
            "table.pattern": "Pattern",
            "table.match_rate": "Match Rate",
            "table.samples": "Samples",
            "chart.data_types": "Data Types Distribution",
            "chart.missing_values": "Top Columns by Missing Values",
            "chart.uniqueness": "Top Columns by Uniqueness",
            "patterns.none": "No patterns detected",
            "patterns.examples": "Examples",
            "correlations.none": "No significant correlations found",
            "recommendations.none": "No specific recommendations at this time",
            "recommendations.validators": "Suggested Validators",
            # Actions
            "action.export": "Export",
            "action.download": "Download",
            "action.share": "Share",
            "action.print": "Print",
            # Common
            "common.yes": "Yes",
            "common.no": "No",
            "common.na": "N/A",
            "common.total": "Total",
            "common.average": "Average",
            "common.percentage": "Percentage",
            "common.unknown": "Unknown",
        },
        metadata={"name": "English", "native": "English", "direction": "ltr"},
    )


def _create_korean_catalog() -> ReportCatalog:
    """Create Korean catalog."""
    return ReportCatalog.from_dict(
        "ko",
        {
            "report.title": "데이터 품질 보고서",
            "validation.execution_title": "실행 오류가 발견되었습니다",
            "validation.execution_message": "검증 중 실행 오류 {count}건이 발생했습니다.",
            "validation.quality_title": "품질 문제가 발견되었습니다",
            "validation.quality_message": "품질 문제 {count}건을 검토해야 합니다.",
            "validation.no_data": "데이터가 없습니다.",
            "validation.label.overview": "개요",
            "validation.label.checks": "검사",
            "validation.label.issues": "품질 문제",
            "validation.label.execution_issues": "실행 오류",
            "validation.label.metadata": "메타데이터",
            "validation.label.status": "상태",
            "validation.label.rows": "행 수",
            "validation.label.columns": "열 수",
            "validation.label.pass_rate": "통과율",
            "validation.label.check": "검사",
            "validation.label.category": "분류",
            "validation.label.issue_count": "문제 수",
            "validation.label.top_severity": "최고 심각도",
            "validation.label.validator": "검증기",
            "validation.label.column": "열",
            "validation.label.issue_type": "문제 유형",
            "validation.label.count": "건수",
            "validation.label.severity": "심각도",
            "validation.label.message": "메시지",
            "validation.label.exception_type": "예외 유형",
            "validation.label.failure_category": "실패 분류",
            "validation.label.retries": "재시도 횟수",
            "validation.label.run_id": "실행 ID",
            "validation.label.run_time": "실행 시각",
            "validation.label.suite": "검증 묶음",
            "validation.label.source": "데이터 소스",
            "validation.label.execution_mode": "실행 방식",
            "validation.label.planned_execution_mode": "계획된 실행 방식",
            "validation.label.result_format": "결과 형식",
            "validation.label.runtime_environment": "실행 환경",
            "validation.label.critical": "심각",
            "validation.label.high": "높음",
            "validation.label.medium": "중간",
            "validation.label.low": "낮음",
            "report.subtitle": "자동화된 데이터 프로파일링 및 검증",
            "report.summary": "요약",
            "report.details": "상세",
            "report.generated_at": "생성 일시",
            "report.generated_by": "생성자",
            "report.generated_by_framework": "Truthound 데이터 품질 프레임워크에서 생성",
            "report.toc": "목차",
            "report.contents": "목차",
            "report.date": "보고서 작성일",
            "report.data_source": "데이터 출처",
            "report.dataset_size": "데이터 규모",
            "report.dataset_size.value": "{rows:,}행 × {columns}열",
            "report.disclaimer": "본 보고서는 자동 생성되었으며, 최종 활용 전 담당자 검토가 필요합니다.",
            "report.profile_description": "데이터 프로파일 보고서",
            "report.executive_summary": "요약문",
            "report.quality_framework": "품질 프레임워크 매핑",
            "report.table_caption": "[표 {number}] {title}",
            "report.figure_caption": "[그림 {number}] {title}",
            "chapter.analysis_overview": "분석 개요",
            "chapter.quality_results": "데이터 품질 진단 결과",
            "chapter.column_diagnostics": "컬럼별 상세 진단",
            "chapter.risk_factors": "이상 패턴 및 위험 요인",
            "chapter.recommendations": "개선 권고사항",
            "chapter.analysis_overview.lead": "본 장에서는 분석 대상 데이터의 규모와 핵심 프로파일 지표를 요약합니다. 컬럼 단위 구조는 {column_table}에 정리하였으며, 관측된 데이터 유형 구성은 {type_figure}에 제시하였습니다.",
            "chapter.quality_results.lead": "본 장에서는 프로파일 기반 품질 신호를 해석합니다. 결측값 패턴은 {missing_figure}에 제시하였고, {quality_table}에서는 측정된 근거와 추가 입력이 필요한 품질 차원을 구분하였습니다.",
            "chapter.column_diagnostics.lead": "본 장에서는 원본 컬럼명과 기술 유형 식별자를 보존하면서 컬럼별 진단 결과를 검토합니다. 감사 목적의 전체 컬럼 프로파일은 {profile_appendix}에 수록하였습니다.",
            "chapter.risk_factors.lead": "본 장에서는 탐지된 패턴, 값 분포 신호, 상관관계, 경고 항목을 검토 근거로 묶어 제시합니다. 고유성 분포는 {uniqueness_figure}에 제시하였으며, 최종 판단 전 업무 규칙에 따른 의미 확인이 필요합니다.",
            "chapter.recommendations.lead": "본 장에서는 프로파일 근거를 우선 검토 조치로 전환합니다. 지표 정의와 재현성 메타데이터는 각각 {metrics_appendix} 및 {repro_appendix}에 수록하였습니다.",
            "summary.subtitle": "목적, 주요 결과, 위험, 조치 및 검토 한계",
            "summary.purpose": "분석 목적",
            "summary.purpose.text": "데이터 프로파일을 점검하고 조치 가능한 데이터 품질 위험을 요약합니다.",
            "summary.data_overview": "입력 데이터 개요",
            "summary.data_overview.text": "분석 대상 데이터는 총 {rows:,}행, {columns}열로 구성되어 있습니다.",
            "summary.key_findings": "주요 결과",
            "summary.key_findings.text": "전체 품질 점수는 {score:.1f}%({grade})이며, 중복 행 비율은 {duplicate_ratio:.2%}입니다.",
            "summary.risks": "주요 위험",
            "summary.risk.missing": "결측값 검토가 필요한 컬럼이 {count}개 확인되었습니다.",
            "summary.risk.duplicates": "중복 행 비율이 {duplicate_ratio:.2%}로 확인되어 원천 적재 및 중복 제거 기준 검토가 필요합니다.",
            "summary.risk.low_uniqueness": "낮은 고유값 비율을 보이는 컬럼이 {count}개 확인되어 코드 목록 또는 상수성 필드 검토가 필요합니다.",
            "summary.risk.none": "제공된 프로파일 기준으로 우선 조치가 필요한 구조적 위험은 확인되지 않았습니다.",
            "summary.priority_actions": "우선 조치",
            "summary.action.missing": "결측률이 높은 컬럼의 수집 절차와 대체 규칙을 우선 검토하십시오.",
            "summary.action.duplicates": "하위 사용 전에 원천 키, 적재 재시도, 중복 제거 규칙을 검토하십시오.",
            "summary.action.low_uniqueness": "낮은 고유값 컬럼이 의도된 범주형 값, 상수 필드, 수집 결함 중 무엇에 해당하는지 확인하십시오.",
            "summary.action.validators": "모니터링 대상 컬럼에 권장 검증 규칙을 유지하십시오.",
            "summary.limitations": "검토 한계",
            "summary.limitations.text": "본 보고서는 제공된 프로파일 메타데이터를 기반으로 하며, 업무상 정확성은 도메인 검증 없이는 확정할 수 없습니다.",
            "quality.framework.subtitle": "공인 데이터 품질 차원에 기반한 해석 계층",
            "quality.framework.note": "아래 매핑은 Truthound 프로파일 신호가 일반적인 데이터 품질 차원과 어떻게 연결되는지 설명합니다. 지표 계산 의미는 변경하지 않습니다.",
            "quality.framework.table": "품질 차원 매핑",
            "quality.framework.caption": "참고 기준: Eurostat ESS/QAF, ISO 8000, Wang and Strong, Batini and Scannapieco, DAMA-DMBOK 데이터 품질 차원.",
            "quality.dimension": "품질 차원",
            "quality.metric": "판단 근거",
            "quality.measurement_status": "측정 상태",
            "quality.note": "비고",
            "quality.status.measured": "측정됨",
            "quality.status.partial": "부분 측정",
            "quality.status.not_measured": "입력 필요",
            "quality.dimension.completeness": "완전성",
            "quality.dimension.completeness.metric": "결측값 비율",
            "quality.dimension.completeness.note": "프로파일링된 컬럼의 결측률을 기준으로 산정합니다.",
            "quality.dimension.completeness.limitation": "완전성은 관측된 결측 여부를 설명하며, 값의 의미상 정확성을 입증하지는 않습니다.",
            "quality.dimension.uniqueness": "고유성",
            "quality.dimension.uniqueness.metric": "서로 다른 값 수 및 고유값 비율",
            "quality.dimension.uniqueness.note": "서로 다른 값 수와 고유값 비율을 기준으로 산정합니다.",
            "quality.dimension.uniqueness.limitation": "고유성은 중복 위험을 파악하는 단서이나, 반복 값이 업무적으로 유효한지는 별도 판단이 필요합니다.",
            "quality.dimension.validity": "유효성",
            "quality.dimension.validity.metric": "탐지 패턴 및 추론 유형 신호",
            "quality.dimension.validity.note": "완전한 판정을 위해서는 명시적인 업무 규칙이 필요합니다.",
            "quality.dimension.validity.limitation": "패턴 및 유형 신호는 근거 자료이며, 완전한 도메인 검증 규칙을 대체하지 않습니다.",
            "quality.dimension.accuracy": "정확성",
            "quality.dimension.accuracy.metric": "업무 기준값 또는 참조 데이터 비교",
            "quality.dimension.accuracy.note": "프로파일만으로는 확정할 수 없으며 참조 데이터가 필요합니다.",
            "quality.dimension.accuracy.limitation": "정확성은 권위 있는 기준 데이터 또는 업무 담당자 검토가 있어야 판단할 수 있습니다.",
            "quality.dimension.timeliness": "시의성",
            "quality.dimension.timeliness.metric": "최신성 및 갱신 SLA",
            "quality.dimension.timeliness.note": "수집 시각 또는 최신성 정책 입력이 필요합니다.",
            "quality.dimension.timeliness.limitation": "시의성은 수집 시각, 기준 시점, 갱신 정책이 제공되어야 평가할 수 있습니다.",
            "appendix.metrics": "지표 정의 및 산식",
            "appendix.reproducibility": "실행 환경 및 재현 정보",
            "appendix.full_profile": "전체 컬럼 프로파일",
            "appendix.quality_coverage": "품질 차원별 측정 가능성 및 한계",
            "appendix.methodology": "진단 기준 및 임계값",
            "appendix.metrics.lead": "본 부록은 본문에서 사용한 지표의 의미와 해석 경계를 정의합니다. Truthound 프로파일 계산 방식을 재정의하지 않습니다.",
            "appendix.reproducibility.lead": "본 부록은 원본 입력 데이터를 노출하지 않고 보고서 산출물을 재현하는 데 필요한 실행 메타데이터를 기록합니다.",
            "appendix.full_profile.lead": "본 부록은 본문 분석에 사용된 전체 컬럼 프로파일을 제공하며, 원본 컬럼명과 기술 유형 식별자를 그대로 보존합니다.",
            "appendix.quality_coverage.lead": "본 부록은 감사 검토를 위해 프로파일에서 측정된 근거와 업무 규칙, 기준 데이터, 최신성 metadata가 필요한 품질 차원을 분리합니다.",
            "appendix.methodology.lead": "본 부록은 자동 생성 경고와 권고사항에 사용한 임계값 정책을 기록하여 보고서 해석을 재현 가능하게 합니다.",
            "appendix.metrics.table": "지표 정의",
            "appendix.full_profile.table": "전체 컬럼 프로파일",
            "appendix.quality_coverage.table": "품질 차원별 측정 가능성",
            "appendix.methodology.table": "진단 기준 및 임계값",
            "appendix.quality_coverage.note": "본 부록은 프로파일 기반 판단 근거와 업무 규칙, 기준 데이터, 최신성 metadata가 필요한 품질 차원을 구분합니다.",
            "appendix.metric": "지표",
            "appendix.definition": "정의",
            "appendix.limitation": "한계",
            "methodology.criterion": "진단 기준",
            "methodology.threshold": "임계값",
            "methodology.interpretation": "해석",
            "methodology.missing_review": "결측값 검토",
            "methodology.missing_review.note": "이 기준 이상인 컬럼은 기본 프로파일 계산을 바꾸지 않고 요약문 검토 지침의 근거가 될 수 있습니다.",
            "methodology.high_missing": "높은 결측값 비율",
            "methodology.high_missing.note": "경고 임계값을 초과한 컬럼은 결측값 검토 대상으로 표시하고, 오류 임계값 이상인 컬럼은 더 높은 심각도로 분류합니다.",
            "methodology.constant_column": "상수 컬럼",
            "methodology.constant_column.threshold": "서로 다른 값 1개",
            "methodology.constant_column.note": "제공된 프로파일에서 상수로 표시된 컬럼은 정보성이 낮을 수 있는 필드로 검토합니다.",
            "methodology.low_uniqueness": "낮은 고유값 비율",
            "methodology.low_uniqueness.threshold": "< {uniqueness}; 행 수 > {rows}",
            "methodology.low_uniqueness.note": "고유값 비율이 임계값보다 낮고 최소 행 수를 초과하는 경우에만 검토 대상으로 표시합니다.",
            "methodology.duplicate_rows": "중복 행",
            "methodology.duplicate_rows.note": "중복 행 비율이 임계값을 초과한 데이터셋은 원천 적재 및 중복 제거 기준 검토 대상으로 표시합니다.",
            "methodology.quality_score_limit": "품질 점수 해석 한계",
            "methodology.quality_score_limit.threshold": "프로파일 기반",
            "methodology.quality_score_limit.note": "품질 점수는 프로파일 메타데이터를 요약하며, 업무상 정확성, 시의성 또는 활용 적합성을 입증하지 않습니다.",
            "appendix.truthound_version": "Truthound 버전",
            "appendix.python_version": "Python 버전",
            "appendix.platform": "플랫폼",
            "appendix.theme": "테마",
            "appendix.language": "언어",
            "appendix.generated_at": "생성 일시",
            "appendix.source_label": "데이터 출처",
            "appendix.input_fingerprint": "입력 fingerprint",
            "appendix.formula.rows": "프로파일링된 전체 행 수입니다.",
            "appendix.formula.columns": "프로파일링된 전체 열 수입니다.",
            "appendix.formula.missing": "프로파일링된 컬럼에서 관측된 전체 결측 셀 수입니다.",
            "appendix.formula.duplicates": "제공된 프로파일에서 중복으로 식별된 행 수입니다.",
            "appendix.formula.duplicate_ratio": "중복 행 수를 전체 행 수로 나눈 값입니다.",
            "appendix.formula.quality_score": "기존 Truthound 프로파일 품질 점수이며, 본 보고서는 계산 방식을 변경하지 않습니다.",
            "section.overview": "개요",
            "section.overview.subtitle": "데이터셋 요약 및 주요 지표",
            "section.quality": "데이터 품질",
            "section.quality.subtitle": "품질 지표 및 평가 결과",
            "section.columns": "컬럼 상세",
            "section.columns.subtitle": "총 {count}개 컬럼 분석",
            "section.alerts": "경고",
            "section.alerts.subtitle": "데이터 품질 이슈 및 주의 사항",
            "section.recommendations": "권장사항",
            "section.recommendations.subtitle": "개선 및 검증 규칙 제안",
            "section.statistics": "통계",
            "section.distribution": "값 분포",
            "section.distribution.subtitle": "컬럼별 분포 분석",
            "section.patterns": "탐지된 패턴",
            "section.patterns.subtitle": "자동 탐지된 데이터 패턴",
            "section.correlations": "상관관계",
            "section.correlations.subtitle": "컬럼 간 관계 및 상관성",
            "quality.score": "품질 점수",
            "quality.completeness": "완전성",
            "quality.uniqueness": "고유성",
            "quality.validity": "유효성",
            "quality.consistency": "일관성",
            "quality.completeness.desc": "결측 없이 채워진 데이터 비율",
            "quality.uniqueness.desc": "고유값 비율 기반 중복 가능성 평가",
            "quality.validity.desc": "데이터 형식의 유효성 평가",
            "quality.consistency.desc": "데이터 일관성 평가",
            "quality.grade": "등급",
            "quality.excellent": "우수",
            "quality.good": "양호",
            "quality.fair": "보통",
            "quality.poor": "미흡",
            "quality.critical": "심각",
            "alert.critical": "심각",
            "alert.warning": "경고",
            "alert.info": "정보",
            "alert.count": "{count}개의 경고가 발견되었습니다",
            "stats.row_count": "행 수",
            "stats.rows": "행 수",
            "stats.columns": "열 수",
            "stats.memory": "메모리",
            "stats.duplicates": "중복 행",
            "stats.duplicate_ratio": "중복 비율",
            "stats.missing": "결측 셀",
            "stats.quality": "품질",
            "stats.rows.desc": "전체 행 수",
            "stats.columns.desc": "전체 열 수",
            "stats.memory.desc": "예상 메모리 사용량",
            "stats.duplicates.desc": "중복 행 수",
            "stats.missing.desc": "전체 결측 셀 수",
            "stats.quality.desc": "전체 데이터 품질",
            "stats.null": "결측",
            "stats.unique": "고유",
            "stats.distinct": "서로 다른 값",
            "stats.column_count": "열 수",
            "stats.null_ratio": "결측률",
            "stats.unique_ratio": "고유값 비율",
            "stats.duplicate_count": "중복 수",
            "stats.mean": "평균",
            "stats.median": "중앙값",
            "stats.std_dev": "표준편차",
            "stats.std.short": "표준편차",
            "stats.min": "최솟값",
            "stats.min.short": "최소",
            "stats.max": "최댓값",
            "stats.max.short": "최대",
            "table.column_summary": "컬럼 요약",
            "table.column": "컬럼",
            "table.type": "유형",
            "table.null_percent": "결측률",
            "table.unique_percent": "고유값 비율",
            "table.distinct": "서로 다른 값",
            "table.pattern": "패턴",
            "table.match_rate": "일치율",
            "table.samples": "예시",
            "chart.data_types": "데이터 유형 분포",
            "chart.missing_values": "결측값 상위 컬럼",
            "chart.uniqueness": "고유성 상위 컬럼",
            "patterns.none": "탐지된 패턴 없음",
            "patterns.examples": "예시",
            "correlations.none": "유의미한 상관관계 없음",
            "recommendations.none": "현재 별도 권장사항 없음",
            "recommendations.validators": "권장 검증 규칙",
            "action.export": "내보내기",
            "action.download": "다운로드",
            "action.share": "공유",
            "action.print": "인쇄",
            "common.yes": "예",
            "common.no": "아니오",
            "common.na": "해당없음",
            "common.total": "합계",
            "common.average": "평균",
            "common.percentage": "백분율",
            "common.unknown": "알 수 없음",
        },
        metadata={"name": "Korean", "native": "한국어", "direction": "ltr"},
    )


def _create_japanese_catalog() -> ReportCatalog:
    """Create Japanese catalog."""
    return ReportCatalog.from_dict(
        "ja",
        {
            "report.title": "データ品質レポート",
            "report.subtitle": "自動データプロファイリングと検証",
            "report.summary": "概要",
            "report.details": "詳細",
            "report.generated_at": "生成日時",
            "report.generated_by": "作成者",
            "section.overview": "概要",
            "section.columns": "カラム分析",
            "section.alerts": "アラート",
            "section.recommendations": "推奨事項",
            "section.statistics": "統計",
            "section.distribution": "分布",
            "section.patterns": "パターン",
            "section.correlations": "相関",
            "quality.score": "品質スコア",
            "quality.grade": "グレード",
            "quality.excellent": "優秀",
            "quality.good": "良好",
            "quality.fair": "普通",
            "quality.poor": "不良",
            "quality.critical": "重大",
            "alert.critical": "重大",
            "alert.warning": "警告",
            "alert.info": "情報",
            "alert.count": "{count}件のアラートが見つかりました",
            "stats.row_count": "行数",
            "stats.column_count": "列数",
            "stats.null_ratio": "欠損率",
            "stats.unique_ratio": "ユニーク率",
            "stats.duplicate_count": "重複数",
            "stats.mean": "平均",
            "stats.median": "中央値",
            "stats.std_dev": "標準偏差",
            "stats.min": "最小値",
            "stats.max": "最大値",
            "action.export": "エクスポート",
            "action.download": "ダウンロード",
            "action.share": "共有",
            "action.print": "印刷",
            "common.yes": "はい",
            "common.no": "いいえ",
            "common.na": "該当なし",
            "common.total": "合計",
            "common.average": "平均",
            "common.percentage": "割合",
        },
        metadata={"name": "Japanese", "native": "日本語", "direction": "ltr"},
    )


def _create_chinese_catalog() -> ReportCatalog:
    """Create Chinese (Simplified) catalog."""
    return ReportCatalog.from_dict(
        "zh",
        {
            "report.title": "数据质量报告",
            "report.subtitle": "自动化数据分析与验证",
            "report.summary": "摘要",
            "report.details": "详情",
            "report.generated_at": "生成时间",
            "report.generated_by": "生成者",
            "section.overview": "概述",
            "section.columns": "列分析",
            "section.alerts": "告警",
            "section.recommendations": "建议",
            "section.statistics": "统计",
            "section.distribution": "分布",
            "section.patterns": "模式",
            "section.correlations": "相关性",
            "quality.score": "质量分数",
            "quality.grade": "等级",
            "quality.excellent": "优秀",
            "quality.good": "良好",
            "quality.fair": "一般",
            "quality.poor": "较差",
            "quality.critical": "严重",
            "alert.critical": "严重",
            "alert.warning": "警告",
            "alert.info": "信息",
            "alert.count": "发现 {count} 个告警",
            "stats.row_count": "行数",
            "stats.column_count": "列数",
            "stats.null_ratio": "缺失率",
            "stats.unique_ratio": "唯一值比率",
            "stats.duplicate_count": "重复数",
            "stats.mean": "平均值",
            "stats.median": "中位数",
            "stats.std_dev": "标准差",
            "stats.min": "最小值",
            "stats.max": "最大值",
            "action.export": "导出",
            "action.download": "下载",
            "action.share": "分享",
            "action.print": "打印",
            "common.yes": "是",
            "common.no": "否",
            "common.na": "不适用",
            "common.total": "总计",
            "common.average": "平均",
            "common.percentage": "百分比",
        },
        metadata={"name": "Chinese", "native": "中文", "direction": "ltr"},
    )


def _create_german_catalog() -> ReportCatalog:
    """Create German catalog."""
    return ReportCatalog.from_dict(
        "de",
        {
            "report.title": "Datenqualitätsbericht",
            "report.subtitle": "Automatisierte Datenprofilierung und Validierung",
            "report.summary": "Zusammenfassung",
            "report.details": "Details",
            "report.generated_at": "Erstellt am",
            "report.generated_by": "Erstellt von",
            "section.overview": "Übersicht",
            "section.columns": "Spaltenanalyse",
            "section.alerts": "Warnungen",
            "section.recommendations": "Empfehlungen",
            "section.statistics": "Statistiken",
            "quality.score": "Qualitätswert",
            "quality.grade": "Bewertung",
            "quality.excellent": "Ausgezeichnet",
            "quality.good": "Gut",
            "quality.fair": "Befriedigend",
            "quality.poor": "Mangelhaft",
            "quality.critical": "Kritisch",
            "alert.critical": "Kritisch",
            "alert.warning": "Warnung",
            "alert.info": "Information",
            "stats.row_count": "Zeilenanzahl",
            "stats.column_count": "Spaltenanzahl",
            "stats.null_ratio": "Fehlende Werte",
            "stats.unique_ratio": "Eindeutige Werte",
            "stats.duplicate_count": "Duplikate",
            "common.yes": "Ja",
            "common.no": "Nein",
            "common.na": "k.A.",
            "common.total": "Gesamt",
        },
        metadata={"name": "German", "native": "Deutsch", "direction": "ltr"},
    )


def _create_french_catalog() -> ReportCatalog:
    """Create French catalog."""
    return ReportCatalog.from_dict(
        "fr",
        {
            "report.title": "Rapport de qualité des données",
            "report.subtitle": "Profilage et validation automatisés des données",
            "report.summary": "Résumé",
            "report.details": "Détails",
            "report.generated_at": "Généré le",
            "report.generated_by": "Généré par",
            "section.overview": "Aperçu",
            "section.columns": "Analyse des colonnes",
            "section.alerts": "Alertes",
            "section.recommendations": "Recommandations",
            "section.statistics": "Statistiques",
            "quality.score": "Score de qualité",
            "quality.grade": "Note",
            "quality.excellent": "Excellent",
            "quality.good": "Bon",
            "quality.fair": "Passable",
            "quality.poor": "Mauvais",
            "quality.critical": "Critique",
            "alert.critical": "Critique",
            "alert.warning": "Avertissement",
            "alert.info": "Information",
            "stats.row_count": "Nombre de lignes",
            "stats.column_count": "Nombre de colonnes",
            "stats.null_ratio": "Taux de valeurs nulles",
            "stats.unique_ratio": "Taux de valeurs uniques",
            "stats.duplicate_count": "Nombre de doublons",
            "common.yes": "Oui",
            "common.no": "Non",
            "common.na": "N/A",
            "common.total": "Total",
        },
        metadata={"name": "French", "native": "Français", "direction": "ltr"},
    )


def _create_spanish_catalog() -> ReportCatalog:
    """Create Spanish catalog."""
    return ReportCatalog.from_dict(
        "es",
        {
            "report.title": "Informe de calidad de datos",
            "report.subtitle": "Perfilado y validación automatizada de datos",
            "report.summary": "Resumen",
            "report.details": "Detalles",
            "report.generated_at": "Generado el",
            "report.generated_by": "Generado por",
            "section.overview": "Resumen",
            "section.columns": "Análisis de columnas",
            "section.alerts": "Alertas",
            "section.recommendations": "Recomendaciones",
            "section.statistics": "Estadísticas",
            "quality.score": "Puntuación de calidad",
            "quality.grade": "Grado",
            "quality.excellent": "Excelente",
            "quality.good": "Bueno",
            "quality.fair": "Regular",
            "quality.poor": "Malo",
            "quality.critical": "Crítico",
            "alert.critical": "Crítico",
            "alert.warning": "Advertencia",
            "alert.info": "Información",
            "stats.row_count": "Número de filas",
            "stats.column_count": "Número de columnas",
            "stats.null_ratio": "Tasa de valores nulos",
            "stats.unique_ratio": "Tasa de valores únicos",
            "stats.duplicate_count": "Número de duplicados",
            "common.yes": "Sí",
            "common.no": "No",
            "common.na": "N/A",
            "common.total": "Total",
        },
        metadata={"name": "Spanish", "native": "Español", "direction": "ltr"},
    )


def _create_portuguese_catalog() -> ReportCatalog:
    """Create Portuguese catalog."""
    return ReportCatalog.from_dict(
        "pt",
        {
            "report.title": "Relatório de Qualidade de Dados",
            "report.subtitle": "Perfilamento e validação automatizada de dados",
            "report.summary": "Resumo",
            "report.details": "Detalhes",
            "report.generated_at": "Gerado em",
            "report.generated_by": "Gerado por",
            "section.overview": "Visão Geral",
            "section.columns": "Análise de Colunas",
            "section.alerts": "Alertas",
            "section.recommendations": "Recomendações",
            "section.statistics": "Estatísticas",
            "quality.score": "Pontuação de Qualidade",
            "quality.grade": "Grau",
            "quality.excellent": "Excelente",
            "quality.good": "Bom",
            "quality.fair": "Regular",
            "quality.poor": "Ruim",
            "quality.critical": "Crítico",
            "alert.critical": "Crítico",
            "alert.warning": "Aviso",
            "alert.info": "Informação",
            "common.yes": "Sim",
            "common.no": "Não",
            "common.na": "N/A",
            "common.total": "Total",
        },
        metadata={"name": "Portuguese", "native": "Português", "direction": "ltr"},
    )


def _create_italian_catalog() -> ReportCatalog:
    """Create Italian catalog."""
    return ReportCatalog.from_dict(
        "it",
        {
            "report.title": "Rapporto sulla qualità dei dati",
            "report.subtitle": "Profilazione e validazione automatizzata dei dati",
            "report.summary": "Riepilogo",
            "report.details": "Dettagli",
            "report.generated_at": "Generato il",
            "report.generated_by": "Generato da",
            "section.overview": "Panoramica",
            "section.columns": "Analisi delle colonne",
            "section.alerts": "Avvisi",
            "section.recommendations": "Raccomandazioni",
            "section.statistics": "Statistiche",
            "quality.score": "Punteggio di qualità",
            "quality.grade": "Grado",
            "quality.excellent": "Eccellente",
            "quality.good": "Buono",
            "quality.fair": "Discreto",
            "quality.poor": "Scarso",
            "quality.critical": "Critico",
            "alert.critical": "Critico",
            "alert.warning": "Avviso",
            "alert.info": "Informazione",
            "common.yes": "Sì",
            "common.no": "No",
            "common.na": "N/D",
            "common.total": "Totale",
        },
        metadata={"name": "Italian", "native": "Italiano", "direction": "ltr"},
    )


def _create_russian_catalog() -> ReportCatalog:
    """Create Russian catalog."""
    return ReportCatalog.from_dict(
        "ru",
        {
            "report.title": "Отчёт о качестве данных",
            "report.subtitle": "Автоматизированное профилирование и валидация данных",
            "report.summary": "Сводка",
            "report.details": "Подробности",
            "report.generated_at": "Создано",
            "report.generated_by": "Автор",
            "section.overview": "Обзор",
            "section.columns": "Анализ столбцов",
            "section.alerts": "Предупреждения",
            "section.recommendations": "Рекомендации",
            "section.statistics": "Статистика",
            "quality.score": "Оценка качества",
            "quality.grade": "Класс",
            "quality.excellent": "Отлично",
            "quality.good": "Хорошо",
            "quality.fair": "Удовлетворительно",
            "quality.poor": "Плохо",
            "quality.critical": "Критично",
            "alert.critical": "Критично",
            "alert.warning": "Предупреждение",
            "alert.info": "Информация",
            "common.yes": "Да",
            "common.no": "Нет",
            "common.na": "Н/Д",
            "common.total": "Всего",
        },
        metadata={"name": "Russian", "native": "Русский", "direction": "ltr"},
    )


def _create_arabic_catalog() -> ReportCatalog:
    """Create Arabic catalog."""
    return ReportCatalog.from_dict(
        "ar",
        {
            "report.title": "تقرير جودة البيانات",
            "report.subtitle": "تحليل وتحقق آلي من البيانات",
            "report.summary": "ملخص",
            "report.details": "تفاصيل",
            "report.generated_at": "تم الإنشاء في",
            "report.generated_by": "تم الإنشاء بواسطة",
            "section.overview": "نظرة عامة",
            "section.columns": "تحليل الأعمدة",
            "section.alerts": "تنبيهات",
            "section.recommendations": "توصيات",
            "section.statistics": "إحصائيات",
            "quality.score": "درجة الجودة",
            "quality.grade": "التقييم",
            "quality.excellent": "ممتاز",
            "quality.good": "جيد",
            "quality.fair": "مقبول",
            "quality.poor": "ضعيف",
            "quality.critical": "حرج",
            "alert.critical": "حرج",
            "alert.warning": "تحذير",
            "alert.info": "معلومات",
            "common.yes": "نعم",
            "common.no": "لا",
            "common.na": "غير متاح",
            "common.total": "المجموع",
        },
        metadata={"name": "Arabic", "native": "العربية", "direction": "rtl"},
    )


def _create_thai_catalog() -> ReportCatalog:
    """Create Thai catalog."""
    return ReportCatalog.from_dict(
        "th",
        {
            "report.title": "รายงานคุณภาพข้อมูล",
            "report.subtitle": "การวิเคราะห์และตรวจสอบข้อมูลอัตโนมัติ",
            "report.summary": "สรุป",
            "report.details": "รายละเอียด",
            "section.overview": "ภาพรวม",
            "section.columns": "การวิเคราะห์คอลัมน์",
            "section.alerts": "การแจ้งเตือน",
            "section.recommendations": "คำแนะนำ",
            "section.statistics": "สถิติ",
            "quality.score": "คะแนนคุณภาพ",
            "quality.grade": "เกรด",
            "quality.excellent": "ดีเยี่ยม",
            "quality.good": "ดี",
            "quality.fair": "พอใช้",
            "quality.poor": "ไม่ดี",
            "quality.critical": "วิกฤต",
            "common.yes": "ใช่",
            "common.no": "ไม่",
            "common.total": "รวม",
        },
        metadata={"name": "Thai", "native": "ไทย", "direction": "ltr"},
    )


def _create_vietnamese_catalog() -> ReportCatalog:
    """Create Vietnamese catalog."""
    return ReportCatalog.from_dict(
        "vi",
        {
            "report.title": "Báo cáo Chất lượng Dữ liệu",
            "report.subtitle": "Phân tích và xác thực dữ liệu tự động",
            "report.summary": "Tóm tắt",
            "report.details": "Chi tiết",
            "section.overview": "Tổng quan",
            "section.columns": "Phân tích cột",
            "section.alerts": "Cảnh báo",
            "section.recommendations": "Khuyến nghị",
            "section.statistics": "Thống kê",
            "quality.score": "Điểm chất lượng",
            "quality.grade": "Hạng",
            "quality.excellent": "Xuất sắc",
            "quality.good": "Tốt",
            "quality.fair": "Trung bình",
            "quality.poor": "Kém",
            "quality.critical": "Nghiêm trọng",
            "common.yes": "Có",
            "common.no": "Không",
            "common.total": "Tổng",
        },
        metadata={"name": "Vietnamese", "native": "Tiếng Việt", "direction": "ltr"},
    )


def _create_indonesian_catalog() -> ReportCatalog:
    """Create Indonesian catalog."""
    return ReportCatalog.from_dict(
        "id",
        {
            "report.title": "Laporan Kualitas Data",
            "report.subtitle": "Profiling dan Validasi Data Otomatis",
            "report.summary": "Ringkasan",
            "report.details": "Detail",
            "section.overview": "Gambaran Umum",
            "section.columns": "Analisis Kolom",
            "section.alerts": "Peringatan",
            "section.recommendations": "Rekomendasi",
            "section.statistics": "Statistik",
            "quality.score": "Skor Kualitas",
            "quality.grade": "Nilai",
            "quality.excellent": "Sangat Baik",
            "quality.good": "Baik",
            "quality.fair": "Cukup",
            "quality.poor": "Buruk",
            "quality.critical": "Kritis",
            "common.yes": "Ya",
            "common.no": "Tidak",
            "common.total": "Total",
        },
        metadata={"name": "Indonesian", "native": "Bahasa Indonesia", "direction": "ltr"},
    )


def _create_turkish_catalog() -> ReportCatalog:
    """Create Turkish catalog."""
    return ReportCatalog.from_dict(
        "tr",
        {
            "report.title": "Veri Kalitesi Raporu",
            "report.subtitle": "Otomatik Veri Profilleme ve Doğrulama",
            "report.summary": "Özet",
            "report.details": "Detaylar",
            "section.overview": "Genel Bakış",
            "section.columns": "Sütun Analizi",
            "section.alerts": "Uyarılar",
            "section.recommendations": "Öneriler",
            "section.statistics": "İstatistikler",
            "quality.score": "Kalite Puanı",
            "quality.grade": "Derece",
            "quality.excellent": "Mükemmel",
            "quality.good": "İyi",
            "quality.fair": "Orta",
            "quality.poor": "Zayıf",
            "quality.critical": "Kritik",
            "common.yes": "Evet",
            "common.no": "Hayır",
            "common.total": "Toplam",
        },
        metadata={"name": "Turkish", "native": "Türkçe", "direction": "ltr"},
    )
