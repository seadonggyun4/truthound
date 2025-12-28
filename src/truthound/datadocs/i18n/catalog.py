"""Message catalog for report internationalization.

This module provides translation catalogs for report UI elements,
supporting 15+ languages with extensible architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


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

    def merge(self, other: "ReportCatalog") -> "ReportCatalog":
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

    def extend(self, messages: dict[str, str]) -> "ReportCatalog":
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
    ) -> "ReportCatalog":
        """Create from dictionary."""
        return cls(
            locale=locale,
            messages=messages.copy(),
            metadata=metadata or {},
        )

    @classmethod
    def from_json(cls, path: Path) -> "ReportCatalog":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            locale=data.get("locale", "en"),
            messages=data.get("messages", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def builder(cls, locale: str) -> "CatalogBuilder":
        """Create a catalog builder."""
        return CatalogBuilder(locale)


class CatalogBuilder:
    """Fluent builder for ReportCatalog."""

    def __init__(self, locale: str) -> None:
        self._locale = locale
        self._messages: dict[str, str] = {}
        self._metadata: dict[str, Any] = {}

    def add(self, key: str, value: str) -> "CatalogBuilder":
        """Add a message."""
        self._messages[key] = value
        return self

    def add_report_section(
        self,
        title: str,
        subtitle: str,
        summary: str,
        details: str,
    ) -> "CatalogBuilder":
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
    ) -> "CatalogBuilder":
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
    ) -> "CatalogBuilder":
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
    ) -> "CatalogBuilder":
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
    ) -> "CatalogBuilder":
        """Add statistics labels."""
        self._messages["stats.row_count"] = row_count
        self._messages["stats.column_count"] = column_count
        self._messages["stats.null_ratio"] = null_ratio
        self._messages["stats.unique_ratio"] = unique_ratio
        self._messages["stats.duplicate_count"] = duplicate_count
        return self

    def with_metadata(self, **metadata: Any) -> "CatalogBuilder":
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
            "report.subtitle": "Automated Data Profiling and Validation",
            "report.summary": "Summary",
            "report.details": "Details",
            "report.generated_at": "Generated at",
            "report.generated_by": "Generated by",
            # Sections
            "section.overview": "Overview",
            "section.columns": "Column Analysis",
            "section.alerts": "Alerts",
            "section.recommendations": "Recommendations",
            "section.statistics": "Statistics",
            "section.distribution": "Distribution",
            "section.patterns": "Patterns",
            "section.correlations": "Correlations",
            # Quality
            "quality.score": "Quality Score",
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
            "stats.column_count": "Column Count",
            "stats.null_ratio": "Null Ratio",
            "stats.unique_ratio": "Unique Ratio",
            "stats.duplicate_count": "Duplicate Count",
            "stats.mean": "Mean",
            "stats.median": "Median",
            "stats.std_dev": "Standard Deviation",
            "stats.min": "Minimum",
            "stats.max": "Maximum",
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
        },
        metadata={"name": "English", "native": "English", "direction": "ltr"},
    )


def _create_korean_catalog() -> ReportCatalog:
    """Create Korean catalog."""
    return ReportCatalog.from_dict(
        "ko",
        {
            "report.title": "데이터 품질 보고서",
            "report.subtitle": "자동화된 데이터 프로파일링 및 검증",
            "report.summary": "요약",
            "report.details": "상세",
            "report.generated_at": "생성 일시",
            "report.generated_by": "생성자",
            "section.overview": "개요",
            "section.columns": "컬럼 분석",
            "section.alerts": "경고",
            "section.recommendations": "권장사항",
            "section.statistics": "통계",
            "section.distribution": "분포",
            "section.patterns": "패턴",
            "section.correlations": "상관관계",
            "quality.score": "품질 점수",
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
            "stats.column_count": "열 수",
            "stats.null_ratio": "결측률",
            "stats.unique_ratio": "고유값 비율",
            "stats.duplicate_count": "중복 수",
            "stats.mean": "평균",
            "stats.median": "중앙값",
            "stats.std_dev": "표준편차",
            "stats.min": "최솟값",
            "stats.max": "최댓값",
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
