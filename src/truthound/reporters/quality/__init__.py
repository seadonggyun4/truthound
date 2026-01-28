"""Quality Reporter module for generating quality score reports.

This module provides a comprehensive quality reporting system with:
- Pluggable reporter architecture (Console, JSON, HTML, Markdown)
- Composable filter system for quality-based filtering
- Trend visualization and analysis
- Integration with datadocs HTML pipeline

Example:
    >>> from truthound.reporters.quality import (
    ...     get_quality_reporter,
    ...     QualityFilter,
    ...     QualityReportEngine,
    ... )
    >>>
    >>> # Generate console report
    >>> reporter = get_quality_reporter("console")
    >>> print(reporter.render(quality_scores))
    >>>
    >>> # Filter by quality level
    >>> filter = QualityFilter.by_level("good").and_(QualityFilter.by_metric("f1_score", ">=", 0.85))
    >>> filtered = filter.apply(quality_scores)
    >>>
    >>> # Full pipeline with engine
    >>> engine = QualityReportEngine()
    >>> report = engine.generate(scores, format="html", output_path="report.html")
"""

from truthound.reporters.quality.protocols import (
    QualityReportable,
    QualityFilterProtocol,
    QualityFormatterProtocol,
    QualityExporterProtocol,
)
from truthound.reporters.quality.config import (
    QualityReporterConfig,
    QualityFilterConfig,
    QualityThresholds,
)
from truthound.reporters.quality.base import (
    BaseQualityReporter,
    QualityReportResult,
)
from truthound.reporters.quality.filters import (
    QualityFilter,
    CompositeFilter,
    LevelFilter,
    MetricFilter,
    ConfidenceFilter,
    ColumnFilter,
    RuleTypeFilter,
    RecommendationFilter,
)
from truthound.reporters.quality.formatters import (
    QualityFormatter,
    ConsoleFormatter,
    JsonFormatter,
    MarkdownFormatter,
    HtmlFormatter,
)
from truthound.reporters.quality.reporters import (
    ConsoleQualityReporter,
    JsonQualityReporter,
    MarkdownQualityReporter,
    HtmlQualityReporter,
)
from truthound.reporters.quality.factory import (
    get_quality_reporter,
    register_quality_reporter,
    list_quality_formats,
)
from truthound.reporters.quality.engine import (
    QualityReportEngine,
    QualityReportContext,
    QualityReportPipeline,
)

__all__ = [
    # Protocols
    "QualityReportable",
    "QualityFilterProtocol",
    "QualityFormatterProtocol",
    "QualityExporterProtocol",
    # Config
    "QualityReporterConfig",
    "QualityFilterConfig",
    "QualityThresholds",
    # Base
    "BaseQualityReporter",
    "QualityReportResult",
    # Filters
    "QualityFilter",
    "CompositeFilter",
    "LevelFilter",
    "MetricFilter",
    "ConfidenceFilter",
    "ColumnFilter",
    "RuleTypeFilter",
    "RecommendationFilter",
    # Formatters
    "QualityFormatter",
    "ConsoleFormatter",
    "JsonFormatter",
    "MarkdownFormatter",
    "HtmlFormatter",
    # Reporters
    "ConsoleQualityReporter",
    "JsonQualityReporter",
    "MarkdownQualityReporter",
    "HtmlQualityReporter",
    # Factory
    "get_quality_reporter",
    "register_quality_reporter",
    "list_quality_formats",
    # Engine
    "QualityReportEngine",
    "QualityReportContext",
    "QualityReportPipeline",
]
