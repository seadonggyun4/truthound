"""Configuration classes for the Quality Reporter system.

This module defines all configuration data classes used throughout the
quality reporting system, providing a clean separation of concerns and
enabling easy customization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class ReportSortOrder(str, Enum):
    """Sort order for quality reports."""

    F1_DESC = "f1_desc"
    F1_ASC = "f1_asc"
    PRECISION_DESC = "precision_desc"
    PRECISION_ASC = "precision_asc"
    RECALL_DESC = "recall_desc"
    RECALL_ASC = "recall_asc"
    CONFIDENCE_DESC = "confidence_desc"
    CONFIDENCE_ASC = "confidence_asc"
    NAME_ASC = "name_asc"
    NAME_DESC = "name_desc"


class QualityDisplayMode(str, Enum):
    """Display mode for quality information."""

    COMPACT = "compact"
    NORMAL = "normal"
    DETAILED = "detailed"
    FULL = "full"


class ChartType(str, Enum):
    """Available chart types for HTML reports."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    DONUT = "donut"
    GAUGE = "gauge"
    RADAR = "radar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"


# =============================================================================
# Threshold Configuration
# =============================================================================


@dataclass
class QualityThresholds:
    """Thresholds for quality level classification.

    Customize these to adjust what constitutes "excellent", "good", etc.
    for your use case.
    """

    excellent: float = 0.95
    good: float = 0.85
    acceptable: float = 0.70
    poor: float = 0.50
    # Below poor is automatically "unacceptable"

    min_confidence: float = 0.5
    """Minimum confidence level for recommendations."""

    min_sample_size: int = 100
    """Minimum sample size for reliable estimates."""

    def classify(self, f1_score: float) -> str:
        """Classify F1 score into quality level.

        Args:
            f1_score: F1 score to classify.

        Returns:
            Quality level string.
        """
        if f1_score >= self.excellent:
            return "excellent"
        elif f1_score >= self.good:
            return "good"
        elif f1_score >= self.acceptable:
            return "acceptable"
        elif f1_score >= self.poor:
            return "poor"
        else:
            return "unacceptable"

    def get_color(self, f1_score: float) -> str:
        """Get color for quality level.

        Args:
            f1_score: F1 score.

        Returns:
            Color name.
        """
        level = self.classify(f1_score)
        return {
            "excellent": "green",
            "good": "blue",
            "acceptable": "yellow",
            "poor": "orange",
            "unacceptable": "red",
        }[level]


# =============================================================================
# Filter Configuration
# =============================================================================


@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering.

    Enables declarative filter configuration from YAML/JSON.
    """

    # Level-based filters
    min_level: str | None = None
    """Minimum quality level (excellent, good, acceptable, poor)."""

    max_level: str | None = None
    """Maximum quality level."""

    # Metric-based filters
    min_f1: float | None = None
    max_f1: float | None = None
    min_precision: float | None = None
    max_precision: float | None = None
    min_recall: float | None = None
    max_recall: float | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None

    # Column/Rule filters
    include_columns: list[str] = field(default_factory=list)
    """Only include scores for these columns."""

    exclude_columns: list[str] = field(default_factory=list)
    """Exclude scores for these columns."""

    include_rule_types: list[str] = field(default_factory=list)
    """Only include scores for these rule types."""

    exclude_rule_types: list[str] = field(default_factory=list)
    """Exclude scores for these rule types."""

    # Recommendation filters
    should_use_only: bool = False
    """Only include rules that should be used."""

    recommendation_contains: str | None = None
    """Filter by recommendation text containing this string."""

    def has_filters(self) -> bool:
        """Check if any filters are configured."""
        return any([
            self.min_level,
            self.max_level,
            self.min_f1,
            self.max_f1,
            self.min_precision,
            self.max_precision,
            self.min_recall,
            self.max_recall,
            self.min_confidence,
            self.max_confidence,
            self.include_columns,
            self.exclude_columns,
            self.include_rule_types,
            self.exclude_rule_types,
            self.should_use_only,
            self.recommendation_contains,
        ])


# =============================================================================
# Reporter Configuration
# =============================================================================


@dataclass
class QualityReporterConfig:
    """Configuration for quality reporters.

    This is the main configuration class that controls all aspects
    of quality report generation.
    """

    # Output settings
    output_path: str | Path | None = None
    """Path to write the report to."""

    title: str = "Quality Score Report"
    """Report title."""

    description: str = ""
    """Optional report description."""

    # Content settings
    include_metrics: bool = True
    """Include detailed metrics (precision, recall, F1, etc.)."""

    include_confusion_matrix: bool = True
    """Include confusion matrix details if available."""

    include_confidence_intervals: bool = True
    """Include confidence intervals for metrics."""

    include_trend_analysis: bool = False
    """Include trend analysis if historical data available."""

    include_recommendations: bool = True
    """Include recommendations for each rule."""

    include_statistics: bool = True
    """Include aggregate statistics."""

    include_summary: bool = True
    """Include summary section."""

    include_charts: bool = True
    """Include charts in HTML reports."""

    # Formatting settings
    metric_precision: int = 4
    """Decimal places for metric values."""

    percentage_format: bool = True
    """Display metrics as percentages."""

    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    """Format for timestamps."""

    # Display settings
    display_mode: QualityDisplayMode = QualityDisplayMode.NORMAL
    """Detail level for display."""

    sort_order: ReportSortOrder = ReportSortOrder.F1_DESC
    """Sort order for scores."""

    max_scores: int | None = None
    """Maximum number of scores to include (None for all)."""

    max_sample_values: int = 5
    """Maximum sample values to show in details."""

    # Theme settings (HTML)
    theme: str = "light"
    """Theme for HTML reports (light, dark, professional)."""

    custom_css: str | None = None
    """Custom CSS to include in HTML reports."""

    chart_library: str = "apexcharts"
    """Chart library for HTML reports (apexcharts, chartjs)."""

    chart_types: list[ChartType] = field(default_factory=lambda: [
        ChartType.BAR,
        ChartType.GAUGE,
        ChartType.RADAR,
    ])
    """Chart types to include."""

    # Thresholds
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    """Quality level thresholds."""

    # Filtering
    filters: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    """Filter configuration."""

    def get_output_path(self) -> Path | None:
        """Get output path as Path object."""
        if self.output_path is None:
            return None
        return Path(self.output_path)

    @classmethod
    def compact(cls) -> "QualityReporterConfig":
        """Create compact configuration."""
        return cls(
            display_mode=QualityDisplayMode.COMPACT,
            include_confusion_matrix=False,
            include_confidence_intervals=False,
            include_trend_analysis=False,
            include_charts=False,
        )

    @classmethod
    def detailed(cls) -> "QualityReporterConfig":
        """Create detailed configuration."""
        return cls(
            display_mode=QualityDisplayMode.DETAILED,
            include_confusion_matrix=True,
            include_confidence_intervals=True,
            include_trend_analysis=True,
            include_charts=True,
        )

    @classmethod
    def full(cls) -> "QualityReporterConfig":
        """Create full configuration with all options."""
        return cls(
            display_mode=QualityDisplayMode.FULL,
            include_confusion_matrix=True,
            include_confidence_intervals=True,
            include_trend_analysis=True,
            include_charts=True,
            chart_types=[
                ChartType.BAR,
                ChartType.GAUGE,
                ChartType.RADAR,
                ChartType.SCATTER,
                ChartType.HEATMAP,
            ],
        )


# =============================================================================
# Engine Configuration
# =============================================================================


@dataclass
class QualityReportEngineConfig:
    """Configuration for the quality report engine."""

    # Pipeline settings
    enable_caching: bool = True
    """Enable result caching."""

    cache_ttl_seconds: int = 3600
    """Cache time-to-live."""

    parallel_processing: bool = True
    """Enable parallel processing for large datasets."""

    max_workers: int = 4
    """Maximum parallel workers."""

    # Event settings
    emit_events: bool = True
    """Emit events during processing."""

    event_handlers: list[Any] = field(default_factory=list)
    """Registered event handlers."""

    # Validation settings
    validate_input: bool = True
    """Validate input data."""

    strict_mode: bool = False
    """Raise on validation errors (vs. warn)."""
