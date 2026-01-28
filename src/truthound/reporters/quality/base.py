"""Base classes for Quality Reporters.

This module provides the abstract base class and common functionality
for all quality reporter implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar

from truthound.reporters.quality.config import (
    QualityReporterConfig,
    ReportSortOrder,
)
from truthound.reporters.quality.protocols import QualityReportable

if TYPE_CHECKING:
    from truthound.profiler.quality import RuleQualityScore


# =============================================================================
# Type Variables
# =============================================================================

ConfigT = TypeVar("ConfigT", bound=QualityReporterConfig)


# =============================================================================
# Exceptions
# =============================================================================


class QualityReporterError(Exception):
    """Base exception for quality reporter errors."""

    pass


class QualityRenderError(QualityReporterError):
    """Raised when rendering fails."""

    pass


class QualityWriteError(QualityReporterError):
    """Raised when writing fails."""

    pass


class QualityValidationError(QualityReporterError):
    """Raised when input validation fails."""

    pass


# =============================================================================
# Report Result
# =============================================================================


@dataclass
class QualityReportResult:
    """Result of generating a quality report.

    Contains the rendered content along with metadata about the
    report generation process.
    """

    content: str
    """Rendered report content."""

    format: str
    """Output format (json, html, console, markdown)."""

    generated_at: datetime = field(default_factory=datetime.now)
    """When the report was generated."""

    output_path: Path | None = None
    """Path where report was written (if applicable)."""

    scores_count: int = 0
    """Number of scores included in the report."""

    filtered_count: int = 0
    """Number of scores filtered out."""

    generation_time_ms: float = 0.0
    """Time to generate report in milliseconds."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format,
            "generated_at": self.generated_at.isoformat(),
            "output_path": str(self.output_path) if self.output_path else None,
            "scores_count": self.scores_count,
            "filtered_count": self.filtered_count,
            "generation_time_ms": self.generation_time_ms,
            "metadata": self.metadata,
        }


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class QualityStatistics:
    """Aggregate statistics for quality scores."""

    total_count: int = 0
    """Total number of scores."""

    # By quality level
    excellent_count: int = 0
    good_count: int = 0
    acceptable_count: int = 0
    poor_count: int = 0
    unacceptable_count: int = 0

    # By recommendation
    should_use_count: int = 0
    should_not_use_count: int = 0

    # Metric aggregates
    avg_f1: float = 0.0
    min_f1: float = 0.0
    max_f1: float = 0.0

    avg_precision: float = 0.0
    min_precision: float = 0.0
    max_precision: float = 0.0

    avg_recall: float = 0.0
    min_recall: float = 0.0
    max_recall: float = 0.0

    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    # By rule type
    by_rule_type: dict[str, int] = field(default_factory=dict)

    # By column
    by_column: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_scores(cls, scores: Sequence[QualityReportable]) -> "QualityStatistics":
        """Calculate statistics from scores.

        Args:
            scores: Sequence of quality scores.

        Returns:
            Calculated statistics.
        """
        if not scores:
            return cls()

        stats = cls(total_count=len(scores))

        # Collect metrics
        f1_scores = []
        precisions = []
        recalls = []
        confidences = []

        for score in scores:
            metrics = score.metrics
            f1_scores.append(metrics.f1_score)
            precisions.append(metrics.precision)
            recalls.append(metrics.recall)
            confidences.append(metrics.confidence)

            # Quality level counts
            level = metrics.quality_level.value.lower()
            if level == "excellent":
                stats.excellent_count += 1
            elif level == "good":
                stats.good_count += 1
            elif level == "acceptable":
                stats.acceptable_count += 1
            elif level == "poor":
                stats.poor_count += 1
            else:
                stats.unacceptable_count += 1

            # Should use counts
            if score.should_use:
                stats.should_use_count += 1
            else:
                stats.should_not_use_count += 1

            # Rule type counts
            if hasattr(score, "rule_type"):
                rule_type = str(score.rule_type.value if hasattr(score.rule_type, "value") else score.rule_type)
                stats.by_rule_type[rule_type] = stats.by_rule_type.get(rule_type, 0) + 1

            # Column counts
            if hasattr(score, "column") and score.column:
                stats.by_column[score.column] = stats.by_column.get(score.column, 0) + 1

        # Calculate aggregates
        if f1_scores:
            stats.avg_f1 = sum(f1_scores) / len(f1_scores)
            stats.min_f1 = min(f1_scores)
            stats.max_f1 = max(f1_scores)

        if precisions:
            stats.avg_precision = sum(precisions) / len(precisions)
            stats.min_precision = min(precisions)
            stats.max_precision = max(precisions)

        if recalls:
            stats.avg_recall = sum(recalls) / len(recalls)
            stats.min_recall = min(recalls)
            stats.max_recall = max(recalls)

        if confidences:
            stats.avg_confidence = sum(confidences) / len(confidences)
            stats.min_confidence = min(confidences)
            stats.max_confidence = max(confidences)

        return stats

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "by_level": {
                "excellent": self.excellent_count,
                "good": self.good_count,
                "acceptable": self.acceptable_count,
                "poor": self.poor_count,
                "unacceptable": self.unacceptable_count,
            },
            "by_recommendation": {
                "should_use": self.should_use_count,
                "should_not_use": self.should_not_use_count,
            },
            "metrics": {
                "f1": {"avg": self.avg_f1, "min": self.min_f1, "max": self.max_f1},
                "precision": {"avg": self.avg_precision, "min": self.min_precision, "max": self.max_precision},
                "recall": {"avg": self.avg_recall, "min": self.min_recall, "max": self.max_recall},
                "confidence": {"avg": self.avg_confidence, "min": self.min_confidence, "max": self.max_confidence},
            },
            "by_rule_type": self.by_rule_type,
            "by_column": self.by_column,
        }


# =============================================================================
# Base Quality Reporter
# =============================================================================


class BaseQualityReporter(ABC, Generic[ConfigT]):
    """Abstract base class for quality reporters.

    All quality reporter implementations should inherit from this class.
    It provides common functionality and defines the interface that
    all reporters must implement.

    Type Parameters:
        ConfigT: Configuration type for this reporter.

    Example:
        >>> class MyQualityReporter(BaseQualityReporter[QualityReporterConfig]):
        ...     name = "my_format"
        ...     file_extension = ".txt"
        ...
        ...     def render(self, data):
        ...         return "My Report"
    """

    # Class-level attributes
    name: str = "base"
    """Reporter identifier."""

    file_extension: str = ".txt"
    """Default file extension."""

    content_type: str = "text/plain"
    """MIME content type."""

    def __init__(
        self,
        config: ConfigT | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the reporter.

        Args:
            config: Reporter configuration. Uses defaults if not provided.
            **kwargs: Additional options to override config values.
        """
        self._config = config or self._default_config()

        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        return QualityReporterConfig()  # type: ignore

    @property
    def config(self) -> ConfigT:
        """Get reporter configuration."""
        return self._config

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def render(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
    ) -> str:
        """Render quality data to string.

        Args:
            data: Quality score(s) to render.

        Returns:
            Rendered report as string.

        Raises:
            QualityRenderError: If rendering fails.
        """
        pass

    # -------------------------------------------------------------------------
    # Concrete Methods
    # -------------------------------------------------------------------------

    def normalize_input(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
    ) -> list[QualityReportable]:
        """Normalize input to list of scores.

        Args:
            data: Single score or sequence of scores.

        Returns:
            List of scores.
        """
        if isinstance(data, (list, tuple)):
            return list(data)
        return [data]

    def sort_scores(
        self,
        scores: list[QualityReportable],
        order: ReportSortOrder | None = None,
    ) -> list[QualityReportable]:
        """Sort scores according to configuration.

        Args:
            scores: List of scores to sort.
            order: Sort order (uses config if not specified).

        Returns:
            Sorted list of scores.
        """
        order = order or self._config.sort_order

        sort_key_map = {
            ReportSortOrder.F1_DESC: lambda s: -s.metrics.f1_score,
            ReportSortOrder.F1_ASC: lambda s: s.metrics.f1_score,
            ReportSortOrder.PRECISION_DESC: lambda s: -s.metrics.precision,
            ReportSortOrder.PRECISION_ASC: lambda s: s.metrics.precision,
            ReportSortOrder.RECALL_DESC: lambda s: -s.metrics.recall,
            ReportSortOrder.RECALL_ASC: lambda s: s.metrics.recall,
            ReportSortOrder.CONFIDENCE_DESC: lambda s: -s.metrics.confidence,
            ReportSortOrder.CONFIDENCE_ASC: lambda s: s.metrics.confidence,
            ReportSortOrder.NAME_ASC: lambda s: s.rule_name,
            ReportSortOrder.NAME_DESC: lambda s: s.rule_name,
        }

        key_fn = sort_key_map.get(order, lambda s: -s.metrics.f1_score)
        reverse = order.value.endswith("_desc") if order.value.endswith(("_desc", "_asc")) else False

        if order in (ReportSortOrder.NAME_DESC,):
            return sorted(scores, key=key_fn, reverse=True)

        return sorted(scores, key=key_fn)

    def calculate_statistics(
        self,
        scores: Sequence[QualityReportable],
    ) -> QualityStatistics:
        """Calculate aggregate statistics.

        Args:
            scores: Scores to analyze.

        Returns:
            Calculated statistics.
        """
        return QualityStatistics.from_scores(scores)

    def render_to_bytes(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
        encoding: str = "utf-8",
    ) -> bytes:
        """Render to bytes.

        Args:
            data: Quality score(s) to render.
            encoding: Character encoding.

        Returns:
            Rendered content as bytes.
        """
        return self.render(data).encode(encoding)

    def write(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
        path: str | Path | None = None,
    ) -> Path:
        """Write rendered report to file.

        Args:
            data: Quality score(s) to render.
            path: Output path (uses config if not specified).

        Returns:
            Path where report was written.

        Raises:
            QualityWriteError: If no path specified or write fails.
        """
        output_path = Path(path) if path else self._config.get_output_path()

        if output_path is None:
            raise QualityWriteError(
                "No output path specified. Either pass a path argument "
                "or set output_path in the reporter configuration."
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            content = self.render(data)
            output_path.write_text(content, encoding="utf-8")
            return output_path

        except (OSError, IOError) as e:
            raise QualityWriteError(f"Failed to write report to {output_path}: {e}")

    def report(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
        path: str | Path | None = None,
    ) -> QualityReportResult:
        """Generate a complete report.

        Renders the report and optionally writes to file.

        Args:
            data: Quality score(s) to render.
            path: Optional output path.

        Returns:
            Report result with content and metadata.
        """
        start_time = datetime.now()

        scores = self.normalize_input(data)
        content = self.render(scores)

        output_path = Path(path) if path else self._config.get_output_path()
        if output_path:
            self.write(data, output_path)

        generation_time = (datetime.now() - start_time).total_seconds() * 1000

        return QualityReportResult(
            content=content,
            format=self.name,
            output_path=output_path,
            scores_count=len(scores),
            generation_time_ms=generation_time,
        )

    def generate_filename(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
        timestamp: bool = True,
    ) -> str:
        """Generate a filename for the report.

        Args:
            data: Data being reported.
            timestamp: Include timestamp in filename.

        Returns:
            Generated filename.
        """
        base_name = f"quality_{self.name}"

        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{ts}"

        return f"{base_name}{self.file_extension}"
