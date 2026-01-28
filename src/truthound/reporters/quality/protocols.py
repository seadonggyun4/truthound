"""Protocol definitions for the Quality Reporter system.

This module defines the core protocols (interfaces) that enable the pluggable
architecture of the quality reporting system. All components are designed
around these protocols to ensure extensibility and testability.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, Sequence, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from truthound.profiler.quality import QualityLevel, QualityMetrics, RuleQualityScore


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
ScoreT = TypeVar("ScoreT", bound="QualityReportable")


# =============================================================================
# Core Data Protocols
# =============================================================================


@runtime_checkable
class QualityReportable(Protocol):
    """Protocol for objects that can be reported on.

    Any object implementing this protocol can be processed by the
    quality reporting system. This allows flexibility in input types
    while maintaining type safety.
    """

    @property
    def rule_name(self) -> str:
        """Name of the rule being scored."""
        ...

    @property
    def metrics(self) -> "QualityMetrics":
        """Quality metrics for the rule."""
        ...

    @property
    def should_use(self) -> bool:
        """Whether the rule should be used based on quality."""
        ...

    @property
    def recommendation(self) -> str:
        """Human-readable recommendation."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        ...


@runtime_checkable
class QualityMetricsProtocol(Protocol):
    """Protocol for quality metrics."""

    @property
    def precision(self) -> float:
        """Precision score (0.0 - 1.0)."""
        ...

    @property
    def recall(self) -> float:
        """Recall score (0.0 - 1.0)."""
        ...

    @property
    def f1_score(self) -> float:
        """F1 score (0.0 - 1.0)."""
        ...

    @property
    def accuracy(self) -> float:
        """Accuracy score (0.0 - 1.0)."""
        ...

    @property
    def confidence(self) -> float:
        """Confidence in the metrics (0.0 - 1.0)."""
        ...

    @property
    def quality_level(self) -> "QualityLevel":
        """Quality level classification."""
        ...


# =============================================================================
# Filter Protocol
# =============================================================================


@runtime_checkable
class QualityFilterProtocol(Protocol[ScoreT]):
    """Protocol for quality score filters.

    Filters implement the predicate pattern for selecting scores
    based on various criteria. Filters can be composed using
    boolean operators (and, or, not).
    """

    @property
    def name(self) -> str:
        """Human-readable name for the filter."""
        ...

    @property
    def description(self) -> str:
        """Description of what the filter does."""
        ...

    def matches(self, score: ScoreT) -> bool:
        """Check if a single score matches this filter.

        Args:
            score: Quality score to check.

        Returns:
            True if the score matches the filter criteria.
        """
        ...

    def apply(self, scores: Sequence[ScoreT]) -> list[ScoreT]:
        """Apply filter to a sequence of scores.

        Args:
            scores: Sequence of scores to filter.

        Returns:
            List of scores that match the filter.
        """
        ...

    def and_(self, other: "QualityFilterProtocol[ScoreT]") -> "QualityFilterProtocol[ScoreT]":
        """Combine with another filter using AND logic."""
        ...

    def or_(self, other: "QualityFilterProtocol[ScoreT]") -> "QualityFilterProtocol[ScoreT]":
        """Combine with another filter using OR logic."""
        ...

    def not_(self) -> "QualityFilterProtocol[ScoreT]":
        """Negate this filter."""
        ...


# =============================================================================
# Formatter Protocol
# =============================================================================


@runtime_checkable
class QualityFormatterProtocol(Protocol):
    """Protocol for quality data formatters.

    Formatters transform quality scores into various representations
    (strings, dictionaries, etc.) for display or serialization.
    """

    @property
    def name(self) -> str:
        """Formatter identifier."""
        ...

    def format_score(self, score: QualityReportable) -> str:
        """Format a single quality score.

        Args:
            score: Quality score to format.

        Returns:
            Formatted string representation.
        """
        ...

    def format_scores(self, scores: Sequence[QualityReportable]) -> str:
        """Format multiple quality scores.

        Args:
            scores: Sequence of scores to format.

        Returns:
            Formatted string representation.
        """
        ...

    def format_summary(
        self,
        scores: Sequence[QualityReportable],
        include_statistics: bool = True,
    ) -> str:
        """Format a summary of quality scores.

        Args:
            scores: Sequence of scores to summarize.
            include_statistics: Whether to include aggregate statistics.

        Returns:
            Formatted summary string.
        """
        ...


# =============================================================================
# Exporter Protocol
# =============================================================================


@runtime_checkable
class QualityExporterProtocol(Protocol):
    """Protocol for exporting quality reports.

    Exporters handle the serialization and output of formatted
    quality reports to various destinations (files, streams, etc.).
    """

    @property
    def name(self) -> str:
        """Exporter identifier."""
        ...

    @property
    def file_extension(self) -> str:
        """Default file extension for this export format."""
        ...

    @property
    def content_type(self) -> str:
        """MIME content type for the export format."""
        ...

    def export(
        self,
        content: str,
        output_path: str | None = None,
    ) -> str:
        """Export content.

        Args:
            content: Formatted content to export.
            output_path: Optional path to write to.

        Returns:
            The exported content (or file path if written).
        """
        ...


# =============================================================================
# Reporter Protocol
# =============================================================================


@runtime_checkable
class QualityReporterProtocol(Protocol):
    """Protocol for quality reporters.

    Reporters combine formatters and exporters to generate
    complete quality reports from score data.
    """

    @property
    def name(self) -> str:
        """Reporter identifier."""
        ...

    @property
    def file_extension(self) -> str:
        """Default file extension."""
        ...

    @property
    def content_type(self) -> str:
        """MIME content type."""
        ...

    def render(self, data: Sequence[QualityReportable] | QualityReportable) -> str:
        """Render quality data to string.

        Args:
            data: Quality score(s) to render.

        Returns:
            Rendered report as string.
        """
        ...

    def write(
        self,
        data: Sequence[QualityReportable] | QualityReportable,
        path: str | None = None,
    ) -> str:
        """Write rendered report to file.

        Args:
            data: Quality score(s) to render.
            path: Output path (uses config if not specified).

        Returns:
            Path where report was written.
        """
        ...


# =============================================================================
# Pipeline Stage Protocols
# =============================================================================


@runtime_checkable
class TransformStageProtocol(Protocol[T]):
    """Protocol for pipeline transformation stages."""

    @property
    def name(self) -> str:
        """Stage identifier."""
        ...

    def transform(self, data: T, context: dict[str, Any]) -> T:
        """Transform data.

        Args:
            data: Data to transform.
            context: Pipeline context.

        Returns:
            Transformed data.
        """
        ...


@runtime_checkable
class RenderStageProtocol(Protocol):
    """Protocol for pipeline render stages."""

    @property
    def name(self) -> str:
        """Stage identifier."""
        ...

    def render(self, data: Any, context: dict[str, Any]) -> str:
        """Render data to string.

        Args:
            data: Data to render.
            context: Pipeline context.

        Returns:
            Rendered string.
        """
        ...


# =============================================================================
# Event Protocol (for extensibility)
# =============================================================================


@dataclass
class QualityReportEvent:
    """Event emitted during report generation."""

    event_type: str
    timestamp: datetime
    data: dict[str, Any]


@runtime_checkable
class QualityReportListener(Protocol):
    """Protocol for report event listeners."""

    def on_event(self, event: QualityReportEvent) -> None:
        """Handle a report event.

        Args:
            event: Event that occurred.
        """
        ...
