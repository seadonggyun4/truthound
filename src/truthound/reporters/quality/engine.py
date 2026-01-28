"""Quality Report Engine - Pipeline-based report generation.

This module provides a pipeline-based engine for generating quality
reports. It supports composable stages, caching, and parallel processing.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from truthound.reporters.quality.protocols import (
    QualityReportable,
    QualityReportEvent,
    QualityReportListener,
    TransformStageProtocol,
    RenderStageProtocol,
)
from truthound.reporters.quality.config import (
    QualityReporterConfig,
    QualityReportEngineConfig,
    QualityFilterConfig,
    ReportSortOrder,
)
from truthound.reporters.quality.base import (
    QualityReportResult,
    QualityStatistics,
    QualityReporterError,
)
from truthound.reporters.quality.filters import QualityFilter, BaseQualityFilter
from truthound.reporters.quality.factory import get_quality_reporter


# =============================================================================
# Pipeline Context
# =============================================================================


@dataclass
class QualityReportContext:
    """Context passed through pipeline stages.

    Contains input data, configuration, and accumulated state.
    """

    # Input data
    scores: list[QualityReportable] = field(default_factory=list)
    original_count: int = 0

    # Configuration
    config: QualityReporterConfig = field(default_factory=QualityReporterConfig)
    engine_config: QualityReportEngineConfig = field(default_factory=QualityReportEngineConfig)

    # Computed state
    statistics: QualityStatistics | None = None
    filtered_count: int = 0

    # Output
    rendered_content: str = ""
    output_path: Path | None = None

    # Metadata
    start_time: datetime = field(default_factory=datetime.now)
    stage_times: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "QualityReportContext":
        """Create a shallow copy of the context."""
        return QualityReportContext(
            scores=list(self.scores),
            original_count=self.original_count,
            config=self.config,
            engine_config=self.engine_config,
            statistics=self.statistics,
            filtered_count=self.filtered_count,
            rendered_content=self.rendered_content,
            output_path=self.output_path,
            start_time=self.start_time,
            stage_times=dict(self.stage_times),
            metadata=dict(self.metadata),
        )


# =============================================================================
# Pipeline Stages
# =============================================================================


class FilterStage:
    """Pipeline stage for filtering scores."""

    name = "filter"

    def __init__(
        self,
        filter_obj: BaseQualityFilter[QualityReportable] | None = None,
        filter_config: QualityFilterConfig | None = None,
    ) -> None:
        self._filter = filter_obj
        self._filter_config = filter_config

    def transform(
        self,
        context: QualityReportContext,
    ) -> QualityReportContext:
        """Apply filtering to scores."""
        start = datetime.now()

        # Build filter from config if not provided
        filter_obj = self._filter
        if filter_obj is None and self._filter_config:
            filter_obj = QualityFilter.from_config(self._filter_config)
        elif filter_obj is None and context.config.filters.has_filters():
            filter_obj = QualityFilter.from_config(context.config.filters)

        # Apply filter
        if filter_obj:
            original_count = len(context.scores)
            context.scores = filter_obj.apply(context.scores)
            context.filtered_count = original_count - len(context.scores)

        context.stage_times[self.name] = (datetime.now() - start).total_seconds() * 1000
        return context


class SortStage:
    """Pipeline stage for sorting scores."""

    name = "sort"

    def __init__(self, order: ReportSortOrder | None = None) -> None:
        self._order = order

    def transform(
        self,
        context: QualityReportContext,
    ) -> QualityReportContext:
        """Sort scores."""
        start = datetime.now()

        order = self._order or context.config.sort_order

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
        reverse = order == ReportSortOrder.NAME_DESC

        context.scores = sorted(context.scores, key=key_fn, reverse=reverse)

        context.stage_times[self.name] = (datetime.now() - start).total_seconds() * 1000
        return context


class LimitStage:
    """Pipeline stage for limiting number of scores."""

    name = "limit"

    def __init__(self, max_scores: int | None = None) -> None:
        self._max_scores = max_scores

    def transform(
        self,
        context: QualityReportContext,
    ) -> QualityReportContext:
        """Limit scores."""
        start = datetime.now()

        max_scores = self._max_scores or context.config.max_scores
        if max_scores and len(context.scores) > max_scores:
            context.scores = context.scores[:max_scores]

        context.stage_times[self.name] = (datetime.now() - start).total_seconds() * 1000
        return context


class StatisticsStage:
    """Pipeline stage for calculating statistics."""

    name = "statistics"

    def transform(
        self,
        context: QualityReportContext,
    ) -> QualityReportContext:
        """Calculate statistics."""
        start = datetime.now()

        context.statistics = QualityStatistics.from_scores(context.scores)

        context.stage_times[self.name] = (datetime.now() - start).total_seconds() * 1000
        return context


class RenderStage:
    """Pipeline stage for rendering output."""

    name = "render"

    def __init__(self, format: str = "console", **kwargs: Any) -> None:
        self._format = format
        self._kwargs = kwargs

    def transform(
        self,
        context: QualityReportContext,
    ) -> QualityReportContext:
        """Render scores."""
        start = datetime.now()

        reporter = get_quality_reporter(
            self._format,
            config=context.config,
            **self._kwargs,
        )
        context.rendered_content = reporter.render(context.scores)

        context.stage_times[self.name] = (datetime.now() - start).total_seconds() * 1000
        return context


class WriteStage:
    """Pipeline stage for writing output to file."""

    name = "write"

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None

    def transform(
        self,
        context: QualityReportContext,
    ) -> QualityReportContext:
        """Write output."""
        start = datetime.now()

        output_path = self._path or context.config.get_output_path()
        if output_path and context.rendered_content:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(context.rendered_content, encoding="utf-8")
            context.output_path = output_path

        context.stage_times[self.name] = (datetime.now() - start).total_seconds() * 1000
        return context


# =============================================================================
# Pipeline
# =============================================================================


class QualityReportPipeline:
    """Composable pipeline for quality report generation.

    Stages are executed in order, with each stage receiving and
    returning a context object.

    Example:
        >>> pipeline = (
        ...     QualityReportPipeline()
        ...     .filter(QualityFilter.by_level("good"))
        ...     .sort(ReportSortOrder.F1_DESC)
        ...     .limit(10)
        ...     .statistics()
        ...     .render("html")
        ...     .write("report.html")
        ... )
        >>> result = pipeline.execute(scores)
    """

    def __init__(self) -> None:
        self._stages: list[Any] = []
        self._listeners: list[QualityReportListener] = []

    def add_stage(self, stage: Any) -> "QualityReportPipeline":
        """Add a stage to the pipeline.

        Args:
            stage: Stage to add.

        Returns:
            Self for chaining.
        """
        self._stages.append(stage)
        return self

    def filter(
        self,
        filter_obj: BaseQualityFilter[QualityReportable] | None = None,
        config: QualityFilterConfig | None = None,
    ) -> "QualityReportPipeline":
        """Add filter stage.

        Args:
            filter_obj: Filter to apply.
            config: Filter configuration.

        Returns:
            Self for chaining.
        """
        return self.add_stage(FilterStage(filter_obj, config))

    def sort(self, order: ReportSortOrder | None = None) -> "QualityReportPipeline":
        """Add sort stage.

        Args:
            order: Sort order.

        Returns:
            Self for chaining.
        """
        return self.add_stage(SortStage(order))

    def limit(self, max_scores: int | None = None) -> "QualityReportPipeline":
        """Add limit stage.

        Args:
            max_scores: Maximum scores to include.

        Returns:
            Self for chaining.
        """
        return self.add_stage(LimitStage(max_scores))

    def statistics(self) -> "QualityReportPipeline":
        """Add statistics stage.

        Returns:
            Self for chaining.
        """
        return self.add_stage(StatisticsStage())

    def render(self, format: str = "console", **kwargs: Any) -> "QualityReportPipeline":
        """Add render stage.

        Args:
            format: Output format.
            **kwargs: Format-specific options.

        Returns:
            Self for chaining.
        """
        return self.add_stage(RenderStage(format, **kwargs))

    def write(self, path: str | Path | None = None) -> "QualityReportPipeline":
        """Add write stage.

        Args:
            path: Output path.

        Returns:
            Self for chaining.
        """
        return self.add_stage(WriteStage(path))

    def add_listener(self, listener: QualityReportListener) -> "QualityReportPipeline":
        """Add event listener.

        Args:
            listener: Listener to add.

        Returns:
            Self for chaining.
        """
        self._listeners.append(listener)
        return self

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to all listeners."""
        event = QualityReportEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
        )
        for listener in self._listeners:
            try:
                listener.on_event(event)
            except Exception:
                pass  # Don't let listener errors break pipeline

    def execute(
        self,
        scores: Sequence[QualityReportable],
        config: QualityReporterConfig | None = None,
    ) -> QualityReportContext:
        """Execute the pipeline.

        Args:
            scores: Quality scores to process.
            config: Optional configuration.

        Returns:
            Pipeline context with results.
        """
        # Initialize context
        context = QualityReportContext(
            scores=list(scores),
            original_count=len(scores),
            config=config or QualityReporterConfig(),
        )

        self._emit_event("pipeline_start", {"stage_count": len(self._stages)})

        # Execute stages
        for i, stage in enumerate(self._stages):
            stage_name = getattr(stage, "name", f"stage_{i}")
            self._emit_event("stage_start", {"stage": stage_name, "index": i})

            try:
                context = stage.transform(context)
                self._emit_event("stage_complete", {
                    "stage": stage_name,
                    "index": i,
                    "duration_ms": context.stage_times.get(stage_name, 0),
                })
            except Exception as e:
                self._emit_event("stage_error", {"stage": stage_name, "error": str(e)})
                raise

        self._emit_event("pipeline_complete", {
            "total_time_ms": sum(context.stage_times.values()),
            "scores_count": len(context.scores),
        })

        return context


# =============================================================================
# Report Engine
# =============================================================================


class QualityReportEngine:
    """High-level engine for generating quality reports.

    Provides a simple interface for common report generation tasks
    while supporting advanced pipeline customization.

    Example:
        >>> engine = QualityReportEngine()
        >>>
        >>> # Simple report
        >>> result = engine.generate(scores, format="html", output_path="report.html")
        >>>
        >>> # With filtering
        >>> result = engine.generate(
        ...     scores,
        ...     format="json",
        ...     filter=QualityFilter.by_level("good"),
        ... )
        >>>
        >>> # Custom pipeline
        >>> result = engine.execute_pipeline(
        ...     scores,
        ...     pipeline=QualityReportPipeline()
        ...         .filter(...)
        ...         .sort(...)
        ...         .render("html"),
        ... )
    """

    def __init__(
        self,
        config: QualityReportEngineConfig | None = None,
    ) -> None:
        self._config = config or QualityReportEngineConfig()
        self._cache: dict[str, QualityReportResult] = {}
        self._lock = threading.Lock()

    @property
    def config(self) -> QualityReportEngineConfig:
        """Get engine configuration."""
        return self._config

    def generate(
        self,
        scores: Sequence[QualityReportable],
        format: str = "console",
        output_path: str | Path | None = None,
        filter: BaseQualityFilter[QualityReportable] | None = None,
        sort_order: ReportSortOrder = ReportSortOrder.F1_DESC,
        max_scores: int | None = None,
        use_cache: bool = True,
        config: QualityReporterConfig | None = None,
        **kwargs: Any,
    ) -> QualityReportResult:
        """Generate a quality report.

        Args:
            scores: Quality scores to report.
            format: Output format (console, json, html, markdown, junit).
            output_path: Optional path to write report.
            filter: Optional filter to apply.
            sort_order: Sort order for scores.
            max_scores: Maximum scores to include.
            use_cache: Whether to use cached results.
            config: Reporter configuration.
            **kwargs: Additional format-specific options.

        Returns:
            Report result with content and metadata.
        """
        start_time = datetime.now()

        # Check cache
        if use_cache and self._config.enable_caching:
            cache_key = self._make_cache_key(scores, format, filter, sort_order, max_scores)
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]

        # Build and execute pipeline
        pipeline = QualityReportPipeline()

        if filter:
            pipeline.filter(filter)
        elif config and config.filters.has_filters():
            pipeline.filter(config=config.filters)

        pipeline.sort(sort_order)

        if max_scores:
            pipeline.limit(max_scores)

        pipeline.statistics()
        pipeline.render(format, **kwargs)

        if output_path:
            pipeline.write(output_path)

        # Execute
        reporter_config = config or QualityReporterConfig(
            output_path=output_path,
            sort_order=sort_order,
            max_scores=max_scores,
        )

        context = pipeline.execute(scores, reporter_config)

        # Build result
        generation_time = (datetime.now() - start_time).total_seconds() * 1000
        result = QualityReportResult(
            content=context.rendered_content,
            format=format,
            output_path=context.output_path,
            scores_count=len(context.scores),
            filtered_count=context.filtered_count,
            generation_time_ms=generation_time,
            metadata={
                "stage_times": context.stage_times,
                "statistics": context.statistics.to_dict() if context.statistics else None,
            },
        )

        # Cache result
        if use_cache and self._config.enable_caching:
            with self._lock:
                self._cache[cache_key] = result

        return result

    def execute_pipeline(
        self,
        scores: Sequence[QualityReportable],
        pipeline: QualityReportPipeline,
        config: QualityReporterConfig | None = None,
    ) -> QualityReportContext:
        """Execute a custom pipeline.

        Args:
            scores: Quality scores to process.
            pipeline: Pipeline to execute.
            config: Optional configuration.

        Returns:
            Pipeline context with results.
        """
        return pipeline.execute(scores, config)

    def clear_cache(self) -> None:
        """Clear the result cache."""
        with self._lock:
            self._cache.clear()

    def _make_cache_key(
        self,
        scores: Sequence[QualityReportable],
        format: str,
        filter: BaseQualityFilter[QualityReportable] | None,
        sort_order: ReportSortOrder,
        max_scores: int | None,
    ) -> str:
        """Create cache key."""
        # Create a simple hash from inputs
        key_parts = [
            str(len(scores)),
            format,
            filter.name if filter else "none",
            sort_order.value,
            str(max_scores),
        ]
        key_str = ":".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_quality_report(
    scores: Sequence[QualityReportable],
    format: str = "console",
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> QualityReportResult:
    """Generate a quality report (convenience function).

    Args:
        scores: Quality scores to report.
        format: Output format.
        output_path: Optional output path.
        **kwargs: Additional options.

    Returns:
        Report result.
    """
    engine = QualityReportEngine()
    return engine.generate(scores, format=format, output_path=output_path, **kwargs)


def filter_quality_scores(
    scores: Sequence[QualityReportable],
    filter: BaseQualityFilter[QualityReportable] | None = None,
    min_level: str | None = None,
    min_f1: float | None = None,
    should_use_only: bool = False,
    **kwargs: Any,
) -> list[QualityReportable]:
    """Filter quality scores (convenience function).

    Args:
        scores: Quality scores to filter.
        filter: Filter to apply.
        min_level: Minimum quality level.
        min_f1: Minimum F1 score.
        should_use_only: Only include recommended rules.
        **kwargs: Additional filter options.

    Returns:
        Filtered list of scores.
    """
    if filter:
        return filter.apply(scores)

    # Build filter from parameters
    filters = []

    if min_level:
        filters.append(QualityFilter.by_level(min_level=min_level))

    if min_f1 is not None:
        filters.append(QualityFilter.by_metric("f1_score", ">=", min_f1))

    if should_use_only:
        filters.append(QualityFilter.by_recommendation(should_use=True))

    if not filters:
        return list(scores)

    combined = QualityFilter.all_of(*filters)
    return combined.apply(scores)


def compare_quality_scores(
    scores: Sequence[QualityReportable],
    sort_by: str = "f1_score",
    descending: bool = True,
) -> list[QualityReportable]:
    """Compare and rank quality scores (convenience function).

    Args:
        scores: Quality scores to compare.
        sort_by: Metric to sort by.
        descending: Sort in descending order.

    Returns:
        Sorted list of scores.
    """
    order_map = {
        "f1_score": ReportSortOrder.F1_DESC if descending else ReportSortOrder.F1_ASC,
        "f1": ReportSortOrder.F1_DESC if descending else ReportSortOrder.F1_ASC,
        "precision": ReportSortOrder.PRECISION_DESC if descending else ReportSortOrder.PRECISION_ASC,
        "recall": ReportSortOrder.RECALL_DESC if descending else ReportSortOrder.RECALL_ASC,
        "confidence": ReportSortOrder.CONFIDENCE_DESC if descending else ReportSortOrder.CONFIDENCE_ASC,
        "name": ReportSortOrder.NAME_DESC if descending else ReportSortOrder.NAME_ASC,
    }

    order = order_map.get(sort_by.lower(), ReportSortOrder.F1_DESC)

    pipeline = QualityReportPipeline().sort(order)
    context = pipeline.execute(scores)

    return context.scores
