"""Composable filter system for quality scores.

This module provides a powerful, composable filter system that allows
users to select quality scores based on various criteria. Filters can
be combined using boolean operators (AND, OR, NOT) to create complex
filtering logic.

Example:
    >>> from truthound.reporters.quality.filters import QualityFilter
    >>>
    >>> # Simple level filter
    >>> good_or_better = QualityFilter.by_level("good")
    >>>
    >>> # Combine filters
    >>> high_confidence_good = (
    ...     QualityFilter.by_level("good")
    ...     .and_(QualityFilter.by_confidence(min_value=0.8))
    ... )
    >>>
    >>> # Apply to scores
    >>> filtered = high_confidence_good.apply(scores)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Sequence, TypeVar

from truthound.reporters.quality.protocols import (
    QualityFilterProtocol,
    QualityReportable,
)
from truthound.reporters.quality.config import QualityFilterConfig

if TYPE_CHECKING:
    from truthound.profiler.quality import QualityLevel, RuleQualityScore


# =============================================================================
# Type Variables
# =============================================================================

ScoreT = TypeVar("ScoreT", bound=QualityReportable)


# =============================================================================
# Quality Level Ordering
# =============================================================================

QUALITY_LEVEL_ORDER = {
    "unacceptable": 0,
    "poor": 1,
    "acceptable": 2,
    "good": 3,
    "excellent": 4,
}


def get_level_value(level: str) -> int:
    """Get numeric value for quality level."""
    return QUALITY_LEVEL_ORDER.get(level.lower(), -1)


# =============================================================================
# Base Filter
# =============================================================================


class BaseQualityFilter(ABC, Generic[ScoreT]):
    """Abstract base class for quality filters.

    Provides common functionality and implements the composite pattern
    for combining filters.
    """

    def __init__(self, name: str = "", description: str = "") -> None:
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        """Filter name."""
        return self._name

    @property
    def description(self) -> str:
        """Filter description."""
        return self._description

    @abstractmethod
    def matches(self, score: ScoreT) -> bool:
        """Check if a score matches this filter.

        Args:
            score: Score to check.

        Returns:
            True if the score matches.
        """
        pass

    def apply(self, scores: Sequence[ScoreT]) -> list[ScoreT]:
        """Apply filter to a sequence of scores.

        Args:
            scores: Scores to filter.

        Returns:
            List of matching scores.
        """
        return [s for s in scores if self.matches(s)]

    def and_(self, other: "BaseQualityFilter[ScoreT]") -> "CompositeFilter[ScoreT]":
        """Combine with another filter using AND.

        Args:
            other: Filter to combine with.

        Returns:
            Combined filter.
        """
        return CompositeFilter(
            filters=[self, other],
            operator="and",
            name=f"({self.name} AND {other.name})",
        )

    def or_(self, other: "BaseQualityFilter[ScoreT]") -> "CompositeFilter[ScoreT]":
        """Combine with another filter using OR.

        Args:
            other: Filter to combine with.

        Returns:
            Combined filter.
        """
        return CompositeFilter(
            filters=[self, other],
            operator="or",
            name=f"({self.name} OR {other.name})",
        )

    def not_(self) -> "NotFilter[ScoreT]":
        """Negate this filter.

        Returns:
            Negated filter.
        """
        return NotFilter(self)

    def __and__(self, other: "BaseQualityFilter[ScoreT]") -> "CompositeFilter[ScoreT]":
        """Support & operator."""
        return self.and_(other)

    def __or__(self, other: "BaseQualityFilter[ScoreT]") -> "CompositeFilter[ScoreT]":
        """Support | operator."""
        return self.or_(other)

    def __invert__(self) -> "NotFilter[ScoreT]":
        """Support ~ operator."""
        return self.not_()


# =============================================================================
# Composite Filters
# =============================================================================


class CompositeFilter(BaseQualityFilter[ScoreT]):
    """Composite filter combining multiple filters."""

    def __init__(
        self,
        filters: list[BaseQualityFilter[ScoreT]],
        operator: str = "and",
        name: str = "",
    ) -> None:
        self._filters = filters
        self._operator = operator.lower()
        super().__init__(
            name=name or f"{operator.upper()}({', '.join(f.name for f in filters)})",
            description=f"Combines {len(filters)} filters with {operator.upper()}",
        )

    @property
    def filters(self) -> list[BaseQualityFilter[ScoreT]]:
        """Get component filters."""
        return self._filters

    @property
    def operator(self) -> str:
        """Get combination operator."""
        return self._operator

    def matches(self, score: ScoreT) -> bool:
        """Check if score matches all/any filters."""
        if self._operator == "and":
            return all(f.matches(score) for f in self._filters)
        elif self._operator == "or":
            return any(f.matches(score) for f in self._filters)
        else:
            raise ValueError(f"Unknown operator: {self._operator}")


class NotFilter(BaseQualityFilter[ScoreT]):
    """Negates another filter."""

    def __init__(self, inner: BaseQualityFilter[ScoreT]) -> None:
        self._inner = inner
        super().__init__(
            name=f"NOT({inner.name})",
            description=f"Negates: {inner.description}",
        )

    @property
    def inner(self) -> BaseQualityFilter[ScoreT]:
        """Get inner filter."""
        return self._inner

    def matches(self, score: ScoreT) -> bool:
        """Check if score does NOT match inner filter."""
        return not self._inner.matches(score)


class AllOfFilter(CompositeFilter[ScoreT]):
    """Requires all filters to match (AND)."""

    def __init__(self, filters: list[BaseQualityFilter[ScoreT]]) -> None:
        super().__init__(filters, operator="and")


class AnyOfFilter(CompositeFilter[ScoreT]):
    """Requires any filter to match (OR)."""

    def __init__(self, filters: list[BaseQualityFilter[ScoreT]]) -> None:
        super().__init__(filters, operator="or")


# =============================================================================
# Specific Filters
# =============================================================================


class LevelFilter(BaseQualityFilter[ScoreT]):
    """Filter by quality level."""

    def __init__(
        self,
        min_level: str | None = None,
        max_level: str | None = None,
        exact_level: str | None = None,
    ) -> None:
        self._min_level = min_level
        self._max_level = max_level
        self._exact_level = exact_level

        if exact_level:
            name = f"level={exact_level}"
        elif min_level and max_level:
            name = f"level[{min_level}..{max_level}]"
        elif min_level:
            name = f"level>={min_level}"
        elif max_level:
            name = f"level<={max_level}"
        else:
            name = "level=*"

        super().__init__(
            name=name,
            description=f"Filter by quality level: {name}",
        )

    def matches(self, score: ScoreT) -> bool:
        """Check if score matches level criteria."""
        level = score.metrics.quality_level.value.lower()
        level_val = get_level_value(level)

        if self._exact_level:
            return level == self._exact_level.lower()

        if self._min_level:
            min_val = get_level_value(self._min_level)
            if level_val < min_val:
                return False

        if self._max_level:
            max_val = get_level_value(self._max_level)
            if level_val > max_val:
                return False

        return True


class MetricFilter(BaseQualityFilter[ScoreT]):
    """Filter by metric value."""

    VALID_METRICS = {
        "f1_score", "f1",
        "precision",
        "recall",
        "accuracy",
        "specificity",
        "mcc",
        "confidence",
    }

    VALID_OPERATORS = {">=", ">", "<=", "<", "==", "!="}

    def __init__(
        self,
        metric: str,
        operator: str,
        value: float,
    ) -> None:
        metric = metric.lower()
        if metric == "f1":
            metric = "f1_score"

        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. "
                f"Valid metrics: {', '.join(self.VALID_METRICS)}"
            )

        if operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator: {operator}. "
                f"Valid operators: {', '.join(self.VALID_OPERATORS)}"
            )

        self._metric = metric
        self._operator = operator
        self._value = value

        super().__init__(
            name=f"{metric}{operator}{value}",
            description=f"Filter by {metric} {operator} {value}",
        )

    def matches(self, score: ScoreT) -> bool:
        """Check if metric matches criteria."""
        metric_value = getattr(score.metrics, self._metric, None)
        if metric_value is None:
            return False

        if self._operator == ">=":
            return metric_value >= self._value
        elif self._operator == ">":
            return metric_value > self._value
        elif self._operator == "<=":
            return metric_value <= self._value
        elif self._operator == "<":
            return metric_value < self._value
        elif self._operator == "==":
            return abs(metric_value - self._value) < 1e-9
        elif self._operator == "!=":
            return abs(metric_value - self._value) >= 1e-9
        else:
            return False


class ConfidenceFilter(BaseQualityFilter[ScoreT]):
    """Filter by confidence level."""

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        self._min_value = min_value
        self._max_value = max_value

        if min_value and max_value:
            name = f"confidence[{min_value}..{max_value}]"
        elif min_value:
            name = f"confidence>={min_value}"
        elif max_value:
            name = f"confidence<={max_value}"
        else:
            name = "confidence=*"

        super().__init__(
            name=name,
            description=f"Filter by confidence: {name}",
        )

    def matches(self, score: ScoreT) -> bool:
        """Check if confidence matches criteria."""
        confidence = score.metrics.confidence

        if self._min_value is not None and confidence < self._min_value:
            return False
        if self._max_value is not None and confidence > self._max_value:
            return False

        return True


class ColumnFilter(BaseQualityFilter[ScoreT]):
    """Filter by column name."""

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        pattern: str | None = None,
    ) -> None:
        import re

        self._include = set(include) if include else None
        self._exclude = set(exclude) if exclude else None
        self._pattern = re.compile(pattern) if pattern else None

        parts = []
        if include:
            parts.append(f"include={include}")
        if exclude:
            parts.append(f"exclude={exclude}")
        if pattern:
            parts.append(f"pattern={pattern}")

        super().__init__(
            name=f"column[{', '.join(parts)}]",
            description="Filter by column name",
        )

    def matches(self, score: ScoreT) -> bool:
        """Check if column matches criteria."""
        column = getattr(score, "column", None)
        if column is None:
            return True  # Table-level scores match by default

        if self._exclude and column in self._exclude:
            return False

        if self._include and column not in self._include:
            return False

        if self._pattern and not self._pattern.match(column):
            return False

        return True


class RuleTypeFilter(BaseQualityFilter[ScoreT]):
    """Filter by rule type."""

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self._include = {t.lower() for t in include} if include else None
        self._exclude = {t.lower() for t in exclude} if exclude else None

        parts = []
        if include:
            parts.append(f"include={include}")
        if exclude:
            parts.append(f"exclude={exclude}")

        super().__init__(
            name=f"rule_type[{', '.join(parts)}]",
            description="Filter by rule type",
        )

    def matches(self, score: ScoreT) -> bool:
        """Check if rule type matches criteria."""
        rule_type = getattr(score, "rule_type", None)
        if rule_type is None:
            return True

        # Handle both enum and string
        type_str = rule_type.value.lower() if hasattr(rule_type, "value") else str(rule_type).lower()

        if self._exclude and type_str in self._exclude:
            return False

        if self._include and type_str not in self._include:
            return False

        return True


class RecommendationFilter(BaseQualityFilter[ScoreT]):
    """Filter by recommendation."""

    def __init__(
        self,
        should_use: bool | None = None,
        contains: str | None = None,
    ) -> None:
        self._should_use = should_use
        self._contains = contains.lower() if contains else None

        parts = []
        if should_use is not None:
            parts.append(f"should_use={should_use}")
        if contains:
            parts.append(f"contains={contains}")

        super().__init__(
            name=f"recommendation[{', '.join(parts)}]",
            description="Filter by recommendation",
        )

    def matches(self, score: ScoreT) -> bool:
        """Check if recommendation matches criteria."""
        if self._should_use is not None:
            if score.should_use != self._should_use:
                return False

        if self._contains:
            if self._contains not in score.recommendation.lower():
                return False

        return True


class CustomFilter(BaseQualityFilter[ScoreT]):
    """Custom filter using a predicate function."""

    def __init__(
        self,
        predicate: Callable[[ScoreT], bool],
        name: str = "custom",
        description: str = "Custom filter function",
    ) -> None:
        self._predicate = predicate
        super().__init__(name=name, description=description)

    def matches(self, score: ScoreT) -> bool:
        """Apply custom predicate."""
        return self._predicate(score)


# =============================================================================
# Filter Factory
# =============================================================================


class QualityFilter:
    """Factory class for creating quality filters.

    Provides a fluent API for creating and combining filters.

    Example:
        >>> # Level filter
        >>> good_filter = QualityFilter.by_level("good")
        >>>
        >>> # Metric filter
        >>> high_f1 = QualityFilter.by_metric("f1_score", ">=", 0.9)
        >>>
        >>> # Combine filters
        >>> combined = good_filter.and_(high_f1)
        >>>
        >>> # From config
        >>> config = QualityFilterConfig(min_level="acceptable", min_f1=0.7)
        >>> filter = QualityFilter.from_config(config)
    """

    @staticmethod
    def by_level(
        level: str | None = None,
        *,
        min_level: str | None = None,
        max_level: str | None = None,
    ) -> LevelFilter[QualityReportable]:
        """Create a level filter.

        Args:
            level: Exact level to match (shorthand for min_level=max_level=level).
            min_level: Minimum level (inclusive).
            max_level: Maximum level (inclusive).

        Returns:
            Level filter.
        """
        if level:
            return LevelFilter(exact_level=level)
        return LevelFilter(min_level=min_level, max_level=max_level)

    @staticmethod
    def by_metric(
        metric: str,
        operator: str,
        value: float,
    ) -> MetricFilter[QualityReportable]:
        """Create a metric filter.

        Args:
            metric: Metric name (f1_score, precision, recall, etc.).
            operator: Comparison operator (>=, >, <=, <, ==, !=).
            value: Value to compare against.

        Returns:
            Metric filter.
        """
        return MetricFilter(metric=metric, operator=operator, value=value)

    @staticmethod
    def by_confidence(
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> ConfidenceFilter[QualityReportable]:
        """Create a confidence filter.

        Args:
            min_value: Minimum confidence.
            max_value: Maximum confidence.

        Returns:
            Confidence filter.
        """
        return ConfidenceFilter(min_value=min_value, max_value=max_value)

    @staticmethod
    def by_column(
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        pattern: str | None = None,
    ) -> ColumnFilter[QualityReportable]:
        """Create a column filter.

        Args:
            include: Columns to include.
            exclude: Columns to exclude.
            pattern: Regex pattern for column names.

        Returns:
            Column filter.
        """
        return ColumnFilter(include=include, exclude=exclude, pattern=pattern)

    @staticmethod
    def by_rule_type(
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> RuleTypeFilter[QualityReportable]:
        """Create a rule type filter.

        Args:
            include: Rule types to include.
            exclude: Rule types to exclude.

        Returns:
            Rule type filter.
        """
        return RuleTypeFilter(include=include, exclude=exclude)

    @staticmethod
    def by_recommendation(
        should_use: bool | None = None,
        contains: str | None = None,
    ) -> RecommendationFilter[QualityReportable]:
        """Create a recommendation filter.

        Args:
            should_use: Filter by should_use flag.
            contains: Filter by recommendation text containing string.

        Returns:
            Recommendation filter.
        """
        return RecommendationFilter(should_use=should_use, contains=contains)

    @staticmethod
    def custom(
        predicate: Callable[[QualityReportable], bool],
        name: str = "custom",
        description: str = "Custom filter function",
    ) -> CustomFilter[QualityReportable]:
        """Create a custom filter.

        Args:
            predicate: Function that returns True for matching scores.
            name: Filter name.
            description: Filter description.

        Returns:
            Custom filter.
        """
        return CustomFilter(predicate=predicate, name=name, description=description)

    @staticmethod
    def all_of(*filters: BaseQualityFilter[QualityReportable]) -> AllOfFilter[QualityReportable]:
        """Create AND combination of filters.

        Args:
            *filters: Filters to combine.

        Returns:
            Combined filter (all must match).
        """
        return AllOfFilter(list(filters))

    @staticmethod
    def any_of(*filters: BaseQualityFilter[QualityReportable]) -> AnyOfFilter[QualityReportable]:
        """Create OR combination of filters.

        Args:
            *filters: Filters to combine.

        Returns:
            Combined filter (any must match).
        """
        return AnyOfFilter(list(filters))

    @staticmethod
    def none() -> CustomFilter[QualityReportable]:
        """Create a filter that matches nothing.

        Returns:
            Filter that always returns False.
        """
        return CustomFilter(lambda _: False, name="none", description="Matches nothing")

    @staticmethod
    def all() -> CustomFilter[QualityReportable]:
        """Create a filter that matches everything.

        Returns:
            Filter that always returns True.
        """
        return CustomFilter(lambda _: True, name="all", description="Matches everything")

    @classmethod
    def from_config(
        cls,
        config: QualityFilterConfig,
    ) -> BaseQualityFilter[QualityReportable]:
        """Create filter from configuration.

        Args:
            config: Filter configuration.

        Returns:
            Combined filter from configuration.
        """
        filters: list[BaseQualityFilter[QualityReportable]] = []

        # Level filters
        if config.min_level or config.max_level:
            filters.append(LevelFilter(
                min_level=config.min_level,
                max_level=config.max_level,
            ))

        # Metric filters
        if config.min_f1 is not None:
            filters.append(MetricFilter("f1_score", ">=", config.min_f1))
        if config.max_f1 is not None:
            filters.append(MetricFilter("f1_score", "<=", config.max_f1))

        if config.min_precision is not None:
            filters.append(MetricFilter("precision", ">=", config.min_precision))
        if config.max_precision is not None:
            filters.append(MetricFilter("precision", "<=", config.max_precision))

        if config.min_recall is not None:
            filters.append(MetricFilter("recall", ">=", config.min_recall))
        if config.max_recall is not None:
            filters.append(MetricFilter("recall", "<=", config.max_recall))

        # Confidence filter
        if config.min_confidence is not None or config.max_confidence is not None:
            filters.append(ConfidenceFilter(
                min_value=config.min_confidence,
                max_value=config.max_confidence,
            ))

        # Column filter
        if config.include_columns or config.exclude_columns:
            filters.append(ColumnFilter(
                include=config.include_columns if config.include_columns else None,
                exclude=config.exclude_columns if config.exclude_columns else None,
            ))

        # Rule type filter
        if config.include_rule_types or config.exclude_rule_types:
            filters.append(RuleTypeFilter(
                include=config.include_rule_types if config.include_rule_types else None,
                exclude=config.exclude_rule_types if config.exclude_rule_types else None,
            ))

        # Recommendation filter
        if config.should_use_only or config.recommendation_contains:
            filters.append(RecommendationFilter(
                should_use=True if config.should_use_only else None,
                contains=config.recommendation_contains,
            ))

        # Combine all filters
        if not filters:
            return cls.all()
        elif len(filters) == 1:
            return filters[0]
        else:
            return AllOfFilter(filters)
