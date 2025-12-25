"""Automatic threshold tuning based on data characteristics.

This module provides intelligent threshold tuning for validation rules:
- Analyzes data distribution to determine optimal thresholds
- Adapts strictness based on data quality
- Supports multiple tuning strategies
- Provides confidence-based recommendations

Key features:
- Statistical analysis for threshold determination
- Outlier detection for boundary setting
- Domain-aware defaults
- A/B testing support for threshold comparison

Example:
    from truthound.profiler.auto_threshold import (
        ThresholdTuner,
        tune_thresholds,
        TuningStrategy,
    )

    # Create tuner
    tuner = ThresholdTuner(strategy="adaptive")

    # Tune thresholds for a profile
    thresholds = tuner.tune(profile)

    print(f"Null threshold: {thresholds.null_threshold}")
    print(f"Uniqueness threshold: {thresholds.uniqueness_threshold}")
"""

from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl

from truthound.profiler.base import (
    ColumnProfile,
    DataType,
    DistributionStats,
    Strictness,
    TableProfile,
)


# =============================================================================
# Types and Enums
# =============================================================================


class TuningStrategy(str, Enum):
    """Threshold tuning strategies."""

    CONSERVATIVE = "conservative"    # Strict thresholds, fewer false positives
    BALANCED = "balanced"           # Balance between precision and recall
    PERMISSIVE = "permissive"       # Loose thresholds, fewer false negatives
    ADAPTIVE = "adaptive"           # Adapt based on data characteristics
    STATISTICAL = "statistical"     # Use statistical methods (percentiles, IQR)
    DOMAIN_AWARE = "domain_aware"   # Use domain-specific knowledge


class ThresholdType(str, Enum):
    """Types of thresholds."""

    NULL_RATIO = "null_ratio"
    UNIQUENESS_RATIO = "uniqueness_ratio"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    PATTERN_MATCH_RATIO = "pattern_match_ratio"
    OUTLIER_RATIO = "outlier_ratio"
    CARDINALITY = "cardinality"


# =============================================================================
# Threshold Configuration
# =============================================================================


@dataclass
class ColumnThresholds:
    """Thresholds for a single column."""

    column_name: str
    null_threshold: float = 0.0
    uniqueness_threshold: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern_match_threshold: float = 0.8
    allowed_values: set[Any] | None = None
    outlier_threshold: float = 0.01
    confidence: float = 0.5
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "null_threshold": self.null_threshold,
            "uniqueness_threshold": self.uniqueness_threshold,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern_match_threshold": self.pattern_match_threshold,
            "allowed_values": list(self.allowed_values) if self.allowed_values else None,
            "outlier_threshold": self.outlier_threshold,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class TableThresholds:
    """Thresholds for an entire table."""

    table_name: str
    columns: dict[str, ColumnThresholds] = field(default_factory=dict)
    duplicate_threshold: float = 0.0
    row_count_min: int | None = None
    row_count_max: int | None = None
    global_null_threshold: float = 0.1
    strategy_used: TuningStrategy = TuningStrategy.BALANCED
    tuned_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_column(self, name: str) -> ColumnThresholds | None:
        """Get thresholds for a column."""
        return self.columns.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "duplicate_threshold": self.duplicate_threshold,
            "row_count_min": self.row_count_min,
            "row_count_max": self.row_count_max,
            "global_null_threshold": self.global_null_threshold,
            "strategy_used": self.strategy_used.value,
            "tuned_at": self.tuned_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Strictness Presets
# =============================================================================


@dataclass
class StrictnessPreset:
    """Preset threshold multipliers for different strictness levels."""

    null_multiplier: float = 1.0
    range_buffer: float = 0.1  # 10% buffer on ranges
    pattern_threshold: float = 0.8
    uniqueness_tolerance: float = 0.05
    outlier_sensitivity: float = 1.0

    @classmethod
    def for_strictness(cls, strictness: Strictness) -> "StrictnessPreset":
        """Get preset for a strictness level."""
        presets = {
            Strictness.LOOSE: cls(
                null_multiplier=1.5,
                range_buffer=0.2,
                pattern_threshold=0.6,
                uniqueness_tolerance=0.1,
                outlier_sensitivity=0.5,
            ),
            Strictness.MEDIUM: cls(
                null_multiplier=1.0,
                range_buffer=0.1,
                pattern_threshold=0.8,
                uniqueness_tolerance=0.05,
                outlier_sensitivity=1.0,
            ),
            Strictness.STRICT: cls(
                null_multiplier=0.5,
                range_buffer=0.05,
                pattern_threshold=0.95,
                uniqueness_tolerance=0.01,
                outlier_sensitivity=2.0,
            ),
        }
        return presets.get(strictness, cls())


# =============================================================================
# Tuning Strategy Protocol
# =============================================================================


class TuningStrategyImpl(ABC):
    """Abstract base for tuning strategies."""

    name: str = "base"

    @abstractmethod
    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any],
    ) -> ColumnThresholds:
        """Tune thresholds for a column.

        Args:
            profile: Column profile
            context: Additional context

        Returns:
            Tuned thresholds
        """
        pass

    @abstractmethod
    def tune_table(
        self,
        profile: TableProfile,
        context: dict[str, Any],
    ) -> TableThresholds:
        """Tune thresholds for a table.

        Args:
            profile: Table profile
            context: Additional context

        Returns:
            Tuned thresholds
        """
        pass


class ConservativeStrategy(TuningStrategyImpl):
    """Conservative tuning - strict thresholds.

    Minimizes false positives at the cost of false negatives.
    """

    name = "conservative"

    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any],
    ) -> ColumnThresholds:
        thresholds = ColumnThresholds(column_name=profile.name)
        reasoning = []

        # Null threshold - very strict
        thresholds.null_threshold = max(0, profile.null_ratio * 0.5)
        reasoning.append(f"Null threshold set to half of observed ({profile.null_ratio:.1%})")

        # Range thresholds - tight bounds
        if profile.distribution:
            dist = profile.distribution
            if dist.min is not None and dist.max is not None:
                range_size = dist.max - dist.min
                buffer = range_size * 0.02  # 2% buffer
                thresholds.min_value = dist.min - buffer
                thresholds.max_value = dist.max + buffer
                reasoning.append(f"Range set with 2% buffer: [{thresholds.min_value:.2f}, {thresholds.max_value:.2f}]")

        # Pattern threshold - high
        thresholds.pattern_match_threshold = 0.95
        reasoning.append("Pattern match threshold: 95%")

        # Uniqueness - if unique, require it
        if profile.is_unique:
            thresholds.uniqueness_threshold = 1.0
            reasoning.append("Column appears unique, requiring 100% uniqueness")

        thresholds.confidence = 0.8
        thresholds.reasoning = reasoning

        return thresholds

    def tune_table(
        self,
        profile: TableProfile,
        context: dict[str, Any],
    ) -> TableThresholds:
        thresholds = TableThresholds(
            table_name=profile.name,
            strategy_used=TuningStrategy.CONSERVATIVE,
        )

        # Tune each column
        for col_profile in profile.columns:
            col_thresholds = self.tune_column(col_profile, context)
            thresholds.columns[col_profile.name] = col_thresholds

        # Table-level thresholds
        thresholds.duplicate_threshold = 0.0  # No duplicates allowed
        thresholds.global_null_threshold = 0.05  # Max 5% nulls overall

        return thresholds


class BalancedStrategy(TuningStrategyImpl):
    """Balanced tuning - middle ground.

    Balances between precision and recall.
    """

    name = "balanced"

    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any],
    ) -> ColumnThresholds:
        thresholds = ColumnThresholds(column_name=profile.name)
        reasoning = []

        # Null threshold - match observed with small buffer
        thresholds.null_threshold = profile.null_ratio * 1.2 + 0.01
        reasoning.append(f"Null threshold: observed + 20% buffer = {thresholds.null_threshold:.1%}")

        # Range thresholds - moderate buffer
        if profile.distribution:
            dist = profile.distribution
            if dist.min is not None and dist.max is not None:
                range_size = dist.max - dist.min
                buffer = range_size * 0.1  # 10% buffer
                thresholds.min_value = dist.min - buffer
                thresholds.max_value = dist.max + buffer
                reasoning.append(f"Range with 10% buffer: [{thresholds.min_value:.2f}, {thresholds.max_value:.2f}]")

        # Length constraints
        if profile.min_length is not None:
            thresholds.min_length = max(0, profile.min_length - 1)
            thresholds.max_length = profile.max_length + 5 if profile.max_length else None
            reasoning.append(f"Length: [{thresholds.min_length}, {thresholds.max_length}]")

        # Pattern threshold
        thresholds.pattern_match_threshold = 0.8
        reasoning.append("Pattern match threshold: 80%")

        # Uniqueness
        if profile.is_unique:
            thresholds.uniqueness_threshold = 0.99  # Allow tiny margin
            reasoning.append("Near-unique required (99%)")
        elif profile.unique_ratio > 0.9:
            thresholds.uniqueness_threshold = 0.9
            reasoning.append("High uniqueness required (90%)")

        thresholds.confidence = 0.7
        thresholds.reasoning = reasoning

        return thresholds

    def tune_table(
        self,
        profile: TableProfile,
        context: dict[str, Any],
    ) -> TableThresholds:
        thresholds = TableThresholds(
            table_name=profile.name,
            strategy_used=TuningStrategy.BALANCED,
        )

        for col_profile in profile.columns:
            col_thresholds = self.tune_column(col_profile, context)
            thresholds.columns[col_profile.name] = col_thresholds

        # Table-level
        thresholds.duplicate_threshold = profile.duplicate_row_ratio * 1.1
        thresholds.global_null_threshold = 0.1

        return thresholds


class PermissiveStrategy(TuningStrategyImpl):
    """Permissive tuning - loose thresholds.

    Minimizes false negatives at the cost of false positives.
    """

    name = "permissive"

    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any],
    ) -> ColumnThresholds:
        thresholds = ColumnThresholds(column_name=profile.name)
        reasoning = []

        # Null threshold - generous
        thresholds.null_threshold = min(1.0, profile.null_ratio * 2 + 0.05)
        reasoning.append(f"Null threshold: 2x observed = {thresholds.null_threshold:.1%}")

        # Range thresholds - wide buffer
        if profile.distribution:
            dist = profile.distribution
            if dist.min is not None and dist.max is not None:
                range_size = dist.max - dist.min
                buffer = range_size * 0.25  # 25% buffer
                thresholds.min_value = dist.min - buffer
                thresholds.max_value = dist.max + buffer
                reasoning.append(f"Wide range: [{thresholds.min_value:.2f}, {thresholds.max_value:.2f}]")

        # Pattern threshold - low
        thresholds.pattern_match_threshold = 0.6
        reasoning.append("Pattern match threshold: 60%")

        thresholds.confidence = 0.6
        thresholds.reasoning = reasoning

        return thresholds

    def tune_table(
        self,
        profile: TableProfile,
        context: dict[str, Any],
    ) -> TableThresholds:
        thresholds = TableThresholds(
            table_name=profile.name,
            strategy_used=TuningStrategy.PERMISSIVE,
        )

        for col_profile in profile.columns:
            col_thresholds = self.tune_column(col_profile, context)
            thresholds.columns[col_profile.name] = col_thresholds

        thresholds.duplicate_threshold = 0.05  # Allow some duplicates
        thresholds.global_null_threshold = 0.2

        return thresholds


class AdaptiveStrategy(TuningStrategyImpl):
    """Adaptive tuning - adjusts based on data characteristics.

    Analyzes data quality signals to choose appropriate thresholds.
    """

    name = "adaptive"

    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any],
    ) -> ColumnThresholds:
        thresholds = ColumnThresholds(column_name=profile.name)
        reasoning = []

        # Determine data quality score
        quality_score = self._assess_quality(profile)
        reasoning.append(f"Data quality score: {quality_score:.2f}")

        # Adjust strictness based on quality
        if quality_score > 0.8:
            # High quality - can be stricter
            null_mult = 0.8
            range_buffer = 0.05
            pattern_threshold = 0.9
        elif quality_score > 0.5:
            # Medium quality - balanced
            null_mult = 1.2
            range_buffer = 0.1
            pattern_threshold = 0.8
        else:
            # Low quality - be permissive
            null_mult = 1.5
            range_buffer = 0.2
            pattern_threshold = 0.6

        # Apply adjusted thresholds
        thresholds.null_threshold = profile.null_ratio * null_mult + 0.01

        if profile.distribution:
            dist = profile.distribution
            if dist.min is not None and dist.max is not None:
                range_size = dist.max - dist.min
                buffer = range_size * range_buffer
                thresholds.min_value = dist.min - buffer
                thresholds.max_value = dist.max + buffer

        thresholds.pattern_match_threshold = pattern_threshold

        # Adaptive uniqueness
        if profile.is_unique:
            thresholds.uniqueness_threshold = 1.0 if quality_score > 0.7 else 0.99
        elif profile.unique_ratio > 0.9:
            thresholds.uniqueness_threshold = profile.unique_ratio * 0.95

        # Length constraints
        if profile.min_length is not None:
            length_buffer = max(1, int(profile.avg_length * 0.1)) if profile.avg_length else 2
            thresholds.min_length = max(0, profile.min_length - length_buffer)
            thresholds.max_length = (profile.max_length or 0) + length_buffer * 2

        thresholds.confidence = quality_score
        thresholds.reasoning = reasoning

        return thresholds

    def tune_table(
        self,
        profile: TableProfile,
        context: dict[str, Any],
    ) -> TableThresholds:
        thresholds = TableThresholds(
            table_name=profile.name,
            strategy_used=TuningStrategy.ADAPTIVE,
        )

        # Calculate overall quality
        col_qualities = []
        for col_profile in profile.columns:
            quality = self._assess_quality(col_profile)
            col_qualities.append(quality)
            col_thresholds = self.tune_column(col_profile, context)
            thresholds.columns[col_profile.name] = col_thresholds

        avg_quality = sum(col_qualities) / len(col_qualities) if col_qualities else 0.5

        # Adaptive table thresholds
        if avg_quality > 0.8:
            thresholds.duplicate_threshold = 0.0
            thresholds.global_null_threshold = 0.05
        elif avg_quality > 0.5:
            thresholds.duplicate_threshold = profile.duplicate_row_ratio * 1.1
            thresholds.global_null_threshold = 0.1
        else:
            thresholds.duplicate_threshold = 0.05
            thresholds.global_null_threshold = 0.2

        thresholds.metadata["overall_quality"] = avg_quality

        return thresholds

    def _assess_quality(self, profile: ColumnProfile) -> float:
        """Assess data quality for a column."""
        scores = []

        # Completeness score (inverse of null ratio)
        completeness = 1.0 - profile.null_ratio
        scores.append(completeness)

        # Consistency score (based on patterns)
        if profile.detected_patterns:
            best_match = max(p.match_ratio for p in profile.detected_patterns)
            scores.append(best_match)

        # Uniqueness appropriateness
        if profile.is_unique or profile.unique_ratio > 0.9:
            # High uniqueness is often good for IDs
            scores.append(0.9)
        elif profile.unique_ratio < 0.01:
            # Very low uniqueness might be categorical (ok) or constant (suspicious)
            scores.append(0.5 if not profile.is_constant else 0.3)
        else:
            scores.append(0.7)

        return sum(scores) / len(scores) if scores else 0.5


class StatisticalStrategy(TuningStrategyImpl):
    """Statistical tuning - uses statistical methods.

    Uses percentiles, IQR, and other statistical measures.
    """

    name = "statistical"

    def __init__(
        self,
        percentile_low: float = 0.01,
        percentile_high: float = 0.99,
        iqr_multiplier: float = 1.5,
    ):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.iqr_multiplier = iqr_multiplier

    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any],
    ) -> ColumnThresholds:
        thresholds = ColumnThresholds(column_name=profile.name)
        reasoning = []

        # Get column data if available
        data = context.get("column_data")

        # Null threshold using binomial confidence interval
        n = profile.row_count
        p = profile.null_ratio
        if n > 0:
            # Wilson score interval
            z = 2.576  # 99% confidence
            denominator = 1 + z * z / n
            centre = p + z * z / (2 * n)
            margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
            upper_bound = min(1.0, (centre + margin) / denominator)
            thresholds.null_threshold = upper_bound
            reasoning.append(f"Null threshold from Wilson CI: {upper_bound:.3f}")

        # Range using percentiles or IQR
        if profile.distribution:
            dist = profile.distribution
            if dist.q1 is not None and dist.q3 is not None:
                # Use IQR method
                iqr = dist.q3 - dist.q1
                lower = dist.q1 - self.iqr_multiplier * iqr
                upper = dist.q3 + self.iqr_multiplier * iqr
                thresholds.min_value = lower
                thresholds.max_value = upper
                reasoning.append(f"Range from IQR ({self.iqr_multiplier}x): [{lower:.2f}, {upper:.2f}]")
            elif dist.min is not None and dist.max is not None:
                # Use min/max with buffer based on std
                if dist.std:
                    buffer = dist.std * 3  # 3 sigma
                else:
                    buffer = (dist.max - dist.min) * 0.1
                thresholds.min_value = dist.min - buffer
                thresholds.max_value = dist.max + buffer
                reasoning.append(f"Range from 3-sigma: [{thresholds.min_value:.2f}, {thresholds.max_value:.2f}]")

        # Pattern threshold based on distribution
        if profile.detected_patterns:
            match_ratios = [p.match_ratio for p in profile.detected_patterns]
            # Use 10th percentile of match ratios
            if len(match_ratios) > 1:
                threshold = sorted(match_ratios)[max(0, len(match_ratios) // 10)]
            else:
                threshold = match_ratios[0] * 0.9
            thresholds.pattern_match_threshold = threshold
            reasoning.append(f"Pattern threshold from distribution: {threshold:.2f}")

        thresholds.confidence = 0.85
        thresholds.reasoning = reasoning

        return thresholds

    def tune_table(
        self,
        profile: TableProfile,
        context: dict[str, Any],
    ) -> TableThresholds:
        thresholds = TableThresholds(
            table_name=profile.name,
            strategy_used=TuningStrategy.STATISTICAL,
        )

        for col_profile in profile.columns:
            col_thresholds = self.tune_column(col_profile, context)
            thresholds.columns[col_profile.name] = col_thresholds

        # Statistical duplicate threshold
        n = profile.row_count
        p = profile.duplicate_row_ratio
        if n > 0 and p > 0:
            z = 2.576
            margin = z * math.sqrt(p * (1 - p) / n)
            thresholds.duplicate_threshold = min(1.0, p + margin)
        else:
            thresholds.duplicate_threshold = 0.01

        return thresholds


class DomainAwareStrategy(TuningStrategyImpl):
    """Domain-aware tuning - uses domain-specific knowledge.

    Applies different rules based on detected data types.
    """

    name = "domain_aware"

    # Domain-specific defaults
    DOMAIN_DEFAULTS: dict[DataType, dict[str, Any]] = {
        DataType.EMAIL: {
            "null_threshold": 0.1,
            "pattern_threshold": 0.95,
            "min_length": 5,
            "max_length": 254,
        },
        DataType.PHONE: {
            "null_threshold": 0.2,
            "pattern_threshold": 0.9,
            "min_length": 7,
            "max_length": 20,
        },
        DataType.UUID: {
            "null_threshold": 0.0,
            "pattern_threshold": 0.99,
            "uniqueness_threshold": 1.0,
            "min_length": 36,
            "max_length": 36,
        },
        DataType.DATE: {
            "null_threshold": 0.1,
            "pattern_threshold": 0.95,
        },
        DataType.IDENTIFIER: {
            "null_threshold": 0.0,
            "uniqueness_threshold": 1.0,
        },
        DataType.CATEGORICAL: {
            "null_threshold": 0.05,
            "max_cardinality": 100,
        },
        DataType.CURRENCY: {
            "null_threshold": 0.05,
            "min_value": 0.0,
        },
        DataType.PERCENTAGE: {
            "null_threshold": 0.05,
            "min_value": 0.0,
            "max_value": 100.0,
        },
        DataType.BOOLEAN: {
            "null_threshold": 0.0,
            "allowed_values": {True, False, 0, 1, "true", "false", "yes", "no"},
        },
        DataType.KOREAN_PHONE: {
            "null_threshold": 0.1,
            "pattern_threshold": 0.95,
            "min_length": 10,
            "max_length": 13,
        },
        DataType.KOREAN_RRN: {
            "null_threshold": 0.0,
            "pattern_threshold": 0.99,
            "min_length": 13,
            "max_length": 14,
        },
    }

    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any],
    ) -> ColumnThresholds:
        thresholds = ColumnThresholds(column_name=profile.name)
        reasoning = []

        # Get domain defaults for this type
        defaults = self.DOMAIN_DEFAULTS.get(profile.inferred_type, {})
        reasoning.append(f"Using domain defaults for {profile.inferred_type.value}")

        # Apply domain defaults
        if "null_threshold" in defaults:
            thresholds.null_threshold = defaults["null_threshold"]
        else:
            thresholds.null_threshold = profile.null_ratio * 1.2 + 0.01

        if "pattern_threshold" in defaults:
            thresholds.pattern_match_threshold = defaults["pattern_threshold"]

        if "min_length" in defaults:
            thresholds.min_length = defaults["min_length"]
        elif profile.min_length is not None:
            thresholds.min_length = profile.min_length

        if "max_length" in defaults:
            thresholds.max_length = defaults["max_length"]
        elif profile.max_length is not None:
            thresholds.max_length = profile.max_length

        if "uniqueness_threshold" in defaults:
            thresholds.uniqueness_threshold = defaults["uniqueness_threshold"]

        if "min_value" in defaults:
            thresholds.min_value = defaults["min_value"]

        if "max_value" in defaults:
            thresholds.max_value = defaults["max_value"]
        elif profile.distribution and profile.distribution.max:
            thresholds.max_value = profile.distribution.max * 1.1

        if "allowed_values" in defaults:
            thresholds.allowed_values = defaults["allowed_values"]

        thresholds.confidence = 0.75
        thresholds.reasoning = reasoning

        return thresholds

    def tune_table(
        self,
        profile: TableProfile,
        context: dict[str, Any],
    ) -> TableThresholds:
        thresholds = TableThresholds(
            table_name=profile.name,
            strategy_used=TuningStrategy.DOMAIN_AWARE,
        )

        for col_profile in profile.columns:
            col_thresholds = self.tune_column(col_profile, context)
            thresholds.columns[col_profile.name] = col_thresholds

        # Table-level thresholds
        # Check if any column is a unique identifier
        has_identifier = any(
            col.inferred_type == DataType.IDENTIFIER or col.is_unique
            for col in profile.columns
        )

        if has_identifier:
            thresholds.duplicate_threshold = 0.0
        else:
            thresholds.duplicate_threshold = profile.duplicate_row_ratio * 1.1

        return thresholds


# =============================================================================
# Strategy Registry
# =============================================================================


class StrategyRegistry:
    """Registry for tuning strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, TuningStrategyImpl] = {}

    def register(self, strategy: TuningStrategyImpl) -> None:
        """Register a strategy."""
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> TuningStrategyImpl:
        """Get strategy by name."""
        if name not in self._strategies:
            raise KeyError(f"Unknown strategy: {name}")
        return self._strategies[name]

    def list_strategies(self) -> list[str]:
        """List available strategies."""
        return list(self._strategies.keys())


# Global registry
strategy_registry = StrategyRegistry()
strategy_registry.register(ConservativeStrategy())
strategy_registry.register(BalancedStrategy())
strategy_registry.register(PermissiveStrategy())
strategy_registry.register(AdaptiveStrategy())
strategy_registry.register(StatisticalStrategy())
strategy_registry.register(DomainAwareStrategy())


# =============================================================================
# Threshold Tuner
# =============================================================================


@dataclass
class TunerConfig:
    """Configuration for threshold tuner."""

    strategy: str = "adaptive"
    strictness: Strictness = Strictness.MEDIUM
    use_domain_hints: bool = True
    min_confidence: float = 0.5
    combine_strategies: bool = False


class ThresholdTuner:
    """Main interface for threshold tuning.

    Analyzes data profiles and determines optimal thresholds.

    Example:
        tuner = ThresholdTuner(strategy="adaptive")
        thresholds = tuner.tune(profile)

        for col_name, col_thresh in thresholds.columns.items():
            print(f"{col_name}: null <= {col_thresh.null_threshold:.1%}")
    """

    def __init__(
        self,
        strategy: str | TuningStrategyImpl = "adaptive",
        config: TunerConfig | None = None,
    ):
        self.config = config or TunerConfig()

        if isinstance(strategy, TuningStrategyImpl):
            self._strategy = strategy
        else:
            self._strategy = strategy_registry.get(strategy)

    def tune(
        self,
        profile: TableProfile,
        context: dict[str, Any] | None = None,
    ) -> TableThresholds:
        """Tune thresholds for a table profile.

        Args:
            profile: Table profile to tune
            context: Additional context

        Returns:
            Tuned thresholds
        """
        context = context or {}

        # Apply strictness preset
        preset = StrictnessPreset.for_strictness(self.config.strictness)
        context["strictness_preset"] = preset

        # Tune using strategy
        thresholds = self._strategy.tune_table(profile, context)

        # Apply strictness multipliers
        self._apply_strictness(thresholds, preset)

        return thresholds

    def tune_column(
        self,
        profile: ColumnProfile,
        context: dict[str, Any] | None = None,
    ) -> ColumnThresholds:
        """Tune thresholds for a single column.

        Args:
            profile: Column profile
            context: Additional context

        Returns:
            Tuned thresholds
        """
        context = context or {}

        preset = StrictnessPreset.for_strictness(self.config.strictness)
        context["strictness_preset"] = preset

        thresholds = self._strategy.tune_column(profile, context)

        # Apply strictness
        thresholds.null_threshold *= preset.null_multiplier
        thresholds.pattern_match_threshold = min(
            1.0,
            thresholds.pattern_match_threshold + (1 - thresholds.pattern_match_threshold) *
            (1 - preset.pattern_threshold)
        )

        return thresholds

    def _apply_strictness(
        self,
        thresholds: TableThresholds,
        preset: StrictnessPreset,
    ) -> None:
        """Apply strictness preset to thresholds."""
        for col_thresholds in thresholds.columns.values():
            # Adjust null threshold
            col_thresholds.null_threshold *= preset.null_multiplier

            # Adjust pattern threshold
            if col_thresholds.pattern_match_threshold < preset.pattern_threshold:
                col_thresholds.pattern_match_threshold = preset.pattern_threshold

            # Adjust range buffer
            if col_thresholds.min_value is not None and col_thresholds.max_value is not None:
                range_size = col_thresholds.max_value - col_thresholds.min_value
                buffer = range_size * preset.range_buffer

                col_thresholds.min_value -= buffer
                col_thresholds.max_value += buffer

    def compare_strategies(
        self,
        profile: TableProfile,
        strategies: list[str] | None = None,
    ) -> dict[str, TableThresholds]:
        """Compare thresholds from different strategies.

        Args:
            profile: Profile to analyze
            strategies: List of strategy names (None = all)

        Returns:
            Dictionary mapping strategy name to thresholds
        """
        if strategies is None:
            strategies = strategy_registry.list_strategies()

        results = {}
        for strategy_name in strategies:
            try:
                strategy = strategy_registry.get(strategy_name)
                thresholds = strategy.tune_table(profile, {})
                results[strategy_name] = thresholds
            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")

        return results


# =============================================================================
# A/B Testing Support
# =============================================================================


@dataclass
class ThresholdTestResult:
    """Result of A/B testing thresholds."""

    threshold_a: TableThresholds
    threshold_b: TableThresholds
    violations_a: int
    violations_b: int
    false_positives_a: int
    false_positives_b: int
    recommendation: str
    details: dict[str, Any] = field(default_factory=dict)


class ThresholdTester:
    """A/B test different threshold configurations.

    Compare how different thresholds perform on validation.
    """

    def __init__(self) -> None:
        pass

    def compare(
        self,
        data: pl.DataFrame,
        threshold_a: TableThresholds,
        threshold_b: TableThresholds,
        ground_truth: dict[str, bool] | None = None,
    ) -> ThresholdTestResult:
        """Compare two threshold configurations.

        Args:
            data: Data to validate
            threshold_a: First threshold set
            threshold_b: Second threshold set
            ground_truth: Optional ground truth (row valid/invalid)

        Returns:
            Comparison results
        """
        violations_a = self._count_violations(data, threshold_a)
        violations_b = self._count_violations(data, threshold_b)

        # If ground truth provided, calculate false positives
        fp_a = 0
        fp_b = 0

        if ground_truth is not None:
            # This would require more sophisticated tracking
            pass

        # Generate recommendation
        if violations_a < violations_b:
            if violations_a * 1.2 < violations_b:
                recommendation = f"Strong preference for A ({violations_a} vs {violations_b} violations)"
            else:
                recommendation = f"Slight preference for A ({violations_a} vs {violations_b} violations)"
        elif violations_b < violations_a:
            if violations_b * 1.2 < violations_a:
                recommendation = f"Strong preference for B ({violations_b} vs {violations_a} violations)"
            else:
                recommendation = f"Slight preference for B ({violations_b} vs {violations_a} violations)"
        else:
            recommendation = "No significant difference"

        return ThresholdTestResult(
            threshold_a=threshold_a,
            threshold_b=threshold_b,
            violations_a=violations_a,
            violations_b=violations_b,
            false_positives_a=fp_a,
            false_positives_b=fp_b,
            recommendation=recommendation,
        )

    def _count_violations(
        self,
        data: pl.DataFrame,
        thresholds: TableThresholds,
    ) -> int:
        """Count threshold violations in data."""
        violations = 0

        for col_name, col_thresh in thresholds.columns.items():
            if col_name not in data.columns:
                continue

            col = data.get_column(col_name)

            # Check null threshold
            null_ratio = col.null_count() / len(col) if len(col) > 0 else 0
            if null_ratio > col_thresh.null_threshold:
                violations += 1

            # Check range
            if col_thresh.min_value is not None:
                below_min = (col < col_thresh.min_value).sum()
                if below_min > 0:
                    violations += below_min

            if col_thresh.max_value is not None:
                above_max = (col > col_thresh.max_value).sum()
                if above_max > 0:
                    violations += above_max

        return violations


# =============================================================================
# Convenience Functions
# =============================================================================


def tune_thresholds(
    profile: TableProfile,
    strategy: str = "adaptive",
    strictness: Strictness = Strictness.MEDIUM,
) -> TableThresholds:
    """Tune thresholds for a profile.

    Args:
        profile: Profile to tune
        strategy: Tuning strategy
        strictness: Desired strictness level

    Returns:
        Tuned thresholds
    """
    config = TunerConfig(strategy=strategy, strictness=strictness)
    tuner = ThresholdTuner(strategy=strategy, config=config)
    return tuner.tune(profile)


def get_available_strategies() -> list[str]:
    """Get list of available tuning strategies."""
    return strategy_registry.list_strategies()


def create_tuner(
    strategy: str = "adaptive",
    strictness: str = "medium",
) -> ThresholdTuner:
    """Create a threshold tuner.

    Args:
        strategy: Strategy name
        strictness: Strictness level ("loose", "medium", "strict")

    Returns:
        Configured tuner
    """
    strictness_enum = Strictness(strictness)
    config = TunerConfig(strategy=strategy, strictness=strictness_enum)
    return ThresholdTuner(strategy=strategy, config=config)
