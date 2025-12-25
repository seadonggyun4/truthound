"""Rule quality scoring with precision/recall estimation.

This module provides comprehensive quality metrics for generated rules:
- Precision and recall estimation
- F1 score calculation
- Confidence scoring
- Rule validation against sample data
- Quality trend analysis

Key features:
- Pluggable quality estimator architecture
- Statistical sampling for large datasets
- Historical quality tracking
- Feedback loop integration

Example:
    from truthound.profiler.quality import (
        RuleQualityScorer,
        QualityMetrics,
        estimate_quality,
    )

    # Score a rule
    scorer = RuleQualityScorer()
    metrics = scorer.score(rule, data)

    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall: {metrics.recall:.2%}")
    print(f"F1 Score: {metrics.f1_score:.2%}")
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar

import polars as pl

from truthound.profiler.base import ColumnProfile, TableProfile, DataType


# =============================================================================
# Types and Enums
# =============================================================================


class QualityLevel(str, Enum):
    """Quality level classification."""

    EXCELLENT = "excellent"  # F1 >= 0.95
    GOOD = "good"           # F1 >= 0.85
    ACCEPTABLE = "acceptable"  # F1 >= 0.70
    POOR = "poor"           # F1 >= 0.50
    UNACCEPTABLE = "unacceptable"  # F1 < 0.50

    @classmethod
    def from_f1(cls, f1_score: float) -> "QualityLevel":
        """Determine quality level from F1 score."""
        if f1_score >= 0.95:
            return cls.EXCELLENT
        elif f1_score >= 0.85:
            return cls.GOOD
        elif f1_score >= 0.70:
            return cls.ACCEPTABLE
        elif f1_score >= 0.50:
            return cls.POOR
        else:
            return cls.UNACCEPTABLE


class RuleType(str, Enum):
    """Types of validation rules."""

    SCHEMA = "schema"
    FORMAT = "format"
    RANGE = "range"
    UNIQUENESS = "uniqueness"
    COMPLETENESS = "completeness"
    PATTERN = "pattern"
    CUSTOM = "custom"


# =============================================================================
# Quality Metrics
# =============================================================================


@dataclass(frozen=True)
class ConfusionMatrix:
    """Confusion matrix for rule evaluation."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def total(self) -> int:
        """Total observations."""
        return (
            self.true_positives + self.true_negatives +
            self.false_positives + self.false_negatives
        )

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total

    @property
    def precision(self) -> float:
        """Calculate precision (PPV)."""
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        """Calculate recall (sensitivity/TPR)."""
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 0.0
        return self.true_positives / denominator

    @property
    def specificity(self) -> float:
        """Calculate specificity (TNR)."""
        denominator = self.true_negatives + self.false_positives
        if denominator == 0:
            return 0.0
        return self.true_negatives / denominator

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def f_beta(self) -> Callable[[float], float]:
        """Calculate F-beta score with given beta."""
        def calc(beta: float) -> float:
            p, r = self.precision, self.recall
            if p + r == 0:
                return 0.0
            beta_sq = beta ** 2
            return (1 + beta_sq) * (p * r) / (beta_sq * p + r)
        return calc

    @property
    def mcc(self) -> float:
        """Calculate Matthews Correlation Coefficient."""
        tp, tn = self.true_positives, self.true_negatives
        fp, fn = self.false_positives, self.false_negatives

        numerator = tp * tn - fp * fn
        denominator = math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1_score": self.f1_score,
            "mcc": self.mcc,
        }


@dataclass
class QualityMetrics:
    """Complete quality metrics for a rule.

    Contains precision, recall, F1, and additional quality indicators.
    """

    # Core metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0

    # Additional metrics
    specificity: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient

    # Confidence intervals (95%)
    precision_ci: tuple[float, float] = (0.0, 0.0)
    recall_ci: tuple[float, float] = (0.0, 0.0)
    f1_ci: tuple[float, float] = (0.0, 0.0)

    # Sample info
    sample_size: int = 0
    population_size: int = 0

    # Quality assessment
    quality_level: QualityLevel = QualityLevel.UNACCEPTABLE
    confidence: float = 0.0  # Confidence in the metrics

    # Confusion matrix
    confusion_matrix: ConfusionMatrix | None = None

    # Metadata
    evaluated_at: datetime = field(default_factory=datetime.now)
    evaluation_duration_ms: float = 0.0

    @classmethod
    def from_confusion_matrix(
        cls,
        matrix: ConfusionMatrix,
        sample_size: int = 0,
        population_size: int = 0,
    ) -> "QualityMetrics":
        """Create metrics from confusion matrix."""
        metrics = cls(
            precision=matrix.precision,
            recall=matrix.recall,
            f1_score=matrix.f1_score,
            accuracy=matrix.accuracy,
            specificity=matrix.specificity,
            mcc=matrix.mcc,
            sample_size=sample_size,
            population_size=population_size,
            quality_level=QualityLevel.from_f1(matrix.f1_score),
            confusion_matrix=matrix,
        )

        # Calculate confidence intervals
        if sample_size > 0:
            metrics.precision_ci = cls._wilson_ci(
                matrix.true_positives,
                matrix.true_positives + matrix.false_positives,
            )
            metrics.recall_ci = cls._wilson_ci(
                matrix.true_positives,
                matrix.true_positives + matrix.false_negatives,
            )

            # Confidence based on sample size
            metrics.confidence = min(1.0, sample_size / max(population_size, 1))

        return metrics

    @staticmethod
    def _wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
        """Calculate Wilson confidence interval."""
        if trials == 0:
            return (0.0, 0.0)

        p = successes / trials
        denominator = 1 + z * z / trials
        centre = p + z * z / (2 * trials)
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)

        lower = max(0.0, (centre - margin) / denominator)
        upper = min(1.0, (centre + margin) / denominator)

        return (lower, upper)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
            "mcc": self.mcc,
            "precision_ci": self.precision_ci,
            "recall_ci": self.recall_ci,
            "f1_ci": self.f1_ci,
            "sample_size": self.sample_size,
            "population_size": self.population_size,
            "quality_level": self.quality_level.value,
            "confidence": self.confidence,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evaluation_duration_ms": self.evaluation_duration_ms,
        }

        if self.confusion_matrix:
            result["confusion_matrix"] = self.confusion_matrix.to_dict()

        return result


# =============================================================================
# Rule Protocol
# =============================================================================


class RuleProtocol(Protocol):
    """Protocol for validation rules."""

    name: str
    rule_type: RuleType
    column: str | None

    def validate(self, value: Any) -> bool:
        """Validate a single value."""
        ...

    def validate_column(self, df: pl.DataFrame, column: str) -> pl.Series:
        """Validate a column, returning boolean series."""
        ...


@dataclass
class ValidationRule:
    """Simple validation rule implementation."""

    name: str
    rule_type: RuleType
    column: str | None = None
    pattern: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: set[Any] | None = None
    nullable: bool = True
    validate_fn: Callable[[Any], bool] | None = None

    def validate(self, value: Any) -> bool:
        """Validate a single value."""
        if value is None:
            return self.nullable

        if self.validate_fn:
            return self.validate_fn(value)

        if self.pattern:
            if not isinstance(value, str):
                return False
            return bool(re.match(self.pattern, value))

        if self.min_value is not None and value < self.min_value:
            return False

        if self.max_value is not None and value > self.max_value:
            return False

        if self.allowed_values is not None and value not in self.allowed_values:
            return False

        return True

    def validate_column(self, df: pl.DataFrame, column: str) -> pl.Series:
        """Validate a column, returning boolean series."""
        col = df.get_column(column)

        # Handle nulls
        is_null = col.is_null()
        if self.nullable:
            valid = is_null  # Nulls are valid if nullable
        else:
            valid = ~is_null  # Nulls are invalid if not nullable

        # Apply rule-specific validation
        non_null = ~is_null

        if self.pattern:
            valid = valid | (non_null & col.cast(pl.Utf8).str.contains(self.pattern))

        elif self.min_value is not None or self.max_value is not None:
            if self.min_value is not None:
                valid = valid & (is_null | (col >= self.min_value))
            if self.max_value is not None:
                valid = valid & (is_null | (col <= self.max_value))

        elif self.allowed_values is not None:
            valid = valid | (non_null & col.is_in(list(self.allowed_values)))

        return valid


# =============================================================================
# Quality Estimator Protocol
# =============================================================================


class QualityEstimator(ABC):
    """Abstract base class for quality estimators.

    Different estimators use different strategies to estimate
    rule quality (sampling, heuristics, etc.)
    """

    name: str = "base"

    @abstractmethod
    def estimate(
        self,
        rule: RuleProtocol,
        data: pl.DataFrame,
        ground_truth: pl.Series | None = None,
    ) -> QualityMetrics:
        """Estimate quality metrics for a rule.

        Args:
            rule: Rule to evaluate
            data: Data to evaluate against
            ground_truth: Optional ground truth labels

        Returns:
            Quality metrics
        """
        pass


class SamplingQualityEstimator(QualityEstimator):
    """Estimates quality using statistical sampling.

    Uses random sampling to estimate precision and recall
    with confidence intervals.
    """

    name = "sampling"

    def __init__(
        self,
        sample_size: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int | None = None,
    ):
        self.sample_size = sample_size
        self.confidence_level = confidence_level
        self.random_seed = random_seed

    def estimate(
        self,
        rule: RuleProtocol,
        data: pl.DataFrame,
        ground_truth: pl.Series | None = None,
    ) -> QualityMetrics:
        """Estimate quality via sampling."""
        start_time = datetime.now()

        column = rule.column
        if column is None or column not in data.columns:
            return QualityMetrics()

        # Sample data if needed
        population_size = len(data)
        if population_size > self.sample_size:
            if self.random_seed is not None:
                random.seed(self.random_seed)
            indices = random.sample(range(population_size), self.sample_size)
            sample = data[indices]
            sample_size = self.sample_size
        else:
            sample = data
            sample_size = population_size

        # Validate sample
        predictions = rule.validate_column(sample, column)

        # If we have ground truth, calculate confusion matrix
        if ground_truth is not None:
            if len(ground_truth) > self.sample_size:
                gt_sample = ground_truth[indices] if population_size > self.sample_size else ground_truth
            else:
                gt_sample = ground_truth

            matrix = self._calculate_confusion_matrix(predictions, gt_sample)
        else:
            # Without ground truth, estimate based on data patterns
            matrix = self._estimate_confusion_matrix(predictions, sample, column)

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        metrics = QualityMetrics.from_confusion_matrix(
            matrix,
            sample_size=sample_size,
            population_size=population_size,
        )
        metrics.evaluation_duration_ms = duration_ms

        return metrics

    def _calculate_confusion_matrix(
        self,
        predictions: pl.Series,
        ground_truth: pl.Series,
    ) -> ConfusionMatrix:
        """Calculate confusion matrix from predictions and ground truth."""
        pred_array = predictions.to_numpy()
        truth_array = ground_truth.to_numpy()

        tp = int(((pred_array == True) & (truth_array == True)).sum())
        tn = int(((pred_array == False) & (truth_array == False)).sum())
        fp = int(((pred_array == True) & (truth_array == False)).sum())
        fn = int(((pred_array == False) & (truth_array == True)).sum())

        return ConfusionMatrix(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
        )

    def _estimate_confusion_matrix(
        self,
        predictions: pl.Series,
        data: pl.DataFrame,
        column: str,
    ) -> ConfusionMatrix:
        """Estimate confusion matrix without ground truth.

        Uses heuristics based on data distribution to estimate
        likely true/false positive rates.
        """
        valid_count = predictions.sum()
        invalid_count = len(predictions) - valid_count

        # Heuristic: assume most valid predictions are true positives
        # and most invalid predictions are true negatives
        # This is a simplification - actual FP/FN rates depend on the rule

        # Estimate FP rate based on rule strictness
        estimated_fp_rate = 0.02  # Conservative estimate
        estimated_fn_rate = 0.05  # Conservative estimate

        tp = int(valid_count * (1 - estimated_fp_rate))
        fp = int(valid_count * estimated_fp_rate)
        tn = int(invalid_count * (1 - estimated_fn_rate))
        fn = int(invalid_count * estimated_fn_rate)

        return ConfusionMatrix(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
        )


class HeuristicQualityEstimator(QualityEstimator):
    """Estimates quality using heuristics and data patterns.

    Useful when ground truth is not available and sampling
    is not practical.
    """

    name = "heuristic"

    def __init__(self, strictness: float = 0.5):
        self.strictness = strictness  # 0.0 = loose, 1.0 = strict

    def estimate(
        self,
        rule: RuleProtocol,
        data: pl.DataFrame,
        ground_truth: pl.Series | None = None,
    ) -> QualityMetrics:
        """Estimate quality using heuristics."""
        start_time = datetime.now()

        column = rule.column
        if column is None or column not in data.columns:
            return QualityMetrics()

        col = data.get_column(column)
        predictions = rule.validate_column(data, column)

        # Calculate base metrics
        valid_ratio = predictions.sum() / len(predictions)
        null_ratio = col.null_count() / len(col)
        unique_ratio = col.n_unique() / len(col)

        # Heuristic quality estimation based on rule type
        if rule.rule_type == RuleType.PATTERN:
            metrics = self._estimate_pattern_quality(
                valid_ratio, null_ratio, unique_ratio
            )
        elif rule.rule_type == RuleType.RANGE:
            metrics = self._estimate_range_quality(
                valid_ratio, null_ratio, col
            )
        elif rule.rule_type == RuleType.UNIQUENESS:
            metrics = self._estimate_uniqueness_quality(
                valid_ratio, unique_ratio
            )
        else:
            metrics = self._estimate_general_quality(
                valid_ratio, null_ratio
            )

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        metrics.sample_size = len(data)
        metrics.population_size = len(data)
        metrics.evaluation_duration_ms = duration_ms
        metrics.quality_level = QualityLevel.from_f1(metrics.f1_score)

        return metrics

    def _estimate_pattern_quality(
        self,
        valid_ratio: float,
        null_ratio: float,
        unique_ratio: float,
    ) -> QualityMetrics:
        """Estimate quality for pattern rules."""
        # Pattern rules with high match ratio are likely good
        # Unless the pattern is too generic (low uniqueness)

        if valid_ratio > 0.95:
            # Very high match - might be too loose
            precision = 0.85 - (valid_ratio - 0.95) * 2
            recall = 0.95
        elif valid_ratio > 0.80:
            # Good match ratio
            precision = 0.90
            recall = valid_ratio
        else:
            # Low match - might be too strict or wrong pattern
            precision = 0.95
            recall = valid_ratio

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            confidence=0.7,  # Heuristic confidence
        )

    def _estimate_range_quality(
        self,
        valid_ratio: float,
        null_ratio: float,
        col: pl.Series,
    ) -> QualityMetrics:
        """Estimate quality for range rules."""
        # Range rules are typically more reliable
        # Quality depends on how well the range fits the data distribution

        # Check if values are near boundaries (potential FN)
        try:
            if col.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                non_null = col.drop_nulls()
                if len(non_null) > 0:
                    std = non_null.std()
                    mean = non_null.mean()
                    # If std is large relative to mean, more uncertainty
                    cv = abs(std / mean) if mean != 0 else 0
                    precision = 0.95 if cv < 0.5 else 0.85
                else:
                    precision = 0.90
            else:
                precision = 0.90
        except Exception:
            precision = 0.90

        recall = valid_ratio
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            confidence=0.8,
        )

    def _estimate_uniqueness_quality(
        self,
        valid_ratio: float,
        unique_ratio: float,
    ) -> QualityMetrics:
        """Estimate quality for uniqueness rules."""
        # Uniqueness rules are binary - either unique or not
        # High precision if unique_ratio is very high
        if unique_ratio > 0.99:
            precision = 0.98
            recall = 0.95
        elif unique_ratio > 0.95:
            precision = 0.90
            recall = 0.90
        else:
            precision = 0.80
            recall = unique_ratio

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            confidence=0.85,
        )

    def _estimate_general_quality(
        self,
        valid_ratio: float,
        null_ratio: float,
    ) -> QualityMetrics:
        """Estimate quality for general rules."""
        # Default estimation
        precision = 0.90
        recall = valid_ratio * (1 - null_ratio)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            confidence=0.6,
        )


class CrossValidationEstimator(QualityEstimator):
    """Estimates quality using cross-validation.

    Splits data into folds and evaluates consistency across folds.
    """

    name = "cross_validation"

    def __init__(
        self,
        n_folds: int = 5,
        random_seed: int | None = None,
    ):
        self.n_folds = n_folds
        self.random_seed = random_seed

    def estimate(
        self,
        rule: RuleProtocol,
        data: pl.DataFrame,
        ground_truth: pl.Series | None = None,
    ) -> QualityMetrics:
        """Estimate quality via cross-validation."""
        start_time = datetime.now()

        column = rule.column
        if column is None or column not in data.columns:
            return QualityMetrics()

        # Create folds
        n = len(data)
        fold_size = n // self.n_folds

        if self.random_seed is not None:
            random.seed(self.random_seed)

        indices = list(range(n))
        random.shuffle(indices)

        # Evaluate on each fold
        fold_metrics: list[float] = []
        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_folds - 1 else n
            fold_indices = indices[start_idx:end_idx]

            fold_data = data[fold_indices]
            predictions = rule.validate_column(fold_data, column)
            valid_ratio = predictions.sum() / len(predictions)
            fold_metrics.append(valid_ratio)

        # Calculate consistency across folds
        mean_valid = sum(fold_metrics) / len(fold_metrics)
        std_valid = (sum((x - mean_valid) ** 2 for x in fold_metrics) / len(fold_metrics)) ** 0.5

        # Low variance = high consistency = likely high precision
        consistency = 1.0 - min(1.0, std_valid * 5)

        precision = 0.85 + consistency * 0.10
        recall = mean_valid
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return QualityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            confidence=consistency,
            sample_size=n,
            population_size=n,
            quality_level=QualityLevel.from_f1(f1),
            evaluation_duration_ms=duration_ms,
        )


# =============================================================================
# Quality Estimator Registry
# =============================================================================


class QualityEstimatorRegistry:
    """Registry for quality estimator factories."""

    def __init__(self) -> None:
        self._estimators: dict[str, type[QualityEstimator]] = {}

    def register(
        self,
        name: str,
        estimator_class: type[QualityEstimator],
    ) -> None:
        """Register an estimator class."""
        self._estimators[name] = estimator_class

    def create(self, name: str, **kwargs: Any) -> QualityEstimator:
        """Create an estimator instance."""
        if name not in self._estimators:
            raise KeyError(
                f"Unknown estimator: {name}. "
                f"Available: {list(self._estimators.keys())}"
            )
        return self._estimators[name](**kwargs)

    def list_estimators(self) -> list[str]:
        """List registered estimator names."""
        return list(self._estimators.keys())


# Global registry
quality_estimator_registry = QualityEstimatorRegistry()
quality_estimator_registry.register("sampling", SamplingQualityEstimator)
quality_estimator_registry.register("heuristic", HeuristicQualityEstimator)
quality_estimator_registry.register("cross_validation", CrossValidationEstimator)


# =============================================================================
# Rule Quality Scorer
# =============================================================================


@dataclass
class ScoringConfig:
    """Configuration for quality scoring."""

    estimator: str = "sampling"
    estimator_options: dict[str, Any] = field(default_factory=dict)
    min_sample_size: int = 100
    min_confidence: float = 0.5
    cache_results: bool = True
    cache_ttl_seconds: int = 3600


@dataclass
class RuleQualityScore:
    """Complete quality score for a rule."""

    rule_name: str
    rule_type: RuleType
    column: str | None
    metrics: QualityMetrics
    recommendation: str
    should_use: bool
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "column": self.column,
            "metrics": self.metrics.to_dict(),
            "recommendation": self.recommendation,
            "should_use": self.should_use,
            "alternatives": self.alternatives,
        }


class RuleQualityScorer:
    """Main interface for scoring rule quality.

    Evaluates rules against data and provides quality recommendations.

    Example:
        scorer = RuleQualityScorer()

        score = scorer.score(rule, data)
        print(f"Should use: {score.should_use}")
        print(f"Recommendation: {score.recommendation}")
    """

    def __init__(
        self,
        estimator: str | QualityEstimator = "sampling",
        estimator_options: dict[str, Any] | None = None,
        min_confidence: float = 0.5,
        quality_threshold: float = 0.70,
    ):
        """Initialize scorer.

        Args:
            estimator: Estimator name or instance
            estimator_options: Options for estimator construction
            min_confidence: Minimum confidence for recommendations
            quality_threshold: Minimum F1 score for rule acceptance
        """
        self.min_confidence = min_confidence
        self.quality_threshold = quality_threshold
        self._cache: dict[str, RuleQualityScore] = {}
        self._lock = threading.Lock()

        if isinstance(estimator, QualityEstimator):
            self._estimator = estimator
        else:
            options = estimator_options or {}
            self._estimator = quality_estimator_registry.create(estimator, **options)

    def score(
        self,
        rule: RuleProtocol | ValidationRule,
        data: pl.DataFrame,
        ground_truth: pl.Series | None = None,
        use_cache: bool = True,
    ) -> RuleQualityScore:
        """Score a rule's quality.

        Args:
            rule: Rule to score
            data: Data to evaluate against
            ground_truth: Optional ground truth labels
            use_cache: Whether to use cached results

        Returns:
            Complete quality score
        """
        # Check cache
        cache_key = self._make_cache_key(rule, data)
        if use_cache:
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]

        # Estimate metrics
        metrics = self._estimator.estimate(rule, data, ground_truth)

        # Generate recommendation
        recommendation, should_use = self._generate_recommendation(metrics, rule)

        # Create score
        score = RuleQualityScore(
            rule_name=rule.name,
            rule_type=rule.rule_type,
            column=rule.column,
            metrics=metrics,
            recommendation=recommendation,
            should_use=should_use,
        )

        # Cache result
        if use_cache:
            with self._lock:
                self._cache[cache_key] = score

        return score

    def score_all(
        self,
        rules: list[RuleProtocol | ValidationRule],
        data: pl.DataFrame,
        ground_truth: pl.Series | None = None,
    ) -> list[RuleQualityScore]:
        """Score multiple rules.

        Args:
            rules: Rules to score
            data: Data to evaluate against
            ground_truth: Optional ground truth labels

        Returns:
            List of quality scores
        """
        return [self.score(rule, data, ground_truth) for rule in rules]

    def compare(
        self,
        rules: list[RuleProtocol | ValidationRule],
        data: pl.DataFrame,
    ) -> list[RuleQualityScore]:
        """Compare multiple rules and rank by quality.

        Args:
            rules: Rules to compare
            data: Data to evaluate against

        Returns:
            Scores sorted by F1 score (best first)
        """
        scores = self.score_all(rules, data)
        return sorted(scores, key=lambda s: s.metrics.f1_score, reverse=True)

    def _generate_recommendation(
        self,
        metrics: QualityMetrics,
        rule: RuleProtocol | ValidationRule,
    ) -> tuple[str, bool]:
        """Generate recommendation based on metrics."""
        f1 = metrics.f1_score
        precision = metrics.precision
        recall = metrics.recall
        confidence = metrics.confidence

        # Check confidence
        if confidence < self.min_confidence:
            return (
                f"Low confidence ({confidence:.0%}). "
                "Consider collecting more data or using ground truth validation.",
                False,
            )

        # Check quality threshold
        if f1 >= self.quality_threshold:
            if f1 >= 0.95:
                return f"Excellent rule quality (F1={f1:.2%}). Safe to use.", True
            elif f1 >= 0.85:
                return f"Good rule quality (F1={f1:.2%}). Recommended for use.", True
            else:
                return f"Acceptable quality (F1={f1:.2%}). Monitor for issues.", True

        # Below threshold - provide specific advice
        if precision < recall:
            return (
                f"Low precision ({precision:.0%}). Rule may be too permissive. "
                "Consider stricter constraints.",
                False,
            )
        elif recall < precision:
            return (
                f"Low recall ({recall:.0%}). Rule may be too strict. "
                "Consider relaxing constraints or checking for edge cases.",
                False,
            )
        else:
            return (
                f"Poor overall quality (F1={f1:.2%}). "
                "Consider redesigning the rule or checking data quality.",
                False,
            )

    def _make_cache_key(
        self,
        rule: RuleProtocol | ValidationRule,
        data: pl.DataFrame,
    ) -> str:
        """Create cache key for rule + data combination."""
        rule_str = f"{rule.name}:{rule.rule_type}:{rule.column}"
        data_hash = hashlib.sha256(
            f"{len(data)}:{data.columns}".encode()
        ).hexdigest()[:16]
        return f"{rule_str}:{data_hash}"

    def clear_cache(self) -> None:
        """Clear the score cache."""
        with self._lock:
            self._cache.clear()


# =============================================================================
# Quality Trend Analyzer
# =============================================================================


@dataclass
class QualityTrendPoint:
    """Single point in quality trend."""

    timestamp: datetime
    metrics: QualityMetrics
    data_size: int
    notes: str = ""


class QualityTrendAnalyzer:
    """Analyzes quality trends over time.

    Tracks how rule quality changes as data evolves.

    Example:
        analyzer = QualityTrendAnalyzer()

        # Record quality over time
        analyzer.record(rule_name, metrics1, datetime.now())
        analyzer.record(rule_name, metrics2, datetime.now())

        # Analyze trend
        trend = analyzer.analyze_trend(rule_name)
        print(f"Quality is {trend.direction}")
    """

    def __init__(self, storage_path: str | Path | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._trends: dict[str, list[QualityTrendPoint]] = defaultdict(list)
        self._lock = threading.Lock()

        if self.storage_path and self.storage_path.exists():
            self._load()

    def record(
        self,
        rule_name: str,
        metrics: QualityMetrics,
        timestamp: datetime | None = None,
        data_size: int = 0,
        notes: str = "",
    ) -> None:
        """Record a quality measurement.

        Args:
            rule_name: Name of the rule
            metrics: Quality metrics
            timestamp: When measured (defaults to now)
            data_size: Size of data evaluated
            notes: Optional notes
        """
        point = QualityTrendPoint(
            timestamp=timestamp or datetime.now(),
            metrics=metrics,
            data_size=data_size,
            notes=notes,
        )

        with self._lock:
            self._trends[rule_name].append(point)
            # Keep last 100 points per rule
            if len(self._trends[rule_name]) > 100:
                self._trends[rule_name] = self._trends[rule_name][-100:]

        if self.storage_path:
            self._save()

    def analyze_trend(
        self,
        rule_name: str,
        window_days: int = 30,
    ) -> dict[str, Any]:
        """Analyze quality trend for a rule.

        Args:
            rule_name: Name of the rule
            window_days: Days to analyze

        Returns:
            Trend analysis results
        """
        with self._lock:
            points = self._trends.get(rule_name, [])

        if not points:
            return {"error": "No data available"}

        # Filter to window
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = [p for p in points if p.timestamp > cutoff]

        if len(recent) < 2:
            return {
                "current": points[-1].metrics.to_dict() if points else None,
                "trend": "insufficient_data",
            }

        # Calculate trend
        f1_values = [p.metrics.f1_score for p in recent]
        first_half = sum(f1_values[:len(f1_values)//2]) / (len(f1_values)//2)
        second_half = sum(f1_values[len(f1_values)//2:]) / (len(f1_values) - len(f1_values)//2)

        change = second_half - first_half
        if change > 0.05:
            direction = "improving"
        elif change < -0.05:
            direction = "degrading"
        else:
            direction = "stable"

        return {
            "current": recent[-1].metrics.to_dict(),
            "trend": direction,
            "change": change,
            "points_analyzed": len(recent),
            "oldest_point": recent[0].timestamp.isoformat(),
            "newest_point": recent[-1].timestamp.isoformat(),
            "f1_min": min(f1_values),
            "f1_max": max(f1_values),
            "f1_mean": sum(f1_values) / len(f1_values),
        }

    def get_history(
        self,
        rule_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get quality history for a rule.

        Args:
            rule_name: Name of the rule
            limit: Maximum points to return

        Returns:
            List of historical measurements
        """
        with self._lock:
            points = self._trends.get(rule_name, [])

        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "metrics": p.metrics.to_dict(),
                "data_size": p.data_size,
                "notes": p.notes,
            }
            for p in points[-limit:]
        ]

    def _save(self) -> None:
        """Save trends to storage."""
        if not self.storage_path:
            return

        data = {}
        with self._lock:
            for rule_name, points in self._trends.items():
                data[rule_name] = [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "metrics": p.metrics.to_dict(),
                        "data_size": p.data_size,
                        "notes": p.notes,
                    }
                    for p in points
                ]

        with open(self.storage_path, "w") as f:
            json.dump(data, f)

    def _load(self) -> None:
        """Load trends from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for rule_name, points in data.items():
                self._trends[rule_name] = [
                    QualityTrendPoint(
                        timestamp=datetime.fromisoformat(p["timestamp"]),
                        metrics=QualityMetrics(**{
                            k: v for k, v in p["metrics"].items()
                            if k in QualityMetrics.__dataclass_fields__
                            and k != "confusion_matrix"
                        }),
                        data_size=p.get("data_size", 0),
                        notes=p.get("notes", ""),
                    )
                    for p in points
                ]
        except Exception:
            pass


# =============================================================================
# Convenience Functions
# =============================================================================


def estimate_quality(
    rule: RuleProtocol | ValidationRule,
    data: pl.DataFrame,
    estimator: str = "sampling",
    **kwargs: Any,
) -> QualityMetrics:
    """Estimate quality metrics for a rule.

    Args:
        rule: Rule to evaluate
        data: Data to evaluate against
        estimator: Estimator type to use
        **kwargs: Estimator options

    Returns:
        Quality metrics
    """
    est = quality_estimator_registry.create(estimator, **kwargs)
    return est.estimate(rule, data)


def score_rule(
    rule: RuleProtocol | ValidationRule,
    data: pl.DataFrame,
    **kwargs: Any,
) -> RuleQualityScore:
    """Score a rule's quality.

    Args:
        rule: Rule to score
        data: Data to evaluate against
        **kwargs: Scorer options

    Returns:
        Complete quality score
    """
    scorer = RuleQualityScorer(**kwargs)
    return scorer.score(rule, data)


def compare_rules(
    rules: list[RuleProtocol | ValidationRule],
    data: pl.DataFrame,
    **kwargs: Any,
) -> list[RuleQualityScore]:
    """Compare multiple rules by quality.

    Args:
        rules: Rules to compare
        data: Data to evaluate against
        **kwargs: Scorer options

    Returns:
        Scores sorted by quality (best first)
    """
    scorer = RuleQualityScorer(**kwargs)
    return scorer.compare(rules, data)
