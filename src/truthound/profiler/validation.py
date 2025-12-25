"""Rule quality validation with labeled data support.

This module provides comprehensive quality validation using labeled datasets:
- Ground truth dataset management
- Statistical validation with confidence intervals
- Cross-validation and bootstrap methods
- A/B testing for rule comparison
- Labeled data collection and annotation

Key features:
- Pluggable validation strategy architecture
- Support for partial labeling (not all rows need labels)
- Integration with existing quality scoring
- Detailed validation reports with actionable insights

Example:
    from truthound.profiler.validation import (
        LabeledDataValidator,
        ValidationDataset,
        create_validation_suite,
    )

    # Create labeled dataset
    dataset = ValidationDataset.from_csv(
        "validation_data.csv",
        label_column="is_valid",
    )

    # Validate rule quality
    validator = LabeledDataValidator()
    result = validator.validate(rule, dataset)

    print(f"Precision: {result.precision:.2%}")
    print(f"Recall: {result.recall:.2%}")
    print(f"Confidence: {result.confidence:.2%}")
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Protocol, TypeVar

import polars as pl

from truthound.profiler.quality import (
    ConfusionMatrix,
    QualityLevel,
    QualityMetrics,
    RuleProtocol,
    RuleType,
    ValidationRule,
)


# =============================================================================
# Types and Enums
# =============================================================================


class LabelType(str, Enum):
    """Types of labels for validation data."""

    BINARY = "binary"  # True/False for valid/invalid
    CATEGORICAL = "categorical"  # Multiple categories
    ORDINAL = "ordinal"  # Ordered categories (e.g., quality scores)
    CONFIDENCE = "confidence"  # Probability/confidence scores


class ValidationMethod(str, Enum):
    """Validation methods available."""

    HOLDOUT = "holdout"  # Simple train/test split
    CROSS_VALIDATION = "cross_validation"  # K-fold cross-validation
    BOOTSTRAP = "bootstrap"  # Bootstrap resampling
    TEMPORAL = "temporal"  # Time-based validation
    STRATIFIED = "stratified"  # Stratified sampling


class ValidationStatus(str, Enum):
    """Status of validation result."""

    PASSED = "passed"  # Rule meets quality threshold
    FAILED = "failed"  # Rule below quality threshold
    INCONCLUSIVE = "inconclusive"  # Not enough data/confidence
    DEGRADED = "degraded"  # Quality dropped from previous


# =============================================================================
# Labeled Data Management
# =============================================================================


@dataclass
class LabeledRow:
    """A single labeled data point."""

    row_id: str | int
    data: dict[str, Any]
    label: bool | str | float
    label_type: LabelType = LabelType.BINARY
    confidence: float = 1.0  # Confidence in the label
    source: str = ""  # Where the label came from
    annotated_at: datetime = field(default_factory=datetime.now)
    annotated_by: str = ""
    notes: str = ""


@dataclass
class ValidationDataset:
    """Dataset with labeled ground truth.

    Manages labeled data for validating rule quality.

    Attributes:
        name: Dataset name
        rows: Labeled data rows
        label_type: Type of labels
        label_column: Name of label column
        metadata: Additional metadata
    """

    name: str
    rows: list[LabeledRow] = field(default_factory=list)
    label_type: LabelType = LabelType.BINARY
    label_column: str = "is_valid"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[LabeledRow]:
        return iter(self.rows)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        label_column: str,
        name: str = "validation_set",
        id_column: str | None = None,
        label_type: LabelType = LabelType.BINARY,
    ) -> "ValidationDataset":
        """Create dataset from a Polars DataFrame.

        Args:
            df: DataFrame with data and labels
            label_column: Column containing labels
            name: Dataset name
            id_column: Column to use as row ID (uses index if None)
            label_type: Type of labels

        Returns:
            ValidationDataset instance
        """
        rows = []
        for i, row in enumerate(df.iter_rows(named=True)):
            label = row.pop(label_column) if label_column in row else None
            row_id = row.get(id_column, i) if id_column else i

            rows.append(LabeledRow(
                row_id=row_id,
                data=row,
                label=label,
                label_type=label_type,
            ))

        return cls(
            name=name,
            rows=rows,
            label_type=label_type,
            label_column=label_column,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        label_column: str,
        name: str | None = None,
        **kwargs: Any,
    ) -> "ValidationDataset":
        """Load dataset from CSV file.

        Args:
            path: Path to CSV file
            label_column: Column containing labels
            name: Dataset name (uses filename if None)
            **kwargs: Additional arguments for from_dataframe

        Returns:
            ValidationDataset instance
        """
        path = Path(path)
        df = pl.read_csv(path)
        return cls.from_dataframe(
            df,
            label_column=label_column,
            name=name or path.stem,
            **kwargs,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "ValidationDataset":
        """Load dataset from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            ValidationDataset instance
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        rows = [
            LabeledRow(
                row_id=r["row_id"],
                data=r["data"],
                label=r["label"],
                label_type=LabelType(r.get("label_type", "binary")),
                confidence=r.get("confidence", 1.0),
                source=r.get("source", ""),
                annotated_at=datetime.fromisoformat(r["annotated_at"])
                if "annotated_at" in r else datetime.now(),
                annotated_by=r.get("annotated_by", ""),
                notes=r.get("notes", ""),
            )
            for r in data.get("rows", [])
        ]

        return cls(
            name=data.get("name", path.stem),
            rows=rows,
            label_type=LabelType(data.get("label_type", "binary")),
            label_column=data.get("label_column", "is_valid"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data else datetime.now(),
            version=data.get("version", "1.0"),
        )

    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame with labels.

        Returns:
            DataFrame with data and label column
        """
        if not self.rows:
            return pl.DataFrame()

        # Collect all data
        data_dicts = [row.data for row in self.rows]
        labels = [row.label for row in self.rows]

        # Create DataFrame
        df = pl.DataFrame(data_dicts)
        df = df.with_columns(pl.Series(self.label_column, labels))

        return df

    def to_json(self, path: str | Path) -> None:
        """Save dataset to JSON file.

        Args:
            path: Output path
        """
        data = {
            "name": self.name,
            "label_type": self.label_type.value,
            "label_column": self.label_column,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "rows": [
                {
                    "row_id": r.row_id,
                    "data": r.data,
                    "label": r.label,
                    "label_type": r.label_type.value,
                    "confidence": r.confidence,
                    "source": r.source,
                    "annotated_at": r.annotated_at.isoformat(),
                    "annotated_by": r.annotated_by,
                    "notes": r.notes,
                }
                for r in self.rows
            ],
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def split(
        self,
        test_ratio: float = 0.2,
        random_seed: int | None = None,
        stratify: bool = True,
    ) -> tuple["ValidationDataset", "ValidationDataset"]:
        """Split dataset into training and test sets.

        Args:
            test_ratio: Ratio of data for test set
            random_seed: Random seed for reproducibility
            stratify: Whether to stratify by label

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if random_seed is not None:
            random.seed(random_seed)

        if stratify and self.label_type == LabelType.BINARY:
            # Stratified split
            positive = [r for r in self.rows if r.label]
            negative = [r for r in self.rows if not r.label]

            random.shuffle(positive)
            random.shuffle(negative)

            n_pos_test = max(1, int(len(positive) * test_ratio))
            n_neg_test = max(1, int(len(negative) * test_ratio))

            test_rows = positive[:n_pos_test] + negative[:n_neg_test]
            train_rows = positive[n_pos_test:] + negative[n_neg_test:]
        else:
            # Random split
            rows = list(self.rows)
            random.shuffle(rows)
            n_test = max(1, int(len(rows) * test_ratio))
            test_rows = rows[:n_test]
            train_rows = rows[n_test:]

        train_ds = ValidationDataset(
            name=f"{self.name}_train",
            rows=train_rows,
            label_type=self.label_type,
            label_column=self.label_column,
            metadata={**self.metadata, "split": "train"},
        )

        test_ds = ValidationDataset(
            name=f"{self.name}_test",
            rows=test_rows,
            label_type=self.label_type,
            label_column=self.label_column,
            metadata={**self.metadata, "split": "test"},
        )

        return train_ds, test_ds

    def get_folds(
        self,
        n_folds: int = 5,
        random_seed: int | None = None,
    ) -> Iterator[tuple["ValidationDataset", "ValidationDataset"]]:
        """Generate k-fold cross-validation splits.

        Args:
            n_folds: Number of folds
            random_seed: Random seed for reproducibility

        Yields:
            Tuples of (train_fold, test_fold)
        """
        if random_seed is not None:
            random.seed(random_seed)

        rows = list(self.rows)
        random.shuffle(rows)

        fold_size = len(rows) // n_folds

        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else len(rows)

            test_rows = rows[start:end]
            train_rows = rows[:start] + rows[end:]

            train_ds = ValidationDataset(
                name=f"{self.name}_fold{i}_train",
                rows=train_rows,
                label_type=self.label_type,
                label_column=self.label_column,
            )

            test_ds = ValidationDataset(
                name=f"{self.name}_fold{i}_test",
                rows=test_rows,
                label_type=self.label_type,
                label_column=self.label_column,
            )

            yield train_ds, test_ds

    def filter_by_confidence(
        self,
        min_confidence: float = 0.8,
    ) -> "ValidationDataset":
        """Filter to high-confidence labels only.

        Args:
            min_confidence: Minimum label confidence

        Returns:
            Filtered dataset
        """
        filtered_rows = [r for r in self.rows if r.confidence >= min_confidence]
        return ValidationDataset(
            name=f"{self.name}_high_confidence",
            rows=filtered_rows,
            label_type=self.label_type,
            label_column=self.label_column,
            metadata={**self.metadata, "min_confidence": min_confidence},
        )

    def get_label_distribution(self) -> dict[Any, int]:
        """Get distribution of labels.

        Returns:
            Dictionary mapping labels to counts
        """
        distribution: dict[Any, int] = defaultdict(int)
        for row in self.rows:
            distribution[row.label] += 1
        return dict(distribution)


# =============================================================================
# Validation Results
# =============================================================================


@dataclass
class ValidationResult:
    """Comprehensive validation result.

    Contains detailed metrics, confidence intervals, and recommendations.
    """

    # Core metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0

    # Confidence intervals (95%)
    precision_ci: tuple[float, float] = (0.0, 1.0)
    recall_ci: tuple[float, float] = (0.0, 1.0)
    f1_ci: tuple[float, float] = (0.0, 1.0)

    # Validation metadata
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    confidence: float = 0.0

    # Confusion matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Status and recommendations
    status: ValidationStatus = ValidationStatus.INCONCLUSIVE
    quality_level: QualityLevel = QualityLevel.UNACCEPTABLE
    recommendation: str = ""
    warnings: list[str] = field(default_factory=list)

    # Method details
    validation_method: ValidationMethod = ValidationMethod.HOLDOUT
    method_details: dict[str, Any] = field(default_factory=dict)

    # Timing
    validated_at: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    @property
    def confusion_matrix(self) -> ConfusionMatrix:
        """Get confusion matrix."""
        return ConfusionMatrix(
            true_positives=self.true_positives,
            true_negatives=self.true_negatives,
            false_positives=self.false_positives,
            false_negatives=self.false_negatives,
        )

    @property
    def specificity(self) -> float:
        """Calculate specificity."""
        if self.true_negatives + self.false_positives == 0:
            return 0.0
        return self.true_negatives / (self.true_negatives + self.false_positives)

    @property
    def mcc(self) -> float:
        """Calculate Matthews Correlation Coefficient."""
        return self.confusion_matrix.mcc

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "precision_ci": self.precision_ci,
            "recall_ci": self.recall_ci,
            "f1_ci": self.f1_ci,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "confidence": self.confidence,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
            },
            "status": self.status.value,
            "quality_level": self.quality_level.value,
            "recommendation": self.recommendation,
            "warnings": self.warnings,
            "validation_method": self.validation_method.value,
            "method_details": self.method_details,
            "validated_at": self.validated_at.isoformat(),
            "duration_ms": self.duration_ms,
        }

    def to_quality_metrics(self) -> QualityMetrics:
        """Convert to QualityMetrics for compatibility."""
        return QualityMetrics.from_confusion_matrix(
            self.confusion_matrix,
            sample_size=self.n_samples,
            population_size=self.n_samples,
        )


@dataclass
class ValidationReport:
    """Complete validation report for multiple rules."""

    rule_results: dict[str, ValidationResult] = field(default_factory=dict)
    overall_status: ValidationStatus = ValidationStatus.INCONCLUSIVE
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    dataset_info: dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def add_result(self, rule_name: str, result: ValidationResult) -> None:
        """Add a rule validation result."""
        self.rule_results[rule_name] = result
        self._update_overall_status()

    def _update_overall_status(self) -> None:
        """Update overall status based on individual results."""
        if not self.rule_results:
            self.overall_status = ValidationStatus.INCONCLUSIVE
            return

        statuses = [r.status for r in self.rule_results.values()]

        if all(s == ValidationStatus.PASSED for s in statuses):
            self.overall_status = ValidationStatus.PASSED
        elif any(s == ValidationStatus.FAILED for s in statuses):
            self.overall_status = ValidationStatus.FAILED
        elif any(s == ValidationStatus.DEGRADED for s in statuses):
            self.overall_status = ValidationStatus.DEGRADED
        else:
            self.overall_status = ValidationStatus.INCONCLUSIVE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_results": {
                name: result.to_dict()
                for name, result in self.rule_results.items()
            },
            "overall_status": self.overall_status.value,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "dataset_info": self.dataset_info,
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# Validation Strategies
# =============================================================================


class ValidationStrategy(ABC):
    """Abstract base class for validation strategies."""

    name: str = "base"

    @abstractmethod
    def validate(
        self,
        rule: RuleProtocol | ValidationRule,
        dataset: ValidationDataset,
        column: str,
    ) -> ValidationResult:
        """Validate a rule against labeled data.

        Args:
            rule: Rule to validate
            dataset: Labeled validation dataset
            column: Column to validate

        Returns:
            Validation result
        """
        pass


class HoldoutValidation(ValidationStrategy):
    """Simple holdout validation strategy."""

    name = "holdout"

    def __init__(
        self,
        quality_threshold: float = 0.70,
        min_samples: int = 30,
    ):
        self.quality_threshold = quality_threshold
        self.min_samples = min_samples

    def validate(
        self,
        rule: RuleProtocol | ValidationRule,
        dataset: ValidationDataset,
        column: str,
    ) -> ValidationResult:
        """Validate using holdout method."""
        start_time = datetime.now()

        if len(dataset) < self.min_samples:
            return ValidationResult(
                status=ValidationStatus.INCONCLUSIVE,
                recommendation=f"Need at least {self.min_samples} samples, got {len(dataset)}",
                n_samples=len(dataset),
            )

        # Convert to DataFrame
        df = dataset.to_dataframe()

        # Get predictions
        predictions = rule.validate_column(df, column)

        # Get ground truth
        ground_truth = df.get_column(dataset.label_column)

        # Calculate confusion matrix
        result = self._calculate_metrics(
            predictions.to_list(),
            ground_truth.to_list(),
        )

        # Determine status
        if result.f1_score >= self.quality_threshold:
            result.status = ValidationStatus.PASSED
            result.recommendation = (
                f"Rule meets quality threshold (F1={result.f1_score:.2%} >= {self.quality_threshold:.0%})"
            )
        else:
            result.status = ValidationStatus.FAILED
            result.recommendation = (
                f"Rule below quality threshold (F1={result.f1_score:.2%} < {self.quality_threshold:.0%})"
            )

        result.quality_level = QualityLevel.from_f1(result.f1_score)
        result.validation_method = ValidationMethod.HOLDOUT
        result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result

    def _calculate_metrics(
        self,
        predictions: list[bool],
        ground_truth: list[bool],
    ) -> ValidationResult:
        """Calculate validation metrics."""
        tp = tn = fp = fn = 0

        for pred, truth in zip(predictions, ground_truth):
            if pred and truth:
                tp += 1
            elif not pred and not truth:
                tn += 1
            elif pred and not truth:
                fp += 1
            else:
                fn += 1

        n = len(predictions)
        n_pos = sum(1 for t in ground_truth if t)
        n_neg = n - n_pos

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n if n > 0 else 0.0

        # Calculate confidence intervals using Wilson score
        precision_ci = self._wilson_ci(tp, tp + fp)
        recall_ci = self._wilson_ci(tp, tp + fn)

        return ValidationResult(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            precision_ci=precision_ci,
            recall_ci=recall_ci,
            n_samples=n,
            n_positive=n_pos,
            n_negative=n_neg,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            confidence=min(1.0, n / 100),  # Higher confidence with more samples
        )

    @staticmethod
    def _wilson_ci(
        successes: int,
        trials: int,
        z: float = 1.96,
    ) -> tuple[float, float]:
        """Calculate Wilson confidence interval."""
        if trials == 0:
            return (0.0, 1.0)

        p = successes / trials
        denominator = 1 + z * z / trials
        centre = p + z * z / (2 * trials)
        margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)

        lower = max(0.0, (centre - margin) / denominator)
        upper = min(1.0, (centre + margin) / denominator)

        return (lower, upper)


class CrossValidationStrategy(ValidationStrategy):
    """K-fold cross-validation strategy."""

    name = "cross_validation"

    def __init__(
        self,
        n_folds: int = 5,
        quality_threshold: float = 0.70,
        random_seed: int | None = None,
    ):
        self.n_folds = n_folds
        self.quality_threshold = quality_threshold
        self.random_seed = random_seed

    def validate(
        self,
        rule: RuleProtocol | ValidationRule,
        dataset: ValidationDataset,
        column: str,
    ) -> ValidationResult:
        """Validate using k-fold cross-validation."""
        start_time = datetime.now()

        fold_metrics: list[dict[str, float]] = []

        for train_ds, test_ds in dataset.get_folds(
            n_folds=self.n_folds,
            random_seed=self.random_seed,
        ):
            df = test_ds.to_dataframe()
            predictions = rule.validate_column(df, column)
            ground_truth = df.get_column(dataset.label_column)

            # Calculate fold metrics
            metrics = self._calculate_fold_metrics(
                predictions.to_list(),
                ground_truth.to_list(),
            )
            fold_metrics.append(metrics)

        # Aggregate across folds
        result = self._aggregate_folds(fold_metrics, len(dataset))

        # Determine status
        if result.f1_score >= self.quality_threshold:
            result.status = ValidationStatus.PASSED
        else:
            result.status = ValidationStatus.FAILED

        result.quality_level = QualityLevel.from_f1(result.f1_score)
        result.validation_method = ValidationMethod.CROSS_VALIDATION
        result.method_details = {
            "n_folds": self.n_folds,
            "fold_f1_scores": [m["f1"] for m in fold_metrics],
        }
        result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result

    def _calculate_fold_metrics(
        self,
        predictions: list[bool],
        ground_truth: list[bool],
    ) -> dict[str, float]:
        """Calculate metrics for a single fold."""
        tp = tn = fp = fn = 0

        for pred, truth in zip(predictions, ground_truth):
            if pred and truth:
                tp += 1
            elif not pred and not truth:
                tn += 1
            elif pred and not truth:
                fp += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def _aggregate_folds(
        self,
        fold_metrics: list[dict[str, float]],
        total_samples: int,
    ) -> ValidationResult:
        """Aggregate metrics across folds."""
        precisions = [m["precision"] for m in fold_metrics]
        recalls = [m["recall"] for m in fold_metrics]
        f1s = [m["f1"] for m in fold_metrics]

        # Calculate mean and confidence intervals
        mean_precision = statistics.mean(precisions)
        mean_recall = statistics.mean(recalls)
        mean_f1 = statistics.mean(f1s)

        std_precision = statistics.stdev(precisions) if len(precisions) > 1 else 0.0
        std_recall = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
        std_f1 = statistics.stdev(f1s) if len(f1s) > 1 else 0.0

        # 95% CI = mean +/- 1.96 * std / sqrt(n)
        z = 1.96
        n = len(fold_metrics)
        margin_precision = z * std_precision / math.sqrt(n) if n > 0 else 0.0
        margin_recall = z * std_recall / math.sqrt(n) if n > 0 else 0.0
        margin_f1 = z * std_f1 / math.sqrt(n) if n > 0 else 0.0

        # Sum confusion matrix across folds
        total_tp = sum(int(m["tp"]) for m in fold_metrics)
        total_tn = sum(int(m["tn"]) for m in fold_metrics)
        total_fp = sum(int(m["fp"]) for m in fold_metrics)
        total_fn = sum(int(m["fn"]) for m in fold_metrics)

        # Calculate consistency-based confidence
        f1_cv = std_f1 / mean_f1 if mean_f1 > 0 else 1.0
        confidence = max(0.0, 1.0 - f1_cv)

        return ValidationResult(
            precision=mean_precision,
            recall=mean_recall,
            f1_score=mean_f1,
            accuracy=(total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
            if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0,
            precision_ci=(
                max(0.0, mean_precision - margin_precision),
                min(1.0, mean_precision + margin_precision),
            ),
            recall_ci=(
                max(0.0, mean_recall - margin_recall),
                min(1.0, mean_recall + margin_recall),
            ),
            f1_ci=(
                max(0.0, mean_f1 - margin_f1),
                min(1.0, mean_f1 + margin_f1),
            ),
            n_samples=total_samples,
            true_positives=total_tp,
            true_negatives=total_tn,
            false_positives=total_fp,
            false_negatives=total_fn,
            confidence=confidence,
            recommendation=f"Cross-validated F1: {mean_f1:.2%} (±{margin_f1:.2%})",
        )


class BootstrapValidation(ValidationStrategy):
    """Bootstrap resampling validation strategy."""

    name = "bootstrap"

    def __init__(
        self,
        n_iterations: int = 1000,
        sample_ratio: float = 0.8,
        quality_threshold: float = 0.70,
        random_seed: int | None = None,
    ):
        self.n_iterations = n_iterations
        self.sample_ratio = sample_ratio
        self.quality_threshold = quality_threshold
        self.random_seed = random_seed

    def validate(
        self,
        rule: RuleProtocol | ValidationRule,
        dataset: ValidationDataset,
        column: str,
    ) -> ValidationResult:
        """Validate using bootstrap resampling."""
        start_time = datetime.now()

        if self.random_seed is not None:
            random.seed(self.random_seed)

        df = dataset.to_dataframe()
        n = len(df)
        sample_size = int(n * self.sample_ratio)

        # Collect bootstrap samples
        f1_scores: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []

        for _ in range(self.n_iterations):
            # Sample with replacement
            indices = [random.randint(0, n - 1) for _ in range(sample_size)]
            sample_df = df[indices]

            predictions = rule.validate_column(sample_df, column)
            ground_truth = sample_df.get_column(dataset.label_column)

            metrics = self._calculate_metrics(
                predictions.to_list(),
                ground_truth.to_list(),
            )

            f1_scores.append(metrics["f1"])
            precisions.append(metrics["precision"])
            recalls.append(metrics["recall"])

        # Calculate percentile confidence intervals
        f1_scores.sort()
        precisions.sort()
        recalls.sort()

        lower_idx = int(0.025 * self.n_iterations)
        upper_idx = int(0.975 * self.n_iterations)

        result = ValidationResult(
            precision=statistics.mean(precisions),
            recall=statistics.mean(recalls),
            f1_score=statistics.mean(f1_scores),
            precision_ci=(precisions[lower_idx], precisions[upper_idx]),
            recall_ci=(recalls[lower_idx], recalls[upper_idx]),
            f1_ci=(f1_scores[lower_idx], f1_scores[upper_idx]),
            n_samples=n,
            confidence=1.0 - statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.5,
            validation_method=ValidationMethod.BOOTSTRAP,
            method_details={
                "n_iterations": self.n_iterations,
                "sample_ratio": self.sample_ratio,
            },
        )

        if result.f1_score >= self.quality_threshold:
            result.status = ValidationStatus.PASSED
        else:
            result.status = ValidationStatus.FAILED

        result.quality_level = QualityLevel.from_f1(result.f1_score)
        result.recommendation = (
            f"Bootstrap F1: {result.f1_score:.2%} "
            f"(95% CI: [{result.f1_ci[0]:.2%}, {result.f1_ci[1]:.2%}])"
        )
        result.duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        return result

    def _calculate_metrics(
        self,
        predictions: list[bool],
        ground_truth: list[bool],
    ) -> dict[str, float]:
        """Calculate metrics for a bootstrap sample."""
        tp = tn = fp = fn = 0

        for pred, truth in zip(predictions, ground_truth):
            if pred and truth:
                tp += 1
            elif not pred and not truth:
                tn += 1
            elif pred and not truth:
                fp += 1
            else:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# Strategy Registry
# =============================================================================


class ValidationStrategyRegistry:
    """Registry for validation strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, type[ValidationStrategy]] = {}

    def register(
        self,
        name: str,
        strategy_class: type[ValidationStrategy],
    ) -> None:
        """Register a validation strategy."""
        self._strategies[name] = strategy_class

    def create(self, name: str, **kwargs: Any) -> ValidationStrategy:
        """Create a strategy instance."""
        if name not in self._strategies:
            raise KeyError(
                f"Unknown strategy: {name}. "
                f"Available: {list(self._strategies.keys())}"
            )
        return self._strategies[name](**kwargs)

    def list_strategies(self) -> list[str]:
        """List registered strategy names."""
        return list(self._strategies.keys())


# Global registry
validation_strategy_registry = ValidationStrategyRegistry()
validation_strategy_registry.register("holdout", HoldoutValidation)
validation_strategy_registry.register("cross_validation", CrossValidationStrategy)
validation_strategy_registry.register("bootstrap", BootstrapValidation)


# =============================================================================
# Main Validator
# =============================================================================


class LabeledDataValidator:
    """Main validator using labeled ground truth data.

    Provides comprehensive rule quality validation with:
    - Multiple validation strategies
    - Detailed confidence intervals
    - Actionable recommendations

    Example:
        validator = LabeledDataValidator(strategy="cross_validation")

        # Validate single rule
        result = validator.validate(rule, dataset, "email_column")
        print(f"F1: {result.f1_score:.2%}")

        # Validate multiple rules
        report = validator.validate_all(rules, dataset)
        print(report.overall_status)
    """

    def __init__(
        self,
        strategy: str | ValidationStrategy = "cross_validation",
        strategy_options: dict[str, Any] | None = None,
        quality_threshold: float = 0.70,
        min_samples: int = 30,
    ):
        """Initialize validator.

        Args:
            strategy: Validation strategy name or instance
            strategy_options: Options for strategy construction
            quality_threshold: Minimum F1 for passing
            min_samples: Minimum samples required
        """
        self.quality_threshold = quality_threshold
        self.min_samples = min_samples
        self._cache: dict[str, ValidationResult] = {}
        self._lock = threading.Lock()

        if isinstance(strategy, ValidationStrategy):
            self._strategy = strategy
        else:
            options = strategy_options or {}
            options.setdefault("quality_threshold", quality_threshold)
            self._strategy = validation_strategy_registry.create(strategy, **options)

    def validate(
        self,
        rule: RuleProtocol | ValidationRule,
        dataset: ValidationDataset,
        column: str | None = None,
        use_cache: bool = True,
    ) -> ValidationResult:
        """Validate a rule against labeled data.

        Args:
            rule: Rule to validate
            dataset: Labeled validation dataset
            column: Column to validate (uses rule.column if None)
            use_cache: Whether to use cached results

        Returns:
            Validation result
        """
        column = column or rule.column
        if column is None:
            return ValidationResult(
                status=ValidationStatus.INCONCLUSIVE,
                recommendation="No column specified for validation",
            )

        # Check cache
        cache_key = self._make_cache_key(rule, dataset, column)
        if use_cache:
            with self._lock:
                if cache_key in self._cache:
                    return self._cache[cache_key]

        # Check sample size
        if len(dataset) < self.min_samples:
            return ValidationResult(
                status=ValidationStatus.INCONCLUSIVE,
                recommendation=f"Need at least {self.min_samples} samples",
                n_samples=len(dataset),
                warnings=[f"Only {len(dataset)} samples available"],
            )

        # Validate
        result = self._strategy.validate(rule, dataset, column)

        # Add warnings for edge cases
        if result.n_positive < 10 or result.n_negative < 10:
            result.warnings.append(
                "Imbalanced labels may affect reliability"
            )

        if result.confidence < 0.7:
            result.warnings.append(
                "Low confidence - consider collecting more labels"
            )

        # Cache result
        if use_cache:
            with self._lock:
                self._cache[cache_key] = result

        return result

    def validate_all(
        self,
        rules: list[RuleProtocol | ValidationRule],
        dataset: ValidationDataset,
    ) -> ValidationReport:
        """Validate multiple rules.

        Args:
            rules: Rules to validate
            dataset: Labeled validation dataset

        Returns:
            Complete validation report
        """
        report = ValidationReport(
            dataset_info={
                "name": dataset.name,
                "size": len(dataset),
                "label_distribution": dataset.get_label_distribution(),
            },
        )

        passed = 0
        failed = 0

        for rule in rules:
            result = self.validate(rule, dataset)
            report.add_result(rule.name, result)

            if result.status == ValidationStatus.PASSED:
                passed += 1
            elif result.status == ValidationStatus.FAILED:
                failed += 1

        # Generate summary
        report.summary = (
            f"Validated {len(rules)} rules: "
            f"{passed} passed, {failed} failed, "
            f"{len(rules) - passed - failed} inconclusive"
        )

        # Generate recommendations
        if failed > 0:
            report.recommendations.append(
                f"{failed} rules failed validation - review and adjust thresholds or rule logic"
            )

        low_confidence = [
            name for name, result in report.rule_results.items()
            if result.confidence < 0.7
        ]
        if low_confidence:
            report.recommendations.append(
                f"Low confidence for rules: {', '.join(low_confidence)} - collect more labels"
            )

        return report

    def compare_rules(
        self,
        rules: list[RuleProtocol | ValidationRule],
        dataset: ValidationDataset,
        column: str,
    ) -> list[tuple[str, ValidationResult]]:
        """Compare multiple rules for the same column.

        Args:
            rules: Rules to compare
            dataset: Labeled validation dataset
            column: Column to validate

        Returns:
            Rules sorted by F1 score (best first)
        """
        results = [
            (rule.name, self.validate(rule, dataset, column))
            for rule in rules
        ]
        return sorted(results, key=lambda x: x[1].f1_score, reverse=True)

    def _make_cache_key(
        self,
        rule: RuleProtocol | ValidationRule,
        dataset: ValidationDataset,
        column: str,
    ) -> str:
        """Create cache key."""
        rule_str = f"{rule.name}:{rule.rule_type}"
        dataset_hash = hashlib.sha256(
            f"{dataset.name}:{len(dataset)}:{dataset.version}".encode()
        ).hexdigest()[:16]
        return f"{rule_str}:{column}:{dataset_hash}"

    def clear_cache(self) -> None:
        """Clear validation cache."""
        with self._lock:
            self._cache.clear()


# =============================================================================
# A/B Testing
# =============================================================================


@dataclass
class ABTestResult:
    """Result of A/B testing between two rules."""

    rule_a_name: str
    rule_b_name: str
    rule_a_f1: float
    rule_b_f1: float
    difference: float
    p_value: float
    significant: bool
    winner: str | None
    confidence_level: float = 0.95
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_a_name": self.rule_a_name,
            "rule_b_name": self.rule_b_name,
            "rule_a_f1": self.rule_a_f1,
            "rule_b_f1": self.rule_b_f1,
            "difference": self.difference,
            "p_value": self.p_value,
            "significant": self.significant,
            "winner": self.winner,
            "confidence_level": self.confidence_level,
            "recommendation": self.recommendation,
        }


class RuleABTester:
    """A/B testing for comparing rule quality.

    Performs statistical tests to determine if one rule
    is significantly better than another.

    Example:
        tester = RuleABTester()

        result = tester.test(rule_a, rule_b, dataset, "column")

        if result.significant:
            print(f"Winner: {result.winner}")
        else:
            print("No significant difference")
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int | None = None,
    ):
        """Initialize A/B tester.

        Args:
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level for significance
            random_seed: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.random_seed = random_seed

    def test(
        self,
        rule_a: RuleProtocol | ValidationRule,
        rule_b: RuleProtocol | ValidationRule,
        dataset: ValidationDataset,
        column: str,
    ) -> ABTestResult:
        """Perform A/B test between two rules.

        Args:
            rule_a: First rule
            rule_b: Second rule
            dataset: Labeled validation dataset
            column: Column to validate

        Returns:
            A/B test result
        """
        if self.random_seed is not None:
            random.seed(self.random_seed)

        df = dataset.to_dataframe()
        n = len(df)

        # Calculate observed F1 scores
        pred_a = rule_a.validate_column(df, column)
        pred_b = rule_b.validate_column(df, column)
        ground_truth = df.get_column(dataset.label_column)

        f1_a = self._calculate_f1(pred_a.to_list(), ground_truth.to_list())
        f1_b = self._calculate_f1(pred_b.to_list(), ground_truth.to_list())
        observed_diff = f1_a - f1_b

        # Bootstrap test
        diff_samples = []
        for _ in range(self.n_iterations):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            sample_df = df[indices]

            sample_pred_a = rule_a.validate_column(sample_df, column)
            sample_pred_b = rule_b.validate_column(sample_df, column)
            sample_gt = sample_df.get_column(dataset.label_column)

            sample_f1_a = self._calculate_f1(
                sample_pred_a.to_list(), sample_gt.to_list()
            )
            sample_f1_b = self._calculate_f1(
                sample_pred_b.to_list(), sample_gt.to_list()
            )

            diff_samples.append(sample_f1_a - sample_f1_b)

        # Calculate p-value (two-tailed)
        # Under null hypothesis, difference centers at 0
        centered = [d - observed_diff for d in diff_samples]
        extreme_count = sum(1 for d in centered if abs(d) >= abs(observed_diff))
        p_value = extreme_count / self.n_iterations

        # Determine significance
        alpha = 1 - self.confidence_level
        significant = p_value < alpha

        # Determine winner
        if significant:
            winner = rule_a.name if observed_diff > 0 else rule_b.name
            recommendation = (
                f"{winner} is significantly better (p={p_value:.4f})"
            )
        else:
            winner = None
            recommendation = (
                f"No significant difference (p={p_value:.4f})"
            )

        return ABTestResult(
            rule_a_name=rule_a.name,
            rule_b_name=rule_b.name,
            rule_a_f1=f1_a,
            rule_b_f1=f1_b,
            difference=observed_diff,
            p_value=p_value,
            significant=significant,
            winner=winner,
            confidence_level=self.confidence_level,
            recommendation=recommendation,
        )

    def _calculate_f1(
        self,
        predictions: list[bool],
        ground_truth: list[bool],
    ) -> float:
        """Calculate F1 score."""
        tp = sum(1 for p, t in zip(predictions, ground_truth) if p and t)
        fp = sum(1 for p, t in zip(predictions, ground_truth) if p and not t)
        fn = sum(1 for p, t in zip(predictions, ground_truth) if not p and t)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_rule(
    rule: RuleProtocol | ValidationRule,
    dataset: ValidationDataset,
    column: str | None = None,
    strategy: str = "cross_validation",
    **kwargs: Any,
) -> ValidationResult:
    """Validate a rule against labeled data.

    Args:
        rule: Rule to validate
        dataset: Labeled validation dataset
        column: Column to validate
        strategy: Validation strategy
        **kwargs: Additional options

    Returns:
        Validation result
    """
    validator = LabeledDataValidator(strategy=strategy, **kwargs)
    return validator.validate(rule, dataset, column)


def create_validation_suite(
    rules: list[RuleProtocol | ValidationRule],
    dataset: ValidationDataset,
    strategy: str = "cross_validation",
) -> ValidationReport:
    """Create a validation suite for multiple rules.

    Args:
        rules: Rules to validate
        dataset: Labeled validation dataset
        strategy: Validation strategy

    Returns:
        Complete validation report
    """
    validator = LabeledDataValidator(strategy=strategy)
    return validator.validate_all(rules, dataset)


def compare_rule_quality(
    rule_a: RuleProtocol | ValidationRule,
    rule_b: RuleProtocol | ValidationRule,
    dataset: ValidationDataset,
    column: str,
) -> ABTestResult:
    """Compare two rules using A/B testing.

    Args:
        rule_a: First rule
        rule_b: Second rule
        dataset: Labeled validation dataset
        column: Column to validate

    Returns:
        A/B test result
    """
    tester = RuleABTester()
    return tester.test(rule_a, rule_b, dataset, column)
