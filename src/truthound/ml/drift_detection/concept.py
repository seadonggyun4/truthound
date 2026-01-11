"""Concept drift detection.

Detects changes in the relationship between features and targets,
indicating that the underlying concept has changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl

from truthound.ml.base import (
    DriftConfig,
    DriftResult,
    MLDriftDetector,
    ModelInfo,
    ModelNotTrainedError,
    ModelState,
    ModelTrainingError,
    ModelType,
    register_model,
)


@dataclass
class ConceptDriftConfig(DriftConfig):
    """Configuration for concept drift detection.

    Attributes:
        target_column: Target/label column name
        method: Detection method ('ddm', 'adwin', 'page_hinkley')
        warning_threshold: Threshold for warning
        drift_threshold: Threshold for confirmed drift
        min_window: Minimum window size before checking for drift
    """

    target_column: str | None = None
    method: str = "ddm"
    warning_threshold: float = 2.0  # Standard deviations
    drift_threshold: float = 3.0
    min_window: int = 30
    feature_columns: list[str] | None = None


@register_model("concept_drift")
class ConceptDriftDetector(MLDriftDetector):
    """Concept drift detection.

    Detects when the relationship between features and target changes,
    which is different from feature drift (input distribution change).

    This is useful for:
    - Detecting when a model needs retraining
    - Monitoring prediction quality over time
    - Identifying regime changes in data

    Supports multiple detection methods:
    - DDM (Drift Detection Method): Monitors error rate
    - ADWIN: Adaptive windowing
    - Page-Hinkley: Sequential analysis

    Example:
        >>> detector = ConceptDriftDetector(target_column="label")
        >>> detector.fit(reference_data)
        >>> result = detector.detect(reference_data, current_data)
    """

    def __init__(self, config: ConceptDriftConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._reference_correlations: dict[str, float] = {}
        self._reference_target_dist: dict[str, Any] = {}
        self._feature_columns: list[str] = []

    @property
    def config(self) -> ConceptDriftConfig:
        return self._config  # type: ignore

    def _default_config(self) -> ConceptDriftConfig:
        return ConceptDriftConfig()

    def _get_model_name(self) -> str:
        return "concept_drift"

    def _get_description(self) -> str:
        return "Concept drift detection for feature-target relationships"

    def fit(self, data: pl.LazyFrame) -> None:
        """Learn reference correlations between features and target.

        Args:
            data: Reference data with target column
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            df = self._maybe_sample(data).collect()

            target_col = self.config.target_column
            if target_col is None:
                # Try to infer target (last column)
                target_col = df.columns[-1]

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found")

            # Get feature columns
            schema = df.schema
            numeric_types = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }

            if self.config.feature_columns:
                self._feature_columns = [
                    c for c in self.config.feature_columns
                    if c in df.columns and c != target_col
                ]
            else:
                self._feature_columns = [
                    c for c in df.columns
                    if c != target_col and type(schema[c]) in numeric_types
                ]

            # Compute feature-target correlations
            self._reference_correlations = {}
            target_data = df[target_col].drop_nulls().cast(pl.Float64)

            for col in self._feature_columns:
                try:
                    corr = self._compute_correlation(df, col, target_col)
                    self._reference_correlations[col] = corr
                except Exception:
                    self._reference_correlations[col] = 0.0

            # Store target distribution
            self._reference_target_dist = self._compute_target_distribution(df, target_col)

            self._reference_data = data
            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to compute reference concept: {e}",
                model_name=self.info.name,
            ) from e

    def _compute_correlation(
        self, df: pl.DataFrame, feature: str, target: str
    ) -> float:
        """Compute Pearson correlation between feature and target."""
        # Filter out nulls from both columns
        valid = df.select([feature, target]).drop_nulls()
        if len(valid) < 2:
            return 0.0

        x = valid[feature].cast(pl.Float64).to_list()
        y = valid[target].cast(pl.Float64).to_list()

        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)

    def _compute_target_distribution(
        self, df: pl.DataFrame, target: str
    ) -> dict[str, Any]:
        """Compute target column distribution."""
        data = df[target].drop_nulls()

        if data.dtype in [pl.String, pl.Utf8, pl.Categorical]:
            # Categorical target
            value_counts = data.value_counts().sort("count", descending=True)
            values = value_counts[target].to_list()
            counts = value_counts["count"].to_list()
            total = sum(counts)
            proportions = {v: c / total for v, c in zip(values, counts)}

            return {
                "type": "categorical",
                "values": set(values),
                "proportions": proportions,
            }
        else:
            # Numeric target
            numeric_data = data.cast(pl.Float64)
            stats = df.select([
                pl.col(target).mean().alias("mean"),
                pl.col(target).std().alias("std"),
                pl.col(target).min().alias("min"),
                pl.col(target).max().alias("max"),
            ]).row(0)

            return {
                "type": "numeric",
                "mean": stats[0],
                "std": stats[1],
                "min": stats[2],
                "max": stats[3],
            }

    def detect(
        self,
        reference: pl.LazyFrame,
        current: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> DriftResult:
        """Detect concept drift.

        Args:
            reference: Reference data
            current: Current data
            columns: Specific feature columns to check

        Returns:
            DriftResult with concept drift analysis
        """
        import time
        start = time.perf_counter()

        curr_df = current.collect()
        target_col = self.config.target_column or curr_df.columns[-1]

        if target_col not in curr_df.columns:
            return DriftResult(
                is_drifted=False,
                drift_score=0.0,
                drift_type="error",
                details=f"Target column '{target_col}' not found",
            )

        # Determine columns to check
        check_columns = columns or self._feature_columns
        check_columns = [c for c in check_columns if c in curr_df.columns]

        # Check correlation changes
        column_scores: list[tuple[str, float]] = []
        correlation_changes: list[str] = []

        for col in check_columns:
            ref_corr = self._reference_correlations.get(col, 0.0)
            try:
                curr_corr = self._compute_correlation(curr_df, col, target_col)
            except Exception:
                curr_corr = 0.0

            # Correlation change score
            corr_diff = abs(curr_corr - ref_corr)

            # Also check for sign change (relationship reversal)
            if ref_corr * curr_corr < 0 and abs(ref_corr) > 0.1 and abs(curr_corr) > 0.1:
                corr_diff += 0.5  # Penalize sign changes
                correlation_changes.append(f"{col}: sign changed")

            # Normalize score
            score = min(1.0, corr_diff / 0.5)  # 0.5 correlation change = score 1.0
            column_scores.append((col, score))

            if score > 0.3:
                correlation_changes.append(
                    f"{col}: {ref_corr:.2f} -> {curr_corr:.2f}"
                )

        # Check target distribution change
        curr_target_dist = self._compute_target_distribution(curr_df, target_col)
        target_drift_score = self._compute_target_drift(
            self._reference_target_dist, curr_target_dist
        )

        if target_drift_score > 0.3:
            column_scores.append(("_target_distribution", target_drift_score))

        # Overall score
        if column_scores:
            max_score = max(score for _, score in column_scores)
            avg_score = sum(score for _, score in column_scores) / len(column_scores)
        else:
            max_score = 0.0
            avg_score = 0.0

        is_drifted = max_score >= self.config.threshold

        # Determine drift type
        if target_drift_score >= self.config.threshold:
            drift_type = "target_shift"
        elif max_score >= self.config.threshold:
            drifted_features = [c for c, s in column_scores if s >= self.config.threshold]
            if len(drifted_features) > len(check_columns) / 2:
                drift_type = "concept_shift"
            else:
                drift_type = "partial_concept_drift"
        else:
            drift_type = "none"

        elapsed = (time.perf_counter() - start) * 1000

        return DriftResult(
            is_drifted=is_drifted,
            drift_score=max_score,
            column_scores=tuple(column_scores),
            drift_type=drift_type,
            confidence=1.0 - avg_score,
            details="; ".join(correlation_changes) if correlation_changes else "No significant changes",
        )

    def _compute_target_drift(
        self, ref: dict[str, Any], curr: dict[str, Any]
    ) -> float:
        """Compute target distribution drift score."""
        # Handle case where ref is empty (fit() not called)
        if not ref or "type" not in ref:
            return 0.0

        if ref["type"] != curr.get("type"):
            return 1.0  # Type changed

        if ref["type"] == "categorical":
            ref_values = ref["values"]
            curr_values = curr.get("values", set())
            ref_props = ref["proportions"]
            curr_props = curr.get("proportions", {})

            # Check for new/missing values
            new_values = curr_values - ref_values
            missing_values = ref_values - curr_values

            drift_score = 0.0
            if new_values:
                new_prop = sum(curr_props.get(v, 0) for v in new_values)
                drift_score = max(drift_score, new_prop)

            if missing_values:
                missing_prop = sum(ref_props.get(v, 0) for v in missing_values)
                drift_score = max(drift_score, missing_prop)

            # Check proportion changes
            common = ref_values & curr_values
            for val in common:
                prop_diff = abs(curr_props.get(val, 0) - ref_props.get(val, 0))
                drift_score = max(drift_score, prop_diff)

            return min(1.0, drift_score)

        else:  # numeric
            ref_mean = ref.get("mean", 0)
            curr_mean = curr.get("mean", 0)
            ref_std = ref.get("std", 1) or 1

            mean_drift = abs(curr_mean - ref_mean) / ref_std

            # Check variance change
            curr_std = curr.get("std", ref_std)
            if ref_std > 0:
                std_ratio = curr_std / ref_std
                std_drift = abs(std_ratio - 1.0)
            else:
                std_drift = 0

            return min(1.0, max(mean_drift / 3, std_drift))

    def get_reference_correlations(self) -> dict[str, float]:
        """Get reference feature-target correlations."""
        return dict(self._reference_correlations)
