"""Feature-level drift detection.

Monitors individual features for distribution changes and
provides detailed feature-level drift analysis.
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


@dataclass(frozen=True)
class FeatureDriftScore:
    """Drift score for a single feature.

    Attributes:
        feature: Feature name
        drift_score: Overall drift score
        statistic_type: Type of statistic used
        reference_stats: Reference statistics
        current_stats: Current statistics
        is_drifted: Whether this feature has drifted
    """

    feature: str
    drift_score: float
    statistic_type: str
    reference_stats: dict[str, Any]
    current_stats: dict[str, Any]
    is_drifted: bool
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "drift_score": round(self.drift_score, 6),
            "statistic_type": self.statistic_type,
            "reference_stats": self.reference_stats,
            "current_stats": self.current_stats,
            "is_drifted": self.is_drifted,
            "details": self.details,
        }


@dataclass
class FeatureDriftConfig(DriftConfig):
    """Configuration for feature drift detection.

    Attributes:
        track_stats: Statistics to track ('mean', 'std', 'min', 'max', etc.)
        relative_threshold: Use relative threshold instead of absolute
        alert_on_new_values: Alert on new categorical values
    """

    track_stats: list[str] = field(default_factory=lambda: [
        "mean", "std", "min", "max", "null_ratio"
    ])
    relative_threshold: bool = True
    alert_on_new_values: bool = True
    categorical_threshold: float = 0.2  # For categorical drift


@register_model("feature_drift")
class FeatureDriftDetector(MLDriftDetector):
    """Feature-level drift detection.

    Monitors individual features for statistical changes and
    provides detailed analysis of which features have drifted
    and how.

    Tracks:
    - Numeric features: mean, std, min, max, quartiles
    - Categorical features: value distribution changes
    - All features: null ratio changes

    Example:
        >>> detector = FeatureDriftDetector()
        >>> detector.fit(reference_data)
        >>> result = detector.detect(reference_data, current_data)
        >>> for col, score in result.column_scores:
        ...     if score > 0.5:
        ...         print(f"{col} has drifted: {score:.2f}")
    """

    def __init__(self, config: FeatureDriftConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._reference_profiles: dict[str, dict] = {}
        self._columns: list[str] = []
        self._numeric_columns: list[str] = []
        self._categorical_columns: list[str] = []

    @property
    def config(self) -> FeatureDriftConfig:
        return self._config  # type: ignore

    def _default_config(self) -> FeatureDriftConfig:
        return FeatureDriftConfig()

    def _get_model_name(self) -> str:
        return "feature_drift"

    def _get_description(self) -> str:
        return "Feature-level drift detection with detailed statistics"

    def fit(self, data: pl.LazyFrame) -> None:
        """Compute reference feature profiles.

        Args:
            data: Reference data
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            df = self._maybe_sample(data).collect()

            # Classify columns
            schema = df.schema
            numeric_types = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }
            categorical_types = {pl.String, pl.Utf8, pl.Categorical}

            self._numeric_columns = [
                c for c in schema.keys()
                if type(schema[c]) in numeric_types
            ]
            self._categorical_columns = [
                c for c in schema.keys()
                if type(schema[c]) in categorical_types
            ]
            self._columns = self._numeric_columns + self._categorical_columns

            # Compute profiles
            self._reference_profiles = {}
            for col in self._numeric_columns:
                self._reference_profiles[col] = self._profile_numeric(df, col)
            for col in self._categorical_columns:
                self._reference_profiles[col] = self._profile_categorical(df, col)

            self._reference_data = data
            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to compute feature profiles: {e}",
                model_name=self.info.name,
            ) from e

    def _profile_numeric(self, df: pl.DataFrame, column: str) -> dict[str, Any]:
        """Profile a numeric column."""
        stats = df.select([
            pl.col(column).count().alias("count"),
            pl.col(column).null_count().alias("null_count"),
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
            pl.col(column).median().alias("median"),
            pl.col(column).quantile(0.25).alias("q1"),
            pl.col(column).quantile(0.75).alias("q3"),
        ]).row(0)

        count = stats[0] or 0
        null_count = stats[1] or 0
        total = count + null_count

        return {
            "type": "numeric",
            "count": count,
            "null_count": null_count,
            "null_ratio": null_count / total if total > 0 else 0,
            "mean": stats[2],
            "std": stats[3],
            "min": stats[4],
            "max": stats[5],
            "median": stats[6],
            "q1": stats[7],
            "q3": stats[8],
        }

    def _profile_categorical(self, df: pl.DataFrame, column: str) -> dict[str, Any]:
        """Profile a categorical column."""
        total = len(df)
        null_count = df[column].null_count()

        # Value counts
        value_counts = (
            df.select(column)
            .drop_nulls()
            .group_by(column)
            .agg(pl.count().alias("count"))
            .sort("count", descending=True)
        )

        values = value_counts[column].to_list()
        counts = value_counts["count"].to_list()
        proportions = {v: c / total for v, c in zip(values, counts)}

        return {
            "type": "categorical",
            "count": total - null_count,
            "null_count": null_count,
            "null_ratio": null_count / total if total > 0 else 0,
            "n_unique": len(values),
            "values": set(values),
            "proportions": proportions,
        }

    def detect(
        self,
        reference: pl.LazyFrame,
        current: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> DriftResult:
        """Detect feature drift.

        Args:
            reference: Reference data
            current: Current data
            columns: Specific columns to check

        Returns:
            DriftResult with per-feature drift scores
        """
        import time
        start = time.perf_counter()

        ref_df = reference.collect()
        curr_df = current.collect()

        # Determine columns to check
        check_columns = columns or self._columns
        check_columns = [c for c in check_columns if c in curr_df.columns]

        # Compute drift per feature
        feature_scores: list[FeatureDriftScore] = []
        column_scores: list[tuple[str, float]] = []

        for col in check_columns:
            ref_profile = self._reference_profiles.get(col)
            if ref_profile is None:
                continue

            if ref_profile["type"] == "numeric":
                curr_profile = self._profile_numeric(curr_df, col)
                score, details = self._compute_numeric_drift(ref_profile, curr_profile)
            else:
                curr_profile = self._profile_categorical(curr_df, col)
                score, details = self._compute_categorical_drift(ref_profile, curr_profile)

            is_drifted = score >= self.config.threshold

            feature_scores.append(FeatureDriftScore(
                feature=col,
                drift_score=score,
                statistic_type=ref_profile["type"],
                reference_stats={k: v for k, v in ref_profile.items() if k != "values"},
                current_stats={k: v for k, v in curr_profile.items() if k != "values"},
                is_drifted=is_drifted,
                details=details,
            ))

            column_scores.append((col, score))

        # Overall analysis
        if column_scores:
            max_score = max(score for _, score in column_scores)
            drifted_count = sum(1 for _, score in column_scores if score >= self.config.threshold)
        else:
            max_score = 0.0
            drifted_count = 0

        is_drifted = drifted_count > 0

        # Determine drift type
        if drifted_count == 0:
            drift_type = "none"
        elif drifted_count == 1:
            drift_type = "single_feature"
        elif drifted_count < len(check_columns) / 2:
            drift_type = "partial"
        else:
            drift_type = "widespread"

        elapsed = (time.perf_counter() - start) * 1000

        return DriftResult(
            is_drifted=is_drifted,
            drift_score=max_score,
            column_scores=tuple(column_scores),
            drift_type=drift_type,
            confidence=1.0 - (drifted_count / len(check_columns)) if check_columns else 1.0,
            details=f"Drifted features: {drifted_count}/{len(check_columns)}",
        )

    def _compute_numeric_drift(
        self,
        ref: dict[str, Any],
        curr: dict[str, Any],
    ) -> tuple[float, str]:
        """Compute drift score for numeric feature."""
        scores = []
        details = []

        # Mean drift
        if ref["mean"] is not None and curr["mean"] is not None:
            ref_mean = ref["mean"]
            curr_mean = curr["mean"]
            ref_std = ref["std"] or 1.0

            if self.config.relative_threshold:
                if ref_std > 0:
                    mean_drift = abs(curr_mean - ref_mean) / ref_std
                else:
                    mean_drift = abs(curr_mean - ref_mean) if curr_mean != ref_mean else 0
                mean_drift = min(1.0, mean_drift / 3.0)  # Normalize to [0, 1]
            else:
                mean_drift = abs(curr_mean - ref_mean) / (abs(ref_mean) + 0.0001)

            scores.append(mean_drift)
            if mean_drift > 0.3:
                details.append(f"mean: {ref_mean:.2f} -> {curr_mean:.2f}")

        # Std drift
        if ref["std"] is not None and curr["std"] is not None:
            ref_std = ref["std"]
            curr_std = curr["std"]

            if ref_std > 0:
                std_ratio = curr_std / ref_std
                std_drift = abs(std_ratio - 1.0)
                std_drift = min(1.0, std_drift)
                scores.append(std_drift)
                if std_drift > 0.3:
                    details.append(f"std: {ref_std:.2f} -> {curr_std:.2f}")

        # Range drift
        if all(x is not None for x in [ref["min"], ref["max"], curr["min"], curr["max"]]):
            ref_range = ref["max"] - ref["min"]
            curr_min, curr_max = curr["min"], curr["max"]

            # Check if current values exceed reference range
            if curr_min < ref["min"] or curr_max > ref["max"]:
                range_drift = 0.5
                details.append(f"range exceeded: [{curr_min:.2f}, {curr_max:.2f}]")
                scores.append(range_drift)

        # Null ratio drift
        null_drift = abs(curr["null_ratio"] - ref["null_ratio"])
        if null_drift > 0.1:
            scores.append(null_drift)
            details.append(f"null_ratio: {ref['null_ratio']:.2%} -> {curr['null_ratio']:.2%}")

        overall_score = max(scores) if scores else 0.0
        return overall_score, "; ".join(details)

    def _compute_categorical_drift(
        self,
        ref: dict[str, Any],
        curr: dict[str, Any],
    ) -> tuple[float, str]:
        """Compute drift score for categorical feature."""
        scores = []
        details = []

        ref_values = ref["values"]
        curr_values = curr.get("values", set())
        ref_props = ref["proportions"]
        curr_props = curr.get("proportions", {})

        # New values
        new_values = curr_values - ref_values
        if new_values and self.config.alert_on_new_values:
            new_prop = sum(curr_props.get(v, 0) for v in new_values)
            if new_prop > 0.05:  # More than 5% of data has new values
                scores.append(new_prop)
                details.append(f"new values: {len(new_values)}")

        # Missing values
        missing_values = ref_values - curr_values
        if missing_values:
            missing_prop = sum(ref_props.get(v, 0) for v in missing_values)
            if missing_prop > 0.1:  # Missing values that were >10% of reference
                scores.append(missing_prop)
                details.append(f"missing values: {len(missing_values)}")

        # Proportion changes
        common_values = ref_values & curr_values
        max_prop_drift = 0.0
        for val in common_values:
            ref_p = ref_props.get(val, 0)
            curr_p = curr_props.get(val, 0)
            prop_drift = abs(curr_p - ref_p)
            max_prop_drift = max(max_prop_drift, prop_drift)

        if max_prop_drift > 0.1:
            scores.append(max_prop_drift)
            details.append(f"max proportion change: {max_prop_drift:.2%}")

        # Null ratio drift
        null_drift = abs(curr.get("null_ratio", 0) - ref.get("null_ratio", 0))
        if null_drift > 0.1:
            scores.append(null_drift)
            details.append(f"null_ratio change: {null_drift:.2%}")

        overall_score = max(scores) if scores else 0.0
        return overall_score, "; ".join(details)

    def get_feature_profiles(self) -> dict[str, dict]:
        """Get reference feature profiles."""
        return dict(self._reference_profiles)

    def get_drifted_features(self, result: DriftResult, threshold: float | None = None) -> list[str]:
        """Get list of drifted features from a result."""
        threshold = threshold or self.config.threshold
        return [col for col, score in result.column_scores if score >= threshold]
