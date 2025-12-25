"""Statistical anomaly detection methods.

Provides anomaly detection based on statistical measures:
- Z-Score: Standard deviations from mean
- IQR: Interquartile range method
- MAD: Median Absolute Deviation

These methods are suitable for univariate data and are
computationally efficient.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from typing import Any

import polars as pl

from truthound.ml.base import (
    AnomalyConfig,
    AnomalyDetector,
    AnomalyResult,
    AnomalyScore,
    AnomalyType,
    InsufficientDataError,
    ModelInfo,
    ModelNotTrainedError,
    ModelState,
    ModelTrainingError,
    ModelType,
    register_model,
)


@dataclass
class StatisticalConfig(AnomalyConfig):
    """Configuration for statistical anomaly detection.

    Attributes:
        z_threshold: Z-score threshold for anomaly (default 3.0)
        iqr_multiplier: IQR multiplier for bounds (default 1.5)
        use_robust_stats: Use median/MAD instead of mean/std
        per_column: Detect anomalies per column independently
    """

    z_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    use_robust_stats: bool = False
    per_column: bool = True
    columns: list[str] | None = None  # Columns to analyze


class StatisticalAnomalyDetector(AnomalyDetector):
    """Base class for statistical anomaly detection.

    Uses statistical measures to identify outliers in numeric data.
    Subclasses implement specific statistical methods.
    """

    def __init__(self, config: StatisticalConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._stats: dict[str, dict[str, float]] = {}
        self._columns: list[str] = []

    @property
    def config(self) -> StatisticalConfig:
        return self._config  # type: ignore

    def _default_config(self) -> StatisticalConfig:
        return StatisticalConfig()

    def fit(self, data: pl.LazyFrame) -> None:
        """Compute statistics from training data.

        Args:
            data: Training data
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            sampled = self._maybe_sample(data)

            # Get numeric columns
            schema = sampled.collect_schema()
            numeric_types = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }

            if self.config.columns:
                self._columns = [
                    c for c in self.config.columns
                    if c in schema and type(schema[c]) in numeric_types
                ]
            else:
                self._columns = [
                    c for c in schema.names()
                    if type(schema[c]) in numeric_types
                ]

            if not self._columns:
                raise InsufficientDataError(
                    "No numeric columns found for anomaly detection",
                    model_name=self.info.name,
                )

            # Compute statistics per column
            self._stats = {}
            for col in self._columns:
                self._stats[col] = self._compute_column_stats(sampled, col)

            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            if isinstance(e, (InsufficientDataError, ModelTrainingError)):
                raise
            raise ModelTrainingError(
                f"Failed to train statistical detector: {e}",
                model_name=self.info.name,
            ) from e

    @abstractmethod
    def _compute_column_stats(
        self, data: pl.LazyFrame, column: str
    ) -> dict[str, float]:
        """Compute statistics for a column.

        Args:
            data: Data to analyze
            column: Column name

        Returns:
            Dict of statistic name to value
        """
        ...

    @abstractmethod
    def _compute_score(
        self, value: float, stats: dict[str, float]
    ) -> float:
        """Compute anomaly score for a value.

        Args:
            value: Value to score
            stats: Column statistics

        Returns:
            Anomaly score (higher = more anomalous)
        """
        ...

    def score(self, data: pl.LazyFrame) -> pl.Series:
        """Compute anomaly scores.

        For multi-column data, returns the maximum score across columns.

        Args:
            data: Data to score

        Returns:
            Series of anomaly scores
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before scoring",
                model_name=self.info.name,
            )

        df = data.collect()
        n_rows = len(df)

        # Compute scores per column
        all_scores: list[list[float]] = []
        for col in self._columns:
            if col not in df.columns:
                continue
            stats = self._stats[col]
            col_data = df[col].to_list()
            col_scores = []
            for val in col_data:
                if val is None:
                    col_scores.append(0.0)  # Nulls get zero score
                else:
                    col_scores.append(self._compute_score(float(val), stats))
            all_scores.append(col_scores)

        # Combine scores: take maximum across columns
        if not all_scores:
            return pl.Series("anomaly_score", [0.0] * n_rows)

        combined = []
        for i in range(n_rows):
            max_score = max(scores[i] for scores in all_scores)
            combined.append(max_score)

        return pl.Series("anomaly_score", combined)

    def get_statistics(self) -> dict[str, dict[str, float]]:
        """Get computed statistics per column."""
        return dict(self._stats)

    def _serialize(self) -> dict[str, Any]:
        base = super()._serialize()
        base["stats"] = self._stats
        base["columns"] = self._columns
        return base

    def _deserialize(self, data: dict[str, Any]) -> None:
        super()._deserialize(data)
        self._stats = data.get("stats", {})
        self._columns = data.get("columns", [])


@register_model("zscore")
class ZScoreAnomalyDetector(StatisticalAnomalyDetector):
    """Z-Score based anomaly detection.

    Identifies outliers based on standard deviations from the mean.
    Points with |z-score| > threshold are classified as anomalies.

    Suitable for normally distributed data.
    """

    def _get_model_name(self) -> str:
        return "zscore"

    def _get_description(self) -> str:
        return "Z-Score based anomaly detection using standard deviations from mean"

    def _compute_column_stats(
        self, data: pl.LazyFrame, column: str
    ) -> dict[str, float]:
        """Compute mean and standard deviation."""
        stats = data.select([
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
        ]).collect()

        return {
            "mean": stats["mean"][0] or 0.0,
            "std": stats["std"][0] or 1.0,  # Avoid division by zero
        }

    def _compute_score(
        self, value: float, stats: dict[str, float]
    ) -> float:
        """Compute absolute z-score."""
        mean = stats["mean"]
        std = stats["std"]

        if std == 0:
            return 0.0 if value == mean else float("inf")

        z_score = abs(value - mean) / std

        # Normalize to [0, 1] using the threshold
        threshold = self.config.z_threshold
        return min(1.0, z_score / threshold)


@register_model("iqr")
class IQRAnomalyDetector(StatisticalAnomalyDetector):
    """Interquartile Range (IQR) based anomaly detection.

    Uses Q1 - k*IQR and Q3 + k*IQR as bounds.
    Points outside these bounds are classified as anomalies.

    More robust to outliers than Z-Score method.
    """

    def _get_model_name(self) -> str:
        return "iqr"

    def _get_description(self) -> str:
        return "IQR-based anomaly detection using interquartile range"

    def _compute_column_stats(
        self, data: pl.LazyFrame, column: str
    ) -> dict[str, float]:
        """Compute quartiles."""
        stats = data.select([
            pl.col(column).quantile(0.25).alias("q1"),
            pl.col(column).quantile(0.50).alias("median"),
            pl.col(column).quantile(0.75).alias("q3"),
        ]).collect()

        q1 = stats["q1"][0] or 0.0
        q3 = stats["q3"][0] or 0.0
        iqr = q3 - q1

        k = self.config.iqr_multiplier
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr

        return {
            "q1": q1,
            "median": stats["median"][0] or 0.0,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def _compute_score(
        self, value: float, stats: dict[str, float]
    ) -> float:
        """Compute score based on distance from IQR bounds."""
        lower = stats["lower_bound"]
        upper = stats["upper_bound"]
        iqr = stats["iqr"]

        if iqr == 0:
            return 0.0 if lower <= value <= upper else 1.0

        if value < lower:
            distance = lower - value
        elif value > upper:
            distance = value - upper
        else:
            return 0.0

        # Normalize by IQR
        return min(1.0, distance / iqr)


@register_model("mad")
class MADAnomalyDetector(StatisticalAnomalyDetector):
    """Median Absolute Deviation (MAD) based anomaly detection.

    Uses median and MAD instead of mean and standard deviation.
    Very robust to outliers in the training data.

    MAD = median(|Xi - median(X)|)
    Modified Z-Score = 0.6745 * (Xi - median) / MAD
    """

    def _get_model_name(self) -> str:
        return "mad"

    def _get_description(self) -> str:
        return "MAD-based anomaly detection using median absolute deviation"

    def _compute_column_stats(
        self, data: pl.LazyFrame, column: str
    ) -> dict[str, float]:
        """Compute median and MAD."""
        # First compute median
        median_result = data.select(
            pl.col(column).median().alias("median")
        ).collect()
        median = median_result["median"][0] or 0.0

        # Then compute MAD
        mad_result = data.select(
            (pl.col(column) - median).abs().median().alias("mad")
        ).collect()
        mad = mad_result["mad"][0] or 0.0

        return {
            "median": median,
            "mad": mad,
        }

    def _compute_score(
        self, value: float, stats: dict[str, float]
    ) -> float:
        """Compute modified z-score using MAD."""
        median = stats["median"]
        mad = stats["mad"]

        if mad == 0:
            return 0.0 if value == median else 1.0

        # Modified z-score
        # 0.6745 is the factor to make MAD consistent with std for normal distribution
        modified_z = 0.6745 * abs(value - median) / mad

        # Normalize to [0, 1]
        threshold = self.config.z_threshold
        return min(1.0, modified_z / threshold)
