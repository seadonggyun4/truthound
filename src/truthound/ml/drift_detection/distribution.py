"""Distribution-based drift detection.

Detects drift by comparing statistical distributions between
reference and current data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import log, sqrt
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
class DistributionDriftConfig(DriftConfig):
    """Configuration for distribution drift detection.

    Attributes:
        method: Detection method ('ks', 'psi', 'jensen_shannon', 'wasserstein')
        n_bins: Number of bins for histogram-based methods
        min_samples: Minimum samples required for detection
    """

    method: str = "psi"
    n_bins: int = 10
    min_samples: int = 30


@register_model("distribution_drift")
class DistributionDriftDetector(MLDriftDetector):
    """Distribution-based drift detection.

    Detects drift by comparing the statistical distribution of features
    between reference and current datasets.

    Supports multiple methods:
    - PSI (Population Stability Index)
    - KS (Kolmogorov-Smirnov test)
    - Jensen-Shannon divergence
    - Wasserstein distance

    Example:
        >>> detector = DistributionDriftDetector(method="psi")
        >>> detector.fit(reference_data)
        >>> result = detector.detect(reference_data, current_data)
    """

    def __init__(self, config: DistributionDriftConfig | None = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._reference_stats: dict[str, dict] = {}
        self._columns: list[str] = []

    @property
    def config(self) -> DistributionDriftConfig:
        return self._config  # type: ignore

    def _default_config(self) -> DistributionDriftConfig:
        return DistributionDriftConfig()

    def _get_model_name(self) -> str:
        return "distribution_drift"

    def _get_description(self) -> str:
        return "Distribution-based drift detection"

    def fit(self, data: pl.LazyFrame) -> None:
        """Compute reference statistics.

        Args:
            data: Reference data
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            df = self._maybe_sample(data).collect()

            # Get numeric columns
            schema = df.schema
            numeric_types = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }

            self._columns = [
                c for c in schema.keys()
                if type(schema[c]) in numeric_types
            ]

            # Compute reference statistics
            self._reference_stats = {}
            for col in self._columns:
                self._reference_stats[col] = self._compute_distribution(df, col)

            self._reference_data = data
            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to compute reference distribution: {e}",
                model_name=self.info.name,
            ) from e

    def _compute_distribution(
        self, df: pl.DataFrame, column: str
    ) -> dict[str, Any]:
        """Compute distribution statistics for a column."""
        data = df[column].drop_nulls().cast(pl.Float64).to_list()

        if not data:
            return {"empty": True}

        stats = {
            "min": min(data),
            "max": max(data),
            "mean": sum(data) / len(data),
            "count": len(data),
            "values": data,  # Store for KS test
        }

        # Compute histogram bins
        min_val, max_val = stats["min"], stats["max"]
        if min_val == max_val:
            stats["bins"] = [1.0]
            stats["bin_edges"] = [min_val, max_val + 1]
        else:
            n_bins = self.config.n_bins
            bin_width = (max_val - min_val) / n_bins
            bin_edges = [min_val + i * bin_width for i in range(n_bins + 1)]
            bin_edges[-1] = max_val + 0.001  # Include max value

            # Count per bin
            counts = [0] * n_bins
            for val in data:
                for i in range(n_bins):
                    if bin_edges[i] <= val < bin_edges[i + 1]:
                        counts[i] += 1
                        break

            # Convert to proportions
            total = sum(counts)
            proportions = [(c + 0.0001) / (total + 0.0001 * n_bins) for c in counts]

            stats["bins"] = proportions
            stats["bin_edges"] = bin_edges

        return stats

    def detect(
        self,
        reference: pl.LazyFrame,
        current: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> DriftResult:
        """Detect drift between reference and current data.

        Args:
            reference: Reference data (or use stored reference if None)
            current: Current data to compare
            columns: Specific columns to check

        Returns:
            DriftResult with drift analysis
        """
        import time
        start = time.perf_counter()

        ref_df = reference.collect()
        curr_df = current.collect()

        # Determine columns to check
        check_columns = columns or self._columns
        check_columns = [c for c in check_columns if c in curr_df.columns]

        method = self.config.method.lower()

        # Compute drift per column
        column_scores: list[tuple[str, float]] = []
        for col in check_columns:
            ref_dist = self._compute_distribution(ref_df, col)
            curr_dist = self._compute_distribution(curr_df, col)

            if ref_dist.get("empty") or curr_dist.get("empty"):
                column_scores.append((col, 0.0))
                continue

            if method == "psi":
                score = self._compute_psi(ref_dist, curr_dist)
            elif method == "ks":
                score = self._compute_ks(ref_dist, curr_dist)
            elif method == "jensen_shannon":
                score = self._compute_js(ref_dist, curr_dist)
            elif method == "wasserstein":
                score = self._compute_wasserstein(ref_dist, curr_dist)
            else:
                score = self._compute_psi(ref_dist, curr_dist)

            column_scores.append((col, score))

        # Overall drift score (max of column scores)
        if column_scores:
            max_score = max(score for _, score in column_scores)
            avg_score = sum(score for _, score in column_scores) / len(column_scores)
        else:
            max_score = 0.0
            avg_score = 0.0

        # Determine if drifted
        is_drifted = max_score >= self.config.threshold

        # Determine drift type
        drifted_cols = [col for col, score in column_scores if score >= self.config.threshold]
        if len(drifted_cols) == 0:
            drift_type = "none"
        elif len(drifted_cols) == 1:
            drift_type = "single_feature"
        elif len(drifted_cols) < len(check_columns) / 2:
            drift_type = "partial"
        else:
            drift_type = "global"

        elapsed = (time.perf_counter() - start) * 1000

        return DriftResult(
            is_drifted=is_drifted,
            drift_score=max_score,
            column_scores=tuple(column_scores),
            drift_type=drift_type,
            confidence=1.0 - avg_score if avg_score < 1 else 0.0,
            details=f"Method: {method}, Drifted columns: {drifted_cols}",
        )

    def _compute_psi(
        self, ref: dict[str, Any], curr: dict[str, Any]
    ) -> float:
        """Compute Population Stability Index."""
        ref_bins = ref["bins"]
        curr_bins = curr.get("bins")

        if curr_bins is None or len(ref_bins) != len(curr_bins):
            # Recompute bins for current using reference edges
            curr_values = curr["values"]
            bin_edges = ref["bin_edges"]
            n_bins = len(ref_bins)

            counts = [0] * n_bins
            for val in curr_values:
                for i in range(n_bins):
                    if i < len(bin_edges) - 1 and bin_edges[i] <= val < bin_edges[i + 1]:
                        counts[i] += 1
                        break

            total = sum(counts)
            curr_bins = [(c + 0.0001) / (total + 0.0001 * n_bins) for c in counts]

        # Calculate PSI
        psi = 0.0
        for ref_p, curr_p in zip(ref_bins, curr_bins):
            if ref_p > 0 and curr_p > 0:
                psi += (curr_p - ref_p) * log(curr_p / ref_p)

        return abs(psi)

    def _compute_ks(
        self, ref: dict[str, Any], curr: dict[str, Any]
    ) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        ref_values = sorted(ref["values"])
        curr_values = sorted(curr["values"])

        n1, n2 = len(ref_values), len(curr_values)
        if n1 == 0 or n2 == 0:
            return 0.0

        all_values = sorted(set(ref_values + curr_values))

        max_diff = 0.0
        for val in all_values:
            ecdf1 = sum(1 for x in ref_values if x <= val) / n1
            ecdf2 = sum(1 for x in curr_values if x <= val) / n2
            max_diff = max(max_diff, abs(ecdf1 - ecdf2))

        return max_diff

    def _compute_js(
        self, ref: dict[str, Any], curr: dict[str, Any]
    ) -> float:
        """Compute Jensen-Shannon divergence."""
        ref_bins = ref["bins"]
        curr_bins = curr.get("bins", ref_bins)

        if len(ref_bins) != len(curr_bins):
            return 0.0

        # Compute mixture distribution
        m = [(p + q) / 2 for p, q in zip(ref_bins, curr_bins)]

        # KL divergences
        def kl(p_dist: list[float], q_dist: list[float]) -> float:
            return sum(
                p * log(p / q) for p, q in zip(p_dist, q_dist)
                if p > 0 and q > 0
            )

        js = 0.5 * kl(ref_bins, m) + 0.5 * kl(curr_bins, m)

        # Normalize to [0, 1]
        return min(1.0, js / 0.693)  # log(2) ≈ 0.693

    def _compute_wasserstein(
        self, ref: dict[str, Any], curr: dict[str, Any]
    ) -> float:
        """Compute 1-Wasserstein (Earth Mover's) distance."""
        ref_values = sorted(ref["values"])
        curr_values = sorted(curr["values"])

        n1, n2 = len(ref_values), len(curr_values)
        if n1 == 0 or n2 == 0:
            return 0.0

        # Compute empirical CDFs
        all_values = sorted(set(ref_values + curr_values))

        total_distance = 0.0
        prev_val = all_values[0]

        for val in all_values[1:]:
            ecdf1 = sum(1 for x in ref_values if x <= prev_val) / n1
            ecdf2 = sum(1 for x in curr_values if x <= prev_val) / n2
            total_distance += abs(ecdf1 - ecdf2) * (val - prev_val)
            prev_val = val

        # Normalize by range
        range_val = ref["max"] - ref["min"]
        if range_val > 0:
            return total_distance / range_val

        return 0.0
