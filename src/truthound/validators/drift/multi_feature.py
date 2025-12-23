"""Multi-feature drift validators.

Validators for detecting drift across multiple columns/features simultaneously.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.drift.base import DriftValidator, NumericDriftMixin, CategoricalDriftMixin
from truthound.validators.registry import register_validator


@register_validator
class FeatureDriftValidator(DriftValidator, NumericDriftMixin, CategoricalDriftMixin):
    """Comprehensive multi-feature drift detection.

    Runs drift detection on multiple columns and provides an aggregated
    drift score. Useful for monitoring ML feature sets.

    Example:
        # Monitor all numeric features
        validator = FeatureDriftValidator(
            columns=["age", "income", "credit_score"],
            reference_data=training_df,
            method="psi",
            threshold=0.25,
            alert_on_any=True,  # Alert if ANY feature drifts
        )

        # Mixed column types
        validator = FeatureDriftValidator(
            columns=["age", "income", "category"],
            reference_data=training_df,
            categorical_columns=["category"],
            method="ks",
        )
    """

    name = "feature_drift"
    category = "drift"

    SUPPORTED_METHODS = ["psi", "ks", "wasserstein", "chi_square"]

    def __init__(
        self,
        columns: list[str],
        reference_data: pl.LazyFrame | pl.DataFrame,
        method: str = "psi",
        threshold: float = 0.25,
        categorical_columns: list[str] | None = None,
        alert_on_any: bool = True,
        min_drift_count: int = 1,
        n_bins: int = 10,
        **kwargs: Any,
    ):
        """Initialize multi-feature drift validator.

        Args:
            columns: List of columns to monitor
            reference_data: Baseline data for comparison
            method: Drift detection method ('psi', 'ks', 'wasserstein', 'chi_square')
            threshold: Threshold for drift detection per column
            categorical_columns: List of categorical columns (others assumed numeric)
            alert_on_any: If True, alert if any column drifts
            min_drift_count: Minimum drifted columns to trigger alert (if alert_on_any=False)
            n_bins: Number of bins for histogram-based methods
            **kwargs: Additional config
        """
        super().__init__(reference_data=reference_data, **kwargs)
        self.columns = columns
        self.method = method.lower()
        self.threshold = threshold
        self.categorical_columns = set(categorical_columns or [])
        self.alert_on_any = alert_on_any
        self.min_drift_count = min_drift_count
        self.n_bins = n_bins

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. Use one of {self.SUPPORTED_METHODS}"
            )

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> dict[str, tuple[float, bool]]:
        """Calculate drift score for each column.

        Returns:
            Dict of column -> (drift_score, is_drifted)
        """
        results = {}

        for col in self.columns:
            is_categorical = col in self.categorical_columns
            score = self._calculate_column_drift(
                reference, current, col, is_categorical
            )
            is_drifted = score > self.threshold
            results[col] = (score, is_drifted)

        return results

    def _calculate_column_drift(
        self,
        reference: pl.LazyFrame,
        current: pl.LazyFrame,
        column: str,
        is_categorical: bool,
    ) -> float:
        """Calculate drift score for a single column."""
        ref_values = reference.select(pl.col(column)).drop_nulls().collect().to_series()
        curr_values = current.select(pl.col(column)).drop_nulls().collect().to_series()

        if len(ref_values) == 0 or len(curr_values) == 0:
            return 0.0

        if self.method == "psi":
            return self._calculate_psi(ref_values, curr_values, is_categorical)
        elif self.method == "ks":
            return self._calculate_ks(ref_values, curr_values)
        elif self.method == "wasserstein":
            return self._calculate_wasserstein(ref_values, curr_values)
        elif self.method == "chi_square":
            return self._calculate_chi_square(ref_values, curr_values, is_categorical)
        else:
            return 0.0

    def _calculate_psi(
        self, ref_values: pl.Series, curr_values: pl.Series, is_categorical: bool
    ) -> float:
        """Calculate PSI for a column."""
        min_freq = 0.0001

        if is_categorical:
            ref_freq = self.compute_category_frequencies(ref_values)
            curr_freq = self.compute_category_frequencies(curr_values)
            ref_aligned, curr_aligned = self.align_categories(ref_freq, curr_freq)
        else:
            ref_arr = ref_values.to_numpy()
            curr_arr = curr_values.to_numpy()

            # Create bins from reference
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(ref_arr, percentiles)
            bin_edges = sorted(set(bin_edges))

            if len(bin_edges) < 2:
                return 0.0

            ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
            curr_counts, _ = np.histogram(curr_arr, bins=bin_edges)

            ref_aligned = (ref_counts / ref_counts.sum()).tolist() if ref_counts.sum() > 0 else []
            curr_aligned = (curr_counts / curr_counts.sum()).tolist() if curr_counts.sum() > 0 else []

        if not ref_aligned or not curr_aligned:
            return 0.0

        psi = 0.0
        for ref_p, curr_p in zip(ref_aligned, curr_aligned):
            ref_p = max(ref_p, min_freq)
            curr_p = max(curr_p, min_freq)
            psi += (curr_p - ref_p) * np.log(curr_p / ref_p)

        return float(psi)

    def _calculate_ks(self, ref_values: pl.Series, curr_values: pl.Series) -> float:
        """Calculate KS statistic for a column."""
        from scipy import stats

        statistic, _ = stats.ks_2samp(
            ref_values.to_numpy(), curr_values.to_numpy()
        )
        return float(statistic)

    def _calculate_wasserstein(
        self, ref_values: pl.Series, curr_values: pl.Series
    ) -> float:
        """Calculate Wasserstein distance for a column."""
        from scipy import stats

        distance = stats.wasserstein_distance(
            ref_values.to_numpy(), curr_values.to_numpy()
        )

        # Normalize by reference std for comparability
        ref_std = ref_values.std()
        if ref_std and ref_std > 0:
            distance = distance / ref_std

        return float(distance)

    def _calculate_chi_square(
        self, ref_values: pl.Series, curr_values: pl.Series, is_categorical: bool
    ) -> float:
        """Calculate chi-square statistic for a column."""
        from scipy import stats

        if is_categorical:
            ref_freq = self.compute_category_frequencies(ref_values)
            curr_freq = self.compute_category_frequencies(curr_values)
            ref_aligned, curr_aligned = self.align_categories(ref_freq, curr_freq)
        else:
            # Bin numeric values
            ref_arr = ref_values.to_numpy()
            curr_arr = curr_values.to_numpy()

            bin_edges = np.histogram_bin_edges(ref_arr, bins=self.n_bins)
            ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
            curr_counts, _ = np.histogram(curr_arr, bins=bin_edges)

            ref_aligned = ref_counts.tolist()
            curr_aligned = curr_counts.tolist()

        if not ref_aligned or not curr_aligned or sum(ref_aligned) == 0:
            return 0.0

        # Scale to comparable counts
        n = len(curr_values)
        observed = np.array([f * n for f in curr_aligned]) if is_categorical else np.array(curr_aligned)
        expected = np.array([f * n for f in ref_aligned]) if is_categorical else np.array(ref_aligned)

        # Filter out zero expected
        mask = expected >= 5
        if not mask.any():
            return 0.0

        observed = observed[mask]
        expected = expected[mask]

        # Normalize expected to match observed sum (required by scipy.stats.chisquare)
        if expected.sum() > 0:
            expected = expected * (observed.sum() / expected.sum())

        chi2, p_value = stats.chisquare(observed, expected)

        # Return normalized chi2 (lower p-value = higher score)
        return float(1 - p_value) if p_value > 0 else 1.0

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        drift_results = self.calculate_drift_score(self.reference_data, lf)

        drifted_columns = [
            (col, score) for col, (score, is_drifted) in drift_results.items()
            if is_drifted
        ]

        # Determine if we should alert
        should_alert = False
        if self.alert_on_any and len(drifted_columns) > 0:
            should_alert = True
        elif not self.alert_on_any and len(drifted_columns) >= self.min_drift_count:
            should_alert = True

        if should_alert:
            # Compute aggregate metrics
            all_scores = [score for score, _ in drift_results.values()]
            avg_drift = np.mean(all_scores)
            max_drift = max(score for _, score in drifted_columns)

            # Build details
            details_list = [f"{col}={score:.4f}" for col, score in drifted_columns[:5]]
            if len(drifted_columns) > 5:
                details_list.append(f"...and {len(drifted_columns) - 5} more")

            issues.append(
                ValidationIssue(
                    column=", ".join(self.columns[:3]) + ("..." if len(self.columns) > 3 else ""),
                    issue_type="feature_drift_detected",
                    count=len(drifted_columns),
                    severity=self._calculate_severity(max_drift, self.threshold),
                    details=f"{len(drifted_columns)}/{len(self.columns)} features drifted. "
                           f"Avg={avg_drift:.4f}, Max={max_drift:.4f}. "
                           f"Drifted: {', '.join(details_list)}",
                    expected=f"All features below threshold ({self.threshold})",
                )
            )

        return issues


@register_validator
class JSDivergenceValidator(DriftValidator, NumericDriftMixin, CategoricalDriftMixin):
    """Jensen-Shannon Divergence for distribution comparison.

    JS divergence is a symmetric and bounded (0 to 1) measure of
    distributional difference. It's the square root of the average
    of KL divergences in both directions.

    Example:
        validator = JSDivergenceValidator(
            column="transaction_amount",
            reference_data=baseline_df,
            threshold=0.1,  # Max JS divergence
        )
    """

    name = "js_divergence"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        threshold: float = 0.1,
        n_bins: int = 10,
        is_categorical: bool = False,
        **kwargs: Any,
    ):
        """Initialize JS divergence validator.

        Args:
            column: Column to check
            reference_data: Baseline data for comparison
            threshold: Maximum allowed JS divergence (0 to 1)
            n_bins: Number of bins for numeric columns
            is_categorical: True if column is categorical
            **kwargs: Additional config
        """
        super().__init__(reference_data=reference_data, **kwargs)
        self.column = column
        self.threshold = threshold
        self.n_bins = n_bins
        self.is_categorical = is_categorical

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> float:
        """Calculate Jensen-Shannon divergence.

        Returns:
            JS divergence (0 to 1, where 0 = identical distributions)
        """
        from scipy import spatial

        ref_values = reference.select(pl.col(self.column)).drop_nulls().collect().to_series()
        curr_values = current.select(pl.col(self.column)).drop_nulls().collect().to_series()

        if len(ref_values) == 0 or len(curr_values) == 0:
            return 0.0

        if self.is_categorical:
            ref_freq = self.compute_category_frequencies(ref_values)
            curr_freq = self.compute_category_frequencies(curr_values)
            ref_aligned, curr_aligned = self.align_categories(ref_freq, curr_freq)
        else:
            ref_arr = ref_values.to_numpy()
            curr_arr = curr_values.to_numpy()

            # Create bins from combined data for fair comparison
            combined = np.concatenate([ref_arr, curr_arr])
            bin_edges = np.histogram_bin_edges(combined, bins=self.n_bins)

            ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
            curr_counts, _ = np.histogram(curr_arr, bins=bin_edges)

            # Normalize to probabilities
            ref_aligned = (ref_counts / ref_counts.sum()).tolist() if ref_counts.sum() > 0 else []
            curr_aligned = (curr_counts / curr_counts.sum()).tolist() if curr_counts.sum() > 0 else []

        if not ref_aligned or not curr_aligned:
            return 0.0

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_probs = np.array(ref_aligned) + epsilon
        curr_probs = np.array(curr_aligned) + epsilon

        # Renormalize
        ref_probs = ref_probs / ref_probs.sum()
        curr_probs = curr_probs / curr_probs.sum()

        # Calculate JS divergence
        js_div = spatial.distance.jensenshannon(ref_probs, curr_probs)

        return float(js_div)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        js_divergence = self.calculate_drift_score(self.reference_data, lf)

        if js_divergence > self.threshold:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="js_divergence_drift_detected",
                    count=1,
                    severity=self._calculate_severity(js_divergence, self.threshold),
                    details=f"Jensen-Shannon divergence: {js_divergence:.4f}",
                    expected=f"JS divergence < {self.threshold}",
                )
            )

        return issues
