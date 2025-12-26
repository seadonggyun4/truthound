"""Population Stability Index (PSI) drift validators.

PSI is widely used in credit risk and ML model monitoring to detect
distribution shifts. It's particularly valued for its interpretability.

Memory Optimization:
    These validators support caching of reference statistics to avoid
    keeping large reference datasets in memory. Use cache_reference=True
    (default) and call cache_and_release() for memory-constrained environments.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.drift.base import (
    ColumnDriftValidator,
    NumericDriftMixin,
    CategoricalDriftMixin,
)
from truthound.validators.cache import NumericStatistics, CategoricalStatistics
from truthound.validators.registry import register_validator


@register_validator
class PSIValidator(ColumnDriftValidator, NumericDriftMixin, CategoricalDriftMixin):
    """Population Stability Index (PSI) for detecting distribution drift.

    PSI is a symmetric measure that quantifies the shift in distribution
    between two datasets. It's widely used in credit risk modeling and
    ML model monitoring.

    Interpretation:
        - PSI < 0.1: No significant drift (stable)
        - 0.1 <= PSI < 0.25: Moderate drift (investigate)
        - PSI >= 0.25: Significant drift (action required)

    Memory Optimization:
        When cache_reference=True (default), reference statistics are cached
        and raw data can be released. This dramatically reduces memory usage
        for large reference datasets.

        # Memory-efficient usage:
        validator = PSIValidator(
            column="credit_score",
            reference_data=large_df,  # 10GB DataFrame
            cache_reference=True,
        )
        validator.cache_and_release()  # Release raw data, keep ~1KB stats

    Example:
        # Numeric column with auto-binning
        validator = PSIValidator(
            column="credit_score",
            reference_data=baseline_df,
            n_bins=10,
            threshold=0.25,
        )

        # Categorical column
        validator = PSIValidator(
            column="product_type",
            reference_data=baseline_df,
            is_categorical=True,
        )
    """

    name = "psi"
    category = "drift"

    # Standard PSI thresholds
    THRESHOLD_STABLE = 0.1
    THRESHOLD_MODERATE = 0.25

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        threshold: float = 0.25,
        n_bins: int = 10,
        is_categorical: bool = False,
        min_frequency: float = 0.0001,
        **kwargs: Any,
    ):
        """Initialize PSI validator.

        Args:
            column: Column to check for drift
            reference_data: Baseline data for comparison
            threshold: PSI threshold for drift detection (default 0.25)
            n_bins: Number of bins for numeric columns
            is_categorical: True if column is categorical
            min_frequency: Minimum frequency to avoid log(0)
            **kwargs: Additional config (including cache_reference)
        """
        # Pass is_categorical to base class for proper caching
        super().__init__(
            column=column,
            reference_data=reference_data,
            is_categorical=is_categorical,
            n_histogram_bins=n_bins,
            **kwargs,
        )
        self.threshold = threshold
        self.n_bins = n_bins
        self.min_frequency = min_frequency

        # Legacy cache fields (for backward compatibility)
        self._ref_distribution: dict | list | None = None
        self._bin_edges: list[float] | None = None

    def _compute_reference_distribution(self) -> None:
        """Pre-compute reference distribution for efficiency.

        This method now uses cached statistics when available,
        dramatically reducing memory usage for large datasets.
        """
        # Try to use cached statistics first
        if self._cache_reference:
            try:
                stats = self.get_reference_statistics()

                if self.is_categorical:
                    if isinstance(stats, CategoricalStatistics):
                        self._ref_distribution = stats.frequencies
                        return
                else:
                    if isinstance(stats, NumericStatistics):
                        self._bin_edges = stats.histogram_edges
                        self._ref_distribution = stats.histogram_counts
                        return
            except (ValueError, AttributeError):
                # Fall back to computing from raw data
                pass

        # Fallback: compute from raw reference data
        if self._reference_data is None:
            self._ref_distribution = [] if not self.is_categorical else {}
            self._bin_edges = []
            return

        ref_values = self._get_column_values(self.reference_data)

        if self.is_categorical:
            self._ref_distribution = self.compute_category_frequencies(ref_values)
        else:
            # Compute histogram bins from reference data
            arr = ref_values.to_numpy()
            arr = arr[~np.isnan(arr)]

            if len(arr) == 0:
                self._ref_distribution = []
                self._bin_edges = []
                return

            # Create bins based on reference data quantiles for robustness
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self._bin_edges = np.percentile(arr, percentiles).tolist()

            # Ensure unique bin edges
            self._bin_edges = sorted(set(self._bin_edges))

            # Compute reference frequencies
            counts, _ = np.histogram(arr, bins=self._bin_edges)
            total = counts.sum()
            if total > 0:
                self._ref_distribution = (counts / total).tolist()
            else:
                self._ref_distribution = [0.0] * (len(self._bin_edges) - 1)

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> float:
        """Calculate PSI score.

        PSI = Σ (P_current - P_reference) * ln(P_current / P_reference)

        This method uses cached statistics when available for memory efficiency.

        Returns:
            PSI score
        """
        # Ensure reference distribution is computed (uses cache if available)
        if self._ref_distribution is None:
            self._compute_reference_distribution()

        curr_values = self._get_column_values(current)

        if self.is_categorical:
            return self._calculate_categorical_psi(curr_values)
        else:
            return self._calculate_numeric_psi(curr_values)

    def _calculate_numeric_psi(self, curr_values: pl.Series) -> float:
        """Calculate PSI for numeric column."""
        if not self._bin_edges or len(self._bin_edges) < 2:
            return 0.0

        arr = curr_values.to_numpy()
        arr = arr[~np.isnan(arr)]

        if len(arr) == 0:
            return 0.0

        # Compute current frequencies using reference bins
        counts, _ = np.histogram(arr, bins=self._bin_edges)
        total = counts.sum()
        if total > 0:
            curr_freq = (counts / total).tolist()
        else:
            return 0.0

        return self._compute_psi(self._ref_distribution, curr_freq)

    def _calculate_categorical_psi(self, curr_values: pl.Series) -> float:
        """Calculate PSI for categorical column."""
        curr_freq = self.compute_category_frequencies(curr_values)

        if not self._ref_distribution or not curr_freq:
            return 0.0

        ref_aligned, curr_aligned = self.align_categories(
            self._ref_distribution, curr_freq
        )

        return self._compute_psi(ref_aligned, curr_aligned)

    def _compute_psi(
        self, ref_freq: list[float], curr_freq: list[float]
    ) -> float:
        """Compute PSI from aligned frequency lists.

        Args:
            ref_freq: Reference frequencies
            curr_freq: Current frequencies

        Returns:
            PSI value
        """
        psi = 0.0

        for ref_p, curr_p in zip(ref_freq, curr_freq):
            # Apply minimum frequency to avoid log(0)
            ref_p = max(ref_p, self.min_frequency)
            curr_p = max(curr_p, self.min_frequency)

            psi += (curr_p - ref_p) * np.log(curr_p / ref_p)

        return float(psi)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        psi = self.calculate_drift_score(self.reference_data, lf)

        if psi >= self.threshold:
            severity = self._get_psi_severity(psi)
            interpretation = self._get_psi_interpretation(psi)

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="psi_drift_detected",
                    count=1,
                    severity=severity,
                    details=f"PSI = {psi:.4f} ({interpretation})",
                    expected=f"PSI < {self.threshold}",
                )
            )

        return issues

    def _get_psi_severity(self, psi: float) -> Severity:
        """Get severity based on PSI value."""
        if psi < self.THRESHOLD_STABLE:
            return Severity.LOW
        elif psi < self.THRESHOLD_MODERATE:
            return Severity.MEDIUM
        elif psi < 0.5:
            return Severity.HIGH
        else:
            return Severity.CRITICAL

    def _get_psi_interpretation(self, psi: float) -> str:
        """Get human-readable interpretation of PSI value."""
        if psi < self.THRESHOLD_STABLE:
            return "No significant drift"
        elif psi < self.THRESHOLD_MODERATE:
            return "Moderate drift - investigate"
        else:
            return "Significant drift - action required"


@register_validator
class CSIValidator(ColumnDriftValidator, NumericDriftMixin, CategoricalDriftMixin):
    """Characteristic Stability Index (CSI) - bin-level PSI analysis.

    CSI provides per-bin PSI contributions, helping identify which
    parts of the distribution are drifting.

    Example:
        validator = CSIValidator(
            column="age",
            reference_data=baseline_df,
            threshold_per_bin=0.05,  # Max PSI contribution per bin
        )
    """

    name = "csi"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        threshold_per_bin: float = 0.05,
        n_bins: int = 10,
        is_categorical: bool = False,
        **kwargs: Any,
    ):
        """Initialize CSI validator.

        Args:
            column: Column to check
            reference_data: Baseline data
            threshold_per_bin: Max allowed PSI contribution per bin
            n_bins: Number of bins for numeric columns
            is_categorical: True if column is categorical
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.threshold_per_bin = threshold_per_bin
        self.n_bins = n_bins
        self.is_categorical = is_categorical

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> list[tuple[str, float]]:
        """Calculate per-bin/category PSI contributions.

        Returns:
            List of (bin_label, psi_contribution) tuples
        """
        ref_values = self._get_column_values(reference)
        curr_values = self._get_column_values(current)

        if self.is_categorical:
            return self._calculate_categorical_csi(ref_values, curr_values)
        else:
            return self._calculate_numeric_csi(ref_values, curr_values)

    def _calculate_numeric_csi(
        self, ref_values: pl.Series, curr_values: pl.Series
    ) -> list[tuple[str, float]]:
        """Calculate CSI for numeric column."""
        ref_arr = ref_values.to_numpy()
        curr_arr = curr_values.to_numpy()

        ref_arr = ref_arr[~np.isnan(ref_arr)]
        curr_arr = curr_arr[~np.isnan(curr_arr)]

        if len(ref_arr) == 0 or len(curr_arr) == 0:
            return []

        # Create bins
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        bin_edges = np.percentile(ref_arr, percentiles)
        bin_edges = sorted(set(bin_edges))

        # Calculate frequencies
        ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
        curr_counts, _ = np.histogram(curr_arr, bins=bin_edges)

        ref_freq = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else ref_counts
        curr_freq = curr_counts / curr_counts.sum() if curr_counts.sum() > 0 else curr_counts

        # Calculate per-bin contributions
        results = []
        min_freq = 0.0001

        for i, (ref_p, curr_p) in enumerate(zip(ref_freq, curr_freq)):
            ref_p = max(ref_p, min_freq)
            curr_p = max(curr_p, min_freq)

            contribution = (curr_p - ref_p) * np.log(curr_p / ref_p)
            bin_label = f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})"
            results.append((bin_label, float(contribution)))

        return results

    def _calculate_categorical_csi(
        self, ref_values: pl.Series, curr_values: pl.Series
    ) -> list[tuple[str, float]]:
        """Calculate CSI for categorical column."""
        ref_freq = self.compute_category_frequencies(ref_values)
        curr_freq = self.compute_category_frequencies(curr_values)

        if not ref_freq or not curr_freq:
            return []

        all_categories = sorted(set(ref_freq.keys()) | set(curr_freq.keys()))
        min_freq = 0.0001

        results = []
        for cat in all_categories:
            ref_p = max(ref_freq.get(cat, 0.0), min_freq)
            curr_p = max(curr_freq.get(cat, 0.0), min_freq)

            contribution = (curr_p - ref_p) * np.log(curr_p / ref_p)
            results.append((str(cat), float(contribution)))

        return results

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        contributions = self.calculate_drift_score(self.reference_data, lf)

        # Find bins/categories exceeding threshold
        problematic = [
            (label, psi) for label, psi in contributions
            if abs(psi) > self.threshold_per_bin
        ]

        if problematic:
            total_psi = sum(psi for _, psi in contributions)
            details_list = [f"{label}: {psi:.4f}" for label, psi in problematic[:5]]

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="csi_drift_detected",
                    count=len(problematic),
                    severity=self._calculate_severity(
                        max(abs(psi) for _, psi in problematic),
                        self.threshold_per_bin
                    ),
                    details=f"Total PSI={total_psi:.4f}. Problematic bins: {', '.join(details_list)}",
                    expected=f"Per-bin PSI < {self.threshold_per_bin}",
                )
            )

        return issues
