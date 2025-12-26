"""Statistical drift validators.

Validators using statistical tests to detect distribution drift.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.drift.base import (
    ColumnDriftValidator,
    NumericDriftMixin,
    CategoricalDriftMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class KSTestValidator(ColumnDriftValidator, NumericDriftMixin):
    """Kolmogorov-Smirnov test for detecting distribution drift in numeric columns.

    The KS test compares the empirical cumulative distribution functions (ECDFs)
    of two samples. It's sensitive to differences in location, scale, and shape.

    Example:
        # Detect drift in purchase amounts
        validator = KSTestValidator(
            column="purchase_amount",
            reference_data=training_df,
            p_value_threshold=0.05,
        )
        issues = validator.validate(production_df.lazy())

        # With custom statistic threshold
        validator = KSTestValidator(
            column="feature_value",
            reference_data=baseline_df,
            statistic_threshold=0.1,  # Max allowed KS statistic
        )
    """

    name = "ks_test"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        p_value_threshold: float = 0.05,
        statistic_threshold: float | None = None,
        **kwargs: Any,
    ):
        """Initialize KS test validator.

        Args:
            column: Numeric column to test
            reference_data: Baseline data for comparison
            p_value_threshold: P-value below which drift is detected (default 0.05)
            statistic_threshold: Optional KS statistic threshold (overrides p-value)
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.p_value_threshold = p_value_threshold
        self.statistic_threshold = statistic_threshold

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> tuple[float, float]:
        """Calculate KS statistic and p-value.

        Returns:
            Tuple of (ks_statistic, p_value)
        """
        from scipy import stats

        ref_values = self._get_column_values(reference).to_numpy()
        curr_values = self._get_column_values(current).to_numpy()

        if len(ref_values) == 0 or len(curr_values) == 0:
            return 0.0, 1.0

        statistic, p_value = stats.ks_2samp(ref_values, curr_values)
        return float(statistic), float(p_value)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        ks_statistic, p_value = self.calculate_drift_score(self.reference_data, lf)

        # Check for drift
        drift_detected = False
        if self.statistic_threshold is not None:
            drift_detected = ks_statistic > self.statistic_threshold
            threshold_desc = f"KS statistic > {self.statistic_threshold}"
        else:
            drift_detected = p_value < self.p_value_threshold
            threshold_desc = f"p-value < {self.p_value_threshold}"

        if drift_detected:
            severity = self._calculate_severity_from_ks(ks_statistic)
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="distribution_drift_detected",
                    count=1,
                    severity=severity,
                    details=f"KS test detected drift: statistic={ks_statistic:.4f}, p-value={p_value:.4e}",
                    expected=f"No significant drift ({threshold_desc})",
                )
            )

        return issues

    def _calculate_severity_from_ks(self, ks_statistic: float) -> Severity:
        """Calculate severity based on KS statistic magnitude."""
        if ks_statistic < 0.1:
            return Severity.LOW
        elif ks_statistic < 0.2:
            return Severity.MEDIUM
        elif ks_statistic < 0.3:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


@register_validator
class StreamingKSTestValidator(ColumnDriftValidator, NumericDriftMixin):
    """Memory-efficient KS test using streaming ECDF.

    This validator uses T-Digest for streaming quantile estimation,
    allowing KS test computation without loading both datasets into memory.

    Memory Complexity:
        - Standard KS test: O(n + m) for n, m samples
        - This implementation: O(compression) constant memory

    Use this for datasets > 1M rows where standard KS test would be slow or OOM.

    Example:
        # For large datasets
        validator = StreamingKSTestValidator(
            column="purchase_amount",
            reference_data=huge_baseline_df,  # Can be very large
            compression=200,  # T-Digest compression factor
        )
    """

    name = "streaming_ks_test"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        p_value_threshold: float = 0.05,
        statistic_threshold: float | None = None,
        compression: float = 200.0,
        chunk_size: int = 100000,
        **kwargs: Any,
    ):
        """Initialize streaming KS test validator.

        Args:
            column: Numeric column to test
            reference_data: Baseline data for comparison
            p_value_threshold: P-value below which drift is detected
            statistic_threshold: Optional KS statistic threshold
            compression: T-Digest compression factor (higher = more accurate)
            chunk_size: Chunk size for streaming processing
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.p_value_threshold = p_value_threshold
        self.statistic_threshold = statistic_threshold
        self.compression = compression
        self.chunk_size = chunk_size

        # Cached reference ECDF
        self._ref_ecdf = None

    def _build_reference_ecdf(self) -> None:
        """Build streaming ECDF from reference data."""
        if self._ref_ecdf is not None:
            return

        from truthound.validators.memory import StreamingECDF
        from truthound.validators.memory.base import DataChunker

        self._ref_ecdf = StreamingECDF(compression=self.compression)

        chunker = DataChunker(
            chunk_size=self.chunk_size,
            columns=[self.column],
            drop_nulls=True,
        )

        for chunk_df in chunker.iterate(self.reference_data):
            values = chunk_df.to_series().to_numpy()
            self._ref_ecdf.update(values)

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> tuple[float, float]:
        """Calculate streaming KS statistic and approximate p-value.

        Returns:
            Tuple of (ks_statistic, p_value)
        """
        import math
        import numpy as np
        from truthound.validators.memory import StreamingECDF
        from truthound.validators.memory.base import DataChunker

        # Build reference ECDF if not cached
        self._build_reference_ecdf()

        # Build current ECDF
        curr_ecdf = StreamingECDF(compression=self.compression)

        chunker = DataChunker(
            chunk_size=self.chunk_size,
            columns=[self.column],
            drop_nulls=True,
        )

        for chunk_df in chunker.iterate(current):
            values = chunk_df.to_series().to_numpy()
            curr_ecdf.update(values)

        if self._ref_ecdf.count == 0 or curr_ecdf.count == 0:
            return 0.0, 1.0

        # Compute max deviation at quantile points
        quantile_points = np.linspace(0.001, 0.999, 1000)

        max_deviation = 0.0
        for q in quantile_points:
            ref_val = self._ref_ecdf.quantile(q)
            curr_val = curr_ecdf.quantile(q)

            # CDF differences at both values
            dev1 = abs(self._ref_ecdf.cdf(ref_val) - curr_ecdf.cdf(ref_val))
            dev2 = abs(self._ref_ecdf.cdf(curr_val) - curr_ecdf.cdf(curr_val))

            max_deviation = max(max_deviation, dev1, dev2)

        ks_stat = max_deviation

        # Approximate p-value using asymptotic distribution
        n_eff = min(self._ref_ecdf.count, curr_ecdf.count)
        lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * ks_stat

        p_value = 0.0
        for k in range(1, 100):
            term = 2 * ((-1) ** (k + 1)) * math.exp(-2 * k * k * lambda_val * lambda_val)
            p_value += term
            if abs(term) < 1e-10:
                break

        p_value = max(0.0, min(1.0, p_value))

        return float(ks_stat), float(p_value)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        ks_statistic, p_value = self.calculate_drift_score(self.reference_data, lf)

        # Check for drift
        drift_detected = False
        if self.statistic_threshold is not None:
            drift_detected = ks_statistic > self.statistic_threshold
            threshold_desc = f"KS statistic > {self.statistic_threshold}"
        else:
            drift_detected = p_value < self.p_value_threshold
            threshold_desc = f"p-value < {self.p_value_threshold}"

        if drift_detected:
            severity = self._calculate_severity_from_ks(ks_statistic)
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="streaming_ks_drift_detected",
                    count=1,
                    severity=severity,
                    details=(
                        f"Streaming KS test detected drift: "
                        f"statistic={ks_statistic:.4f}, p-value={p_value:.4e} "
                        f"(compression={self.compression})"
                    ),
                    expected=f"No significant drift ({threshold_desc})",
                )
            )

        return issues

    def _calculate_severity_from_ks(self, ks_statistic: float) -> Severity:
        """Calculate severity based on KS statistic magnitude."""
        if ks_statistic < 0.1:
            return Severity.LOW
        elif ks_statistic < 0.2:
            return Severity.MEDIUM
        elif ks_statistic < 0.3:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


@register_validator
class ChiSquareDriftValidator(ColumnDriftValidator, CategoricalDriftMixin):
    """Chi-square test for detecting drift in categorical columns.

    Compares the frequency distribution of categories between reference
    and current data using Pearson's chi-square test.

    Example:
        # Detect drift in product categories
        validator = ChiSquareDriftValidator(
            column="product_category",
            reference_data=baseline_df,
            p_value_threshold=0.05,
        )
    """

    name = "chi_square_drift"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        p_value_threshold: float = 0.05,
        min_expected_frequency: float = 5.0,
        **kwargs: Any,
    ):
        """Initialize chi-square drift validator.

        Args:
            column: Categorical column to test
            reference_data: Baseline data for comparison
            p_value_threshold: P-value below which drift is detected
            min_expected_frequency: Minimum expected frequency per category
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.p_value_threshold = p_value_threshold
        self.min_expected_frequency = min_expected_frequency

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> tuple[float, float]:
        """Calculate chi-square statistic and p-value.

        Returns:
            Tuple of (chi2_statistic, p_value)
        """
        from scipy import stats
        import numpy as np

        ref_values = self._get_column_values(reference)
        curr_values = self._get_column_values(current)

        ref_freq = self.compute_category_frequencies(ref_values)
        curr_freq = self.compute_category_frequencies(curr_values)

        if not ref_freq or not curr_freq:
            return 0.0, 1.0

        ref_aligned, curr_aligned = self.align_categories(ref_freq, curr_freq)

        # Convert to counts
        n_curr = len(curr_values)
        observed = np.array([f * n_curr for f in curr_aligned])
        expected = np.array([f * n_curr for f in ref_aligned])

        # Handle zero expected frequencies
        mask = expected >= self.min_expected_frequency
        if not mask.any():
            return 0.0, 1.0

        observed = observed[mask]
        expected = expected[mask]

        if len(observed) < 2:
            return 0.0, 1.0

        # Normalize expected to match observed sum (required by scipy.stats.chisquare)
        expected = expected * (observed.sum() / expected.sum())

        chi2, p_value = stats.chisquare(observed, expected)
        return float(chi2), float(p_value)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        chi2_statistic, p_value = self.calculate_drift_score(self.reference_data, lf)

        if p_value < self.p_value_threshold:
            severity = self._calculate_severity_from_pvalue(p_value)
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="categorical_drift_detected",
                    count=1,
                    severity=severity,
                    details=f"Chi-square test detected drift: χ²={chi2_statistic:.2f}, p-value={p_value:.4e}",
                    expected=f"No significant drift (p-value >= {self.p_value_threshold})",
                )
            )

        return issues

    def _calculate_severity_from_pvalue(self, p_value: float) -> Severity:
        """Calculate severity based on p-value."""
        if p_value >= 0.01:
            return Severity.MEDIUM
        elif p_value >= 0.001:
            return Severity.HIGH
        else:
            return Severity.CRITICAL


@register_validator
class WassersteinDriftValidator(ColumnDriftValidator, NumericDriftMixin):
    """Wasserstein distance (Earth Mover's Distance) for numeric drift detection.

    The Wasserstein distance measures the minimum "work" needed to transform
    one distribution into another. It's more interpretable than KS test
    as it's in the same units as the data.

    Example:
        # Detect drift with interpretable distance
        validator = WassersteinDriftValidator(
            column="price",
            reference_data=baseline_df,
            threshold=10.0,  # Maximum allowed distance in price units
        )
    """

    name = "wasserstein_drift"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        threshold: float,
        normalize: bool = False,
        **kwargs: Any,
    ):
        """Initialize Wasserstein drift validator.

        Args:
            column: Numeric column to test
            reference_data: Baseline data for comparison
            threshold: Maximum allowed Wasserstein distance
            normalize: If True, normalize by reference std deviation
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.threshold = threshold
        self.normalize = normalize

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> float:
        """Calculate Wasserstein distance.

        Returns:
            Wasserstein distance
        """
        from scipy import stats

        ref_values = self._get_column_values(reference).to_numpy()
        curr_values = self._get_column_values(current).to_numpy()

        if len(ref_values) == 0 or len(curr_values) == 0:
            return 0.0

        distance = stats.wasserstein_distance(ref_values, curr_values)

        if self.normalize:
            ref_std = ref_values.std()
            if ref_std > 0:
                distance = distance / ref_std

        return float(distance)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        distance = self.calculate_drift_score(self.reference_data, lf)

        if distance > self.threshold:
            severity = self._calculate_severity(distance, self.threshold)
            unit_desc = "std deviations" if self.normalize else "units"
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="wasserstein_drift_detected",
                    count=1,
                    severity=severity,
                    details=f"Wasserstein distance: {distance:.4f} {unit_desc}",
                    expected=f"Distance <= {self.threshold} {unit_desc}",
                )
            )

        return issues
