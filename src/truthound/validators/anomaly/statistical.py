"""Statistical anomaly detection validators.

Pure statistical methods for outlier/anomaly detection that don't
require external ML libraries.
"""

from typing import Any

import polars as pl
import numpy as np

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.registry import register_validator
from truthound.validators.anomaly.base import (
    ColumnAnomalyValidator,
    StatisticalAnomalyMixin,
)


@register_validator
class IQRAnomalyValidator(ColumnAnomalyValidator, StatisticalAnomalyMixin):
    """Enhanced IQR-based anomaly detection with configurable sensitivity.

    The Interquartile Range (IQR) method is robust to outliers and works well
    for detecting anomalies in skewed distributions.

    Bounds: [Q1 - k*IQR, Q3 + k*IQR]
    - k=1.5: Standard outliers (Tukey's fences)
    - k=3.0: Extreme outliers

    Example:
        # Standard outlier detection
        validator = IQRAnomalyValidator(
            column="transaction_amount",
            iqr_multiplier=1.5,
            max_anomaly_ratio=0.05,
        )

        # Extreme outliers only
        validator = IQRAnomalyValidator(
            column="sensor_reading",
            iqr_multiplier=3.0,
        )
    """

    name = "iqr_anomaly"

    def __init__(
        self,
        column: str,
        iqr_multiplier: float = 1.5,
        max_anomaly_ratio: float = 0.1,
        detect_lower: bool = True,
        detect_upper: bool = True,
        **kwargs: Any,
    ):
        """Initialize IQR anomaly validator.

        Args:
            column: Column to check for anomalies
            iqr_multiplier: IQR multiplier (1.5=standard, 3.0=extreme)
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            detect_lower: Whether to detect lower bound anomalies
            detect_upper: Whether to detect upper bound anomalies
            **kwargs: Additional config
        """
        super().__init__(column=column, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.iqr_multiplier = iqr_multiplier
        self.detect_lower = detect_lower
        self.detect_upper = detect_upper

    def detect_column_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using IQR method."""
        lower, upper = self.compute_iqr_bounds(values, self.iqr_multiplier)

        anomaly_mask = np.zeros(len(values), dtype=bool)

        if self.detect_lower:
            anomaly_mask |= values < lower
        if self.detect_upper:
            anomaly_mask |= values > upper

        return anomaly_mask, {
            "lower_bound": lower,
            "upper_bound": upper,
            "iqr_multiplier": self.iqr_multiplier,
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Get non-null values
        values = (
            lf.select(pl.col(self.column).drop_nulls())
            .collect()
            .to_series()
            .to_numpy()
        )

        if len(values) < 4:
            return issues

        anomaly_mask, info = self.detect_column_anomalies(values)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(values)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            # Get sample anomalous values
            anomaly_values = values[anomaly_mask]
            sample_values = anomaly_values[:5].tolist() if len(anomaly_values) > 0 else []

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="iqr_anomaly_detected",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"IQR bounds: [{info['lower_bound']:.4f}, {info['upper_bound']:.4f}], "
                        f"anomaly ratio: {anomaly_ratio:.2%}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                    sample_values=sample_values,
                )
            )

        return issues


@register_validator
class MADAnomalyValidator(ColumnAnomalyValidator, StatisticalAnomalyMixin):
    """Median Absolute Deviation (MAD) based anomaly detection.

    MAD is more robust than standard deviation because it uses the median
    instead of the mean, making it resistant to outliers.

    Modified Z-score = 0.6745 * (x - median) / MAD

    The constant 0.6745 makes MAD consistent with std for normal distributions.

    Example:
        # Detect values with modified Z-score > 3.5
        validator = MADAnomalyValidator(
            column="measurement",
            threshold=3.5,
        )
    """

    name = "mad_anomaly"

    def __init__(
        self,
        column: str,
        threshold: float = 3.5,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize MAD anomaly validator.

        Args:
            column: Column to check for anomalies
            threshold: Modified Z-score threshold (default 3.5)
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(column=column, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.threshold = threshold

    def detect_column_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies using modified Z-score."""
        median, mad = self.compute_mad(values)

        if mad == 0:
            # Fall back to mean absolute deviation
            mad = np.mean(np.abs(values - median))
            if mad == 0:
                return np.zeros(len(values), dtype=bool), {
                    "median": median,
                    "mad": 0,
                    "threshold": self.threshold,
                }

        modified_zscores = np.abs(0.6745 * (values - median) / mad)
        anomaly_mask = modified_zscores > self.threshold

        return anomaly_mask, {
            "median": median,
            "mad": mad,
            "threshold": self.threshold,
            "max_zscore": float(np.max(modified_zscores)),
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        values = (
            lf.select(pl.col(self.column).drop_nulls())
            .collect()
            .to_series()
            .to_numpy()
        )

        if len(values) < 3:
            return issues

        anomaly_mask, info = self.detect_column_anomalies(values)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(values)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="mad_anomaly_detected",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"MAD={info['mad']:.4f}, median={info['median']:.4f}, "
                        f"max modified Z-score={info['max_zscore']:.2f}, "
                        f"anomaly ratio: {anomaly_ratio:.2%}"
                    ),
                    expected=f"|Modified Z-score| <= {self.threshold}",
                )
            )

        return issues


@register_validator
class GrubbsTestValidator(ColumnAnomalyValidator):
    """Grubbs' Test for detecting single outliers.

    Grubbs' test (also known as the ESD test) detects a single outlier in a
    univariate dataset that follows an approximately normal distribution.

    The test statistic is:
    G = max|X_i - mean| / std

    This validator can run iteratively to detect multiple outliers.

    Example:
        # Test for outliers with significance level 0.05
        validator = GrubbsTestValidator(
            column="measurement",
            alpha=0.05,
            max_iterations=10,
        )
    """

    name = "grubbs_test"

    def __init__(
        self,
        column: str,
        alpha: float = 0.05,
        max_iterations: int = 10,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize Grubbs test validator.

        Args:
            column: Column to check for outliers
            alpha: Significance level (default 0.05)
            max_iterations: Maximum iterations for iterative detection
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(column=column, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.alpha = alpha
        self.max_iterations = max_iterations

    def _grubbs_critical_value(self, n: int, alpha: float) -> float:
        """Calculate critical value for Grubbs test.

        Uses the t-distribution to compute the critical value.
        """
        from scipy import stats

        t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
        return g_critical

    def detect_column_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect outliers using iterative Grubbs test."""
        from scipy import stats

        anomaly_indices: list[int] = []
        remaining_values = values.copy()
        remaining_indices = np.arange(len(values))

        for iteration in range(self.max_iterations):
            n = len(remaining_values)
            if n < 3:
                break

            mean = np.mean(remaining_values)
            std = np.std(remaining_values, ddof=1)

            if std == 0:
                break

            # Find the most extreme value
            deviations = np.abs(remaining_values - mean)
            max_idx = np.argmax(deviations)
            g_statistic = deviations[max_idx] / std

            # Get critical value
            g_critical = self._grubbs_critical_value(n, self.alpha)

            if g_statistic > g_critical:
                # Mark as outlier
                anomaly_indices.append(remaining_indices[max_idx])
                # Remove from remaining
                remaining_values = np.delete(remaining_values, max_idx)
                remaining_indices = np.delete(remaining_indices, max_idx)
            else:
                # No more outliers
                break

        # Create anomaly mask
        anomaly_mask = np.zeros(len(values), dtype=bool)
        anomaly_mask[anomaly_indices] = True

        return anomaly_mask, {
            "outliers_found": len(anomaly_indices),
            "alpha": self.alpha,
            "iterations": iteration + 1,
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        values = (
            lf.select(pl.col(self.column).drop_nulls())
            .collect()
            .to_series()
            .to_numpy()
        )

        if len(values) < 3:
            return issues

        anomaly_mask, info = self.detect_column_anomalies(values)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(values)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            # Get sample anomalous values
            anomaly_values = values[anomaly_mask]
            sample_values = anomaly_values[:5].tolist() if len(anomaly_values) > 0 else []

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="grubbs_outlier_detected",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Grubbs test (alpha={self.alpha}) found {anomaly_count} outliers "
                        f"in {info['iterations']} iterations, anomaly ratio: {anomaly_ratio:.2%}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                    sample_values=sample_values,
                )
            )

        return issues


@register_validator
class TukeyFencesValidator(ColumnAnomalyValidator, StatisticalAnomalyMixin):
    """Tukey's Fences for outlier detection with inner and outer fences.

    Tukey's method defines two sets of fences:
    - Inner fences: [Q1 - 1.5*IQR, Q3 + 1.5*IQR] - mild outliers outside
    - Outer fences: [Q1 - 3.0*IQR, Q3 + 3.0*IQR] - extreme outliers outside

    This validator separately tracks mild and extreme outliers.

    Example:
        validator = TukeyFencesValidator(
            column="price",
            detect_mild=True,
            detect_extreme=True,
        )
    """

    name = "tukey_fences"

    def __init__(
        self,
        column: str,
        detect_mild: bool = True,
        detect_extreme: bool = True,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize Tukey's fences validator.

        Args:
            column: Column to check for outliers
            detect_mild: Whether to report mild outliers (k=1.5)
            detect_extreme: Whether to report extreme outliers (k=3.0)
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(column=column, max_anomaly_ratio=max_anomaly_ratio, **kwargs)
        self.detect_mild = detect_mild
        self.detect_extreme = detect_extreme

    def detect_column_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect mild and extreme outliers using Tukey's fences."""
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        # Inner fences (k=1.5)
        inner_lower = q1 - 1.5 * iqr
        inner_upper = q3 + 1.5 * iqr

        # Outer fences (k=3.0)
        outer_lower = q1 - 3.0 * iqr
        outer_upper = q3 + 3.0 * iqr

        # Classify outliers
        extreme_mask = (values < outer_lower) | (values > outer_upper)
        mild_mask = ((values < inner_lower) | (values > inner_upper)) & ~extreme_mask

        # Combined anomaly mask based on settings
        anomaly_mask = np.zeros(len(values), dtype=bool)
        if self.detect_extreme:
            anomaly_mask |= extreme_mask
        if self.detect_mild:
            anomaly_mask |= mild_mask

        return anomaly_mask, {
            "inner_fences": (inner_lower, inner_upper),
            "outer_fences": (outer_lower, outer_upper),
            "mild_count": int(mild_mask.sum()),
            "extreme_count": int(extreme_mask.sum()),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        values = (
            lf.select(pl.col(self.column).drop_nulls())
            .collect()
            .to_series()
            .to_numpy()
        )

        if len(values) < 4:
            return issues

        anomaly_mask, info = self.detect_column_anomalies(values)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(values)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            inner = info["inner_fences"]
            outer = info["outer_fences"]

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="tukey_outlier_detected",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Inner fences: [{inner[0]:.4f}, {inner[1]:.4f}], "
                        f"Outer fences: [{outer[0]:.4f}, {outer[1]:.4f}], "
                        f"Mild: {info['mild_count']}, Extreme: {info['extreme_count']}, "
                        f"Total anomaly ratio: {anomaly_ratio:.2%}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues


@register_validator
class PercentileAnomalyValidator(ColumnAnomalyValidator):
    """Percentile-based anomaly detection.

    Flags values outside the specified percentile range as anomalies.
    Useful when you have domain knowledge about acceptable ranges.

    Example:
        # Flag values outside 1st and 99th percentiles
        validator = PercentileAnomalyValidator(
            column="response_time",
            lower_percentile=1,
            upper_percentile=99,
        )
    """

    name = "percentile_anomaly"

    def __init__(
        self,
        column: str,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        max_anomaly_ratio: float = 0.1,
        **kwargs: Any,
    ):
        """Initialize percentile anomaly validator.

        Args:
            column: Column to check for anomalies
            lower_percentile: Lower percentile threshold (0-100)
            upper_percentile: Upper percentile threshold (0-100)
            max_anomaly_ratio: Maximum acceptable ratio of anomalies
            **kwargs: Additional config
        """
        super().__init__(column=column, max_anomaly_ratio=max_anomaly_ratio, **kwargs)

        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError(
                f"Invalid percentile range: [{lower_percentile}, {upper_percentile}]"
            )

        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def detect_column_anomalies(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect anomalies outside percentile bounds."""
        lower_bound = np.percentile(values, self.lower_percentile)
        upper_bound = np.percentile(values, self.upper_percentile)

        anomaly_mask = (values < lower_bound) | (values > upper_bound)

        return anomaly_mask, {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
        }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        values = (
            lf.select(pl.col(self.column).drop_nulls())
            .collect()
            .to_series()
            .to_numpy()
        )

        if len(values) < 10:
            return issues

        anomaly_mask, info = self.detect_column_anomalies(values)
        anomaly_count = int(anomaly_mask.sum())
        anomaly_ratio = anomaly_count / len(values)

        if anomaly_ratio > self.max_anomaly_ratio:
            severity = self._calculate_severity(anomaly_ratio)

            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="percentile_anomaly_detected",
                    count=anomaly_count,
                    severity=severity,
                    details=(
                        f"Bounds at P{self.lower_percentile}-P{self.upper_percentile}: "
                        f"[{info['lower_bound']:.4f}, {info['upper_bound']:.4f}], "
                        f"anomaly ratio: {anomaly_ratio:.2%}"
                    ),
                    expected=f"Anomaly ratio <= {self.max_anomaly_ratio:.2%}",
                )
            )

        return issues
