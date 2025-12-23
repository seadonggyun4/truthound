"""Numeric drift validators.

Validators for detecting drift in numeric column statistics.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue
from truthound.validators.drift.base import ColumnDriftValidator, DriftValidator
from truthound.validators.registry import register_validator


@register_validator
class MeanDriftValidator(ColumnDriftValidator):
    """Detects drift in column mean value.

    Example:
        # Detect if mean changes by more than 10%
        validator = MeanDriftValidator(
            column="price",
            reference_data=baseline_df,
            threshold_pct=10.0,
        )

        # Detect if mean changes by more than 5 units
        validator = MeanDriftValidator(
            column="temperature",
            reference_data=baseline_df,
            threshold_abs=5.0,
        )
    """

    name = "mean_drift"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        threshold_pct: float | None = None,
        threshold_abs: float | None = None,
        **kwargs: Any,
    ):
        """Initialize mean drift validator.

        Args:
            column: Numeric column to check
            reference_data: Baseline data for comparison
            threshold_pct: Maximum allowed percentage change
            threshold_abs: Maximum allowed absolute change
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.threshold_pct = threshold_pct
        self.threshold_abs = threshold_abs

        if threshold_pct is None and threshold_abs is None:
            raise ValueError("At least one of 'threshold_pct' or 'threshold_abs' required")

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> tuple[float, float, float]:
        """Calculate mean values and drift.

        Returns:
            Tuple of (ref_mean, curr_mean, pct_change)
        """
        ref_mean = reference.select(pl.col(self.column).mean()).collect().item()
        curr_mean = current.select(pl.col(self.column).mean()).collect().item()

        if ref_mean is None or curr_mean is None:
            return 0.0, 0.0, 0.0

        abs_change = abs(curr_mean - ref_mean)
        pct_change = (abs_change / abs(ref_mean) * 100) if ref_mean != 0 else 0.0

        return float(ref_mean), float(curr_mean), float(pct_change)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        ref_mean, curr_mean, pct_change = self.calculate_drift_score(
            self.reference_data, lf
        )
        abs_change = abs(curr_mean - ref_mean)

        drift_detected = False
        threshold_desc = ""

        if self.threshold_pct is not None and pct_change > self.threshold_pct:
            drift_detected = True
            threshold_desc = f"{pct_change:.2f}% > {self.threshold_pct}%"

        if self.threshold_abs is not None and abs_change > self.threshold_abs:
            drift_detected = True
            threshold_desc = f"{abs_change:.4f} > {self.threshold_abs}"

        if drift_detected:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="mean_drift_detected",
                    count=1,
                    severity=self._calculate_severity(pct_change, self.threshold_pct or 10.0),
                    details=f"Mean changed from {ref_mean:.4f} to {curr_mean:.4f} ({pct_change:+.2f}%)",
                    expected=f"Mean change within threshold ({threshold_desc})",
                )
            )

        return issues


@register_validator
class VarianceDriftValidator(ColumnDriftValidator):
    """Detects drift in column variance/standard deviation.

    Example:
        validator = VarianceDriftValidator(
            column="score",
            reference_data=baseline_df,
            threshold_pct=20.0,  # Max 20% change in variance
        )
    """

    name = "variance_drift"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        threshold_pct: float = 20.0,
        use_std: bool = True,
        **kwargs: Any,
    ):
        """Initialize variance drift validator.

        Args:
            column: Numeric column to check
            reference_data: Baseline data for comparison
            threshold_pct: Maximum allowed percentage change
            use_std: If True, compare std deviation instead of variance
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.threshold_pct = threshold_pct
        self.use_std = use_std

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> tuple[float, float, float]:
        """Calculate variance/std values and drift.

        Returns:
            Tuple of (ref_value, curr_value, pct_change)
        """
        if self.use_std:
            ref_val = reference.select(pl.col(self.column).std()).collect().item()
            curr_val = current.select(pl.col(self.column).std()).collect().item()
        else:
            ref_val = reference.select(pl.col(self.column).var()).collect().item()
            curr_val = current.select(pl.col(self.column).var()).collect().item()

        if ref_val is None or curr_val is None:
            return 0.0, 0.0, 0.0

        abs_change = abs(curr_val - ref_val)
        pct_change = (abs_change / ref_val * 100) if ref_val != 0 else 0.0

        return float(ref_val), float(curr_val), float(pct_change)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        ref_val, curr_val, pct_change = self.calculate_drift_score(
            self.reference_data, lf
        )

        metric_name = "Std deviation" if self.use_std else "Variance"

        if pct_change > self.threshold_pct:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="variance_drift_detected",
                    count=1,
                    severity=self._calculate_severity(pct_change, self.threshold_pct),
                    details=f"{metric_name} changed from {ref_val:.4f} to {curr_val:.4f} ({pct_change:+.2f}%)",
                    expected=f"{metric_name} change < {self.threshold_pct}%",
                )
            )

        return issues


@register_validator
class QuantileDriftValidator(ColumnDriftValidator):
    """Detects drift in specific quantiles of a distribution.

    Example:
        # Monitor median and 95th percentile
        validator = QuantileDriftValidator(
            column="response_time",
            reference_data=baseline_df,
            quantiles=[0.5, 0.95],
            threshold_pct=15.0,
        )
    """

    name = "quantile_drift"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        quantiles: list[float] = [0.25, 0.5, 0.75],
        threshold_pct: float = 15.0,
        **kwargs: Any,
    ):
        """Initialize quantile drift validator.

        Args:
            column: Numeric column to check
            reference_data: Baseline data for comparison
            quantiles: List of quantiles to monitor (0.0 to 1.0)
            threshold_pct: Maximum allowed percentage change per quantile
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.quantiles = quantiles
        self.threshold_pct = threshold_pct

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> dict[float, tuple[float, float, float]]:
        """Calculate quantile values and drift.

        Returns:
            Dict of quantile -> (ref_value, curr_value, pct_change)
        """
        results = {}

        for q in self.quantiles:
            ref_val = reference.select(
                pl.col(self.column).quantile(q)
            ).collect().item()
            curr_val = current.select(
                pl.col(self.column).quantile(q)
            ).collect().item()

            if ref_val is None or curr_val is None:
                results[q] = (0.0, 0.0, 0.0)
                continue

            abs_change = abs(curr_val - ref_val)
            pct_change = (abs_change / abs(ref_val) * 100) if ref_val != 0 else 0.0

            results[q] = (float(ref_val), float(curr_val), float(pct_change))

        return results

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        drift_results = self.calculate_drift_score(self.reference_data, lf)

        drifted_quantiles = []
        for q, (ref_val, curr_val, pct_change) in drift_results.items():
            if pct_change > self.threshold_pct:
                drifted_quantiles.append(
                    f"Q{int(q*100)}%: {ref_val:.2f}→{curr_val:.2f} ({pct_change:+.1f}%)"
                )

        if drifted_quantiles:
            max_pct = max(pct for _, _, pct in drift_results.values())
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="quantile_drift_detected",
                    count=len(drifted_quantiles),
                    severity=self._calculate_severity(max_pct, self.threshold_pct),
                    details=f"Quantile drift: {'; '.join(drifted_quantiles)}",
                    expected=f"All quantile changes < {self.threshold_pct}%",
                )
            )

        return issues


@register_validator
class RangeDriftValidator(ColumnDriftValidator):
    """Detects drift in data range (min/max values).

    Example:
        validator = RangeDriftValidator(
            column="age",
            reference_data=baseline_df,
            check_min=True,
            check_max=True,
            allow_expansion=True,  # Allow range to grow, but not shrink
        )
    """

    name = "range_drift"
    category = "drift"

    def __init__(
        self,
        column: str,
        reference_data: pl.LazyFrame | pl.DataFrame,
        check_min: bool = True,
        check_max: bool = True,
        allow_expansion: bool = False,
        threshold_pct: float = 10.0,
        **kwargs: Any,
    ):
        """Initialize range drift validator.

        Args:
            column: Numeric column to check
            reference_data: Baseline data for comparison
            check_min: Check minimum value drift
            check_max: Check maximum value drift
            allow_expansion: If True, only alert on range shrinkage
            threshold_pct: Maximum allowed percentage change
            **kwargs: Additional config
        """
        super().__init__(column=column, reference_data=reference_data, **kwargs)
        self.check_min = check_min
        self.check_max = check_max
        self.allow_expansion = allow_expansion
        self.threshold_pct = threshold_pct

    def calculate_drift_score(
        self, reference: pl.LazyFrame, current: pl.LazyFrame
    ) -> dict[str, tuple[float, float]]:
        """Calculate min/max values.

        Returns:
            Dict of 'min'/'max' -> (ref_value, curr_value)
        """
        results = {}

        if self.check_min:
            ref_min = reference.select(pl.col(self.column).min()).collect().item()
            curr_min = current.select(pl.col(self.column).min()).collect().item()
            results['min'] = (float(ref_min or 0), float(curr_min or 0))

        if self.check_max:
            ref_max = reference.select(pl.col(self.column).max()).collect().item()
            curr_max = current.select(pl.col(self.column).max()).collect().item()
            results['max'] = (float(ref_max or 0), float(curr_max or 0))

        return results

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        range_values = self.calculate_drift_score(self.reference_data, lf)
        drift_details = []

        for stat, (ref_val, curr_val) in range_values.items():
            if ref_val == 0:
                continue

            pct_change = abs(curr_val - ref_val) / abs(ref_val) * 100

            # Check if drift is significant
            if pct_change > self.threshold_pct:
                # If allow_expansion, only alert on concerning changes
                if self.allow_expansion:
                    if stat == 'min' and curr_val > ref_val:  # min increased (shrinkage)
                        drift_details.append(f"min: {ref_val:.2f}→{curr_val:.2f}")
                    elif stat == 'max' and curr_val < ref_val:  # max decreased (shrinkage)
                        drift_details.append(f"max: {ref_val:.2f}→{curr_val:.2f}")
                else:
                    drift_details.append(f"{stat}: {ref_val:.2f}→{curr_val:.2f}")

        if drift_details:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="range_drift_detected",
                    count=len(drift_details),
                    severity=Severity.MEDIUM,
                    details=f"Range drift: {', '.join(drift_details)}",
                    expected=f"Range change < {self.threshold_pct}%",
                )
            )

        return issues
