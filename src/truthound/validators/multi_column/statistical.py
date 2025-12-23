"""Multi-column statistical validators.

Validators for checking statistical relationships between columns.
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ColumnCorrelationValidator(Validator):
    """Validates correlation between two numeric columns.

    Example:
        # Height and weight should be positively correlated
        validator = ColumnCorrelationValidator(
            column_a="height",
            column_b="weight",
            min_correlation=0.3,
        )

        # No strong correlation between independent variables
        validator = ColumnCorrelationValidator(
            column_a="feature_1",
            column_b="feature_2",
            max_correlation=0.7,
        )
    """

    name = "column_correlation"
    category = "multi_column"

    def __init__(
        self,
        column_a: str,
        column_b: str,
        min_correlation: float | None = None,
        max_correlation: float | None = None,
        expected_correlation: float | None = None,
        tolerance: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column_a = column_a
        self.column_b = column_b
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.expected_correlation = expected_correlation
        self.tolerance = tolerance

        if min_correlation is None and max_correlation is None and expected_correlation is None:
            raise ValueError(
                "At least one of 'min_correlation', 'max_correlation', or 'expected_correlation' required"
            )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Calculate Pearson correlation
        df = lf.select([self.column_a, self.column_b]).drop_nulls().collect()

        if len(df) < 3:
            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="insufficient_data_for_correlation",
                    count=len(df),
                    severity=Severity.MEDIUM,
                    details="Need at least 3 data points for correlation",
                    expected=">= 3 data points",
                )
            )
            return issues

        # Compute correlation using Polars
        correlation = df.select(
            pl.corr(self.column_a, self.column_b).alias("corr")
        )["corr"][0]

        if correlation is None:
            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="correlation_computation_failed",
                    count=1,
                    severity=Severity.HIGH,
                    details="Could not compute correlation (possibly constant values)",
                    expected="Computable correlation",
                )
            )
            return issues

        # Check bounds
        if self.expected_correlation is not None:
            if abs(correlation - self.expected_correlation) > self.tolerance:
                issues.append(
                    ValidationIssue(
                        column=f"{self.column_a}, {self.column_b}",
                        issue_type="correlation_mismatch",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=f"Correlation {correlation:.3f} differs from expected {self.expected_correlation}",
                        expected=f"Correlation ≈ {self.expected_correlation} ± {self.tolerance}",
                    )
                )

        if self.min_correlation is not None and correlation < self.min_correlation:
            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="correlation_too_low",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=f"Correlation {correlation:.3f} is below minimum {self.min_correlation}",
                    expected=f"Correlation >= {self.min_correlation}",
                )
            )

        if self.max_correlation is not None and correlation > self.max_correlation:
            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="correlation_too_high",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=f"Correlation {correlation:.3f} exceeds maximum {self.max_correlation}",
                    expected=f"Correlation <= {self.max_correlation}",
                )
            )

        return issues


@register_validator
class ColumnCovarianceValidator(Validator):
    """Validates covariance between two numeric columns.

    Example:
        # Check covariance is positive
        validator = ColumnCovarianceValidator(
            column_a="x",
            column_b="y",
            min_value=0,
        )
    """

    name = "column_covariance"
    category = "multi_column"

    def __init__(
        self,
        column_a: str,
        column_b: str,
        min_value: float | None = None,
        max_value: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column_a = column_a
        self.column_b = column_b
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        df = lf.select([self.column_a, self.column_b]).drop_nulls().collect()

        if len(df) < 2:
            return issues

        # Compute covariance
        covariance = df.select(
            pl.cov(self.column_a, self.column_b).alias("cov")
        )["cov"][0]

        if covariance is None:
            return issues

        if self.min_value is not None and covariance < self.min_value:
            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="covariance_too_low",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=f"Covariance {covariance:.3f} < minimum {self.min_value}",
                    expected=f"Covariance >= {self.min_value}",
                )
            )

        if self.max_value is not None and covariance > self.max_value:
            issues.append(
                ValidationIssue(
                    column=f"{self.column_a}, {self.column_b}",
                    issue_type="covariance_too_high",
                    count=1,
                    severity=Severity.MEDIUM,
                    details=f"Covariance {covariance:.3f} > maximum {self.max_value}",
                    expected=f"Covariance <= {self.max_value}",
                )
            )

        return issues


@register_validator
class MultiColumnVarianceValidator(Validator):
    """Validates variance relationship across multiple columns.

    Example:
        # Variance of columns should be similar
        validator = MultiColumnVarianceValidator(
            columns=["score_1", "score_2", "score_3"],
            max_variance_ratio=2.0,  # Max ratio between highest and lowest variance
        )
    """

    name = "multi_column_variance"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        max_variance_ratio: float | None = None,
        min_variance: float | None = None,
        max_variance: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.columns = columns
        self.max_variance_ratio = max_variance_ratio
        self.min_variance = min_variance
        self.max_variance = max_variance

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Calculate variance for each column
        var_exprs = [pl.col(c).var().alias(f"var_{c}") for c in self.columns]
        result = lf.select(var_exprs).collect()

        variances = {c: result[f"var_{c}"][0] for c in self.columns}
        valid_variances = {k: v for k, v in variances.items() if v is not None}

        if len(valid_variances) < 2:
            return issues

        # Check variance bounds
        for col, var in valid_variances.items():
            if self.min_variance is not None and var < self.min_variance:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="variance_too_low",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=f"Variance {var:.3f} < minimum {self.min_variance}",
                        expected=f"Variance >= {self.min_variance}",
                    )
                )

            if self.max_variance is not None and var > self.max_variance:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="variance_too_high",
                        count=1,
                        severity=Severity.MEDIUM,
                        details=f"Variance {var:.3f} > maximum {self.max_variance}",
                        expected=f"Variance <= {self.max_variance}",
                    )
                )

        # Check variance ratio
        if self.max_variance_ratio is not None:
            max_var = max(valid_variances.values())
            min_var = min(valid_variances.values())

            if min_var > 0:
                ratio = max_var / min_var
                if ratio > self.max_variance_ratio:
                    issues.append(
                        ValidationIssue(
                            column=", ".join(self.columns),
                            issue_type="variance_ratio_exceeded",
                            count=1,
                            severity=Severity.MEDIUM,
                            details=f"Variance ratio {ratio:.2f} exceeds maximum {self.max_variance_ratio}",
                            expected=f"Variance ratio <= {self.max_variance_ratio}",
                        )
                    )

        return issues
