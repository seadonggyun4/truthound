"""NaN (Not a Number) validators for float columns.

NaN values are distinct from NULL/None in Polars:
- NULL (None): Missing value, works with all types
- NaN: IEEE 754 floating-point "Not a Number", only for Float32/Float64

This module provides validators specifically for detecting NaN values,
which are often introduced by:
- Division by zero (0/0)
- Invalid math operations (sqrt(-1))
- Data parsing errors
- External data sources
"""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    FloatValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class NaNValidator(Validator, FloatValidatorMixin):
    """Detects NaN (Not a Number) values in float columns.

    Unlike NullValidator which detects None/missing values, this validator
    specifically finds IEEE 754 NaN values in Float32/Float64 columns.

    Example:
        validator = NaNValidator()
        issues = validator.validate(lf)

        # Only check specific columns
        validator = NaNValidator(columns=["price", "ratio"])

        # With severity override
        validator = NaNValidator(severity_override=Severity.CRITICAL)
    """

    name = "nan"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_float_columns(lf)

        if not columns:
            return issues

        # Single optimized query for all float columns
        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).is_nan().sum().alias(f"_nan_{c}") for c in columns],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            nan_count = result[f"_nan_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]

            if nan_count > 0:
                # Calculate ratio against non-null values
                ratio = nan_count / non_null_count if non_null_count > 0 else 1.0

                # Check mostly threshold
                if self._passes_mostly(nan_count, non_null_count):
                    continue

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="nan",
                        count=nan_count,
                        severity=self._calculate_severity(ratio),
                        details=f"{ratio:.1%} of non-null values are NaN",
                        expected="No NaN values",
                        actual=f"{nan_count} NaN values",
                    )
                )

        return issues


@register_validator
class NotNaNValidator(Validator, FloatValidatorMixin):
    """Validates that float columns contain no NaN values.

    Similar to NotNullValidator but for NaN values specifically.
    Reports HIGH severity by default since NaN often indicates data corruption.

    Example:
        validator = NotNaNValidator(columns=["price"])
        issues = validator.validate(lf)
    """

    name = "not_nan"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_float_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).is_nan().sum().alias(f"_nan_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            nan_count = result[f"_nan_{col}"][0]

            if nan_count > 0:
                nan_pct = nan_count / total_rows
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_nan_violation",
                        count=nan_count,
                        severity=Severity.HIGH,
                        details=f"Expected no NaN, found {nan_count} ({nan_pct:.1%})",
                        expected=0,
                        actual=nan_count,
                    )
                )

        return issues


@register_validator
class NaNRatioValidator(Validator, FloatValidatorMixin):
    """Validates that NaN ratio is below a threshold.

    Useful when some NaN values are acceptable but need to be controlled.

    Example:
        # Allow up to 5% NaN values
        validator = NaNRatioValidator(max_ratio=0.05)

        # Check only specific columns
        validator = NaNRatioValidator(columns=["score"], max_ratio=0.01)
    """

    name = "nan_ratio"
    category = "completeness"

    def __init__(
        self,
        max_ratio: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if not 0.0 <= max_ratio <= 1.0:
            raise ValueError(f"max_ratio must be between 0 and 1, got {max_ratio}")
        self.max_ratio = max_ratio

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_float_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).is_nan().sum().alias(f"_nan_{c}") for c in columns],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            nan_count = result[f"_nan_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]

            if non_null_count == 0:
                continue

            ratio = nan_count / non_null_count

            if ratio > self.max_ratio:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="nan_ratio_exceeded",
                        count=nan_count,
                        severity=self._calculate_severity(
                            ratio - self.max_ratio,
                            thresholds=(0.3, 0.1, 0.01),
                        ),
                        details=f"NaN ratio {ratio:.1%} > max {self.max_ratio:.1%}",
                        expected=f"<= {self.max_ratio:.1%}",
                        actual=f"{ratio:.1%}",
                    )
                )

        return issues


@register_validator
class InfinityValidator(Validator, FloatValidatorMixin):
    """Detects infinity values (inf, -inf) in float columns.

    Infinity values can cause issues in calculations and aggregations.
    This validator detects both positive and negative infinity.

    Example:
        validator = InfinityValidator()
        issues = validator.validate(lf)
    """

    name = "infinity"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_float_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[pl.col(c).is_infinite().sum().alias(f"_inf_{c}") for c in columns],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            inf_count = result[f"_inf_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]

            if inf_count > 0:
                ratio = inf_count / non_null_count if non_null_count > 0 else 1.0

                if self._passes_mostly(inf_count, non_null_count):
                    continue

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="infinity",
                        count=inf_count,
                        severity=self._calculate_severity(ratio),
                        details=f"{inf_count} infinite values detected",
                        expected="No infinity values",
                        actual=f"{inf_count} inf values ({ratio:.1%})",
                    )
                )

        return issues


@register_validator
class FiniteValidator(Validator, FloatValidatorMixin):
    """Validates that float columns contain only finite values.

    Checks for both NaN and infinity values at once. This is equivalent
    to running both NaNValidator and InfinityValidator but more efficient.

    Example:
        validator = FiniteValidator(columns=["price", "quantity"])
        issues = validator.validate(lf)
    """

    name = "finite"
    category = "completeness"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_float_columns(lf)

        if not columns:
            return issues

        # Check for non-finite values (NaN or Inf)
        exprs = [
            pl.len().alias("_total"),
            *[
                (pl.col(c).is_nan() | pl.col(c).is_infinite()).sum().alias(f"_nonfinite_{c}")
                for c in columns
            ],
            *[pl.col(c).count().alias(f"_count_{c}") for c in columns],
        ]
        result = lf.select(exprs).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        for col in columns:
            nonfinite_count = result[f"_nonfinite_{col}"][0]
            non_null_count = result[f"_count_{col}"][0]

            if nonfinite_count > 0:
                ratio = nonfinite_count / non_null_count if non_null_count > 0 else 1.0

                if self._passes_mostly(nonfinite_count, non_null_count):
                    continue

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_finite",
                        count=nonfinite_count,
                        severity=Severity.HIGH,
                        details=f"{nonfinite_count} non-finite values (NaN or Inf)",
                        expected="All finite values",
                        actual=f"{nonfinite_count} non-finite ({ratio:.1%})",
                    )
                )

        return issues
