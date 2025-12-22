"""Range and boundary validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    NumericValidatorMixin,
)
from truthound.validators.registry import register_validator


@register_validator
class BetweenValidator(Validator, NumericValidatorMixin):
    """Validates that numeric values are within a specified range."""

    name = "between"
    category = "distribution"

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        exprs: list[pl.Expr] = [pl.len().alias("_total")]

        for col in columns:
            if self.inclusive:
                below = (pl.col(col) < self.min_value) if self.min_value is not None else pl.lit(False)
                above = (pl.col(col) > self.max_value) if self.max_value is not None else pl.lit(False)
            else:
                below = (pl.col(col) <= self.min_value) if self.min_value is not None else pl.lit(False)
                above = (pl.col(col) >= self.max_value) if self.max_value is not None else pl.lit(False)

            exprs.append(
                ((below | above) & pl.col(col).is_not_null()).sum().alias(f"_out_{col}")
            )

        result = lf.select(exprs).collect()
        total_rows = result["_total"][0]

        if total_rows == 0:
            return issues

        for col in columns:
            out_count = result[f"_out_{col}"][0]
            if out_count > 0:
                ratio = out_count / total_rows
                range_str = f"[{self.min_value}, {self.max_value}]"
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="out_of_range",
                        count=out_count,
                        severity=self._calculate_severity(ratio, (0.1, 0.05, 0.01)),
                        details=f"{out_count} values outside {range_str}",
                        expected=range_str,
                    )
                )

        return issues


@register_validator
class RangeValidator(Validator, NumericValidatorMixin):
    """Auto-detects expected ranges based on column names."""

    name = "range"
    category = "distribution"

    KNOWN_RANGES: dict[str, tuple[float | None, float | None]] = {
        "age": (0, 150),
        "price": (0, None),
        "quantity": (0, None),
        "count": (0, None),
        "amount": (0, None),
        "percentage": (0, 100),
        "percent": (0, 100),
        "pct": (0, 100),
        "rate": (0, 100),
        "score": (0, 100),
        "rating": (0, 5),
        "year": (1900, 2100),
        "month": (1, 12),
        "day": (1, 31),
        "hour": (0, 23),
        "minute": (0, 59),
        "second": (0, 59),
    }

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        # Collect data once
        df = lf.collect()
        total_rows = len(df)

        if total_rows == 0:
            return issues

        for col in columns:
            col_lower = col.lower()

            # Find matching range
            min_val, max_val = None, None
            for pattern, (pmin, pmax) in self.KNOWN_RANGES.items():
                if pattern in col_lower:
                    min_val, max_val = pmin, pmax
                    break

            if min_val is None and max_val is None:
                continue

            col_data = df.get_column(col)
            out_count = 0

            if min_val is not None:
                out_count += col_data.filter(col_data < min_val).drop_nulls().len()
            if max_val is not None:
                out_count += col_data.filter(col_data > max_val).drop_nulls().len()

            if out_count > 0:
                oor_pct = out_count / total_rows
                range_desc = ""
                if min_val is not None and max_val is not None:
                    range_desc = f"[{min_val}, {max_val}]"
                elif min_val is not None:
                    range_desc = f">= {min_val}"
                else:
                    range_desc = f"<= {max_val}"

                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="out_of_range",
                        count=out_count,
                        severity=self._calculate_severity(oor_pct, (0.1, 0.05, 0.01)),
                        details=f"Expected {range_desc}",
                        expected=range_desc,
                    )
                )

        return issues


@register_validator
class PositiveValidator(Validator, NumericValidatorMixin):
    """Validates that numeric values are positive (> 0)."""

    name = "positive"
    category = "distribution"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[
                ((pl.col(c) <= 0) & pl.col(c).is_not_null()).sum().alias(f"_neg_{c}")
                for c in columns
            ],
        ]
        result = lf.select(exprs).collect()
        total_rows = result["_total"][0]

        if total_rows == 0:
            return issues

        for col in columns:
            neg_count = result[f"_neg_{col}"][0]
            if neg_count > 0:
                ratio = neg_count / total_rows
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="not_positive",
                        count=neg_count,
                        severity=self._calculate_severity(ratio),
                        details=f"{neg_count} non-positive values",
                        expected="> 0",
                    )
                )

        return issues


@register_validator
class NonNegativeValidator(Validator, NumericValidatorMixin):
    """Validates that numeric values are non-negative (>= 0)."""

    name = "non_negative"
    category = "distribution"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        columns = self._get_numeric_columns(lf)

        if not columns:
            return issues

        exprs = [
            pl.len().alias("_total"),
            *[
                ((pl.col(c) < 0) & pl.col(c).is_not_null()).sum().alias(f"_neg_{c}")
                for c in columns
            ],
        ]
        result = lf.select(exprs).collect()
        total_rows = result["_total"][0]

        if total_rows == 0:
            return issues

        for col in columns:
            neg_count = result[f"_neg_{col}"][0]
            if neg_count > 0:
                ratio = neg_count / total_rows
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="negative",
                        count=neg_count,
                        severity=self._calculate_severity(ratio),
                        details=f"{neg_count} negative values",
                        expected=">= 0",
                    )
                )

        return issues
