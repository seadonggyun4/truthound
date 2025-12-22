"""Date order validators."""

from typing import Any

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class DateOrderValidator(Validator):
    """Validates that one date column is before/after another.

    Example:
        # start_date should be before end_date
        validator = DateOrderValidator(
            first_column="start_date",
            second_column="end_date",
            allow_equal=True,
        )
    """

    name = "date_order"
    category = "datetime"

    def __init__(
        self,
        first_column: str,
        second_column: str,
        allow_equal: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.first_column = first_column
        self.second_column = second_column
        self.allow_equal = allow_equal

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.allow_equal:
            violation_expr = pl.col(self.first_column) > pl.col(self.second_column)
        else:
            violation_expr = pl.col(self.first_column) >= pl.col(self.second_column)

        result = lf.select([
            pl.len().alias("_total"),
            (
                violation_expr
                & pl.col(self.first_column).is_not_null()
                & pl.col(self.second_column).is_not_null()
            ).sum().alias("_violations"),
        ]).collect()

        total_rows = result["_total"][0]
        violations = result["_violations"][0]

        if violations > 0:
            op = "<=" if self.allow_equal else "<"
            issues.append(
                ValidationIssue(
                    column=f"{self.first_column}, {self.second_column}",
                    issue_type="date_order_violation",
                    count=violations,
                    severity=Severity.HIGH,
                    details=f"{self.first_column} should be {op} {self.second_column}",
                    expected=f"{self.first_column} {op} {self.second_column}",
                )
            )

        return issues
