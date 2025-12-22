"""Conditional null validators."""

from typing import Any, Callable

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class ConditionalNullValidator(Validator):
    """Validates null values based on conditions in other columns.

    Example:
        # Email should not be null when subscription_type is 'newsletter'
        validator = ConditionalNullValidator(
            column="email",
            condition_column="subscription_type",
            condition_values=["newsletter"],
        )
    """

    name = "conditional_null"
    category = "completeness"

    def __init__(
        self,
        column: str,
        condition_column: str,
        condition_values: list[Any],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.condition_column = condition_column
        self.condition_values = condition_values

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Filter rows where condition is met
        filtered = lf.filter(
            pl.col(self.condition_column).is_in(self.condition_values)
        )

        result = filtered.select([
            pl.len().alias("_total"),
            pl.col(self.column).null_count().alias("_null"),
        ]).collect()

        total_rows = result["_total"][0]
        if total_rows == 0:
            return issues

        null_count = result["_null"][0]
        if null_count > 0:
            null_pct = null_count / total_rows
            condition_str = f"{self.condition_column} in {self.condition_values}"
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="conditional_null",
                    count=null_count,
                    severity=Severity.HIGH,
                    details=f"Found {null_count} nulls when {condition_str}",
                    expected=0,
                    actual=null_count,
                )
            )

        return issues
