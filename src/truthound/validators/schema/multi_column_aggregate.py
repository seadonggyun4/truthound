"""Multi-column aggregate validators."""

from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class MultiColumnSumValidator(Validator):
    """Validates that sum of multiple columns equals an expected value or another column.

    Example:
        # Sum of parts should equal total
        validator = MultiColumnSumValidator(
            columns=["part1", "part2", "part3"],
            equals_column="total",
        )

        # Or equals a fixed value
        validator = MultiColumnSumValidator(
            columns=["q1", "q2", "q3", "q4"],
            equals_value=100,
        )
    """

    name = "multi_column_sum"
    category = "schema"

    def __init__(
        self,
        columns: list[str],
        equals_column: str | None = None,
        equals_value: float | None = None,
        tolerance: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.sum_columns = columns
        self.equals_column = equals_column
        self.equals_value = equals_value
        self.tolerance = tolerance

        if equals_column is None and equals_value is None:
            raise ValueError("Either equals_column or equals_value must be provided")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Build sum expression
        sum_expr = pl.lit(0)
        for col in self.sum_columns:
            sum_expr = sum_expr + pl.col(col).fill_null(0)

        if self.equals_column:
            # Compare sum to another column
            violation_expr = (
                (sum_expr - pl.col(self.equals_column)).abs() > self.tolerance
            )
            expected_desc = f"sum({', '.join(self.sum_columns)}) = {self.equals_column}"
        else:
            # Compare sum to fixed value
            violation_expr = (sum_expr - self.equals_value).abs() > self.tolerance
            expected_desc = f"sum({', '.join(self.sum_columns)}) = {self.equals_value}"

        result = lf.select([
            pl.len().alias("_total"),
            violation_expr.sum().alias("_violations"),
        ]).collect()

        violations = result["_violations"][0]
        total = result["_total"][0]

        if violations > 0:
            if self._passes_mostly(violations, total):
                return issues

            ratio = violations / total if total > 0 else 0
            col_desc = ", ".join(self.sum_columns)

            issues.append(
                ValidationIssue(
                    column=f"[{col_desc}]",
                    issue_type="multi_column_sum_mismatch",
                    count=violations,
                    severity=self._calculate_severity(ratio),
                    details=expected_desc,
                    expected=expected_desc,
                )
            )

        return issues


@register_validator
class MultiColumnCalculationValidator(Validator):
    """Validates custom calculations across multiple columns.

    Example:
        # profit = revenue - cost
        validator = MultiColumnCalculationValidator(
            expression="revenue - cost",
            equals_column="profit",
        )
    """

    name = "multi_column_calculation"
    category = "schema"

    OPERATORS = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
    }

    def __init__(
        self,
        left_column: str,
        operator: Literal["+", "-", "*", "/"],
        right_column: str,
        equals_column: str | None = None,
        equals_value: float | None = None,
        tolerance: float = 0.001,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.left_column = left_column
        self.operator = operator
        self.right_column = right_column
        self.equals_column = equals_column
        self.equals_value = equals_value
        self.tolerance = tolerance

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        op_func = self.OPERATORS[self.operator]
        calc_expr = op_func(pl.col(self.left_column), pl.col(self.right_column))

        if self.equals_column:
            violation_expr = (calc_expr - pl.col(self.equals_column)).abs() > self.tolerance
            expected_desc = f"{self.left_column} {self.operator} {self.right_column} = {self.equals_column}"
        else:
            violation_expr = (calc_expr - self.equals_value).abs() > self.tolerance
            expected_desc = f"{self.left_column} {self.operator} {self.right_column} = {self.equals_value}"

        result = lf.select([
            pl.len().alias("_total"),
            violation_expr.sum().alias("_violations"),
        ]).collect()

        violations = result["_violations"][0]
        total = result["_total"][0]

        if violations > 0:
            if self._passes_mostly(violations, total):
                return issues

            ratio = violations / total if total > 0 else 0

            issues.append(
                ValidationIssue(
                    column=f"{self.left_column}, {self.right_column}",
                    issue_type="calculation_mismatch",
                    count=violations,
                    severity=self._calculate_severity(ratio),
                    details=expected_desc,
                )
            )

        return issues
