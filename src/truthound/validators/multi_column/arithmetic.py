"""Multi-column arithmetic validators.

Validators for checking arithmetic relationships between columns.
"""

from typing import Any

import polars as pl

from truthound.validators.multi_column.base import ColumnArithmeticValidator
from truthound.validators.registry import register_validator


@register_validator
class ColumnSumValidator(ColumnArithmeticValidator):
    """Validates that the sum of columns equals expected value or column.

    Example:
        # Sum of parts equals total
        validator = ColumnSumValidator(
            columns=["part_a", "part_b", "part_c"],
            result_column="total",
        )

        # Sum equals constant
        validator = ColumnSumValidator(
            columns=["q1", "q2", "q3", "q4"],
            expected_value=100,
        )

        # Sum within range
        validator = ColumnSumValidator(
            columns=["score_1", "score_2"],
            min_value=0,
            max_value=200,
        )
    """

    name = "column_sum"
    category = "multi_column"

    def get_computed_expression(self) -> pl.Expr:
        """Return sum of columns."""
        result = pl.col(self.columns[0])
        for col in self.columns[1:]:
            result = result + pl.col(col)
        return result

    def _get_issue_type(self) -> str:
        return "column_sum_mismatch"

    def _get_expected(self) -> str:
        if self.result_column:
            return f"sum({', '.join(self.columns)}) = {self.result_column}"
        elif self.expected_value is not None:
            return f"sum({', '.join(self.columns)}) = {self.expected_value}"
        else:
            bounds = []
            if self.min_value is not None:
                bounds.append(f">= {self.min_value}")
            if self.max_value is not None:
                bounds.append(f"<= {self.max_value}")
            return f"sum({', '.join(self.columns)}) {' and '.join(bounds)}"


@register_validator
class ColumnProductValidator(ColumnArithmeticValidator):
    """Validates that the product of columns equals expected value or column.

    Example:
        # Quantity * price = total
        validator = ColumnProductValidator(
            columns=["quantity", "unit_price"],
            result_column="total_price",
        )

        # With tolerance for floating point
        validator = ColumnProductValidator(
            columns=["rate", "hours"],
            result_column="earnings",
            tolerance=0.01,
        )
    """

    name = "column_product"
    category = "multi_column"

    def get_computed_expression(self) -> pl.Expr:
        """Return product of columns."""
        result = pl.col(self.columns[0])
        for col in self.columns[1:]:
            result = result * pl.col(col)
        return result

    def _get_issue_type(self) -> str:
        return "column_product_mismatch"

    def _get_expected(self) -> str:
        if self.result_column:
            return f"{' * '.join(self.columns)} = {self.result_column}"
        elif self.expected_value is not None:
            return f"{' * '.join(self.columns)} = {self.expected_value}"
        else:
            bounds = []
            if self.min_value is not None:
                bounds.append(f">= {self.min_value}")
            if self.max_value is not None:
                bounds.append(f"<= {self.max_value}")
            return f"{' * '.join(self.columns)} {' and '.join(bounds)}"


@register_validator
class ColumnDifferenceValidator(ColumnArithmeticValidator):
    """Validates that the difference between columns meets criteria.

    Example:
        # End date - start date should be positive
        validator = ColumnDifferenceValidator(
            columns=["end_value", "start_value"],
            min_value=0,
        )

        # Difference equals another column
        validator = ColumnDifferenceValidator(
            columns=["gross", "tax"],
            result_column="net",
        )
    """

    name = "column_difference"
    category = "multi_column"

    def get_computed_expression(self) -> pl.Expr:
        """Return difference of columns (first - rest)."""
        result = pl.col(self.columns[0])
        for col in self.columns[1:]:
            result = result - pl.col(col)
        return result

    def _get_issue_type(self) -> str:
        return "column_difference_mismatch"

    def _get_expected(self) -> str:
        diff_expr = f"{self.columns[0]} - " + " - ".join(self.columns[1:])
        if self.result_column:
            return f"({diff_expr}) = {self.result_column}"
        elif self.expected_value is not None:
            return f"({diff_expr}) = {self.expected_value}"
        else:
            bounds = []
            if self.min_value is not None:
                bounds.append(f">= {self.min_value}")
            if self.max_value is not None:
                bounds.append(f"<= {self.max_value}")
            return f"({diff_expr}) {' and '.join(bounds)}"


@register_validator
class ColumnRatioValidator(ColumnArithmeticValidator):
    """Validates that the ratio between columns meets criteria.

    Example:
        # Profit margin check
        validator = ColumnRatioValidator(
            columns=["profit", "revenue"],
            min_value=0.1,  # At least 10% margin
            max_value=0.5,  # At most 50% margin
        )

        # Ratio equals expected column
        validator = ColumnRatioValidator(
            columns=["numerator", "denominator"],
            result_column="ratio",
            tolerance=0.001,
        )
    """

    name = "column_ratio"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        handle_zero_division: str = "skip",
        **kwargs: Any,
    ):
        """Initialize ratio validator.

        Args:
            columns: [numerator, denominator] columns
            handle_zero_division: How to handle zero denominator: "skip", "fail", "allow"
            **kwargs: Additional config
        """
        if len(columns) != 2:
            raise ValueError("ColumnRatioValidator requires exactly 2 columns [numerator, denominator]")
        super().__init__(columns=columns, **kwargs)
        self.handle_zero_division = handle_zero_division

    def get_computed_expression(self) -> pl.Expr:
        """Return ratio of columns."""
        return pl.col(self.columns[0]) / pl.col(self.columns[1])

    def get_validation_expression(self) -> pl.Expr:
        """Return validation expression handling zero division."""
        denominator = pl.col(self.columns[1])

        if self.handle_zero_division == "skip":
            # Skip rows where denominator is zero (treat as valid)
            zero_check = denominator == 0
            arithmetic_check = super().get_validation_expression()
            return zero_check | arithmetic_check
        elif self.handle_zero_division == "fail":
            # Fail if denominator is zero
            zero_check = denominator != 0
            arithmetic_check = super().get_validation_expression()
            return zero_check & arithmetic_check
        else:  # allow
            return super().get_validation_expression()

    def _get_issue_type(self) -> str:
        return "column_ratio_mismatch"

    def _get_expected(self) -> str:
        ratio_expr = f"{self.columns[0]} / {self.columns[1]}"
        if self.result_column:
            return f"({ratio_expr}) = {self.result_column}"
        elif self.expected_value is not None:
            return f"({ratio_expr}) = {self.expected_value}"
        else:
            bounds = []
            if self.min_value is not None:
                bounds.append(f">= {self.min_value}")
            if self.max_value is not None:
                bounds.append(f"<= {self.max_value}")
            return f"({ratio_expr}) {' and '.join(bounds)}"


@register_validator
class ColumnPercentageValidator(ColumnArithmeticValidator):
    """Validates that column represents valid percentage of total.

    Example:
        # Part should be percentage of total
        validator = ColumnPercentageValidator(
            columns=["part", "total"],
            min_value=0,
            max_value=100,
        )
    """

    name = "column_percentage"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        **kwargs: Any,
    ):
        if len(columns) != 2:
            raise ValueError("ColumnPercentageValidator requires exactly 2 columns [part, total]")
        super().__init__(columns=columns, **kwargs)

    def get_computed_expression(self) -> pl.Expr:
        """Return percentage (part/total * 100)."""
        return (pl.col(self.columns[0]) / pl.col(self.columns[1])) * 100

    def get_validation_expression(self) -> pl.Expr:
        """Handle zero division."""
        total = pl.col(self.columns[1])
        zero_check = total == 0
        arithmetic_check = super().get_validation_expression()
        return zero_check | arithmetic_check

    def _get_issue_type(self) -> str:
        return "column_percentage_mismatch"

    def _get_expected(self) -> str:
        pct_expr = f"({self.columns[0]} / {self.columns[1]}) * 100"
        bounds = []
        if self.min_value is not None:
            bounds.append(f">= {self.min_value}%")
        if self.max_value is not None:
            bounds.append(f"<= {self.max_value}%")
        return f"{pct_expr} {' and '.join(bounds)}" if bounds else pct_expr
