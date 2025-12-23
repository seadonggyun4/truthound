"""Multi-column comparison validators.

Validators for comparing values across multiple columns.
"""

from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.multi_column.base import MultiColumnValidator
from truthound.validators.registry import register_validator


@register_validator
class ColumnComparisonValidator(MultiColumnValidator):
    """Validates comparison relationship between two columns.

    Example:
        # End date must be after start date
        validator = ColumnComparisonValidator(
            columns=["start_date", "end_date"],
            operator="<",
        )

        # Price must not exceed max_price
        validator = ColumnComparisonValidator(
            columns=["price", "max_price"],
            operator="<=",
        )
    """

    name = "column_comparison"
    category = "multi_column"

    OPERATORS = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(
        self,
        columns: list[str],
        operator: Literal["<", "<=", ">", ">=", "==", "!="],
        **kwargs: Any,
    ):
        if len(columns) != 2:
            raise ValueError("ColumnComparisonValidator requires exactly 2 columns")
        super().__init__(columns=columns, **kwargs)
        self.operator = operator

        if operator not in self.OPERATORS:
            raise ValueError(f"Invalid operator: {operator}")

    def get_validation_expression(self) -> pl.Expr:
        """Return comparison expression."""
        op_func = self.OPERATORS[self.operator]
        col_a = pl.col(self.columns[0])
        col_b = pl.col(self.columns[1])

        # Handle nulls - null comparison should be skipped
        return col_a.is_null() | col_b.is_null() | op_func(col_a, col_b)

    def _get_issue_type(self) -> str:
        return "column_comparison_failed"

    def _get_expected(self) -> str:
        return f"{self.columns[0]} {self.operator} {self.columns[1]}"


@register_validator
class ColumnChainComparisonValidator(MultiColumnValidator):
    """Validates chain comparison across multiple columns.

    Example:
        # min <= value <= max
        validator = ColumnChainComparisonValidator(
            columns=["min_val", "value", "max_val"],
            operators=["<=", "<="],
        )

        # Ordered sequence: a < b < c < d
        validator = ColumnChainComparisonValidator(
            columns=["a", "b", "c", "d"],
            operators=["<", "<", "<"],
        )
    """

    name = "column_chain_comparison"
    category = "multi_column"

    OPERATORS = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
    }

    def __init__(
        self,
        columns: list[str],
        operators: list[str],
        **kwargs: Any,
    ):
        super().__init__(columns=columns, **kwargs)
        self.operators = operators

        if len(operators) != len(columns) - 1:
            raise ValueError(
                f"Number of operators ({len(operators)}) must be "
                f"one less than number of columns ({len(columns)})"
            )

        for op in operators:
            if op not in self.OPERATORS:
                raise ValueError(f"Invalid operator: {op}")

    def get_validation_expression(self) -> pl.Expr:
        """Return chain comparison expression."""
        result = pl.lit(True)

        for i, op in enumerate(self.operators):
            op_func = self.OPERATORS[op]
            col_a = pl.col(self.columns[i])
            col_b = pl.col(self.columns[i + 1])

            # Add null check and comparison
            comparison = col_a.is_null() | col_b.is_null() | op_func(col_a, col_b)
            result = result & comparison

        return result

    def _get_issue_type(self) -> str:
        return "column_chain_comparison_failed"

    def _get_expected(self) -> str:
        parts = []
        for i, op in enumerate(self.operators):
            if i == 0:
                parts.append(self.columns[i])
            parts.append(op)
            parts.append(self.columns[i + 1])
        return " ".join(parts)


@register_validator
class ColumnMaxValidator(MultiColumnValidator):
    """Validates that a column contains the maximum of other columns.

    Example:
        # max_score should be max of all scores
        validator = ColumnMaxValidator(
            columns=["score_1", "score_2", "score_3"],
            result_column="max_score",
        )
    """

    name = "column_max"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        result_column: str,
        tolerance: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(columns=columns, **kwargs)
        self.result_column = result_column
        self.tolerance = tolerance

    def get_validation_expression(self) -> pl.Expr:
        """Return max comparison expression."""
        max_expr = pl.max_horizontal(*[pl.col(c) for c in self.columns])
        result = pl.col(self.result_column)

        if self.tolerance > 0:
            return (max_expr - result).abs() <= self.tolerance
        return max_expr == result

    def _get_issue_type(self) -> str:
        return "column_max_mismatch"

    def _get_expected(self) -> str:
        return f"{self.result_column} = max({', '.join(self.columns)})"


@register_validator
class ColumnMinValidator(MultiColumnValidator):
    """Validates that a column contains the minimum of other columns.

    Example:
        # min_price should be min of all prices
        validator = ColumnMinValidator(
            columns=["price_a", "price_b", "price_c"],
            result_column="min_price",
        )
    """

    name = "column_min"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        result_column: str,
        tolerance: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(columns=columns, **kwargs)
        self.result_column = result_column
        self.tolerance = tolerance

    def get_validation_expression(self) -> pl.Expr:
        """Return min comparison expression."""
        min_expr = pl.min_horizontal(*[pl.col(c) for c in self.columns])
        result = pl.col(self.result_column)

        if self.tolerance > 0:
            return (min_expr - result).abs() <= self.tolerance
        return min_expr == result

    def _get_issue_type(self) -> str:
        return "column_min_mismatch"

    def _get_expected(self) -> str:
        return f"{self.result_column} = min({', '.join(self.columns)})"


@register_validator
class ColumnMeanValidator(MultiColumnValidator):
    """Validates that a column contains the mean of other columns.

    Example:
        # avg_score should be average of all scores
        validator = ColumnMeanValidator(
            columns=["score_1", "score_2", "score_3"],
            result_column="avg_score",
            tolerance=0.01,
        )
    """

    name = "column_mean"
    category = "multi_column"

    def __init__(
        self,
        columns: list[str],
        result_column: str,
        tolerance: float = 0.001,
        **kwargs: Any,
    ):
        super().__init__(columns=columns, **kwargs)
        self.result_column = result_column
        self.tolerance = tolerance

    def get_validation_expression(self) -> pl.Expr:
        """Return mean comparison expression."""
        mean_expr = pl.mean_horizontal(*[pl.col(c) for c in self.columns])
        result = pl.col(self.result_column)

        return (mean_expr - result).abs() <= self.tolerance

    def _get_issue_type(self) -> str:
        return "column_mean_mismatch"

    def _get_expected(self) -> str:
        return f"{self.result_column} = mean({', '.join(self.columns)}) ± {self.tolerance}"
