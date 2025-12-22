"""Cross-table aggregate validators."""

from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator


@register_validator
class CrossTableAggregateValidator(Validator):
    """Validates that aggregate values match between tables.

    Example:
        # Sum of order amounts should match between fact and summary tables
        validator = CrossTableAggregateValidator(
            column="amount",
            reference_data=summary_df,
            reference_column="total_amount",
            aggregate="sum",
        )
    """

    name = "cross_table_aggregate"
    category = "cross_table"

    AGGREGATES = {
        "sum": pl.Expr.sum,
        "mean": pl.Expr.mean,
        "min": pl.Expr.min,
        "max": pl.Expr.max,
        "count": pl.Expr.count,
        "n_unique": pl.Expr.n_unique,
    }

    def __init__(
        self,
        column: str,
        reference_data: pl.DataFrame | pl.LazyFrame,
        reference_column: str | None = None,
        aggregate: Literal["sum", "mean", "min", "max", "count", "n_unique"] = "sum",
        reference_name: str = "reference",
        tolerance: float = 0.0,
        tolerance_type: Literal["absolute", "relative"] = "absolute",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.reference_column = reference_column or column
        self.aggregate = aggregate
        self.reference_name = reference_name
        self.tolerance = tolerance
        self.tolerance_type = tolerance_type

        # Compute reference aggregate
        agg_func = self.AGGREGATES[aggregate]
        if isinstance(reference_data, pl.LazyFrame):
            ref_lf = reference_data
        else:
            ref_lf = reference_data.lazy()

        self.reference_value = (
            ref_lf.select(agg_func(pl.col(self.reference_column)))
            .collect()
            .item()
        )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        agg_func = self.AGGREGATES[self.aggregate]
        source_value = lf.select(agg_func(pl.col(self.column))).collect().item()

        # Calculate difference
        if self.reference_value is None or source_value is None:
            diff = float("inf")
        else:
            diff = abs(source_value - self.reference_value)

        # Check tolerance
        if self.tolerance_type == "relative" and self.reference_value:
            threshold = abs(self.reference_value) * self.tolerance
        else:
            threshold = self.tolerance

        if diff > threshold:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="cross_table_aggregate_mismatch",
                    count=1,
                    severity=Severity.HIGH,
                    details=f"{self.aggregate}({self.column}) != {self.reference_name}.{self.reference_column}",
                    expected=self.reference_value,
                    actual=source_value,
                )
            )

        return issues


@register_validator
class CrossTableDistinctCountValidator(Validator):
    """Validates that distinct value counts match between tables.

    Example:
        # Number of unique customers should match between orders and customers
        validator = CrossTableDistinctCountValidator(
            column="customer_id",
            reference_data=customers_df,
            reference_column="id",
        )
    """

    name = "cross_table_distinct_count"
    category = "cross_table"

    def __init__(
        self,
        column: str,
        reference_data: pl.DataFrame | pl.LazyFrame,
        reference_column: str | None = None,
        reference_name: str = "reference",
        tolerance: int = 0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.column = column
        self.reference_column = reference_column or column
        self.reference_name = reference_name
        self.tolerance = tolerance

        # Compute reference distinct count
        if isinstance(reference_data, pl.LazyFrame):
            ref_lf = reference_data
        else:
            ref_lf = reference_data.lazy()

        self.reference_count = (
            ref_lf.select(pl.col(self.reference_column).n_unique())
            .collect()
            .item()
        )

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        source_count = lf.select(pl.col(self.column).n_unique()).collect().item()
        diff = abs(source_count - self.reference_count)

        if diff > self.tolerance:
            issues.append(
                ValidationIssue(
                    column=self.column,
                    issue_type="cross_table_distinct_count_mismatch",
                    count=diff,
                    severity=Severity.HIGH,
                    details=f"Distinct count mismatch with {self.reference_name}.{self.reference_column}",
                    expected=self.reference_count,
                    actual=source_count,
                )
            )

        return issues
