"""Cross-table aggregate validators."""

from typing import Any, Literal

import polars as pl

from truthound.types import Severity
from truthound.validators.base import ValidationIssue, Validator
from truthound.validators.registry import register_validator
from truthound.validators.optimization import LazyAggregationMixin, AggregationExpressionBuilder


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


@register_validator
class OptimizedCrossTableAggregateValidator(Validator, LazyAggregationMixin):
    """Optimized cross-table aggregate validator using lazy evaluation.

    Uses LazyAggregationMixin for memory-efficient cross-table validation:
    - Lazy aggregation pushdown to avoid materializing intermediate results
    - Streaming joins for large tables
    - Caching of frequently used aggregations

    Example:
        validator = OptimizedCrossTableAggregateValidator(
            source_column="amount",
            reference_data=summary_df,
            reference_column="total_amount",
            group_by="customer_id",
            aggregate="sum",
        )
    """

    name = "optimized_cross_table_aggregate"
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
        source_column: str,
        reference_data: pl.DataFrame | pl.LazyFrame,
        reference_column: str | None = None,
        group_by: str | list[str] | None = None,
        aggregate: Literal["sum", "mean", "min", "max", "count", "n_unique"] = "sum",
        reference_name: str = "reference",
        tolerance: float = 0.0,
        tolerance_type: Literal["absolute", "relative"] = "absolute",
        cache_aggregates: bool = True,
        **kwargs: Any,
    ):
        """Initialize optimized cross-table aggregate validator.

        Args:
            source_column: Column in source data to aggregate
            reference_data: Reference/summary table
            reference_column: Column in reference with expected values
            group_by: Group by columns for per-group validation
            aggregate: Aggregation type
            reference_name: Name for reference table in messages
            tolerance: Allowed difference
            tolerance_type: 'absolute' or 'relative'
            cache_aggregates: Whether to cache aggregate results
            **kwargs: Additional config
        """
        super().__init__(**kwargs)
        self.source_column = source_column
        self.reference_column = reference_column or source_column
        self.group_by = [group_by] if isinstance(group_by, str) else group_by
        self.aggregate = aggregate
        self.reference_name = reference_name
        self.tolerance = tolerance
        self.tolerance_type = tolerance_type
        self._cache_enabled = cache_aggregates

        # Store reference data as lazy
        if isinstance(reference_data, pl.LazyFrame):
            self._reference_lf = reference_data
        else:
            self._reference_lf = reference_data.lazy()

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Build aggregation expression using builder
        builder = AggregationExpressionBuilder()
        if self.aggregate == "sum":
            builder.sum(self.source_column, alias="_agg_value")
        elif self.aggregate == "mean":
            builder.mean(self.source_column, alias="_agg_value")
        elif self.aggregate == "min":
            builder.min(self.source_column, alias="_agg_value")
        elif self.aggregate == "max":
            builder.max(self.source_column, alias="_agg_value")
        elif self.aggregate == "count":
            builder.count(alias="_agg_value")
        elif self.aggregate == "n_unique":
            builder.n_unique(self.source_column, alias="_agg_value")

        agg_exprs = builder.build()

        if self.group_by:
            # Per-group validation using lazy aggregation
            source_agg_result = self.aggregate_lazy(
                lf,
                group_by=self.group_by,
                agg_exprs=agg_exprs,
                cache_key="source_agg" if self._cache_enabled else None,
            )

            # Compare with reference
            mismatches = self.compare_aggregates(
                source=self._reference_lf,
                aggregated=source_agg_result,
                key_column=self.group_by[0] if len(self.group_by) == 1 else self.group_by,
                source_column=self.reference_column,
                agg_column="_agg_value",
                tolerance=self.tolerance if self.tolerance_type == "absolute" else 0.0,
            )

            if len(mismatches) > 0:
                issues.append(
                    ValidationIssue(
                        column=self.source_column,
                        issue_type="optimized_cross_table_aggregate_mismatch",
                        count=len(mismatches),
                        severity=Severity.HIGH,
                        details=(
                            f"{len(mismatches)} groups have {self.aggregate}({self.source_column}) "
                            f"!= {self.reference_name}.{self.reference_column} (lazy aggregation)"
                        ),
                        expected=f"All groups match within tolerance {self.tolerance}",
                    )
                )
        else:
            # Global aggregation
            agg_func = self.AGGREGATES[self.aggregate]
            source_value = lf.select(agg_func(pl.col(self.source_column)).alias("_v")).collect().item()
            reference_value = (
                self._reference_lf.select(agg_func(pl.col(self.reference_column)).alias("_v"))
                .collect()
                .item()
            )

            if source_value is None or reference_value is None:
                diff = float("inf")
            else:
                diff = abs(source_value - reference_value)

            if self.tolerance_type == "relative" and reference_value:
                threshold = abs(reference_value) * self.tolerance
            else:
                threshold = self.tolerance

            if diff > threshold:
                issues.append(
                    ValidationIssue(
                        column=self.source_column,
                        issue_type="optimized_cross_table_aggregate_mismatch",
                        count=1,
                        severity=Severity.HIGH,
                        details=(
                            f"{self.aggregate}({self.source_column}) != "
                            f"{self.reference_name}.{self.reference_column} (lazy evaluation)"
                        ),
                        expected=reference_value,
                        actual=source_value,
                    )
                )

        return issues
