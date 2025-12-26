"""Lazy aggregation optimization for cross-table validators.

This module provides memory-efficient aggregation patterns that leverage
Polars' lazy evaluation to minimize memory usage during cross-table
validation operations.

Key Optimizations:
    - Lazy aggregation pushdown to avoid materializing intermediate results
    - Streaming joins for large tables
    - Incremental aggregation for grouped operations
    - Caching of frequently used aggregations

Usage:
    class OptimizedCrossTableValidator(CrossTableValidator, LazyAggregationMixin):
        def validate_aggregates(self, orders, order_items):
            return self.streaming_aggregate_join(
                left=orders,
                right=order_items,
                join_key="order_id",
                agg_exprs=[pl.col("quantity").sum()],
            )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
from functools import lru_cache
import hashlib

import polars as pl


@dataclass
class AggregationResult:
    """Result of an aggregation operation.

    Attributes:
        data: Aggregated data (lazy or collected)
        row_count: Number of rows in result
        memory_estimate_mb: Estimated memory usage
        was_streaming: Whether streaming was used
    """

    data: pl.LazyFrame | pl.DataFrame
    row_count: int | None = None
    memory_estimate_mb: float | None = None
    was_streaming: bool = False

    def collect(self) -> pl.DataFrame:
        """Collect lazy result if needed."""
        if isinstance(self.data, pl.LazyFrame):
            return self.data.collect()
        return self.data

    def lazy(self) -> pl.LazyFrame:
        """Convert to lazy if needed."""
        if isinstance(self.data, pl.DataFrame):
            return self.data.lazy()
        return self.data


@dataclass
class JoinStrategy:
    """Configuration for join operations.

    Attributes:
        method: Join method ('hash', 'sort_merge', 'broadcast')
        streaming: Whether to use streaming join
        slice_size: Slice size for streaming joins
        parallel: Enable parallel processing
    """

    method: str = "hash"
    streaming: bool = False
    slice_size: int = 100000
    parallel: bool = True


class LazyAggregationMixin:
    """Mixin providing lazy aggregation operations.

    Use in validators that perform cross-table aggregations or joins.
    Leverages Polars lazy evaluation for memory efficiency.

    Features:
        - Lazy expression aggregation
        - Streaming joins for large tables
        - Aggregation result caching
        - Memory-efficient grouped operations

    Example:
        class CrossTableValidator(BaseValidator, LazyAggregationMixin):
            def validate(self, orders, order_items):
                # Compute aggregates without materializing full join
                result = self.aggregate_with_join(
                    left=orders.lazy(),
                    right=order_items.lazy(),
                    left_on="order_id",
                    right_on="order_id",
                    agg_exprs=[
                        pl.col("quantity").sum().alias("total_qty"),
                        pl.col("price").sum().alias("total_price"),
                    ],
                )
                return self.compare_aggregates(orders, result)
    """

    # Configuration
    _agg_cache: dict[str, AggregationResult] = {}
    _cache_enabled: bool = True
    _default_join_strategy: JoinStrategy = field(default_factory=JoinStrategy)

    def aggregate_lazy(
        self,
        lf: pl.LazyFrame,
        group_by: str | list[str],
        agg_exprs: list[pl.Expr],
        maintain_order: bool = False,
        cache_key: str | None = None,
    ) -> AggregationResult:
        """Perform lazy aggregation with optional caching.

        Args:
            lf: Input LazyFrame
            group_by: Column(s) to group by
            agg_exprs: Aggregation expressions
            maintain_order: Whether to maintain group order
            cache_key: Optional key for caching result

        Returns:
            AggregationResult with lazy frame
        """
        if cache_key and cache_key in self._agg_cache:
            return self._agg_cache[cache_key]

        if isinstance(group_by, str):
            group_by = [group_by]

        result_lf = lf.group_by(group_by, maintain_order=maintain_order).agg(agg_exprs)

        result = AggregationResult(
            data=result_lf,
            was_streaming=False,
        )

        if cache_key and self._cache_enabled:
            self._agg_cache[cache_key] = result

        return result

    def aggregate_with_join(
        self,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        left_on: str | list[str],
        right_on: str | list[str] | None = None,
        agg_exprs: list[pl.Expr] | None = None,
        how: str = "left",
    ) -> AggregationResult:
        """Join and aggregate in a single lazy operation.

        Optimizes by pushing aggregation before materialization.

        Args:
            left: Left LazyFrame
            right: Right LazyFrame
            left_on: Left join key(s)
            right_on: Right join key(s) (defaults to left_on)
            agg_exprs: Aggregation expressions for right side
            how: Join type ('left', 'inner', 'outer')

        Returns:
            AggregationResult
        """
        right_on = right_on or left_on

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        # Pre-aggregate right side if aggregations specified
        if agg_exprs:
            right_agg = right.group_by(right_on).agg(agg_exprs)
        else:
            right_agg = right

        # Perform join
        result_lf = left.join(
            right_agg,
            left_on=left_on,
            right_on=right_on,
            how=how,
        )

        return AggregationResult(
            data=result_lf,
            was_streaming=False,
        )

    def streaming_aggregate_join(
        self,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        join_key: str | list[str],
        agg_exprs: list[pl.Expr],
        slice_size: int = 100000,
    ) -> AggregationResult:
        """Streaming join with aggregation for large tables.

        Processes data in slices to limit memory usage.

        Args:
            left: Left LazyFrame (smaller, iterated over)
            right: Right LazyFrame (aggregated first)
            join_key: Join key column(s)
            agg_exprs: Aggregation expressions
            slice_size: Rows per slice

        Returns:
            AggregationResult
        """
        if isinstance(join_key, str):
            join_key = [join_key]

        # First, aggregate the right side (done once)
        right_agg = right.group_by(join_key).agg(agg_exprs)

        # Collect right side as it's typically smaller after aggregation
        right_df = right_agg.collect()

        # Process left side in slices
        left_df = left.collect()
        n_rows = len(left_df)

        result_dfs = []
        for start in range(0, n_rows, slice_size):
            end = min(start + slice_size, n_rows)
            slice_df = left_df.slice(start, end - start)

            # Join slice with aggregated right
            joined = slice_df.join(
                right_df,
                on=join_key,
                how="left",
            )
            result_dfs.append(joined)

        # Combine results
        if result_dfs:
            result_df = pl.concat(result_dfs)
        else:
            result_df = left_df.join(right_df, on=join_key, how="left")

        return AggregationResult(
            data=result_df,
            row_count=len(result_df),
            was_streaming=True,
        )

    def compare_aggregates(
        self,
        source: pl.LazyFrame | pl.DataFrame,
        aggregated: AggregationResult,
        key_column: str,
        source_column: str,
        agg_column: str,
        tolerance: float = 0.0,
    ) -> pl.DataFrame:
        """Compare source values with aggregated values.

        Useful for cross-table consistency checks.

        Args:
            source: Source data with expected values
            aggregated: Aggregation result to compare
            key_column: Join key
            source_column: Column in source with expected value
            agg_column: Column in aggregated with actual value
            tolerance: Allowed difference (absolute)

        Returns:
            DataFrame with mismatches
        """
        source_lf = source.lazy() if isinstance(source, pl.DataFrame) else source
        agg_lf = aggregated.lazy()

        # Join and compare
        compared = source_lf.join(
            agg_lf.select([key_column, agg_column]),
            on=key_column,
            how="left",
        ).with_columns([
            (pl.col(source_column) - pl.col(agg_column).fill_null(0))
            .abs()
            .alias("_diff"),
        ])

        # Filter mismatches
        mismatches = compared.filter(
            pl.col("_diff") > tolerance
        ).collect()

        return mismatches

    def incremental_aggregate(
        self,
        existing: AggregationResult | None,
        new_data: pl.LazyFrame,
        group_by: str | list[str],
        sum_columns: list[str] | None = None,
        count_column: str | None = None,
    ) -> AggregationResult:
        """Incrementally update aggregation with new data.

        For sum and count aggregations, combines existing and new
        without reprocessing all data.

        Args:
            existing: Existing aggregation result (or None)
            new_data: New data to incorporate
            group_by: Grouping columns
            sum_columns: Columns to sum
            count_column: Column name for count (if tracking)

        Returns:
            Updated AggregationResult
        """
        if isinstance(group_by, str):
            group_by = [group_by]

        sum_columns = sum_columns or []

        # Build aggregation expressions
        agg_exprs = [pl.col(c).sum().alias(c) for c in sum_columns]
        if count_column:
            agg_exprs.append(pl.len().alias(count_column))

        # Aggregate new data
        new_agg = new_data.group_by(group_by).agg(agg_exprs)

        if existing is None:
            return AggregationResult(data=new_agg, was_streaming=False)

        # Combine with existing
        existing_lf = existing.lazy()

        # Union and re-aggregate
        combined = pl.concat([existing_lf, new_agg])
        final_agg = combined.group_by(group_by).agg(agg_exprs)

        return AggregationResult(data=final_agg, was_streaming=False)

    def window_aggregate(
        self,
        lf: pl.LazyFrame,
        partition_by: str | list[str],
        agg_exprs: list[pl.Expr],
        order_by: str | None = None,
    ) -> pl.LazyFrame:
        """Apply window aggregations.

        More memory-efficient than self-join for computing
        group-level statistics.

        Args:
            lf: Input LazyFrame
            partition_by: Partition columns
            agg_exprs: Window aggregation expressions
            order_by: Optional ordering within partitions

        Returns:
            LazyFrame with window columns added
        """
        if isinstance(partition_by, str):
            partition_by = [partition_by]

        # Convert aggregations to window functions
        window_exprs = []
        for expr in agg_exprs:
            window_expr = expr.over(partition_by)
            window_exprs.append(window_expr)

        return lf.with_columns(window_exprs)

    def semi_join_filter(
        self,
        main: pl.LazyFrame,
        filter_by: pl.LazyFrame,
        on: str | list[str],
        anti: bool = False,
    ) -> pl.LazyFrame:
        """Filter using semi-join (more efficient than regular join).

        Args:
            main: Main data to filter
            filter_by: Data to filter by
            on: Join key(s)
            anti: If True, return rows NOT in filter_by

        Returns:
            Filtered LazyFrame
        """
        how = "anti" if anti else "semi"
        return main.join(filter_by.select(on if isinstance(on, list) else [on]), on=on, how=how)

    def multi_table_aggregate(
        self,
        tables: dict[str, pl.LazyFrame],
        joins: list[tuple[str, str, str | list[str]]],
        final_agg: list[pl.Expr],
        final_group_by: str | list[str] | None = None,
    ) -> AggregationResult:
        """Aggregate across multiple tables with optimized join order.

        Args:
            tables: Named tables {"orders": orders_lf, "items": items_lf, ...}
            joins: List of (left_table, right_table, join_keys)
            final_agg: Final aggregation expressions
            final_group_by: Final grouping columns

        Returns:
            AggregationResult
        """
        # Start with first table
        if not joins:
            raise ValueError("At least one join required")

        result = tables[joins[0][0]]

        # Apply joins in order
        for left_name, right_name, keys in joins:
            right = tables[right_name]
            if isinstance(keys, str):
                keys = [keys]

            result = result.join(right, on=keys, how="left")

        # Apply final aggregation
        if final_group_by:
            if isinstance(final_group_by, str):
                final_group_by = [final_group_by]
            result = result.group_by(final_group_by).agg(final_agg)
        else:
            result = result.select(final_agg)

        return AggregationResult(data=result, was_streaming=False)

    def clear_aggregation_cache(self) -> None:
        """Clear cached aggregation results."""
        self._agg_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_items": len(self._agg_cache),
            "cache_keys": list(self._agg_cache.keys()),
        }


class AggregationExpressionBuilder:
    """Builder for common aggregation expressions.

    Provides a fluent interface for building aggregation expressions
    with proper naming and type handling.

    Example:
        builder = AggregationExpressionBuilder()
        exprs = (
            builder
            .sum("quantity", alias="total_qty")
            .mean("price", alias="avg_price")
            .count()
            .build()
        )
    """

    def __init__(self) -> None:
        self._exprs: list[pl.Expr] = []

    def sum(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add sum aggregation."""
        expr = pl.col(column).sum()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def mean(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add mean aggregation."""
        expr = pl.col(column).mean()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def min(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add min aggregation."""
        expr = pl.col(column).min()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def max(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add max aggregation."""
        expr = pl.col(column).max()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def count(self, alias: str = "count") -> "AggregationExpressionBuilder":
        """Add count aggregation."""
        self._exprs.append(pl.len().alias(alias))
        return self

    def std(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add standard deviation aggregation."""
        expr = pl.col(column).std()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def first(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add first value aggregation."""
        expr = pl.col(column).first()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def last(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add last value aggregation."""
        expr = pl.col(column).last()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def n_unique(self, column: str, alias: str | None = None) -> "AggregationExpressionBuilder":
        """Add unique count aggregation."""
        expr = pl.col(column).n_unique()
        if alias:
            expr = expr.alias(alias)
        self._exprs.append(expr)
        return self

    def custom(self, expr: pl.Expr) -> "AggregationExpressionBuilder":
        """Add custom expression."""
        self._exprs.append(expr)
        return self

    def build(self) -> list[pl.Expr]:
        """Build and return expression list."""
        return self._exprs.copy()

    def clear(self) -> "AggregationExpressionBuilder":
        """Clear expressions."""
        self._exprs.clear()
        return self
