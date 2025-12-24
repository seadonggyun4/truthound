"""Polars execution engine implementation.

This module provides the primary execution engine based on Polars,
which is the default and most feature-complete engine in Truthound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from truthound.execution.base import (
    BaseExecutionEngine,
    ExecutionConfig,
)
from truthound.execution._protocols import AggregationType

if TYPE_CHECKING:
    import numpy as np


class PolarsExecutionEngine(BaseExecutionEngine[ExecutionConfig]):
    """Execution engine based on Polars LazyFrame.

    This is the primary execution engine that leverages Polars' lazy
    evaluation and query optimization for high performance.

    Example:
        >>> engine = PolarsExecutionEngine(lf)
        >>> null_count = engine.count_nulls("column_name")
        >>> stats = engine.get_stats("numeric_column")
    """

    engine_type = "polars"

    def __init__(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        config: ExecutionConfig | None = None,
    ) -> None:
        """Initialize Polars execution engine.

        Args:
            data: Polars LazyFrame or DataFrame.
            config: Optional configuration.
        """
        super().__init__(config)

        if isinstance(data, pl.DataFrame):
            self._lf = data.lazy()
        else:
            self._lf = data

        self._schema = self._lf.collect_schema()

    @property
    def lazyframe(self) -> pl.LazyFrame:
        """Get the underlying LazyFrame."""
        return self._lf

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count total number of rows."""
        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._lf.select(pl.len()).collect().item()
        self._set_cached(cache_key, result)
        return result

    def get_columns(self) -> list[str]:
        """Get list of column names."""
        return list(self._schema.names())

    def count_nulls(self, column: str) -> int:
        """Count null values in a column."""
        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._lf.select(pl.col(column).null_count()).collect().item()
        self._set_cached(cache_key, result)
        return result

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls for all columns in a single query."""
        cache_key = self._cache_key("count_nulls_all")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        exprs = [pl.col(c).null_count().alias(f"_null_{c}") for c in self.get_columns()]
        result = self._lf.select(exprs).collect()

        null_counts = {
            col: result[f"_null_{col}"][0]
            for col in self.get_columns()
        }
        self._set_cached(cache_key, null_counts)
        return null_counts

    def count_distinct(self, column: str) -> int:
        """Count distinct values in a column."""
        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result = self._lf.select(pl.col(column).n_unique()).collect().item()
        self._set_cached(cache_key, result)
        return result

    # -------------------------------------------------------------------------
    # Statistical Operations
    # -------------------------------------------------------------------------

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get comprehensive statistics for a numeric column."""
        cache_key = self._cache_key("get_stats", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        exprs = [
            pl.col(column).count().alias("count"),
            pl.col(column).null_count().alias("null_count"),
            pl.col(column).mean().alias("mean"),
            pl.col(column).std().alias("std"),
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
            pl.col(column).median().alias("median"),
            pl.col(column).quantile(0.25, interpolation="linear").alias("q25"),
            pl.col(column).quantile(0.75, interpolation="linear").alias("q75"),
            pl.col(column).sum().alias("sum"),
        ]

        result = self._lf.select(exprs).collect()

        stats = {
            "count": result["count"][0],
            "null_count": result["null_count"][0],
            "mean": result["mean"][0],
            "std": result["std"][0],
            "min": result["min"][0],
            "max": result["max"][0],
            "median": result["median"][0],
            "q25": result["q25"][0],
            "q75": result["q75"][0],
            "sum": result["sum"][0],
        }

        self._set_cached(cache_key, stats)
        return stats

    def get_quantiles(
        self,
        column: str,
        quantiles: list[float],
    ) -> list[float]:
        """Get specific quantiles for a column."""
        exprs = [
            pl.col(column).quantile(q, interpolation="linear").alias(f"q_{i}")
            for i, q in enumerate(quantiles)
        ]
        result = self._lf.select(exprs).collect()
        return [result[f"q_{i}"][0] for i in range(len(quantiles))]

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value frequency counts for a column."""
        counts = (
            self._lf.group_by(column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        if limit:
            counts = counts.limit(limit)
        result = counts.collect()
        return dict(zip(result[column].to_list(), result["count"].to_list()))

    # -------------------------------------------------------------------------
    # Aggregation Operations (Optimized)
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        aggregations: dict[str, AggregationType],
    ) -> dict[str, Any]:
        """Perform multiple aggregations in a single query."""
        agg_map = {
            AggregationType.COUNT: lambda c: pl.col(c).count(),
            AggregationType.SUM: lambda c: pl.col(c).sum(),
            AggregationType.MEAN: lambda c: pl.col(c).mean(),
            AggregationType.MEDIAN: lambda c: pl.col(c).median(),
            AggregationType.MIN: lambda c: pl.col(c).min(),
            AggregationType.MAX: lambda c: pl.col(c).max(),
            AggregationType.STD: lambda c: pl.col(c).std(),
            AggregationType.VAR: lambda c: pl.col(c).var(),
            AggregationType.FIRST: lambda c: pl.col(c).first(),
            AggregationType.LAST: lambda c: pl.col(c).last(),
            AggregationType.COUNT_DISTINCT: lambda c: pl.col(c).n_unique(),
            AggregationType.NULL_COUNT: lambda c: pl.col(c).null_count(),
        }

        exprs = []
        keys = []
        for col, agg_type in aggregations.items():
            key = f"{col}_{agg_type.value}"
            expr = agg_map[agg_type](col).alias(key)
            exprs.append(expr)
            keys.append(key)

        result = self._lf.select(exprs).collect()
        return {key: result[key][0] for key in keys}

    # -------------------------------------------------------------------------
    # Value Operations
    # -------------------------------------------------------------------------

    def get_distinct_values(
        self,
        column: str,
        limit: int | None = None,
    ) -> list[Any]:
        """Get distinct values from a column."""
        unique = self._lf.select(column).unique()
        if limit:
            unique = unique.limit(limit)
        return unique.collect()[column].to_list()

    def get_column_values(
        self,
        column: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Any]:
        """Get values from a column."""
        lf = self._lf
        if offset > 0:
            lf = lf.slice(offset, limit or 1_000_000)
        elif limit:
            lf = lf.limit(limit)
        return lf.select(column).collect()[column].to_list()

    def get_sample_values(self, column: str, n: int = 5) -> list[Any]:
        """Get sample non-null values from a column."""
        result = (
            self._lf
            .select(column)
            .filter(pl.col(column).is_not_null())
            .limit(n)
            .collect()
        )
        return result[column].to_list()

    # -------------------------------------------------------------------------
    # Filter Operations
    # -------------------------------------------------------------------------

    def filter_by_condition(self, condition: str) -> "PolarsExecutionEngine":
        """Create a new engine with filtered data.

        Supports SQL-like conditions that are converted to Polars expressions.
        """
        # For now, use Polars SQL context for parsing
        # This provides SQL-like syntax support
        try:
            ctx = pl.SQLContext(data=self._lf)
            filtered = ctx.execute(f"SELECT * FROM data WHERE {condition}")
            return PolarsExecutionEngine(filtered, self._config)
        except Exception:
            # Fallback: try direct Polars expression
            # This is limited to simple column expressions
            raise NotImplementedError(
                f"Condition '{condition}' cannot be parsed. "
                "Use Polars expressions directly on the LazyFrame."
            )

    def count_matching(self, condition: str) -> int:
        """Count rows matching a condition using SQL context."""
        try:
            ctx = pl.SQLContext(data=self._lf)
            result = ctx.execute(
                f"SELECT COUNT(*) as cnt FROM data WHERE {condition}"
            ).collect()
            return result["cnt"][0]
        except Exception:
            raise NotImplementedError(
                f"Condition '{condition}' cannot be parsed. "
                "Use Polars expressions directly."
            )

    # -------------------------------------------------------------------------
    # Pattern Matching
    # -------------------------------------------------------------------------

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching a regex pattern."""
        result = (
            self._lf
            .select(
                pl.col(column)
                .str.contains(pattern)
                .fill_null(False)
                .sum()
                .alias("count")
            )
            .collect()
        )
        return result["count"][0]

    def count_not_matching_regex(self, column: str, pattern: str) -> int:
        """Count values not matching a regex pattern."""
        result = (
            self._lf
            .select(
                pl.col(column)
                .str.contains(pattern)
                .fill_null(True)
                .not_()
                .sum()
                .alias("count")
            )
            .collect()
        )
        return result["count"][0]

    # -------------------------------------------------------------------------
    # Range Operations
    # -------------------------------------------------------------------------

    def count_in_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        inclusive: bool = True,
    ) -> int:
        """Count values within a range."""
        conditions = []

        if min_value is not None:
            if inclusive:
                conditions.append(pl.col(column) >= min_value)
            else:
                conditions.append(pl.col(column) > min_value)

        if max_value is not None:
            if inclusive:
                conditions.append(pl.col(column) <= max_value)
            else:
                conditions.append(pl.col(column) < max_value)

        if not conditions:
            return self.count_rows()

        combined = conditions[0]
        for cond in conditions[1:]:
            combined = combined & cond

        result = self._lf.select(combined.sum().alias("count")).collect()
        return result["count"][0]

    # -------------------------------------------------------------------------
    # Set Operations
    # -------------------------------------------------------------------------

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values that are in a set."""
        result = (
            self._lf
            .select(pl.col(column).is_in(list(values)).sum().alias("count"))
            .collect()
        )
        return result["count"][0]

    # -------------------------------------------------------------------------
    # Duplicate Operations
    # -------------------------------------------------------------------------

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicate rows based on specified columns."""
        total = self.count_rows()
        unique = self._lf.select(columns).unique().select(pl.len()).collect().item()
        return total - unique

    def get_duplicate_values(
        self,
        columns: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get sample duplicate values."""
        dupes = (
            self._lf
            .group_by(columns)
            .agg(pl.len().alias("_count"))
            .filter(pl.col("_count") > 1)
            .sort("_count", descending=True)
            .limit(limit)
            .drop("_count")
            .collect()
        )
        return dupes.to_dicts()

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Return the underlying LazyFrame."""
        return self._lf

    def to_polars_dataframe(self) -> pl.DataFrame:
        """Collect and return as DataFrame."""
        return self._lf.collect()

    def to_numpy(self, columns: list[str] | None = None) -> "np.ndarray":
        """Convert to numpy array."""
        from truthound.execution.base import ExecutionSizeError

        row_count = self.count_rows()
        if row_count > self._config.max_rows_for_numpy:
            raise ExecutionSizeError(
                operation="to_numpy",
                current_size=row_count,
                max_size=self._config.max_rows_for_numpy,
            )

        if columns is None:
            # Get numeric columns
            numeric_types = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }
            columns = [
                col for col, dtype in self._schema.items()
                if type(dtype) in numeric_types
            ]

        if not columns:
            import numpy as np
            return np.array([])

        return self._lf.select(columns).collect().to_numpy()

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "PolarsExecutionEngine":
        """Create a new engine with sampled data."""
        row_count = self.count_rows()

        if row_count <= n:
            # No need to sample
            return PolarsExecutionEngine(self._lf, self._config)

        # Sample using Polars
        fraction = min(n / row_count, 1.0)
        sampled = self._lf.collect().sample(fraction=fraction, seed=seed).lazy()

        return PolarsExecutionEngine(sampled, self._config)

    # -------------------------------------------------------------------------
    # Advanced Polars Operations
    # -------------------------------------------------------------------------

    def select(self, *exprs: pl.Expr | str) -> "PolarsExecutionEngine":
        """Create a new engine with selected columns/expressions."""
        return PolarsExecutionEngine(self._lf.select(*exprs), self._config)

    def filter(self, expr: pl.Expr) -> "PolarsExecutionEngine":
        """Create a new engine with filtered data using Polars expression."""
        return PolarsExecutionEngine(self._lf.filter(expr), self._config)

    def with_columns(self, *exprs: pl.Expr) -> "PolarsExecutionEngine":
        """Create a new engine with additional columns."""
        return PolarsExecutionEngine(self._lf.with_columns(*exprs), self._config)

    def group_by(self, *columns: str) -> pl.LazyGroupBy:
        """Get a Polars LazyGroupBy for custom aggregations."""
        return self._lf.group_by(*columns)
