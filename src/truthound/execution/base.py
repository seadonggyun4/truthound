"""Base classes for execution engines.

This module provides the abstract base class for execution engines,
which are responsible for running validation operations on data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from truthound.execution._protocols import (
    AggregationType,
    ExecutionEngineProtocol,
)

if TYPE_CHECKING:
    import numpy as np
    import polars as pl
    from truthound.datasources.base import BaseDataSource


# =============================================================================
# Exceptions
# =============================================================================


class ExecutionError(Exception):
    """Base exception for execution engine errors."""

    pass


class UnsupportedOperationError(ExecutionError):
    """Raised when an operation is not supported by the engine."""

    def __init__(self, engine_type: str, operation: str) -> None:
        self.engine_type = engine_type
        self.operation = operation
        super().__init__(f"Operation '{operation}' not supported by {engine_type} engine")


class ExecutionSizeError(ExecutionError):
    """Raised when data size exceeds limits for an operation."""

    def __init__(self, operation: str, current_size: int, max_size: int) -> None:
        self.operation = operation
        self.current_size = current_size
        self.max_size = max_size
        super().__init__(
            f"Operation '{operation}' requires sampling: "
            f"current size ({current_size:,}) exceeds limit ({max_size:,})"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExecutionConfig:
    """Configuration for execution engines.

    Attributes:
        max_rows_for_numpy: Maximum rows for numpy conversion.
        max_rows_for_polars: Maximum rows for Polars conversion.
        sample_size_for_stats: Sample size for statistical operations on large data.
        cache_results: Whether to cache intermediate results.
        parallel: Whether to use parallel execution where possible.
    """

    max_rows_for_numpy: int = 100_000
    max_rows_for_polars: int = 10_000_000
    sample_size_for_stats: int = 100_000
    cache_results: bool = True
    parallel: bool = True


ConfigT = TypeVar("ConfigT", bound=ExecutionConfig)


# =============================================================================
# Abstract Base Execution Engine
# =============================================================================


class BaseExecutionEngine(ABC, Generic[ConfigT]):
    """Abstract base class for all execution engines.

    Execution engines provide the actual implementation for running
    validation operations. Different engines optimize for different
    backends (Polars, Pandas, SQL, etc.).

    Type Parameters:
        ConfigT: The configuration type for this engine.
    """

    engine_type: str = "base"

    def __init__(self, config: ConfigT | None = None) -> None:
        """Initialize the execution engine.

        Args:
            config: Optional configuration.
        """
        self._config = config or self._default_config()
        self._cache: dict[str, Any] = {}

    @classmethod
    def _default_config(cls) -> ConfigT:
        """Create default configuration."""
        return ExecutionConfig()  # type: ignore

    @property
    def config(self) -> ConfigT:
        """Get the engine configuration."""
        return self._config

    @property
    def supports_sql_pushdown(self) -> bool:
        """Check if this engine supports SQL pushdown.

        Override in SQL-capable engines.
        """
        return False

    # -------------------------------------------------------------------------
    # Abstract Methods - Core Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def count_rows(self) -> int:
        """Count total number of rows."""
        pass

    @abstractmethod
    def get_columns(self) -> list[str]:
        """Get list of column names."""
        pass

    @abstractmethod
    def count_nulls(self, column: str) -> int:
        """Count null values in a column."""
        pass

    @abstractmethod
    def count_distinct(self, column: str) -> int:
        """Count distinct values in a column."""
        pass

    @abstractmethod
    def get_stats(self, column: str) -> dict[str, Any]:
        """Get comprehensive statistics for a numeric column."""
        pass

    @abstractmethod
    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame."""
        pass

    @abstractmethod
    def sample(self, n: int = 1000, seed: int | None = None) -> "BaseExecutionEngine":
        """Create a new engine with sampled data."""
        pass

    # -------------------------------------------------------------------------
    # Default Implementations - Null Operations
    # -------------------------------------------------------------------------

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls for all columns.

        Default implementation calls count_nulls for each column.
        Override for optimized batch operations.
        """
        return {col: self.count_nulls(col) for col in self.get_columns()}

    def get_completeness_ratio(self, column: str) -> float:
        """Get completeness ratio (non-null / total) for a column."""
        total = self.count_rows()
        if total == 0:
            return 1.0
        nulls = self.count_nulls(column)
        return (total - nulls) / total

    # -------------------------------------------------------------------------
    # Default Implementations - Value Operations
    # -------------------------------------------------------------------------

    def get_distinct_values(
        self,
        column: str,
        limit: int | None = None,
    ) -> list[Any]:
        """Get distinct values from a column.

        Default implementation uses Polars. Override for optimization.
        """
        lf = self.to_polars_lazyframe()
        result = lf.select(column).unique().collect()
        values = result[column].to_list()
        if limit:
            return values[:limit]
        return values

    def get_column_values(
        self,
        column: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Any]:
        """Get values from a column.

        Default implementation uses Polars. Override for optimization.
        """
        lf = self.to_polars_lazyframe()
        if offset > 0:
            lf = lf.slice(offset, limit or 1_000_000)
        elif limit:
            lf = lf.limit(limit)
        return lf.select(column).collect()[column].to_list()

    def get_sample_values(self, column: str, n: int = 5) -> list[Any]:
        """Get sample values from a column."""
        return self.get_column_values(column, limit=n)

    # -------------------------------------------------------------------------
    # Default Implementations - Statistical Operations
    # -------------------------------------------------------------------------

    def get_quantiles(
        self,
        column: str,
        quantiles: list[float],
    ) -> list[float]:
        """Get specific quantiles for a column.

        Default implementation uses Polars.
        """
        import polars as pl

        lf = self.to_polars_lazyframe()
        exprs = [
            pl.col(column).quantile(q, interpolation="linear").alias(f"q_{i}")
            for i, q in enumerate(quantiles)
        ]
        result = lf.select(exprs).collect()
        return [result[f"q_{i}"][0] for i in range(len(quantiles))]

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value frequency counts for a column."""
        import polars as pl

        lf = self.to_polars_lazyframe()
        counts = (
            lf.group_by(column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        if limit:
            counts = counts.limit(limit)
        result = counts.collect()
        return dict(zip(result[column].to_list(), result["count"].to_list()))

    # -------------------------------------------------------------------------
    # Default Implementations - Aggregation Operations
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        aggregations: dict[str, AggregationType],
    ) -> dict[str, Any]:
        """Perform multiple aggregations at once.

        Default implementation uses Polars.
        """
        import polars as pl

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

        lf = self.to_polars_lazyframe()
        result = lf.select(exprs).collect()

        return {key: result[key][0] for key in keys}

    def aggregate_column(
        self,
        column: str,
        agg_type: AggregationType,
    ) -> Any:
        """Perform a single aggregation on a column."""
        result = self.aggregate({column: agg_type})
        return result[f"{column}_{agg_type.value}"]

    # -------------------------------------------------------------------------
    # Default Implementations - Filter Operations
    # -------------------------------------------------------------------------

    def count_matching(self, condition: str) -> int:
        """Count rows matching a condition.

        Default implementation uses filter + count. Override for SQL pushdown.
        """
        filtered = self.filter_by_condition(condition)
        return filtered.count_rows()

    def count_not_matching(self, condition: str) -> int:
        """Count rows not matching a condition."""
        return self.count_rows() - self.count_matching(condition)

    def filter_by_condition(self, condition: str) -> "BaseExecutionEngine":
        """Create a new engine with filtered data.

        Default implementation parses condition for Polars.
        Override for SQL-native filtering.
        """
        # This is a basic implementation - complex conditions need override
        raise NotImplementedError(
            f"{self.engine_type} engine doesn't support condition filtering. "
            "Use Polars expressions directly."
        )

    # -------------------------------------------------------------------------
    # Default Implementations - Pattern Matching
    # -------------------------------------------------------------------------

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching a regex pattern."""
        import polars as pl

        lf = self.to_polars_lazyframe()
        result = (
            lf.select(pl.col(column).str.contains(pattern).sum().alias("count"))
            .collect()
        )
        return result["count"][0]

    def count_not_matching_regex(self, column: str, pattern: str) -> int:
        """Count values not matching a regex pattern."""
        total = self.count_rows()
        matching = self.count_matching_regex(column, pattern)
        return total - matching

    # -------------------------------------------------------------------------
    # Default Implementations - Range Operations
    # -------------------------------------------------------------------------

    def count_in_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
        inclusive: bool = True,
    ) -> int:
        """Count values within a range."""
        import polars as pl

        lf = self.to_polars_lazyframe()
        expr = pl.col(column)

        if min_value is not None:
            if inclusive:
                expr = expr.ge(min_value)
            else:
                expr = expr.gt(min_value)

        if max_value is not None:
            max_expr = pl.col(column)
            if inclusive:
                max_expr = max_expr.le(max_value)
            else:
                max_expr = max_expr.lt(max_value)
            if min_value is not None:
                expr = expr & max_expr
            else:
                expr = max_expr

        result = lf.select(expr.sum().alias("count")).collect()
        return result["count"][0]

    def count_outside_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
    ) -> int:
        """Count values outside a range."""
        total = self.count_rows()
        in_range = self.count_in_range(column, min_value, max_value)
        return total - in_range

    # -------------------------------------------------------------------------
    # Default Implementations - Set Operations
    # -------------------------------------------------------------------------

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values that are in a set."""
        import polars as pl

        lf = self.to_polars_lazyframe()
        result = (
            lf.select(pl.col(column).is_in(list(values)).sum().alias("count"))
            .collect()
        )
        return result["count"][0]

    def count_not_in_set(self, column: str, values: set[Any]) -> int:
        """Count values that are not in a set."""
        total = self.count_rows()
        in_set = self.count_in_set(column, values)
        return total - in_set

    # -------------------------------------------------------------------------
    # Default Implementations - Duplicate Operations
    # -------------------------------------------------------------------------

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicate rows based on specified columns."""
        import polars as pl

        lf = self.to_polars_lazyframe()
        total = self.count_rows()
        unique = lf.select(columns).unique().select(pl.len()).collect().item()
        return total - unique

    def get_duplicate_values(
        self,
        columns: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get sample duplicate values."""
        import polars as pl

        lf = self.to_polars_lazyframe()

        # Find duplicates
        dupes = (
            lf.group_by(columns)
            .agg(pl.len().alias("_count"))
            .filter(pl.col("_count") > 1)
            .sort("_count", descending=True)
            .limit(limit)
            .drop("_count")
            .collect()
        )

        return dupes.to_dicts()

    # -------------------------------------------------------------------------
    # Default Implementations - Conversion Methods
    # -------------------------------------------------------------------------

    def to_numpy(self, columns: list[str] | None = None) -> "np.ndarray":
        """Convert to numpy array.

        Default implementation uses Polars as intermediary.
        """
        import numpy as np

        lf = self.to_polars_lazyframe()

        # Check size limit
        row_count = self.count_rows()
        if row_count > self._config.max_rows_for_numpy:
            raise ExecutionSizeError(
                operation="to_numpy",
                current_size=row_count,
                max_size=self._config.max_rows_for_numpy,
            )

        if columns is None:
            # Get numeric columns only
            import polars as pl
            schema = lf.collect_schema()
            numeric_types = {
                pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                pl.Float32, pl.Float64,
            }
            columns = [
                col for col, dtype in schema.items()
                if type(dtype) in numeric_types
            ]

        if not columns:
            return np.array([])

        df = lf.select(columns).collect()
        return df.to_numpy()

    def to_numpy_safe(
        self,
        columns: list[str] | None = None,
        max_rows: int | None = None,
        seed: int | None = 42,
    ) -> "np.ndarray":
        """Convert to numpy array with automatic sampling if needed.

        This is a safer version that automatically samples large datasets.
        """
        import numpy as np

        max_rows = max_rows or self._config.max_rows_for_numpy
        row_count = self.count_rows()

        engine = self
        if row_count > max_rows:
            engine = self.sample(n=max_rows, seed=seed)

        return engine.to_numpy(columns)

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _cache_key(self, operation: str, *args: Any) -> str:
        """Generate a cache key for an operation."""
        return f"{operation}:{':'.join(str(a) for a in args)}"

    def _get_cached(self, key: str) -> Any | None:
        """Get a cached result."""
        if self._config.cache_results:
            return self._cache.get(key)
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        """Set a cached result."""
        if self._config.cache_results:
            self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "BaseExecutionEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.clear_cache()

    def __repr__(self) -> str:
        """Get string representation."""
        return f"{self.__class__.__name__}(type='{self.engine_type}')"
