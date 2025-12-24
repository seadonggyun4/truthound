"""Protocol definitions for execution engines.

This module defines the structural typing protocols that all execution engine
implementations should follow.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import polars as pl


class ValidatorCapability(Enum):
    """Capabilities for validators indicating which engines they support.

    This enables a tier system where validators declare their compatibility:
    - POLARS_ONLY: Only works with Polars (e.g., ML-based validators)
    - SQL_CAPABLE: Can be pushed down to SQL databases
    - UNIVERSAL: Works with any execution engine
    """

    POLARS_ONLY = "polars_only"
    SQL_CAPABLE = "sql_capable"
    UNIVERSAL = "universal"


class AggregationType(Enum):
    """Types of aggregation operations."""

    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    VAR = "var"
    FIRST = "first"
    LAST = "last"
    COUNT_DISTINCT = "count_distinct"
    NULL_COUNT = "null_count"


@runtime_checkable
class ExecutionEngineProtocol(Protocol):
    """Protocol defining the interface for all execution engines.

    Execution engines are responsible for:
    - Running validation operations on data
    - Providing optimized implementations for different backends
    - Converting to Polars for validators that require it
    """

    @property
    def engine_type(self) -> str:
        """Get the type of execution engine."""
        ...

    @property
    def supports_sql_pushdown(self) -> bool:
        """Check if this engine supports SQL pushdown."""
        ...

    # -------------------------------------------------------------------------
    # Row Operations
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count total number of rows."""
        ...

    def get_columns(self) -> list[str]:
        """Get list of column names."""
        ...

    # -------------------------------------------------------------------------
    # Null Operations
    # -------------------------------------------------------------------------

    def count_nulls(self, column: str) -> int:
        """Count null values in a column."""
        ...

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls for all columns."""
        ...

    # -------------------------------------------------------------------------
    # Distinct/Unique Operations
    # -------------------------------------------------------------------------

    def count_distinct(self, column: str) -> int:
        """Count distinct values in a column."""
        ...

    def get_distinct_values(
        self,
        column: str,
        limit: int | None = None,
    ) -> list[Any]:
        """Get distinct values from a column."""
        ...

    # -------------------------------------------------------------------------
    # Value Access
    # -------------------------------------------------------------------------

    def get_column_values(
        self,
        column: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Any]:
        """Get values from a column."""
        ...

    def get_sample_values(
        self,
        column: str,
        n: int = 5,
    ) -> list[Any]:
        """Get sample values from a column (for error reporting)."""
        ...

    # -------------------------------------------------------------------------
    # Statistical Operations
    # -------------------------------------------------------------------------

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get comprehensive statistics for a numeric column.

        Returns dict with: count, null_count, mean, std, min, max, median,
                          q25, q50, q75
        """
        ...

    def get_quantiles(
        self,
        column: str,
        quantiles: list[float],
    ) -> list[float]:
        """Get specific quantiles for a column."""
        ...

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value frequency counts for a column."""
        ...

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        aggregations: dict[str, AggregationType],
    ) -> dict[str, Any]:
        """Perform multiple aggregations at once.

        Args:
            aggregations: Mapping of column name to aggregation type.

        Returns:
            Mapping of "column_aggregation" to result value.
        """
        ...

    def aggregate_column(
        self,
        column: str,
        agg_type: AggregationType,
    ) -> Any:
        """Perform a single aggregation on a column."""
        ...

    # -------------------------------------------------------------------------
    # Filter Operations
    # -------------------------------------------------------------------------

    def filter_by_condition(
        self,
        condition: str,
    ) -> "ExecutionEngineProtocol":
        """Create a new engine with filtered data.

        Args:
            condition: Filter condition (SQL-like syntax).

        Returns:
            New execution engine with filtered data.
        """
        ...

    def count_matching(self, condition: str) -> int:
        """Count rows matching a condition."""
        ...

    def count_not_matching(self, condition: str) -> int:
        """Count rows not matching a condition."""
        ...

    # -------------------------------------------------------------------------
    # Pattern Matching (String Operations)
    # -------------------------------------------------------------------------

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching a regex pattern."""
        ...

    def count_not_matching_regex(self, column: str, pattern: str) -> int:
        """Count values not matching a regex pattern."""
        ...

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
        ...

    def count_outside_range(
        self,
        column: str,
        min_value: Any | None = None,
        max_value: Any | None = None,
    ) -> int:
        """Count values outside a range."""
        ...

    # -------------------------------------------------------------------------
    # Set Operations
    # -------------------------------------------------------------------------

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values that are in a set."""
        ...

    def count_not_in_set(self, column: str, values: set[Any]) -> int:
        """Count values that are not in a set."""
        ...

    # -------------------------------------------------------------------------
    # Duplicate Operations
    # -------------------------------------------------------------------------

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicate rows based on specified columns."""
        ...

    def get_duplicate_values(
        self,
        columns: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get sample duplicate values."""
        ...

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame.

        This is the fallback for validators that require Polars.
        """
        ...

    def to_numpy(self, columns: list[str] | None = None) -> "np.ndarray":
        """Convert to numpy array.

        Args:
            columns: Columns to include. If None, includes all numeric columns.

        Returns:
            2D numpy array.
        """
        ...

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "ExecutionEngineProtocol":
        """Create a new engine with sampled data."""
        ...


@runtime_checkable
class SQLExecutionEngineProtocol(ExecutionEngineProtocol, Protocol):
    """Extended protocol for SQL-based execution engines."""

    def execute_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute a raw SQL query."""
        ...

    def build_count_query(self, condition: str | None = None) -> str:
        """Build a COUNT query with optional condition."""
        ...

    def build_aggregate_query(
        self,
        aggregations: dict[str, AggregationType],
    ) -> str:
        """Build an aggregation query."""
        ...
