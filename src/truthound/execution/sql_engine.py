"""SQL execution engine implementation.

This module provides an execution engine for SQL data sources,
enabling SQL pushdown optimization where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from truthound.execution.base import (
    BaseExecutionEngine,
    ExecutionConfig,
    ExecutionSizeError,
)
from truthound.execution._protocols import AggregationType

if TYPE_CHECKING:
    import numpy as np
    import polars as pl
    from truthound.datasources.sql.base import BaseSQLDataSource


class SQLExecutionEngine(BaseExecutionEngine[ExecutionConfig]):
    """Execution engine for SQL data sources.

    This engine pushes operations to the database where possible,
    minimizing data transfer and leveraging database optimization.

    Example:
        >>> from truthound.datasources.sql import PostgreSQLDataSource
        >>> source = PostgreSQLDataSource(table="users", ...)
        >>> engine = SQLExecutionEngine(source)
        >>> null_count = engine.count_nulls("email")  # Runs in database
    """

    engine_type = "sql"

    def __init__(
        self,
        datasource: "BaseSQLDataSource",
        config: ExecutionConfig | None = None,
    ) -> None:
        """Initialize SQL execution engine.

        Args:
            datasource: SQL data source.
            config: Optional configuration.
        """
        super().__init__(config)
        self._source = datasource

    @property
    def datasource(self) -> "BaseSQLDataSource":
        """Get the underlying data source."""
        return self._source

    @property
    def supports_sql_pushdown(self) -> bool:
        """SQL engine supports pushdown."""
        return True

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _quote(self, identifier: str) -> str:
        """Quote an identifier using the data source's method."""
        return self._source._quote_identifier(identifier)

    def _execute_scalar(self, query: str) -> Any:
        """Execute a query and return single value."""
        return self._source.execute_scalar(query)

    def _execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a query and return results."""
        return self._source.execute_query(query)

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def count_rows(self) -> int:
        """Count total number of rows."""
        cache_key = self._cache_key("count_rows")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        query = self._source.build_count_query()
        result = self._execute_scalar(query)
        self._set_cached(cache_key, result)
        return result

    def get_columns(self) -> list[str]:
        """Get list of column names."""
        return self._source.columns

    def count_nulls(self, column: str) -> int:
        """Count null values using SQL."""
        cache_key = self._cache_key("count_nulls", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        query = self._source.build_null_count_query(column)
        result = self._execute_scalar(query)
        self._set_cached(cache_key, result)
        return result

    def count_nulls_all(self) -> dict[str, int]:
        """Count nulls for all columns in a single query."""
        cache_key = self._cache_key("count_nulls_all")
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Build query with all null counts
        columns = self.get_columns()
        select_parts = [
            f"SUM(CASE WHEN {self._quote(col)} IS NULL THEN 1 ELSE 0 END) as null_{i}"
            for i, col in enumerate(columns)
        ]
        query = f"SELECT {', '.join(select_parts)} FROM {self._source.full_table_name}"

        result = self._execute_query(query)
        if not result:
            return {col: 0 for col in columns}

        row = result[0]
        null_counts = {col: row[f"null_{i}"] or 0 for i, col in enumerate(columns)}
        self._set_cached(cache_key, null_counts)
        return null_counts

    def count_distinct(self, column: str) -> int:
        """Count distinct values using SQL."""
        cache_key = self._cache_key("count_distinct", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        query = self._source.build_distinct_count_query(column)
        result = self._execute_scalar(query)
        self._set_cached(cache_key, result)
        return result

    # -------------------------------------------------------------------------
    # Statistical Operations
    # -------------------------------------------------------------------------

    def get_stats(self, column: str) -> dict[str, Any]:
        """Get statistics using SQL aggregations."""
        cache_key = self._cache_key("get_stats", column)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        query = self._source.build_stats_query(column)
        result = self._execute_query(query)

        if not result:
            return {}

        row = result[0]
        stats = {
            "count": row.get("count", 0),
            "null_count": row.get("null_count", 0),
            "mean": row.get("mean"),
            "min": row.get("min"),
            "max": row.get("max"),
            "sum": row.get("sum"),
            # Standard SQL doesn't have median/std - set to None
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
        }

        self._set_cached(cache_key, stats)
        return stats

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value frequency counts using SQL."""
        col = self._quote(column)
        query = f"""
            SELECT {col} as value, COUNT(*) as cnt
            FROM {self._source.full_table_name}
            GROUP BY {col}
            ORDER BY cnt DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        result = self._execute_query(query)
        return {row["value"]: row["cnt"] for row in result}

    # -------------------------------------------------------------------------
    # Aggregation Operations
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        aggregations: dict[str, AggregationType],
    ) -> dict[str, Any]:
        """Perform multiple aggregations in a single SQL query."""
        agg_sql = {
            AggregationType.COUNT: lambda c: f"COUNT({c})",
            AggregationType.SUM: lambda c: f"SUM({c})",
            AggregationType.MEAN: lambda c: f"AVG({c})",
            AggregationType.MIN: lambda c: f"MIN({c})",
            AggregationType.MAX: lambda c: f"MAX({c})",
            AggregationType.COUNT_DISTINCT: lambda c: f"COUNT(DISTINCT {c})",
            AggregationType.NULL_COUNT: lambda c: f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END)",
        }

        select_parts = []
        keys = []

        for col, agg_type in aggregations.items():
            key = f"{col}_{agg_type.value}"
            keys.append(key)

            quoted_col = self._quote(col)
            if agg_type in agg_sql:
                select_parts.append(f"{agg_sql[agg_type](quoted_col)} as {key}")
            else:
                # Unsupported in SQL, will return None
                select_parts.append(f"NULL as {key}")

        query = f"SELECT {', '.join(select_parts)} FROM {self._source.full_table_name}"
        result = self._execute_query(query)

        if not result:
            return {key: None for key in keys}

        return {key: result[0].get(key) for key in keys}

    # -------------------------------------------------------------------------
    # Value Operations
    # -------------------------------------------------------------------------

    def get_distinct_values(
        self,
        column: str,
        limit: int | None = None,
    ) -> list[Any]:
        """Get distinct values using SQL."""
        col = self._quote(column)
        query = f"SELECT DISTINCT {col} FROM {self._source.full_table_name}"
        if limit:
            query += f" LIMIT {limit}"

        result = self._execute_query(query)
        return [row[column] for row in result]

    def get_column_values(
        self,
        column: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Any]:
        """Get values from a column using SQL."""
        col = self._quote(column)
        query = f"SELECT {col} FROM {self._source.full_table_name}"
        if limit:
            query += f" LIMIT {limit}"
        if offset > 0:
            query += f" OFFSET {offset}"

        result = self._execute_query(query)
        return [row[column] for row in result]

    def get_sample_values(self, column: str, n: int = 5) -> list[Any]:
        """Get sample non-null values."""
        col = self._quote(column)
        query = f"""
            SELECT {col}
            FROM {self._source.full_table_name}
            WHERE {col} IS NOT NULL
            LIMIT {n}
        """
        result = self._execute_query(query)
        return [row[column] for row in result]

    # -------------------------------------------------------------------------
    # Filter Operations
    # -------------------------------------------------------------------------

    def count_matching(self, condition: str) -> int:
        """Count rows matching a SQL condition."""
        query = self._source.build_count_query(condition)
        return self._execute_scalar(query)

    def count_not_matching(self, condition: str) -> int:
        """Count rows not matching a condition."""
        total = self.count_rows()
        matching = self.count_matching(condition)
        return total - matching

    def filter_by_condition(self, condition: str) -> "SQLExecutionEngine":
        """Create a filtered view using SQL.

        Note: This creates a subquery-based view, not a new table.
        """
        # Create a wrapped data source with the condition
        from truthound.datasources.sql.base import SampledSQLDataSource

        # For now, return self with a note that full filtering
        # would require more complex implementation
        raise NotImplementedError(
            "Full SQL filter_by_condition not yet implemented. "
            "Use count_matching() for condition-based counting."
        )

    # -------------------------------------------------------------------------
    # Pattern Matching
    # -------------------------------------------------------------------------

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Count values matching a regex pattern.

        Note: Regex syntax varies by database. This uses REGEXP for MySQL
        and ~ for PostgreSQL.
        """
        col = self._quote(column)
        source_type = self._source.source_type

        if source_type == "postgresql":
            query = f"""
                SELECT COUNT(*)
                FROM {self._source.full_table_name}
                WHERE {col} ~ '{pattern}'
            """
        elif source_type == "mysql":
            query = f"""
                SELECT COUNT(*)
                FROM {self._source.full_table_name}
                WHERE {col} REGEXP '{pattern}'
            """
        elif source_type == "sqlite":
            # SQLite doesn't have native regex support
            # Fall back to Polars
            return super().count_matching_regex(column, pattern)
        else:
            return super().count_matching_regex(column, pattern)

        return self._execute_scalar(query)

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
        """Count values within a range using SQL."""
        col = self._quote(column)
        conditions = []

        if min_value is not None:
            op = ">=" if inclusive else ">"
            conditions.append(f"{col} {op} {min_value}")

        if max_value is not None:
            op = "<=" if inclusive else "<"
            conditions.append(f"{col} {op} {max_value}")

        if not conditions:
            return self.count_rows()

        condition = " AND ".join(conditions)
        return self.count_matching(condition)

    # -------------------------------------------------------------------------
    # Set Operations
    # -------------------------------------------------------------------------

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values in a set using SQL IN clause."""
        if not values:
            return 0

        col = self._quote(column)

        # Format values for SQL
        formatted = []
        for v in values:
            if isinstance(v, str):
                escaped = v.replace("'", "''")
                formatted.append(f"'{escaped}'")
            elif v is None:
                continue  # NULL handled separately
            else:
                formatted.append(str(v))

        if not formatted:
            return 0

        condition = f"{col} IN ({', '.join(formatted)})"
        return self.count_matching(condition)

    # -------------------------------------------------------------------------
    # Duplicate Operations
    # -------------------------------------------------------------------------

    def count_duplicates(self, columns: list[str]) -> int:
        """Count duplicate rows using SQL."""
        cols = ", ".join(self._quote(c) for c in columns)
        query = f"""
            SELECT COUNT(*) - COUNT(DISTINCT ({cols}))
            FROM {self._source.full_table_name}
        """
        return self._execute_scalar(query)

    def get_duplicate_values(
        self,
        columns: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get sample duplicate values using SQL."""
        cols = ", ".join(self._quote(c) for c in columns)
        query = f"""
            SELECT {cols}, COUNT(*) as cnt
            FROM {self._source.full_table_name}
            GROUP BY {cols}
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
            LIMIT {limit}
        """
        result = self._execute_query(query)
        # Remove the count column from results
        return [{k: v for k, v in row.items() if k != "cnt"} for row in result]

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame by fetching data.

        Warning: This loads data into memory.
        """
        row_count = self.count_rows()
        if row_count > self._config.max_rows_for_polars:
            raise ExecutionSizeError(
                operation="to_polars_lazyframe",
                current_size=row_count,
                max_size=self._config.max_rows_for_polars,
            )

        return self._source.to_polars_lazyframe()

    def to_numpy(self, columns: list[str] | None = None) -> "np.ndarray":
        """Convert to numpy array.

        This fetches data from database and converts to numpy.
        """
        import numpy as np

        row_count = self.count_rows()
        if row_count > self._config.max_rows_for_numpy:
            raise ExecutionSizeError(
                operation="to_numpy",
                current_size=row_count,
                max_size=self._config.max_rows_for_numpy,
            )

        if columns is None:
            # Get numeric columns
            from truthound.datasources._protocols import ColumnType
            schema = self._source.schema
            numeric_types = {ColumnType.INTEGER, ColumnType.FLOAT, ColumnType.DECIMAL}
            columns = [col for col, dtype in schema.items() if dtype in numeric_types]

        if not columns:
            return np.array([])

        # Fetch data
        cols = ", ".join(self._quote(c) for c in columns)
        query = f"SELECT {cols} FROM {self._source.full_table_name}"
        result = self._execute_query(query)

        # Convert to numpy
        data = [[row[col] for col in columns] for row in result]
        return np.array(data, dtype=float)

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "SQLExecutionEngine":
        """Create a new engine with sampled data."""
        sampled_source = self._source.sample(n, seed)
        return SQLExecutionEngine(sampled_source, self._config)

    # -------------------------------------------------------------------------
    # SQL-specific Methods
    # -------------------------------------------------------------------------

    def execute_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute a raw SQL query."""
        return self._execute_query(query)

    def build_count_query(self, condition: str | None = None) -> str:
        """Build a COUNT query."""
        return self._source.build_count_query(condition)
