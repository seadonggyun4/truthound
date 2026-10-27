"""SQL execution engine implementation.

This module provides an execution engine for SQL data sources,
enabling SQL pushdown optimization where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from truthound.execution._protocols import AggregationType
from truthound.execution.base import (
    BaseExecutionEngine,
    ExecutionConfig,
    ExecutionSizeError,
)

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
        datasource: BaseSQLDataSource,
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
    def datasource(self) -> BaseSQLDataSource:
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

        # Complete statistics share Core's numeric/null semantics and bounded
        # materialization contract instead of fabricating unavailable values.
        from truthound.execution.polars_engine import PolarsExecutionEngine

        stats = PolarsExecutionEngine(self.to_polars_lazyframe(), self._config).get_stats(column)

        self._set_cached(cache_key, stats)
        return stats

    def get_value_counts(
        self,
        column: str,
        limit: int | None = None,
    ) -> dict[Any, int]:
        """Get value frequency counts using SQL."""
        self._validate_window(limit)
        if limit == 0:
            return {}
        col = self._quote(column)
        query = f"""
            SELECT {col} as value, COUNT(*) as cnt
            FROM {self._source.full_table_name}
            GROUP BY {col}
            ORDER BY cnt DESC
        """
        result = self._execute_query(self._page_query(query, column, limit, projection="*"))
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

        if not aggregations:
            return {}
        if set(aggregations) - set(self.get_columns()):
            raise ValueError("Unknown aggregation column")
        selected = {column: AggregationType(kind) for column, kind in aggregations.items()}
        fallback = {column: kind for column, kind in selected.items() if kind not in agg_sql}
        # Reuse the Core implementation and its conversion bound. Unsupported
        # pushdown must not look like a successfully computed null statistic.
        result = super().aggregate(fallback) if fallback else {}
        parts, keys = [], []
        for column, kind in selected.items():
            if kind not in agg_sql:
                continue
            key = f"{column}_{kind.value}"
            keys.append(key)
            expression = agg_sql[kind](self._quote(column))
            if kind == AggregationType.NULL_COUNT:
                expression = f"COALESCE({expression}, 0)"
            parts.append(f"{expression} AS {self._quote(key)}")
        if parts:
            rows = self._execute_query(
                f"SELECT {', '.join(parts)} FROM {self._source.full_table_name}"
            )
            result.update({key: rows[0].get(key) if rows else None for key in keys})
        return {
            f"{column}_{kind.value}": result[f"{column}_{kind.value}"]
            for column, kind in selected.items()
        }

    # -------------------------------------------------------------------------
    # Value Operations
    # -------------------------------------------------------------------------

    def _page_query(
        self,
        query: str,
        column: str,
        limit: int | None,
        offset: int = 0,
        *,
        projection: str | None = None,
    ) -> str:
        self._validate_window(limit, offset)
        query = query.strip()
        dialect = getattr(self._source, "materialization_dialect", "limit")
        col = self._quote(column)
        if dialect == "top":
            if limit == 0:
                prefix = "SELECT DISTINCT " if query.startswith("SELECT DISTINCT ") else "SELECT "
                query = prefix + "TOP (0) " + query[len(prefix) :]
            elif offset:
                query += f" ORDER BY {col} OFFSET {offset} ROWS"
                if limit is not None:
                    query += f" FETCH NEXT {limit} ROWS ONLY"
            elif limit is not None:
                prefix = "SELECT DISTINCT " if query.startswith("SELECT DISTINCT ") else "SELECT "
                query = prefix + f"TOP ({limit}) " + query[len(prefix) :]
        elif dialect == "rownum":
            from uuid import uuid4

            number = "th_page_" + uuid4().hex
            source = self._source._alias_subquery(query, "th_source")
            query = f"SELECT th_source.*, ROWNUM {number} FROM {source}"
            if limit is not None:
                query += f" WHERE ROWNUM <= {offset + limit}"
            query = f"SELECT {projection or col} FROM {self._source._alias_subquery(query, 'th_page')} WHERE {number} > {offset}"
        else:
            if limit is not None:
                query += f" LIMIT {limit}"
            elif offset:
                root = self._source
                seen = set()
                while hasattr(root, "_parent") and id(root) not in seen:
                    seen.add(id(root))
                    root = root._parent
                kind = root.source_type
                if kind == "sqlite":
                    query += " LIMIT -1"
                elif kind == "mysql":
                    query += " LIMIT 18446744073709551615"
                elif kind not in {"postgresql", "redshift", "duckdb", "snowflake"}:
                    query += " LIMIT 9223372036854775807"
            if offset:
                query += f" OFFSET {offset}"
        return query

    def _page_values(
        self, query: str, column: str, limit: int | None, offset: int = 0
    ) -> list[Any]:
        return [
            row[column]
            for row in self._execute_query(self._page_query(query, column, limit, offset))
        ]

    def get_distinct_values(self, column: str, limit: int | None = None) -> list[Any]:
        """Get distinct values using provider pagination."""
        query = f"SELECT DISTINCT {self._quote(column)} FROM {self._source.full_table_name}"
        return self._page_values(query, column, limit)

    def get_column_values(
        self, column: str, limit: int | None = None, offset: int = 0
    ) -> list[Any]:
        """Get values with an explicit zero limit and provider-native offsets."""
        query = f"SELECT {self._quote(column)} FROM {self._source.full_table_name}"
        return self._page_values(query, column, limit, offset)

    def get_sample_values(self, column: str, n: int = 5) -> list[Any]:
        """Get leading non-null values using provider pagination."""
        col = self._quote(column)
        query = f"SELECT {col} FROM {self._source.full_table_name} WHERE {col} IS NOT NULL"
        return self._page_values(query, column, n)

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

    def filter_by_condition(self, condition: str) -> SQLExecutionEngine:
        """Create a filtered view using SQL.

        Note: This creates a subquery-based view, not a new table.
        """
        from truthound.datasources.sql.base import FilteredSQLDataSource

        return SQLExecutionEngine(FilteredSQLDataSource(self._source, condition), self._config)

    # -------------------------------------------------------------------------
    # Pattern Matching
    # -------------------------------------------------------------------------

    def _bound_parameter_style(self):
        from truthound.datasources.sql.base import FilteredSQLDataSource, SampledSQLDataSource

        source = self._source
        while isinstance(source, (FilteredSQLDataSource, SampledSQLDataSource)):
            source = source._parent
        return {"sqlite": "?", "duckdb": "?", "postgresql": "%s", "mysql": "%s"}.get(
            source.source_type
        ), source.source_type

    @staticmethod
    def _parameter_sql_text(text, placeholder):
        # Percent-style drivers also parse percent characters inside quoted SQL.
        return text.replace("%", "%%") if placeholder == "%s" else text

    def _bound_count_query(self, condition, placeholder):
        table = self._parameter_sql_text(self._source.full_table_name, placeholder)
        return f"SELECT COUNT(*) FROM {table} WHERE {condition}"

    def count_matching_regex(self, column: str, pattern: str) -> int:
        """Use native bound regex operands where supported, otherwise bounded Polars."""
        placeholder, source_type = self._bound_parameter_style()
        if source_type not in {"postgresql", "mysql"}:
            return super().count_matching_regex(column, pattern)
        operator = "~" if source_type == "postgresql" else "REGEXP"
        col = self._parameter_sql_text(self._quote(column), placeholder)
        query = self._bound_count_query(f"{col} {operator} {placeholder}", placeholder)
        return self._source.execute_scalar(query, (pattern,))

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
        if min_value is None and max_value is None:
            return self.count_rows()
        placeholder, _ = self._bound_parameter_style()
        if placeholder is None:
            return super().count_in_range(column, min_value, max_value, inclusive)
        col = self._parameter_sql_text(self._quote(column), placeholder)
        conditions, parameters = [], []
        if min_value is not None:
            conditions.append(f"{col} {'>=' if inclusive else '>'} {placeholder}")
            parameters.append(min_value)
        if max_value is not None:
            conditions.append(f"{col} {'<=' if inclusive else '<'} {placeholder}")
            parameters.append(max_value)
        query = self._bound_count_query(" AND ".join(conditions), placeholder)
        return self._source.execute_scalar(query, tuple(parameters))

    # -------------------------------------------------------------------------
    # Set Operations
    # -------------------------------------------------------------------------

    def count_in_set(self, column: str, values: set[Any]) -> int:
        """Count values in a set using SQL IN clause."""
        if not values:
            return 0

        placeholder, _ = self._bound_parameter_style()
        if placeholder is None:
            return super().count_in_set(column, values)
        parameters = tuple(value for value in values if value is not None)
        if not parameters:
            return 0
        col = self._parameter_sql_text(self._quote(column), placeholder)
        condition = f"{col} IN ({', '.join([placeholder] * len(parameters))})"
        return self._source.execute_scalar(self._bound_count_query(condition, placeholder), parameters)

    # -------------------------------------------------------------------------
    # Duplicate Operations
    # -------------------------------------------------------------------------

    def count_duplicates(self, columns: list[str]) -> int:
        """Count excess rows in native groups, including null-bearing groups."""
        alias = self._quote(self._duplicate_count_alias(columns))
        cols = ", ".join(self._quote(column) for column in columns)
        grouped = f"SELECT COUNT(*) AS {alias} FROM {self._source.full_table_name} GROUP BY {cols}"
        source = self._source._alias_subquery(grouped, "th_duplicates")
        return self._execute_scalar(f"SELECT COALESCE(SUM({alias} - 1), 0) FROM {source}")

    def get_duplicate_values(self, columns: list[str], limit: int = 10) -> list[dict[str, Any]]:
        """Return native duplicate groups with provider-specific pagination."""
        self._validate_window(limit)
        alias = self._quote(self._duplicate_count_alias(columns))
        if limit == 0:
            return []
        cols = ", ".join(self._quote(column) for column in columns)
        query = (
            f"SELECT {cols}, COUNT(*) AS {alias} FROM {self._source.full_table_name} "
            f"GROUP BY {cols} HAVING COUNT(*) > 1 ORDER BY {alias} DESC"
        )
        result = self._execute_query(self._page_query(query, columns[0], limit, projection=cols))
        return [{column: row[column] for column in columns} for row in result]

    # -------------------------------------------------------------------------
    # Conversion Methods
    # -------------------------------------------------------------------------

    def to_polars_lazyframe(self) -> pl.LazyFrame:
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

    def to_numpy(self, columns: list[str] | None = None) -> np.ndarray:
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
    ) -> SQLExecutionEngine:
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
