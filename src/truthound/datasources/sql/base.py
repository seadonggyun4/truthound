"""Base classes for SQL data sources.

This module provides the abstract base class for SQL-based data sources,
with connection pooling and common SQL operations.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
from typing import TYPE_CHECKING, Any

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
)
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    DataSourceConnectionError,
    DataSourceError,
    DataSourceSchemaError,
    DataSourceSizeError,
    sql_type_to_column_type,
)

if TYPE_CHECKING:
    import polars as pl

    from truthound.execution.base import BaseExecutionEngine


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SQLDataSourceConfig(DataSourceConfig):
    """Configuration for SQL data sources.

    Attributes:
        pool_size: Number of connections in the pool.
        pool_timeout: Timeout for acquiring a connection from pool.
        query_timeout: Timeout for query execution.
        fetch_size: Number of rows to fetch at a time.
        materialization_row_limit: Maximum rows allowed in an in-memory
            Polars fallback. SQL pushdown operations are not limited by this
            value.
        use_server_side_cursor: Use server-side cursor for large results.
        schema_name: Database schema name (for PostgreSQL, etc.).
    """

    pool_size: int = 5
    pool_timeout: float = 30.0
    query_timeout: float = 300.0
    fetch_size: int = 10000
    materialization_row_limit: int = 100_000
    use_server_side_cursor: bool = False
    schema_name: str | None = None


def _row_mapping(row: Any, columns: list[str]) -> dict[str, Any]:
    """Normalize DB-API, mapping, and SQLAlchemy-style rows by column name."""

    if len(columns) != len(set(columns)):
        raise DataSourceSchemaError(
            "SQL result contains duplicate column names. Alias duplicate columns "
            "before converting the result to a mapping."
        )

    mapping: Mapping[Any, Any] | None = None
    if isinstance(row, Mapping):
        mapping = row
    else:
        candidate = getattr(row, "_mapping", None)
        if isinstance(candidate, Mapping):
            mapping = candidate

    if mapping is not None:
        if all(column in mapping for column in columns):
            return {column: mapping[column] for column in columns}

        casefolded = {str(key).casefold(): value for key, value in mapping.items()}
        if all(column.casefold() in casefolded for column in columns):
            return {column: casefolded[column.casefold()] for column in columns}

        values = list(mapping.values())
        if len(values) == len(columns):
            return dict(zip(columns, values, strict=True))
        raise DataSourceSchemaError(
            "SQL row keys do not match cursor description columns."
        )

    try:
        values = list(row)
    except TypeError as exc:
        raise DataSourceSchemaError(
            f"Unsupported SQL row type: {type(row).__name__}"
        ) from exc
    if len(values) != len(columns):
        raise DataSourceSchemaError(
            "SQL row value count does not match cursor description columns."
        )
    return dict(zip(columns, values, strict=True))


def _scalar_value(row: Any) -> Any:
    """Extract the first selected value without assuming a tuple row."""

    if row is None:
        return None
    if isinstance(row, Mapping):
        return next(iter(row.values()), None)
    candidate = getattr(row, "_mapping", None)
    if isinstance(candidate, Mapping):
        return next(iter(candidate.values()), None)
    try:
        return row[0]
    except (KeyError, IndexError, TypeError):
        try:
            return next(iter(row))
        except TypeError:
            return row


def _schema_pair(row: Any) -> tuple[str, str]:
    """Normalize a schema query row to ``(column_name, data_type)``."""

    if isinstance(row, Mapping):
        lowered = {str(key).casefold(): value for key, value in row.items()}
        name = lowered.get("column_name", lowered.get("name"))
        data_type = lowered.get("data_type", lowered.get("type"))
        if name is not None and data_type is not None:
            return str(name), str(data_type)
        values = list(row.values())
    else:
        candidate = getattr(row, "_mapping", None)
        if isinstance(candidate, Mapping):
            return _schema_pair(candidate)
        try:
            values = list(row)
        except TypeError as exc:
            raise DataSourceSchemaError(
                f"Unsupported SQL schema row type: {type(row).__name__}"
            ) from exc
    if len(values) < 2:
        raise DataSourceSchemaError("SQL schema row must contain a name and data type.")
    return str(values[0]), str(values[1])


# =============================================================================
# Connection Pool
# =============================================================================


class SQLConnectionPool:
    """Thread-safe connection pool for SQL databases.

    This pool manages database connections, reusing them across
    operations for better performance.

    Example:
        >>> pool = SQLConnectionPool(create_connection, size=5)
        >>> with pool.acquire() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM users")
    """

    def __init__(
        self,
        connection_factory: callable,
        size: int = 5,
        timeout: float = 30.0,
    ) -> None:
        """Initialize connection pool.

        Args:
            connection_factory: Callable that creates a new connection.
            size: Maximum number of connections.
            timeout: Timeout for acquiring a connection.
        """
        self._factory = connection_factory
        self._size = size
        self._timeout = timeout
        self._pool: Queue = Queue(maxsize=size)
        self._lock = Lock()
        self._created = 0
        self._closed = False

    def _create_connection(self) -> Any:
        """Create a new connection."""
        with self._lock:
            if self._created < self._size:
                conn = self._factory()
                self._created += 1
                return conn
        return None

    @contextmanager
    def acquire(self) -> Iterator[Any]:
        """Acquire a connection from the pool.

        Yields:
            Database connection.

        Raises:
            DataSourceConnectionError: If unable to acquire connection.
        """
        if self._closed:
            raise DataSourceConnectionError("pool", "Connection pool is closed")

        conn = None

        # Try to get from pool first
        try:
            conn = self._pool.get_nowait()
        except Empty:
            # Try to create new connection
            conn = self._create_connection()

            if conn is None:
                # Pool is full, wait for available connection
                try:
                    conn = self._pool.get(timeout=self._timeout)
                except Empty:
                    raise DataSourceConnectionError(
                        "pool",
                        f"Timeout waiting for connection after {self._timeout}s"
                    ) from None

        try:
            yield conn
        finally:
            # Return connection to pool
            if conn is not None and not self._closed:
                try:
                    self._pool.put_nowait(conn)
                except Exception:
                    # Pool is full or closed, close the connection
                    with suppress(Exception):
                        conn.close()

    def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
                with suppress(Exception):
                    conn.close()
            except Empty:
                break

    @property
    def size(self) -> int:
        """Get pool size."""
        return self._size

    @property
    def available(self) -> int:
        """Get number of available connections."""
        return self._pool.qsize()


# =============================================================================
# Abstract Base SQL Data Source
# =============================================================================


class BaseSQLDataSource(BaseDataSource[SQLDataSourceConfig]):
    """Abstract base class for SQL-based data sources.

    This class provides common functionality for all SQL databases,
    including connection pooling, schema introspection, and query execution.

    Supports two modes of operation:
    - **Table mode**: Validate an existing table or view
    - **Query mode**: Validate results from a custom SQL query

    Subclasses must implement:
    - _create_connection(): Create a database connection
    - _get_table_schema_query() or _fetch_schema(): Discover columns and types
    - _get_row_count_query(): Get SQL for counting rows
    - _quote_identifier(): Quote an identifier for the provider dialect

    Example:
        >>> # Table mode
        >>> source = SQLiteDataSource(database="db.sqlite", table="users")
        >>>
        >>> # Query mode
        >>> source = SQLiteDataSource(
        ...     database="db.sqlite",
        ...     query="SELECT id, name FROM users WHERE active = 1"
        ... )
    """

    source_type = "sql"
    materialization_dialect = "limit"
    subquery_alias_keyword = "AS"

    def __init__(
        self,
        table: str | None = None,
        query: str | None = None,
        config: SQLDataSourceConfig | None = None,
    ) -> None:
        """Initialize SQL data source.

        Args:
            table: Table or view name to validate. Mutually exclusive with query.
            query: Custom SQL query to validate. Mutually exclusive with table.
            config: Optional configuration.

        Raises:
            ValueError: If neither or both table and query are provided.
        """
        super().__init__(config)
        self._validate_schema_discovery_strategy()

        # Validate mutually exclusive parameters
        if table is None and query is None:
            raise ValueError("Either 'table' or 'query' must be provided")
        if table is not None and query is not None:
            raise ValueError("'table' and 'query' are mutually exclusive; provide only one")

        self._table = table
        self._query = query
        self._is_query_mode = query is not None
        self._pool: SQLConnectionPool | None = None
        self._db_schema: list[tuple[str, str]] | None = None

    @classmethod
    def _default_config(cls) -> SQLDataSourceConfig:
        return SQLDataSourceConfig()

    def _validate_schema_discovery_strategy(self) -> None:
        """Require either query-based or provider-native schema discovery."""

        provider_class = type(self)
        has_query_strategy = (
            provider_class._get_table_schema_query
            is not BaseSQLDataSource._get_table_schema_query
        )
        has_native_strategy = (
            provider_class._fetch_schema is not BaseSQLDataSource._fetch_schema
        )
        if not (has_query_strategy or has_native_strategy):
            raise TypeError(
                f"{provider_class.__name__} must implement a schema discovery strategy: "
                "override _get_table_schema_query() or _fetch_schema()."
            )

    @property
    def table_name(self) -> str | None:
        """Get the table name (None if in query mode)."""
        return self._table

    @property
    def query_sql(self) -> str | None:
        """Get the custom SQL query (None if in table mode)."""
        return self._query

    def _clean_query_sql(self) -> str:
        """Return a custom query safe to embed as a subquery."""

        return str(self._query or "").strip().rstrip(";")

    def _alias_subquery(self, query: str, alias: str) -> str:
        """Wrap and alias a query using the provider's SQL dialect."""

        keyword = f" {self.subquery_alias_keyword}" if self.subquery_alias_keyword else ""
        return f"({query}){keyword} {alias}"

    @property
    def is_query_mode(self) -> bool:
        """Check if data source is in query mode."""
        return self._is_query_mode

    @property
    def name(self) -> str:
        """Get the data source name."""
        if self._config.name:
            return self._config.name
        if self._is_query_mode:
            # Truncate query for display
            query_preview = self._query[:50] + "..." if len(self._query) > 50 else self._query
            return f"{self.source_type}:query({query_preview})"
        return f"{self.source_type}:{self._table}"

    @property
    def full_table_name(self) -> str:
        """Get the fully qualified table name or subquery expression.

        For query mode, returns a subquery wrapped in parentheses with alias.
        """
        if self._is_query_mode:
            # Wrap query as subquery with alias for use in FROM clauses
            return self._alias_subquery(self._clean_query_sql(), "_query_result")
        if self._config.schema_name:
            return f"{self._config.schema_name}.{self._table}"
        return self._table

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def _create_connection(self) -> Any:
        """Create a new database connection.

        Returns:
            Database connection object.
        """
        pass

    def _get_table_schema_query(self) -> str:
        """Get SQL used by the common query-based schema discovery strategy.

        Providers with a native metadata API may override ``_fetch_schema``
        instead. Construction validates that at least one strategy is present.

        Returns:
            SQL query that returns (column_name, data_type) rows.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_row_count_query(self) -> str:
        """Get SQL query to count rows.

        Returns:
            SQL query that returns a single count value.
        """
        pass

    @abstractmethod
    def _quote_identifier(self, identifier: str) -> str:
        """Quote a SQL identifier (table/column name).

        Args:
            identifier: The identifier to quote.

        Returns:
            Quoted identifier.
        """
        pass

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def _connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is None:
            self._pool = SQLConnectionPool(
                connection_factory=self._create_connection,
                size=self._config.pool_size,
                timeout=self._config.pool_timeout,
            )
        self._is_connected = True

    def _disconnect(self) -> None:
        """Close connection pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._is_connected = False

    @contextmanager
    def _get_connection(self) -> Iterator[Any]:
        """Get a connection from the pool."""
        if self._pool is None:
            self._connect()

        with self._pool.acquire() as conn:
            yield conn

    # -------------------------------------------------------------------------
    # Schema Operations
    # -------------------------------------------------------------------------

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from database.

        In table mode, uses the table schema query.
        In query mode, infers schema from query result metadata.
        """
        if self._is_query_mode:
            return self._fetch_schema_from_query()

        return self._fetch_schema_query_rows(self._get_table_schema_query())

    def _fetch_schema_query_rows(self, query: str) -> list[tuple[str, str]]:
        """Execute a schema query with shared row normalization and cleanup."""

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                return [_schema_pair(row) for row in cursor.fetchall()]
            finally:
                cursor.close()

    def _fetch_schema_from_query(self) -> list[tuple[str, str]]:
        """Infer schema from query result metadata.

        Executes the query with LIMIT 0 to get column info without data.
        """
        # Use LIMIT 0 to get metadata without fetching actual data
        # Wrap in subquery to ensure LIMIT works with any query
        schema_source = self._alias_subquery(
            self._clean_query_sql(),
            "_schema_check",
        )
        schema_query = f"SELECT * FROM {schema_source} LIMIT 0"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(schema_query)
            except Exception:
                # Fallback: try direct query (some DBs don't support LIMIT 0 well)
                cursor.execute(self._clean_query_sql())

            # Get column names and types from cursor description
            if cursor.description is None:
                cursor.close()
                return []

            result = []
            for desc in cursor.description:
                col_name = desc[0]
                # Type info varies by database driver
                # desc[1] is type_code in DB-API 2.0
                col_type = self._get_type_name_from_description(desc)
                result.append((col_name, col_type))

            cursor.close()
            return result

    def _get_type_name_from_description(self, desc: tuple) -> str:
        """Convert cursor description to type name.

        Override in subclasses for database-specific type mapping.

        Args:
            desc: Cursor description tuple (name, type_code, ...).

        Returns:
            SQL type name string.
        """
        # Default implementation - return type_code as string
        # Subclasses should override for proper type mapping
        type_code = desc[1] if len(desc) > 1 else None
        if type_code is None:
            return "UNKNOWN"
        return str(type_code)

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Get the schema as column name to type mapping."""
        if self._cached_schema is None:
            if self._db_schema is None:
                self._db_schema = self._fetch_schema()

            self._cached_schema = {
                col_name: sql_type_to_column_type(col_type)
                for col_name, col_type in self._db_schema
            }
        return self._cached_schema

    @property
    def sql_schema(self) -> dict[str, str]:
        """Get the native SQL schema (column -> SQL type)."""
        if self._db_schema is None:
            self._db_schema = self._fetch_schema()
        return {col_name: col_type for col_name, col_type in self._db_schema}

    # -------------------------------------------------------------------------
    # Row Count
    # -------------------------------------------------------------------------

    @property
    def row_count(self) -> int | None:
        """Get row count from database."""
        if self._cached_row_count is None:
            value = self.execute_scalar(self._get_effective_row_count_query())
            try:
                self._cached_row_count = int(value or 0)
            except (TypeError, ValueError) as exc:
                raise DataSourceError(
                    f"Invalid row count returned by {self.source_type}: {type(value).__name__}"
                ) from exc
        return self._cached_row_count

    def _get_effective_row_count_query(self) -> str:
        """Get the row count query for current mode.

        In query mode, wraps the custom query to count its results.
        """
        if self._is_query_mode:
            source = self._alias_subquery(self._clean_query_sql(), "_count_query")
            return f"SELECT COUNT(*) FROM {source}"
        return self._get_row_count_query()

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def execute_query(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results.

        Args:
            query: SQL query to execute.
            params: Optional query parameters.

        Returns:
            List of dictionaries with column names as keys.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params is not None:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                columns = [str(desc[0]) for desc in (cursor.description or ())]
                rows = cursor.fetchall()
                return [_row_mapping(row, columns) for row in rows]
            finally:
                cursor.close()

    def execute_scalar(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> Any:
        """Execute a query and return a single value.

        Args:
            query: SQL query to execute.
            params: Optional query parameters.

        Returns:
            Single value from the first row/column.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params is not None:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return _scalar_value(cursor.fetchone())
            finally:
                cursor.close()

    # -------------------------------------------------------------------------
    # Data Source Interface
    # -------------------------------------------------------------------------

    @property
    def capabilities(self) -> set[DataSourceCapability]:
        """Get data source capabilities."""
        return {
            DataSourceCapability.SQL_PUSHDOWN,
            DataSourceCapability.SAMPLING,
            DataSourceCapability.SCHEMA_INFERENCE,
            DataSourceCapability.ROW_COUNT,
        }

    def get_execution_engine(self) -> BaseExecutionEngine:
        """Get a SQL execution engine."""
        from truthound.execution.sql_engine import SQLExecutionEngine
        return SQLExecutionEngine(self)

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> BaseSQLDataSource:
        """Create a sampled view of the data.

        Note: Most SQL databases don't support seeded sampling,
        so the seed parameter may be ignored.
        """
        # Return a wrapper that limits queries
        return SampledSQLDataSource(self, n, seed)

    def _materialization_limit(self) -> int:
        limit = min(
            int(self._config.max_rows),
            int(self._config.materialization_row_limit),
        )
        if limit <= 0:
            raise DataSourceError("materialization_row_limit must be greater than zero")
        return limit

    def _materialization_source(self) -> str:
        if self._is_query_mode:
            return self._alias_subquery(
                self._clean_query_sql(),
                "_truthound_materialized_source",
            )
        return self.full_table_name

    def _build_bounded_materialization_query(self, limit: int) -> str:
        """Build a provider-safe query that can return at most ``limit`` rows."""

        source = self._materialization_source()
        if self.materialization_dialect == "top":
            return f"SELECT TOP ({limit}) * FROM {source}"
        if self.materialization_dialect == "rownum":
            return f"SELECT * FROM {source} WHERE ROWNUM <= {limit}"
        return f"SELECT * FROM {source} LIMIT {limit}"

    def _fetch_bounded_rows(
        self,
        query: str,
        *,
        max_rows: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Fetch no more than ``max_rows`` using DB-API batches."""

        batch_size = max(1, min(int(self._config.fetch_size), max_rows))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                columns = [str(desc[0]) for desc in (cursor.description or ())]
                rows: list[dict[str, Any]] = []
                while len(rows) < max_rows:
                    requested = min(batch_size, max_rows - len(rows))
                    batch = cursor.fetchmany(requested)
                    if not batch:
                        break
                    rows.extend(_row_mapping(row, columns) for row in batch)
                return columns, rows
            finally:
                cursor.close()

    def to_polars_lazyframe(self) -> pl.LazyFrame:
        """Convert a safely bounded SQL result to a Polars LazyFrame.

        Exact SQL operations continue to use pushdown. Validators that require
        an in-memory Polars fallback must fit within
        ``materialization_row_limit`` or use an explicit sampled data source.
        """
        import polars as pl

        self.check_size_limits()
        limit = self._materialization_limit()
        row_count = self.row_count
        if row_count is not None and row_count > limit:
            raise DataSourceSizeError(row_count, limit, "rows for SQL materialization")

        query = self._build_bounded_materialization_query(limit + 1)
        columns, rows = self._fetch_bounded_rows(query, max_rows=limit + 1)
        if len(rows) > limit:
            raise DataSourceSizeError(len(rows), limit, "rows for SQL materialization")
        if rows:
            return pl.DataFrame(rows).lazy()
        return pl.DataFrame({column: [] for column in columns or self.columns}).lazy()

    def validate_connection(self) -> bool:
        """Validate database connection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # SQL Query Builders
    # -------------------------------------------------------------------------

    def build_count_query(self, condition: str | None = None) -> str:
        """Build a COUNT query.

        Args:
            condition: Optional WHERE condition.

        Returns:
            SQL query string.
        """
        query = f"SELECT COUNT(*) FROM {self.full_table_name}"
        if condition:
            query += f" WHERE {condition}"
        return query

    def build_distinct_count_query(self, column: str) -> str:
        """Build a COUNT DISTINCT query."""
        col = self._quote_identifier(column)
        return f"SELECT COUNT(DISTINCT {col}) FROM {self.full_table_name}"

    def build_null_count_query(self, column: str) -> str:
        """Build a NULL count query."""
        col = self._quote_identifier(column)
        return f"SELECT COUNT(*) FROM {self.full_table_name} WHERE {col} IS NULL"

    def build_stats_query(self, column: str) -> str:
        """Build a statistics query for a numeric column."""
        col = self._quote_identifier(column)
        return f"""
            SELECT
                COUNT({col}) as count,
                COUNT(*) - COUNT({col}) as null_count,
                AVG({col}) as mean,
                MIN({col}) as min,
                MAX({col}) as max,
                SUM({col}) as sum
            FROM {self.full_table_name}
        """


# =============================================================================
# Sampled SQL Data Source
# =============================================================================


class SampledSQLDataSource(BaseSQLDataSource):
    """A sampled view of a SQL data source.

    This wraps another SQL data source and limits query results.
    """

    source_type = "sql_sampled"

    def __init__(
        self,
        parent: BaseSQLDataSource,
        sample_size: int,
        seed: int | None = None,
    ) -> None:
        """Initialize sampled SQL data source.

        Args:
            parent: Parent SQL data source.
            sample_size: Maximum number of rows.
            seed: Random seed (may be ignored by database).
        """
        super().__init__(
            table=parent.table_name or "_truthound_sampled_query",
            config=parent.config,
        )
        self._parent = parent
        self._sample_size = sample_size
        self._seed = seed
        self.materialization_dialect = parent.materialization_dialect

    def _create_connection(self) -> Any:
        return self._parent._create_connection()

    def _get_table_schema_query(self) -> str:
        return self._parent._get_table_schema_query()

    def _fetch_schema(self) -> list[tuple[str, str]]:
        return list(self._parent.sql_schema.items())

    def _get_row_count_query(self) -> str:
        # Return sample size as row count
        return f"SELECT LEAST({self._sample_size}, ({self._parent._get_row_count_query()}))"

    def _quote_identifier(self, identifier: str) -> str:
        return self._parent._quote_identifier(identifier)

    @property
    def full_table_name(self) -> str:
        """Get sampled table expression."""
        query = self._parent._build_bounded_materialization_query(self._sample_size)
        return self._parent._alias_subquery(query, "sampled")

    @property
    def row_count(self) -> int | None:
        """Get sample row count."""
        parent_count = self._parent.row_count
        if parent_count is None:
            return self._sample_size
        return min(parent_count, self._sample_size)

    def sample(self, n: int = 1000, seed: int | None = None) -> SampledSQLDataSource:
        """Create a smaller sample."""
        new_size = min(n, self._sample_size)
        return SampledSQLDataSource(self._parent, new_size, seed)

    def execute_query(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> list[dict[str, Any]]:
        if params is None:
            return self._parent.execute_query(query)
        return self._parent.execute_query(query, params)

    def execute_scalar(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> Any:
        if params is None:
            return self._parent.execute_scalar(query)
        return self._parent.execute_scalar(query, params)

    def _fetch_bounded_rows(
        self,
        query: str,
        *,
        max_rows: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        return self._parent._fetch_bounded_rows(query, max_rows=max_rows)
