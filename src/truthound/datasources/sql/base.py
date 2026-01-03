"""Base classes for SQL data sources.

This module provides the abstract base class for SQL-based data sources,
with connection pooling and common SQL operations.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterator

from truthound.datasources._protocols import (
    ColumnType,
    DataSourceCapability,
)
from truthound.datasources.base import (
    BaseDataSource,
    DataSourceConfig,
    DataSourceConnectionError,
    DataSourceError,
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
        use_server_side_cursor: Use server-side cursor for large results.
        schema_name: Database schema name (for PostgreSQL, etc.).
    """

    pool_size: int = 5
    pool_timeout: float = 30.0
    query_timeout: float = 300.0
    fetch_size: int = 10000
    use_server_side_cursor: bool = False
    schema_name: str | None = None


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
                    )

        try:
            yield conn
        finally:
            # Return connection to pool
            if conn is not None and not self._closed:
                try:
                    self._pool.put_nowait(conn)
                except Exception:
                    # Pool is full or closed, close the connection
                    try:
                        conn.close()
                    except Exception:
                        pass

    def close(self) -> None:
        """Close all connections in the pool."""
        self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
                try:
                    conn.close()
                except Exception:
                    pass
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
    - _get_table_schema(): Get column names and types from database
    - _get_row_count_query(): Get SQL for counting rows

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

    @property
    def table_name(self) -> str | None:
        """Get the table name (None if in query mode)."""
        return self._table

    @property
    def query_sql(self) -> str | None:
        """Get the custom SQL query (None if in table mode)."""
        return self._query

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
            return f"({self._query}) AS _query_result"
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

    @abstractmethod
    def _get_table_schema_query(self) -> str:
        """Get SQL query to retrieve table schema.

        Returns:
            SQL query that returns (column_name, data_type) rows.
        """
        pass

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

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._get_table_schema_query())
            result = cursor.fetchall()
            cursor.close()
            return result

    def _fetch_schema_from_query(self) -> list[tuple[str, str]]:
        """Infer schema from query result metadata.

        Executes the query with LIMIT 0 to get column info without data.
        """
        # Use LIMIT 0 to get metadata without fetching actual data
        # Wrap in subquery to ensure LIMIT works with any query
        schema_query = f"SELECT * FROM ({self._query}) AS _schema_check LIMIT 0"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(schema_query)
            except Exception:
                # Fallback: try direct query (some DBs don't support LIMIT 0 well)
                cursor.execute(self._query)

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
            with self._get_connection() as conn:
                cursor = conn.cursor()
                query = self._get_effective_row_count_query()
                cursor.execute(query)
                result = cursor.fetchone()
                cursor.close()
                self._cached_row_count = result[0] if result else 0
        return self._cached_row_count

    def _get_effective_row_count_query(self) -> str:
        """Get the row count query for current mode.

        In query mode, wraps the custom query to count its results.
        """
        if self._is_query_mode:
            return f"SELECT COUNT(*) FROM ({self._query}) AS _count_query"
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
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            return [dict(zip(columns, row)) for row in rows]

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
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None

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

    def get_execution_engine(self) -> "BaseExecutionEngine":
        """Get a SQL execution engine."""
        from truthound.execution.sql_engine import SQLExecutionEngine
        return SQLExecutionEngine(self)

    def sample(
        self,
        n: int = 1000,
        seed: int | None = None,
    ) -> "BaseSQLDataSource":
        """Create a sampled view of the data.

        Note: Most SQL databases don't support seeded sampling,
        so the seed parameter may be ignored.
        """
        # Return a wrapper that limits queries
        return SampledSQLDataSource(self, n, seed)

    def to_polars_lazyframe(self) -> "pl.LazyFrame":
        """Convert to Polars LazyFrame by fetching all data.

        Warning: This loads all data into memory.

        In query mode, executes the custom query directly.
        In table mode, fetches all rows from the table.
        """
        import polars as pl

        # Check size limits
        self.check_size_limits()

        # Fetch all data
        if self._is_query_mode:
            # In query mode, execute the custom query directly
            query = self._query
        else:
            query = f"SELECT * FROM {self.full_table_name}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

        # Convert to Polars
        data = {col: [row[i] for row in rows] for i, col in enumerate(columns)}
        return pl.DataFrame(data).lazy()

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
            table=parent.table_name,
            config=parent.config,
        )
        self._parent = parent
        self._sample_size = sample_size
        self._seed = seed

    def _create_connection(self) -> Any:
        return self._parent._create_connection()

    def _get_table_schema_query(self) -> str:
        return self._parent._get_table_schema_query()

    def _get_row_count_query(self) -> str:
        # Return sample size as row count
        return f"SELECT LEAST({self._sample_size}, ({self._parent._get_row_count_query()}))"

    def _quote_identifier(self, identifier: str) -> str:
        return self._parent._quote_identifier(identifier)

    @property
    def full_table_name(self) -> str:
        """Get sampled table expression."""
        # Subquery with LIMIT
        return f"(SELECT * FROM {self._parent.full_table_name} LIMIT {self._sample_size}) AS sampled"

    @property
    def row_count(self) -> int | None:
        """Get sample row count."""
        parent_count = self._parent.row_count
        if parent_count is None:
            return self._sample_size
        return min(parent_count, self._sample_size)

    def sample(self, n: int = 1000, seed: int | None = None) -> "SampledSQLDataSource":
        """Create a smaller sample."""
        new_size = min(n, self._sample_size)
        return SampledSQLDataSource(self._parent, new_size, seed)
