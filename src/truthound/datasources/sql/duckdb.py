"""DuckDB data source implementation.

This module provides a data source for DuckDB databases.
DuckDB is an in-process analytical database with excellent
Polars integration.

Requires: pip install duckdb
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from truthound.datasources.sql.base import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
)
from truthound.datasources.base import DataSourceError

try:
    import duckdb
except ImportError as e:
    raise ImportError(
        "DuckDB is required for DuckDBDataSource. "
        "Install it with: pip install duckdb"
    ) from e


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DuckDBDataSourceConfig(SQLDataSourceConfig):
    """Configuration for DuckDB data sources.

    Attributes:
        database: Path to DuckDB database file, or ":memory:".
        read_only: Open database in read-only mode.
        threads: Number of threads for query execution (None = auto).
        memory_limit: Maximum memory usage (e.g., "4GB").
    """

    database: str = ":memory:"
    read_only: bool = False
    threads: int | None = None
    memory_limit: str | None = None


# =============================================================================
# DuckDB Data Source
# =============================================================================


class DuckDBDataSource(BaseSQLDataSource):
    """Data source for DuckDB databases.

    DuckDB is an in-process OLAP database that's designed for analytical
    workloads. It has excellent Polars integration and can directly read
    Parquet, CSV, and JSON files.

    Supports two modes:
    - **Table mode**: Validate an existing table
    - **Query mode**: Validate results from a custom SQL query

    Example:
        >>> # Table mode
        >>> source = DuckDBDataSource(
        ...     table="users",
        ...     database="analytics.duckdb",
        ... )
        >>> report = th.check(source=source)

        >>> # Query mode
        >>> source = DuckDBDataSource(
        ...     database="analytics.duckdb",
        ...     query="SELECT * FROM users WHERE active = true",
        ... )
        >>> lf = source.to_polars_lazyframe()

        >>> # In-memory database
        >>> source = DuckDBDataSource(
        ...     table="test_table",
        ...     database=":memory:",
        ... )

        >>> # Read directly from Parquet file
        >>> source = DuckDBDataSource(
        ...     database=":memory:",
        ...     query="SELECT * FROM 'data.parquet'",
        ... )
    """

    source_type = "duckdb"

    def __init__(
        self,
        table: str | None = None,
        database: str = ":memory:",
        query: str | None = None,
        config: DuckDBDataSourceConfig | None = None,
        read_only: bool = False,
    ) -> None:
        """Initialize DuckDB data source.

        Args:
            table: Table name to validate. Mutually exclusive with query.
            database: Path to database file or ":memory:".
            query: Custom SQL query to validate. Mutually exclusive with table.
            config: Optional configuration.
            read_only: Open database in read-only mode.

        Raises:
            ValueError: If neither or both table and query are provided.
            DataSourceError: If database file does not exist (non-memory mode).
        """
        if config is None:
            config = DuckDBDataSourceConfig(database=database, read_only=read_only)
        else:
            config.database = database
            config.read_only = read_only

        self._database = database
        self._read_only = read_only

        # Validate database exists (unless in-memory)
        if database != ":memory:" and not database.startswith(":"):
            db_path = Path(database)
            if not db_path.exists() and not read_only:
                # DuckDB will create the file if it doesn't exist (unless read_only)
                pass
            elif read_only and not db_path.exists():
                raise DataSourceError(f"Database file not found: {database}")

        super().__init__(table=table, query=query, config=config)

    @classmethod
    def _default_config(cls) -> DuckDBDataSourceConfig:
        return DuckDBDataSourceConfig()

    @property
    def database(self) -> str:
        """Get the database path."""
        return self._database

    def _create_connection(self) -> duckdb.DuckDBPyConnection:
        """Create a new DuckDB connection."""
        cfg: DuckDBDataSourceConfig = self._config  # type: ignore

        conn = duckdb.connect(
            database=cfg.database,
            read_only=cfg.read_only,
        )

        # Configure connection settings
        if cfg.threads is not None:
            conn.execute(f"SET threads = {cfg.threads}")
        if cfg.memory_limit is not None:
            conn.execute(f"SET memory_limit = '{cfg.memory_limit}'")

        return conn

    def _get_table_schema_query(self) -> str:
        """Get DuckDB schema query."""
        if self._table is None:
            raise ValueError("Cannot get table schema in query mode")
        return f"DESCRIBE {self._quote_identifier(self._table)}"

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from DuckDB database.

        DuckDB's DESCRIBE returns:
        (column_name, column_type, null, key, default, extra)
        """
        if self._is_query_mode:
            return self._fetch_schema_from_query()

        with self._get_connection() as conn:
            result = conn.execute(self._get_table_schema_query()).fetchall()
            # Extract (name, type) from DESCRIBE result
            return [(row[0], row[1]) for row in result]

    def _get_type_name_from_description(self, desc: tuple) -> str:
        """Convert DuckDB cursor description to type name."""
        # desc format: (name, type_code, display_size, internal_size, precision, scale, null_ok)
        type_code = desc[1] if len(desc) > 1 else None

        if type_code is None:
            return "VARCHAR"

        # DuckDB type codes are type objects
        type_name = str(type_code).upper()
        return type_name

    def _get_row_count_query(self) -> str:
        """Get DuckDB row count query for table mode."""
        if self._table is None:
            raise ValueError("Cannot get row count query in query mode")
        return f"SELECT COUNT(*) FROM {self._quote_identifier(self._table)}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote DuckDB identifier with double quotes."""
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    # -------------------------------------------------------------------------
    # DuckDB-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> list[dict[str, Any]]:
        """Get detailed table information."""
        query = f"DESCRIBE {self._quote_identifier(self._table)}"
        return self.execute_query(query)

    def get_tables(self) -> list[str]:
        """Get list of all tables in the database."""
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        result = self.execute_query(query)
        return [row["table_name"] for row in result]

    def explain(self, analyze: bool = False) -> str:
        """Get query execution plan.

        Args:
            analyze: If True, actually execute and show timing.

        Returns:
            Query plan as string.
        """
        if self._is_query_mode:
            query = self._query
        else:
            query = f"SELECT * FROM {self.full_table_name}"
        explain_cmd = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
        with self._get_connection() as conn:
            result = conn.execute(f"{explain_cmd} {query}").fetchall()
            return "\n".join(str(row[0]) for row in result)

    @classmethod
    def from_parquet(
        cls,
        path: str,
        database: str = ":memory:",
    ) -> "DuckDBDataSource":
        """Create DuckDB data source from a Parquet file.

        DuckDB can directly query Parquet files without loading them.

        Args:
            path: Path to Parquet file or glob pattern.
            database: Database path (default: in-memory).

        Returns:
            DuckDBDataSource querying the Parquet file.

        Example:
            >>> source = DuckDBDataSource.from_parquet("data/*.parquet")
            >>> report = th.check(source=source)
        """
        return cls(
            database=database,
            query=f"SELECT * FROM '{path}'",
        )

    @classmethod
    def from_csv(
        cls,
        path: str,
        database: str = ":memory:",
        **read_options: Any,
    ) -> "DuckDBDataSource":
        """Create DuckDB data source from a CSV file.

        Args:
            path: Path to CSV file or glob pattern.
            database: Database path (default: in-memory).
            **read_options: Additional read_csv options (header, delimiter, etc.)

        Returns:
            DuckDBDataSource querying the CSV file.

        Example:
            >>> source = DuckDBDataSource.from_csv("data.csv", header=True)
        """
        options = ", ".join(f"{k}={v!r}" for k, v in read_options.items())
        if options:
            query = f"SELECT * FROM read_csv('{path}', {options})"
        else:
            query = f"SELECT * FROM read_csv_auto('{path}')"

        return cls(database=database, query=query)

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        table: str,
        database: str = ":memory:",
    ) -> "DuckDBDataSource":
        """Create DuckDB data source from a DataFrame.

        DuckDB has native support for Polars and Pandas DataFrames.

        Args:
            df: Polars or Pandas DataFrame.
            table: Table name to create.
            database: Database path (default: in-memory).

        Returns:
            DuckDBDataSource with data loaded.

        Example:
            >>> import polars as pl
            >>> df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            >>> source = DuckDBDataSource.from_dataframe(df, "test_table")
        """
        import tempfile

        df_type = type(df).__module__

        # For in-memory database, we need to keep the connection alive
        # So we use a temporary file instead
        if database == ":memory:":
            temp_dir = tempfile.mkdtemp()
            database = str(Path(temp_dir) / "temp.duckdb")

        conn = duckdb.connect(database)

        if "polars" in df_type:
            import polars as pl
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            # DuckDB can directly register Polars DataFrames
            conn.register("temp_df", df)
            conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM temp_df')
            conn.unregister("temp_df")
        elif "pandas" in df_type:
            conn.register("temp_df", df)
            conn.execute(f'CREATE TABLE "{table}" AS SELECT * FROM temp_df')
            conn.unregister("temp_df")
        else:
            conn.close()
            raise DataSourceError(f"Unsupported DataFrame type: {type(df)}")

        conn.close()

        return cls(table=table, database=database)
