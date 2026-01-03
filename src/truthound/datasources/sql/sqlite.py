"""SQLite data source implementation.

This module provides a data source for SQLite databases.
SQLite is included in Python's standard library, so no additional
dependencies are required.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from truthound.datasources.sql.base import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
)
from truthound.datasources.base import DataSourceError


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SQLiteDataSourceConfig(SQLDataSourceConfig):
    """Configuration for SQLite data sources.

    Attributes:
        database: Path to SQLite database file, or ":memory:".
        timeout: Connection timeout in seconds.
        detect_types: SQLite type detection flags.
        isolation_level: Transaction isolation level.
    """

    database: str = ":memory:"
    timeout: float = 5.0
    detect_types: int = sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    isolation_level: str | None = None


# =============================================================================
# SQLite Data Source
# =============================================================================


class SQLiteDataSource(BaseSQLDataSource):
    """Data source for SQLite databases.

    SQLite is a file-based database that requires no server setup.
    It's great for testing and smaller datasets.

    Supports two modes:
    - **Table mode**: Validate an existing table
    - **Query mode**: Validate results from a custom SQL query

    Example:
        >>> # Table mode
        >>> source = SQLiteDataSource(
        ...     table="users",
        ...     database="data.db",
        ... )
        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())

        >>> # Query mode
        >>> source = SQLiteDataSource(
        ...     database="data.db",
        ...     query="SELECT id, name FROM users WHERE active = 1",
        ... )
        >>> lf = source.to_polars_lazyframe()

        >>> # In-memory database
        >>> source = SQLiteDataSource(
        ...     table="test_table",
        ...     database=":memory:",
        ... )
    """

    source_type = "sqlite"

    def __init__(
        self,
        table: str | None = None,
        database: str = ":memory:",
        query: str | None = None,
        config: SQLiteDataSourceConfig | None = None,
    ) -> None:
        """Initialize SQLite data source.

        Args:
            table: Table name to validate. Mutually exclusive with query.
            database: Path to database file or ":memory:".
            query: Custom SQL query to validate. Mutually exclusive with table.
            config: Optional configuration.

        Raises:
            ValueError: If neither or both table and query are provided.
            DataSourceError: If database file does not exist.
        """
        if config is None:
            config = SQLiteDataSourceConfig(database=database)
        else:
            config.database = database

        self._database = database

        # Validate database exists (unless in-memory)
        if database != ":memory:":
            db_path = Path(database)
            if not db_path.exists():
                raise DataSourceError(f"Database file not found: {database}")

        super().__init__(table=table, query=query, config=config)

    @classmethod
    def _default_config(cls) -> SQLiteDataSourceConfig:
        return SQLiteDataSourceConfig()

    @property
    def database(self) -> str:
        """Get the database path."""
        return self._database

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection."""
        cfg: SQLiteDataSourceConfig = self._config  # type: ignore

        conn = sqlite3.connect(
            cfg.database,
            timeout=cfg.timeout,
            detect_types=cfg.detect_types,
            isolation_level=cfg.isolation_level,
        )

        # Enable row factory for dict-like access
        conn.row_factory = sqlite3.Row

        return conn

    def _get_table_schema_query(self) -> str:
        """Get SQLite schema query using PRAGMA."""
        # SQLite uses PRAGMA table_info
        if self._table is None:
            raise ValueError("Cannot get table schema in query mode")
        return f"PRAGMA table_info({self._quote_identifier(self._table)})"

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from SQLite database.

        In query mode, uses the base class method to infer from query results.
        In table mode, uses PRAGMA table_info.

        SQLite's PRAGMA table_info returns:
        (cid, name, type, notnull, dflt_value, pk)
        """
        if self._is_query_mode:
            return self._fetch_schema_from_query()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._get_table_schema_query())
            result = cursor.fetchall()
            cursor.close()

            # Extract (name, type) from PRAGMA result
            return [(row[1], row[2]) for row in result]

    def _get_type_name_from_description(self, desc: tuple) -> str:
        """Convert SQLite cursor description to type name.

        SQLite type affinity: TEXT, NUMERIC, INTEGER, REAL, BLOB
        """
        # SQLite cursor description doesn't include type info reliably
        # We need to infer from sample data or return a default
        # desc format: (name, type_code, display_size, internal_size, precision, scale, null_ok)
        type_code = desc[1] if len(desc) > 1 else None

        # SQLite type_code is often None, so we return a generic type
        if type_code is None:
            return "TEXT"

        # Map Python type codes to SQL types
        type_map = {
            str: "TEXT",
            int: "INTEGER",
            float: "REAL",
            bytes: "BLOB",
            type(None): "NULL",
        }
        return type_map.get(type_code, "TEXT")

    def _get_row_count_query(self) -> str:
        """Get SQLite row count query for table mode."""
        if self._table is None:
            raise ValueError("Cannot get row count query in query mode")
        return f"SELECT COUNT(*) FROM {self._quote_identifier(self._table)}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote SQLite identifier with double quotes."""
        # SQLite uses double quotes for identifiers
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    # -------------------------------------------------------------------------
    # SQLite-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> list[dict[str, Any]]:
        """Get detailed table information from PRAGMA."""
        query = f"PRAGMA table_info({self._quote_identifier(self._table)})"
        return self.execute_query(query)

    def get_index_info(self) -> list[dict[str, Any]]:
        """Get index information for the table."""
        query = f"PRAGMA index_list({self._quote_identifier(self._table)})"
        return self.execute_query(query)

    def get_foreign_keys(self) -> list[dict[str, Any]]:
        """Get foreign key information for the table."""
        query = f"PRAGMA foreign_key_list({self._quote_identifier(self._table)})"
        return self.execute_query(query)

    def vacuum(self) -> None:
        """Run VACUUM to optimize database file."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")

    def analyze(self) -> None:
        """Run ANALYZE to update table statistics."""
        with self._get_connection() as conn:
            conn.execute(f"ANALYZE {self._quote_identifier(self._table)}")

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        table: str,
        database: str | None = None,
    ) -> "SQLiteDataSource":
        """Create SQLite data source from a DataFrame.

        This is useful for testing SQL operations on in-memory data.

        Args:
            df: Pandas or Polars DataFrame.
            table: Table name to create.
            database: Database path. If None, creates a temporary file.

        Returns:
            SQLiteDataSource with data loaded.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            >>> source = SQLiteDataSource.from_dataframe(df, "test_table")
        """
        import tempfile

        # Determine DataFrame type and convert
        df_type = type(df).__module__

        if "polars" in df_type:
            import polars as pl
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            pandas_df = df.to_pandas()
        elif "pandas" in df_type:
            pandas_df = df
        else:
            raise DataSourceError(f"Unsupported DataFrame type: {type(df)}")

        # Use temporary file if no database specified (in-memory doesn't persist across connections)
        if database is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            database = temp_file.name
            temp_file.close()

        # Create database and insert data
        conn = sqlite3.connect(database)
        pandas_df.to_sql(table, conn, if_exists="replace", index=False)
        conn.close()

        return cls(table=table, database=database)
