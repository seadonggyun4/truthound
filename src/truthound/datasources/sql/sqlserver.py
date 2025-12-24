"""Microsoft SQL Server data source implementation.

This module provides a data source for Microsoft SQL Server,
supporting both SQL Server Authentication and Windows Authentication.

Requires: pip install pyodbc or pymssql
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from truthound.datasources.sql.base import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
)
from truthound.datasources.base import (
    DataSourceConnectionError,
    DataSourceError,
)

if TYPE_CHECKING:
    pass


def _check_sqlserver_available() -> tuple[str, Any]:
    """Check if SQL Server driver is available.

    Returns:
        Tuple of (driver_name, driver_module).
    """
    # Try pyodbc first (more feature-complete)
    try:
        import pyodbc
        return "pyodbc", pyodbc
    except ImportError:
        pass

    # Fall back to pymssql
    try:
        import pymssql
        return "pymssql", pymssql
    except ImportError:
        pass

    raise ImportError(
        "pyodbc or pymssql is required for SQLServerDataSource. "
        "Install with: pip install pyodbc  OR  pip install pymssql"
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SQLServerConfig(SQLDataSourceConfig):
    """Configuration for SQL Server data source.

    Attributes:
        host: Server hostname or IP.
        port: Server port (default: 1433).
        database: Database name.
        user: Username (for SQL Server auth).
        password: Password (for SQL Server auth).
        trusted_connection: Use Windows Authentication.
        driver: ODBC driver name (for pyodbc).
        encrypt: Encrypt connection.
        trust_server_certificate: Trust self-signed certificates.
        application_name: Application name for connection.
        connection_timeout: Connection timeout in seconds.
    """

    host: str | None = None
    port: int = 1433
    database: str | None = None
    user: str | None = None
    password: str | None = None
    trusted_connection: bool = False
    driver: str = "ODBC Driver 17 for SQL Server"
    encrypt: bool = True
    trust_server_certificate: bool = False
    application_name: str = "Truthound"
    connection_timeout: int = 30


# =============================================================================
# SQL Server Data Source
# =============================================================================


class SQLServerDataSource(BaseSQLDataSource):
    """Data source for Microsoft SQL Server.

    Supports:
    - SQL Server Authentication (username/password)
    - Windows Authentication (trusted connection)
    - Azure SQL Database
    - Both pyodbc and pymssql drivers

    Example:
        >>> # SQL Server Authentication
        >>> source = SQLServerDataSource(
        ...     table="Users",
        ...     host="sqlserver.example.com",
        ...     database="MyDB",
        ...     user="sa",
        ...     password="password",
        ... )

        >>> # Windows Authentication
        >>> source = SQLServerDataSource(
        ...     table="Users",
        ...     host="sqlserver.example.com",
        ...     database="MyDB",
        ...     trusted_connection=True,
        ... )

        >>> # Azure SQL Database
        >>> source = SQLServerDataSource(
        ...     table="Users",
        ...     host="myserver.database.windows.net",
        ...     database="MyDB",
        ...     user="admin@myserver",
        ...     password="password",
        ...     encrypt=True,
        ... )

        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
    """

    source_type = "sqlserver"

    def __init__(
        self,
        table: str,
        host: str,
        database: str,
        user: str | None = None,
        password: str | None = None,
        port: int = 1433,
        schema: str = "dbo",
        trusted_connection: bool = False,
        encrypt: bool = True,
        config: SQLServerConfig | None = None,
    ) -> None:
        """Initialize SQL Server data source.

        Args:
            table: Table name.
            host: Server hostname.
            database: Database name.
            user: Username.
            password: Password.
            port: Server port.
            schema: Schema name (default: dbo).
            trusted_connection: Use Windows Authentication.
            encrypt: Encrypt connection.
            config: Optional configuration.
        """
        self._driver_name, self._driver = _check_sqlserver_available()

        if config is None:
            config = SQLServerConfig()

        config.host = host
        config.port = port
        config.database = database
        config.user = user
        config.password = password
        config.schema_name = schema
        config.trusted_connection = trusted_connection
        config.encrypt = encrypt
        config.name = config.name or f"{host}/{database}.{schema}.{table}"

        self._schema = schema

        super().__init__(table=table, config=config)

    @classmethod
    def _default_config(cls) -> SQLServerConfig:
        return SQLServerConfig()

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f"[{self._config.database}].[{self._schema}].[{self._table}]"

    def _create_connection(self) -> Any:
        """Create SQL Server connection."""
        if self._driver_name == "pyodbc":
            return self._create_pyodbc_connection()
        else:
            return self._create_pymssql_connection()

    def _create_pyodbc_connection(self) -> Any:
        """Create connection using pyodbc."""
        import pyodbc

        conn_str_parts = [
            f"DRIVER={{{self._config.driver}}}",
            f"SERVER={self._config.host},{self._config.port}",
            f"DATABASE={self._config.database}",
        ]

        if self._config.trusted_connection:
            conn_str_parts.append("Trusted_Connection=yes")
        else:
            conn_str_parts.append(f"UID={self._config.user}")
            conn_str_parts.append(f"PWD={self._config.password}")

        if self._config.encrypt:
            conn_str_parts.append("Encrypt=yes")
            if self._config.trust_server_certificate:
                conn_str_parts.append("TrustServerCertificate=yes")

        conn_str_parts.append(f"APP={self._config.application_name}")
        conn_str_parts.append(f"Connection Timeout={self._config.connection_timeout}")

        conn_str = ";".join(conn_str_parts)
        return pyodbc.connect(conn_str)

    def _create_pymssql_connection(self) -> Any:
        """Create connection using pymssql."""
        import pymssql

        return pymssql.connect(
            server=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            database=self._config.database,
            appname=self._config.application_name,
            login_timeout=self._config.connection_timeout,
        )

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from SQL Server."""
        query = f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self._schema}'
            AND TABLE_NAME = '{self._table}'
            ORDER BY ORDINAL_POSITION
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return [(row[0], row[1]) for row in result]

    def _get_row_count_query(self) -> str:
        """Get SQL Server row count query."""
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote SQL Server identifier with brackets."""
        escaped = identifier.replace("]", "]]")
        return f"[{escaped}]"

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute SQL Server query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return [dict(zip(columns, row)) for row in results]

    def execute_scalar(self, query: str) -> Any:
        """Execute SQL Server query returning single value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None

    def to_polars_lazyframe(self):
        """Convert SQL Server table to Polars LazyFrame."""
        import polars as pl

        query = f"SELECT * FROM {self.full_table_name}"
        if self._config.max_rows:
            query = f"SELECT TOP {self._config.max_rows} * FROM {self.full_table_name}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()

        df_dict = {col: [row[i] for row in data] for i, col in enumerate(columns)}
        return pl.DataFrame(df_dict).lazy()

    def validate_connection(self) -> bool:
        """Validate SQL Server connection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # SQL Server-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> dict[str, Any]:
        """Get detailed table information."""
        query = f"""
            SELECT
                t.TABLE_CATALOG,
                t.TABLE_SCHEMA,
                t.TABLE_NAME,
                t.TABLE_TYPE,
                p.rows AS row_count,
                SUM(a.total_pages) * 8 AS total_space_kb,
                SUM(a.used_pages) * 8 AS used_space_kb
            FROM INFORMATION_SCHEMA.TABLES t
            INNER JOIN sys.tables st ON t.TABLE_NAME = st.name
            INNER JOIN sys.partitions p ON st.object_id = p.object_id
            INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
            WHERE t.TABLE_SCHEMA = '{self._schema}'
            AND t.TABLE_NAME = '{self._table}'
            GROUP BY t.TABLE_CATALOG, t.TABLE_SCHEMA, t.TABLE_NAME, t.TABLE_TYPE, p.rows
        """
        results = self.execute_query(query)
        return results[0] if results else {}

    def get_indexes(self) -> list[dict[str, Any]]:
        """Get table indexes."""
        query = f"""
            SELECT
                i.name AS index_name,
                i.type_desc AS index_type,
                i.is_unique,
                i.is_primary_key,
                STRING_AGG(c.name, ', ') AS columns
            FROM sys.indexes i
            INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
            INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
            INNER JOIN sys.tables t ON i.object_id = t.object_id
            INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
            WHERE s.name = '{self._schema}'
            AND t.name = '{self._table}'
            GROUP BY i.name, i.type_desc, i.is_unique, i.is_primary_key
        """
        return self.execute_query(query)

    def get_constraints(self) -> list[dict[str, Any]]:
        """Get table constraints."""
        query = f"""
            SELECT
                CONSTRAINT_NAME,
                CONSTRAINT_TYPE
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
            WHERE TABLE_SCHEMA = '{self._schema}'
            AND TABLE_NAME = '{self._table}'
        """
        return self.execute_query(query)

    def get_statistics(self) -> list[dict[str, Any]]:
        """Get table statistics."""
        query = f"""
            SELECT
                s.name AS stats_name,
                STATS_DATE(s.object_id, s.stats_id) AS last_updated,
                s.auto_created,
                s.user_created
            FROM sys.stats s
            INNER JOIN sys.tables t ON s.object_id = t.object_id
            INNER JOIN sys.schemas sc ON t.schema_id = sc.schema_id
            WHERE sc.name = '{self._schema}'
            AND t.name = '{self._table}'
        """
        return self.execute_query(query)

    def update_statistics(self) -> None:
        """Update table statistics."""
        query = f"UPDATE STATISTICS {self.full_table_name}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()

    @classmethod
    def from_connection_string(
        cls,
        table: str,
        connection_string: str,
        schema: str = "dbo",
    ) -> "SQLServerDataSource":
        """Create data source from connection string.

        Args:
            table: Table name.
            connection_string: ODBC connection string.
            schema: Schema name.

        Returns:
            SQLServerDataSource.
        """
        # Parse connection string for required parameters
        parts = dict(p.split("=", 1) for p in connection_string.split(";") if "=" in p)

        host = parts.get("SERVER", parts.get("Data Source", ""))
        if "," in host:
            host, port = host.rsplit(",", 1)
            port = int(port)
        else:
            port = 1433

        database = parts.get("DATABASE", parts.get("Initial Catalog", ""))
        user = parts.get("UID", parts.get("User ID"))
        password = parts.get("PWD", parts.get("Password"))
        trusted = parts.get("Trusted_Connection", "").lower() == "yes"

        return cls(
            table=table,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            schema=schema,
            trusted_connection=trusted,
        )
