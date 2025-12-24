"""MySQL data source implementation.

This module provides a data source for MySQL databases.
Requires: pip install pymysql (or mysql-connector-python)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from truthound.datasources.sql.base import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
)
from truthound.datasources.base import DataSourceConnectionError


def _check_pymysql_available() -> None:
    """Check if pymysql is available."""
    try:
        import pymysql  # noqa: F401
    except ImportError:
        raise ImportError(
            "pymysql is required for MySQLDataSource. "
            "Install with: pip install pymysql"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MySQLDataSourceConfig(SQLDataSourceConfig):
    """Configuration for MySQL data sources.

    Attributes:
        host: Database host.
        port: Database port.
        database: Database name.
        user: Database user.
        password: Database password.
        charset: Character set.
        ssl: SSL configuration dictionary.
        autocommit: Whether to use autocommit mode.
    """

    host: str = "localhost"
    port: int = 3306
    database: str = "mysql"
    user: str = "root"
    password: str = ""
    charset: str = "utf8mb4"
    ssl: dict | None = None
    autocommit: bool = True


# =============================================================================
# MySQL Data Source
# =============================================================================


class MySQLDataSource(BaseSQLDataSource):
    """Data source for MySQL databases.

    MySQL is a popular open-source relational database widely used
    in web applications.

    Example:
        >>> source = MySQLDataSource(
        ...     table="users",
        ...     host="localhost",
        ...     database="mydb",
        ...     user="root",
        ...     password="secret",
        ... )
        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())

        >>> # Using connection string
        >>> source = MySQLDataSource.from_connection_string(
        ...     "mysql://user:pass@host:3306/db",
        ...     table="users",
        ... )
    """

    source_type = "mysql"

    def __init__(
        self,
        table: str,
        host: str = "localhost",
        port: int = 3306,
        database: str = "mysql",
        user: str = "root",
        password: str = "",
        config: MySQLDataSourceConfig | None = None,
    ) -> None:
        """Initialize MySQL data source.

        Args:
            table: Table name to validate.
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            config: Optional configuration (overrides other params).
        """
        _check_pymysql_available()

        if config is None:
            config = MySQLDataSourceConfig(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
            )

        super().__init__(table=table, config=config)

    @classmethod
    def _default_config(cls) -> MySQLDataSourceConfig:
        return MySQLDataSourceConfig()

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        table: str,
    ) -> "MySQLDataSource":
        """Create data source from connection string.

        Args:
            connection_string: MySQL connection string.
            table: Table name to validate.

        Returns:
            MySQLDataSource instance.

        Example:
            >>> source = MySQLDataSource.from_connection_string(
            ...     "mysql://user:pass@host:3306/db",
            ...     table="users",
            ... )
        """
        _check_pymysql_available()
        from urllib.parse import urlparse

        parsed = urlparse(connection_string)

        config = MySQLDataSourceConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            database=parsed.path.lstrip("/") or "mysql",
            user=parsed.username or "root",
            password=parsed.password or "",
        )

        return cls(table=table, config=config)

    def _create_connection(self) -> Any:
        """Create a new MySQL connection."""
        import pymysql

        cfg: MySQLDataSourceConfig = self._config  # type: ignore

        try:
            conn = pymysql.connect(
                host=cfg.host,
                port=cfg.port,
                database=cfg.database,
                user=cfg.user,
                password=cfg.password,
                charset=cfg.charset,
                autocommit=cfg.autocommit,
                ssl=cfg.ssl,
                cursorclass=pymysql.cursors.DictCursor,
            )
            return conn
        except pymysql.Error as e:
            raise DataSourceConnectionError("mysql", str(e))

    def _get_table_schema_query(self) -> str:
        """Get MySQL schema query from information_schema."""
        cfg: MySQLDataSourceConfig = self._config  # type: ignore

        return f"""
            SELECT COLUMN_NAME as column_name, DATA_TYPE as data_type
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = '{cfg.database}'
              AND TABLE_NAME = '{self._table}'
            ORDER BY ORDINAL_POSITION
        """

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from MySQL database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(self._get_table_schema_query())
            result = cursor.fetchall()
            cursor.close()

            # Result is list of dicts due to DictCursor
            return [(row["column_name"], row["data_type"]) for row in result]

    def _get_row_count_query(self) -> str:
        """Get MySQL row count query."""
        return f"SELECT COUNT(*) as cnt FROM {self._quote_identifier(self._table)}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote MySQL identifier with backticks."""
        escaped = identifier.replace("`", "``")
        return f"`{escaped}`"

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        cfg: MySQLDataSourceConfig = self._config  # type: ignore
        return f"{self._quote_identifier(cfg.database)}.{self._quote_identifier(self._table)}"

    def execute_query(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return list(result)

    def execute_scalar(
        self,
        query: str,
        params: tuple | dict | None = None,
    ) -> Any:
        """Execute a query and return a single value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()

            if result is None:
                return None

            # DictCursor returns dict, get first value
            if isinstance(result, dict):
                return list(result.values())[0]
            return result[0]

    # -------------------------------------------------------------------------
    # MySQL-specific Methods
    # -------------------------------------------------------------------------

    def get_table_status(self) -> dict[str, Any]:
        """Get table status information."""
        cfg: MySQLDataSourceConfig = self._config  # type: ignore
        query = f"SHOW TABLE STATUS FROM {self._quote_identifier(cfg.database)} LIKE '{self._table}'"
        result = self.execute_query(query)
        return result[0] if result else {}

    def get_table_size(self) -> dict[str, Any]:
        """Get table size information."""
        cfg: MySQLDataSourceConfig = self._config  # type: ignore
        query = f"""
            SELECT
                ROUND(data_length / 1024 / 1024, 2) as data_size_mb,
                ROUND(index_length / 1024 / 1024, 2) as index_size_mb,
                ROUND((data_length + index_length) / 1024 / 1024, 2) as total_size_mb,
                TABLE_ROWS as approx_rows
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = '{cfg.database}'
              AND TABLE_NAME = '{self._table}'
        """
        result = self.execute_query(query)
        return result[0] if result else {}

    def get_index_info(self) -> list[dict[str, Any]]:
        """Get index information for the table."""
        query = f"SHOW INDEX FROM {self._quote_identifier(self._table)}"
        return self.execute_query(query)

    def get_create_table(self) -> str:
        """Get the CREATE TABLE statement."""
        query = f"SHOW CREATE TABLE {self._quote_identifier(self._table)}"
        result = self.execute_query(query)
        if result:
            return result[0].get("Create Table", "")
        return ""

    def analyze(self) -> None:
        """Run ANALYZE to update table statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"ANALYZE TABLE {self._quote_identifier(self._table)}")
            cursor.close()

    def optimize(self) -> None:
        """Run OPTIMIZE to defragment the table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"OPTIMIZE TABLE {self._quote_identifier(self._table)}")
            cursor.close()
