"""PostgreSQL data source implementation.

This module provides a data source for PostgreSQL databases.
Requires: pip install psycopg2-binary (or psycopg2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from truthound.datasources.sql.base import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
)
from truthound.datasources.base import DataSourceConnectionError


def _check_psycopg2_available() -> None:
    """Check if psycopg2 is available."""
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        raise ImportError(
            "psycopg2 is required for PostgreSQLDataSource. "
            "Install with: pip install psycopg2-binary"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PostgreSQLDataSourceConfig(SQLDataSourceConfig):
    """Configuration for PostgreSQL data sources.

    Attributes:
        host: Database host.
        port: Database port.
        database: Database name.
        user: Database user.
        password: Database password.
        sslmode: SSL mode (disable, require, verify-ca, verify-full).
        application_name: Application name for connection.
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "postgres"
    user: str = "postgres"
    password: str = ""
    sslmode: str = "prefer"
    application_name: str = "truthound"

    # Override default schema_name
    schema_name: str | None = "public"


# =============================================================================
# PostgreSQL Data Source
# =============================================================================


class PostgreSQLDataSource(BaseSQLDataSource):
    """Data source for PostgreSQL databases.

    PostgreSQL is a powerful, open-source relational database with
    excellent support for advanced data types and operations.

    Example:
        >>> source = PostgreSQLDataSource(
        ...     table="users",
        ...     host="localhost",
        ...     database="mydb",
        ...     user="postgres",
        ...     password="secret",
        ... )
        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())

        >>> # Using connection string
        >>> source = PostgreSQLDataSource.from_connection_string(
        ...     "postgresql://user:pass@host:5432/db",
        ...     table="users",
        ... )
    """

    source_type = "postgresql"

    def __init__(
        self,
        table: str,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        schema_name: str = "public",
        config: PostgreSQLDataSourceConfig | None = None,
    ) -> None:
        """Initialize PostgreSQL data source.

        Args:
            table: Table name to validate.
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            schema_name: Schema name (default: public).
            config: Optional configuration (overrides other params).
        """
        _check_psycopg2_available()

        if config is None:
            config = PostgreSQLDataSourceConfig(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                schema_name=schema_name,
            )

        super().__init__(table=table, config=config)

    @classmethod
    def _default_config(cls) -> PostgreSQLDataSourceConfig:
        return PostgreSQLDataSourceConfig()

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        table: str,
        schema_name: str = "public",
    ) -> "PostgreSQLDataSource":
        """Create data source from connection string.

        Args:
            connection_string: PostgreSQL connection string.
            table: Table name to validate.
            schema_name: Schema name.

        Returns:
            PostgreSQLDataSource instance.

        Example:
            >>> source = PostgreSQLDataSource.from_connection_string(
            ...     "postgresql://user:pass@host:5432/db",
            ...     table="users",
            ... )
        """
        _check_psycopg2_available()
        from urllib.parse import urlparse

        parsed = urlparse(connection_string)

        config = PostgreSQLDataSourceConfig(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            database=parsed.path.lstrip("/") or "postgres",
            user=parsed.username or "postgres",
            password=parsed.password or "",
            schema_name=schema_name,
        )

        return cls(table=table, config=config)

    def _create_connection(self) -> Any:
        """Create a new PostgreSQL connection."""
        import psycopg2
        import psycopg2.extras

        cfg: PostgreSQLDataSourceConfig = self._config  # type: ignore

        try:
            conn = psycopg2.connect(
                host=cfg.host,
                port=cfg.port,
                database=cfg.database,
                user=cfg.user,
                password=cfg.password,
                sslmode=cfg.sslmode,
                application_name=cfg.application_name,
            )
            # Set autocommit for read operations
            conn.autocommit = True
            return conn
        except psycopg2.Error as e:
            raise DataSourceConnectionError("postgresql", str(e))

    def _get_table_schema_query(self) -> str:
        """Get PostgreSQL schema query from information_schema."""
        cfg: PostgreSQLDataSourceConfig = self._config  # type: ignore
        schema = cfg.schema_name or "public"

        return f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
              AND table_name = '{self._table}'
            ORDER BY ordinal_position
        """

    def _get_row_count_query(self) -> str:
        """Get PostgreSQL row count query."""
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote PostgreSQL identifier with double quotes."""
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        cfg: PostgreSQLDataSourceConfig = self._config  # type: ignore
        if cfg.schema_name:
            return f'{self._quote_identifier(cfg.schema_name)}.{self._quote_identifier(self._table)}'
        return self._quote_identifier(self._table)

    # -------------------------------------------------------------------------
    # PostgreSQL-specific Methods
    # -------------------------------------------------------------------------

    def get_table_size(self) -> dict[str, Any]:
        """Get table size information.

        Returns:
            Dictionary with size information.
        """
        query = f"""
            SELECT
                pg_size_pretty(pg_total_relation_size('{self.full_table_name}')) as total_size,
                pg_size_pretty(pg_relation_size('{self.full_table_name}')) as table_size,
                pg_size_pretty(pg_indexes_size('{self.full_table_name}')) as indexes_size
        """
        result = self.execute_query(query)
        return result[0] if result else {}

    def get_table_statistics(self) -> dict[str, Any]:
        """Get table statistics from pg_stat_user_tables."""
        cfg: PostgreSQLDataSourceConfig = self._config  # type: ignore
        query = f"""
            SELECT
                n_live_tup as live_rows,
                n_dead_tup as dead_rows,
                last_vacuum,
                last_autovacuum,
                last_analyze,
                last_autoanalyze
            FROM pg_stat_user_tables
            WHERE schemaname = '{cfg.schema_name or 'public'}'
              AND relname = '{self._table}'
        """
        result = self.execute_query(query)
        return result[0] if result else {}

    def get_index_info(self) -> list[dict[str, Any]]:
        """Get index information for the table."""
        cfg: PostgreSQLDataSourceConfig = self._config  # type: ignore
        query = f"""
            SELECT
                i.relname as index_name,
                a.attname as column_name,
                ix.indisunique as is_unique,
                ix.indisprimary as is_primary
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE t.relname = '{self._table}'
              AND n.nspname = '{cfg.schema_name or 'public'}'
            ORDER BY i.relname, a.attnum
        """
        return self.execute_query(query)

    def get_constraints(self) -> list[dict[str, Any]]:
        """Get constraint information for the table."""
        cfg: PostgreSQLDataSourceConfig = self._config  # type: ignore
        query = f"""
            SELECT
                c.conname as constraint_name,
                c.contype as constraint_type,
                pg_get_constraintdef(c.oid) as definition
            FROM pg_constraint c
            JOIN pg_namespace n ON n.oid = c.connamespace
            JOIN pg_class t ON t.oid = c.conrelid
            WHERE t.relname = '{self._table}'
              AND n.nspname = '{cfg.schema_name or 'public'}'
        """
        return self.execute_query(query)

    def analyze(self) -> None:
        """Run ANALYZE to update table statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"ANALYZE {self.full_table_name}")
            cursor.close()

    def vacuum(self, full: bool = False) -> None:
        """Run VACUUM to reclaim storage.

        Args:
            full: If True, run VACUUM FULL (more aggressive but locks table).
        """
        with self._get_connection() as conn:
            # VACUUM cannot run inside a transaction
            old_isolation = conn.isolation_level
            conn.set_isolation_level(0)
            cursor = conn.cursor()
            if full:
                cursor.execute(f"VACUUM FULL {self.full_table_name}")
            else:
                cursor.execute(f"VACUUM {self.full_table_name}")
            cursor.close()
            conn.set_isolation_level(old_isolation)
