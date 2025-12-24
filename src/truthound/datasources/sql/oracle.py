"""Oracle Database data source implementation.

This module provides a data source for Oracle Database,
supporting both thick and thin client modes.

Requires: pip install oracledb
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


def _check_oracle_available() -> None:
    """Check if Oracle client is available."""
    try:
        import oracledb  # noqa: F401
    except ImportError:
        raise ImportError(
            "oracledb is required for OracleDataSource. "
            "Install with: pip install oracledb"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OracleConfig(SQLDataSourceConfig):
    """Configuration for Oracle data source.

    Attributes:
        host: Database host.
        port: Database port (default: 1521).
        service_name: Oracle service name.
        sid: Oracle SID (alternative to service_name).
        user: Username.
        password: Password.
        dsn: Full DSN string (alternative to host/port/service).
        wallet_location: Path to Oracle wallet for mutual TLS.
        wallet_password: Wallet password.
        thick_mode: Use thick mode with Oracle Client libraries.
        encoding: Character encoding.
    """

    host: str | None = None
    port: int = 1521
    service_name: str | None = None
    sid: str | None = None
    user: str | None = None
    password: str | None = None
    dsn: str | None = None
    wallet_location: str | None = None
    wallet_password: str | None = None
    thick_mode: bool = False
    encoding: str = "UTF-8"


# =============================================================================
# Oracle Data Source
# =============================================================================


class OracleDataSource(BaseSQLDataSource):
    """Data source for Oracle Database.

    Supports:
    - Thin mode (pure Python, no Oracle Client needed)
    - Thick mode (requires Oracle Client libraries)
    - Oracle Wallet authentication
    - Connection pooling

    Example:
        >>> # Using host/service_name
        >>> source = OracleDataSource(
        ...     table="USERS",
        ...     host="oracle.example.com",
        ...     service_name="ORCL",
        ...     user="myuser",
        ...     password="mypassword",
        ... )

        >>> # Using DSN
        >>> source = OracleDataSource(
        ...     table="USERS",
        ...     dsn="oracle.example.com:1521/ORCL",
        ...     user="myuser",
        ...     password="mypassword",
        ... )

        >>> # Using Oracle Wallet (Autonomous Database)
        >>> source = OracleDataSource(
        ...     table="USERS",
        ...     dsn="mydb_high",
        ...     user="ADMIN",
        ...     password="password",
        ...     wallet_location="/path/to/wallet",
        ... )

        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
    """

    source_type = "oracle"

    def __init__(
        self,
        table: str,
        host: str | None = None,
        port: int = 1521,
        service_name: str | None = None,
        sid: str | None = None,
        user: str | None = None,
        password: str | None = None,
        dsn: str | None = None,
        schema: str | None = None,
        wallet_location: str | None = None,
        thick_mode: bool = False,
        config: OracleConfig | None = None,
    ) -> None:
        """Initialize Oracle data source.

        Args:
            table: Table name.
            host: Database host.
            port: Database port.
            service_name: Oracle service name.
            sid: Oracle SID (alternative to service_name).
            user: Username.
            password: Password.
            dsn: Full DSN string.
            schema: Schema/owner name.
            wallet_location: Path to Oracle wallet.
            thick_mode: Use thick mode.
            config: Optional configuration.
        """
        _check_oracle_available()

        if config is None:
            config = OracleConfig()

        config.host = host
        config.port = port
        config.service_name = service_name
        config.sid = sid
        config.user = user
        config.password = password
        config.dsn = dsn
        config.schema_name = schema
        config.wallet_location = wallet_location
        config.thick_mode = thick_mode

        # Build name
        db_name = dsn or service_name or sid or host or "oracle"
        config.name = config.name or f"{db_name}.{schema or user}.{table}"

        self._schema = schema or user  # Oracle uses user as default schema
        self._thick_mode = thick_mode

        super().__init__(table=table, config=config)

        # Initialize thick mode if requested
        if thick_mode:
            import oracledb
            oracledb.init_oracle_client()

    @classmethod
    def _default_config(cls) -> OracleConfig:
        return OracleConfig()

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        if self._schema:
            return f'"{self._schema}"."{self._table}"'
        return f'"{self._table}"'

    def _create_connection(self) -> Any:
        """Create Oracle connection."""
        import oracledb

        conn_params = {
            "user": self._config.user,
            "password": self._config.password,
        }

        # Build DSN if not provided
        if self._config.dsn:
            conn_params["dsn"] = self._config.dsn
        elif self._config.service_name:
            conn_params["dsn"] = oracledb.makedsn(
                self._config.host,
                self._config.port,
                service_name=self._config.service_name,
            )
        elif self._config.sid:
            conn_params["dsn"] = oracledb.makedsn(
                self._config.host,
                self._config.port,
                sid=self._config.sid,
            )
        else:
            raise DataSourceError("Either dsn, service_name, or sid is required")

        # Handle wallet authentication
        if self._config.wallet_location:
            conn_params["config_dir"] = self._config.wallet_location
            conn_params["wallet_location"] = self._config.wallet_location
            if self._config.wallet_password:
                conn_params["wallet_password"] = self._config.wallet_password

        return oracledb.connect(**conn_params)

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from Oracle."""
        schema_filter = self._schema or self._config.user
        query = f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM ALL_TAB_COLUMNS
            WHERE OWNER = UPPER('{schema_filter}')
            AND TABLE_NAME = UPPER('{self._table}')
            ORDER BY COLUMN_ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return [(row[0], row[1]) for row in result]

    def _get_row_count_query(self) -> str:
        """Get Oracle row count query."""
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote Oracle identifier with double quotes."""
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute Oracle query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return [dict(zip(columns, row)) for row in results]

    def execute_scalar(self, query: str) -> Any:
        """Execute Oracle query returning single value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None

    def to_polars_lazyframe(self):
        """Convert Oracle table to Polars LazyFrame."""
        import polars as pl

        query = f"SELECT * FROM {self.full_table_name}"
        if self._config.max_rows:
            query = f"SELECT * FROM ({query}) WHERE ROWNUM <= {self._config.max_rows}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()

        df_dict = {col: [row[i] for row in data] for i, col in enumerate(columns)}
        return pl.DataFrame(df_dict).lazy()

    def validate_connection(self) -> bool:
        """Validate Oracle connection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                cursor.close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Oracle-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> dict[str, Any]:
        """Get detailed table information."""
        schema_filter = self._schema or self._config.user
        query = f"""
            SELECT
                OWNER,
                TABLE_NAME,
                TABLESPACE_NAME,
                STATUS,
                NUM_ROWS,
                BLOCKS,
                AVG_ROW_LEN,
                LAST_ANALYZED,
                PARTITIONED,
                COMPRESSION
            FROM ALL_TABLES
            WHERE OWNER = UPPER('{schema_filter}')
            AND TABLE_NAME = UPPER('{self._table}')
        """
        results = self.execute_query(query)
        return results[0] if results else {}

    def get_constraints(self) -> list[dict[str, Any]]:
        """Get table constraints."""
        schema_filter = self._schema or self._config.user
        query = f"""
            SELECT
                CONSTRAINT_NAME,
                CONSTRAINT_TYPE,
                STATUS,
                VALIDATED
            FROM ALL_CONSTRAINTS
            WHERE OWNER = UPPER('{schema_filter}')
            AND TABLE_NAME = UPPER('{self._table}')
        """
        return self.execute_query(query)

    def get_indexes(self) -> list[dict[str, Any]]:
        """Get table indexes."""
        schema_filter = self._schema or self._config.user
        query = f"""
            SELECT
                INDEX_NAME,
                INDEX_TYPE,
                UNIQUENESS,
                STATUS,
                TABLESPACE_NAME
            FROM ALL_INDEXES
            WHERE OWNER = UPPER('{schema_filter}')
            AND TABLE_NAME = UPPER('{self._table}')
        """
        return self.execute_query(query)

    def get_partition_info(self) -> list[dict[str, Any]]:
        """Get partition information for partitioned tables."""
        schema_filter = self._schema or self._config.user
        query = f"""
            SELECT
                PARTITION_NAME,
                PARTITION_POSITION,
                TABLESPACE_NAME,
                NUM_ROWS,
                BLOCKS
            FROM ALL_TAB_PARTITIONS
            WHERE TABLE_OWNER = UPPER('{schema_filter}')
            AND TABLE_NAME = UPPER('{self._table}')
            ORDER BY PARTITION_POSITION
        """
        return self.execute_query(query)

    def analyze_table(self) -> None:
        """Gather table statistics."""
        schema_filter = self._schema or self._config.user
        query = f"""
            BEGIN
                DBMS_STATS.GATHER_TABLE_STATS(
                    ownname => '{schema_filter}',
                    tabname => '{self._table}'
                );
            END;
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()

    @classmethod
    def from_tns(
        cls,
        table: str,
        tns_name: str,
        user: str,
        password: str,
        schema: str | None = None,
        tns_admin: str | None = None,
    ) -> "OracleDataSource":
        """Create data source from TNS name.

        Args:
            table: Table name.
            tns_name: TNS alias from tnsnames.ora.
            user: Username.
            password: Password.
            schema: Schema name.
            tns_admin: Path to TNS_ADMIN directory.

        Returns:
            OracleDataSource configured with TNS.
        """
        import os
        if tns_admin:
            os.environ["TNS_ADMIN"] = tns_admin

        return cls(
            table=table,
            dsn=tns_name,
            user=user,
            password=password,
            schema=schema,
        )
