"""Snowflake data source implementation.

This module provides a data source for Snowflake,
supporting various authentication methods including password, SSO, and key-pair.

Requires: pip install snowflake-connector-python
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from truthound.datasources.sql.cloud_base import (
    CloudDWConfig,
    CloudDWDataSource,
    load_credentials_from_env,
)
from truthound.datasources.base import (
    DataSourceConnectionError,
    DataSourceError,
)

if TYPE_CHECKING:
    import snowflake.connector


def _check_snowflake_available() -> None:
    """Check if Snowflake connector is available."""
    try:
        import snowflake.connector  # noqa: F401
    except ImportError:
        raise ImportError(
            "snowflake-connector-python is required for SnowflakeDataSource. "
            "Install with: pip install snowflake-connector-python"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SnowflakeConfig(CloudDWConfig):
    """Configuration for Snowflake data source.

    Attributes:
        account: Snowflake account identifier (e.g., 'xy12345.us-east-1').
        user: Username for authentication.
        password: Password (for password auth).
        database: Database name.
        schema_name: Schema name (default: PUBLIC).
        warehouse: Virtual warehouse name.
        role: Role to use.
        authenticator: Authentication method ('snowflake', 'externalbrowser', 'oauth').
        private_key_path: Path to private key file (for key-pair auth).
        private_key_passphrase: Passphrase for encrypted private key.
        token: OAuth token (for OAuth auth).
        client_session_keep_alive: Keep connection alive.
    """

    account: str | None = None
    user: str | None = None
    password: str | None = None
    database: str | None = None
    authenticator: str = "snowflake"
    private_key_path: str | None = None
    private_key_passphrase: str | None = None
    token: str | None = None
    client_session_keep_alive: bool = True


# =============================================================================
# Snowflake Data Source
# =============================================================================


class SnowflakeDataSource(CloudDWDataSource):
    """Data source for Snowflake.

    Supports authentication via:
    - Username/Password
    - External Browser (SSO)
    - Key-Pair Authentication
    - OAuth Token

    Example:
        >>> # Using password authentication
        >>> source = SnowflakeDataSource(
        ...     table="USERS",
        ...     account="xy12345.us-east-1",
        ...     user="myuser",
        ...     password="mypassword",
        ...     database="MYDB",
        ...     schema="PUBLIC",
        ...     warehouse="COMPUTE_WH",
        ... )

        >>> # Using SSO
        >>> source = SnowflakeDataSource(
        ...     table="USERS",
        ...     account="xy12345.us-east-1",
        ...     user="myuser@company.com",
        ...     database="MYDB",
        ...     authenticator="externalbrowser",
        ... )

        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
    """

    source_type = "snowflake"

    def __init__(
        self,
        table: str,
        account: str,
        user: str,
        database: str,
        schema: str = "PUBLIC",
        warehouse: str | None = None,
        password: str | None = None,
        role: str | None = None,
        authenticator: str = "snowflake",
        private_key_path: str | None = None,
        config: SnowflakeConfig | None = None,
    ) -> None:
        """Initialize Snowflake data source.

        Args:
            table: Table name.
            account: Snowflake account identifier.
            user: Username.
            database: Database name.
            schema: Schema name (default: PUBLIC).
            warehouse: Virtual warehouse name.
            password: Password (for password auth).
            role: Role to use.
            authenticator: Auth method ('snowflake', 'externalbrowser', 'oauth').
            private_key_path: Path to private key file.
            config: Optional configuration.
        """
        _check_snowflake_available()

        if config is None:
            config = SnowflakeConfig()

        config.account = account
        config.user = user
        config.password = password
        config.database = database
        config.schema_name = schema
        config.warehouse = warehouse
        config.role = role
        config.authenticator = authenticator
        config.private_key_path = private_key_path
        config.name = config.name or f"{database}.{schema}.{table}"

        self._account = account
        self._user = user
        self._database = database
        self._schema = schema
        self._warehouse = warehouse
        self._role = role
        self._authenticator = authenticator

        super().__init__(table=table, config=config)

    @classmethod
    def _default_config(cls) -> SnowflakeConfig:
        return SnowflakeConfig()

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f'"{self._database}"."{self._schema}"."{self._table}"'

    def _validate_credentials(self) -> bool:
        """Validate Snowflake credentials."""
        try:
            conn = self._create_connection()
            conn.cursor().execute("SELECT 1")
            conn.close()
            return True
        except Exception as e:
            raise DataSourceConnectionError(
                source_type="snowflake",
                details=f"Failed to authenticate: {e}",
            )

    def _create_connection(self) -> Any:
        """Create Snowflake connection."""
        import snowflake.connector

        conn_params = {
            "account": self._account,
            "user": self._user,
            "database": self._database,
            "schema": self._schema,
            "client_session_keep_alive": self._config.client_session_keep_alive,
        }

        if self._warehouse:
            conn_params["warehouse"] = self._warehouse
        if self._role:
            conn_params["role"] = self._role

        # Handle different authentication methods
        if self._authenticator == "snowflake":
            if self._config.password:
                conn_params["password"] = self._config.password
            else:
                raise DataSourceError("Password required for snowflake authenticator")

        elif self._authenticator == "externalbrowser":
            conn_params["authenticator"] = "externalbrowser"

        elif self._authenticator == "oauth":
            if self._config.token:
                conn_params["authenticator"] = "oauth"
                conn_params["token"] = self._config.token
            else:
                raise DataSourceError("Token required for OAuth authentication")

        elif self._config.private_key_path:
            # Key-pair authentication
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            with open(self._config.private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=self._config.private_key_passphrase.encode()
                    if self._config.private_key_passphrase
                    else None,
                    backend=default_backend(),
                )

            pkb = private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            conn_params["private_key"] = pkb

        return snowflake.connector.connect(**conn_params)

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from Snowflake."""
        query = f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM {self._database}.INFORMATION_SCHEMA.COLUMNS
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
        """Get Snowflake row count query."""
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote Snowflake identifier with double quotes."""
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _get_cost_estimate(self, query: str) -> dict[str, Any] | None:
        """Snowflake doesn't have direct cost estimation via dry run."""
        # Could implement EXPLAIN-based estimation
        return None

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute Snowflake query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return [dict(zip(columns, row)) for row in results]

    def execute_scalar(self, query: str) -> Any:
        """Execute Snowflake query returning single value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None

    def to_polars_lazyframe(self):
        """Convert Snowflake table to Polars LazyFrame."""
        import polars as pl

        query = f"SELECT * FROM {self.full_table_name}"
        if self._config.max_rows:
            query += f" LIMIT {self._config.max_rows}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            cursor.close()

        # Convert to DataFrame
        df_dict = {col: [row[i] for row in data] for i, col in enumerate(columns)}
        return pl.DataFrame(df_dict).lazy()

    def validate_connection(self) -> bool:
        """Validate Snowflake connection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Snowflake-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> dict[str, Any]:
        """Get detailed table information."""
        query = f"""
            SELECT
                TABLE_CATALOG,
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE,
                ROW_COUNT,
                BYTES,
                CREATED,
                LAST_ALTERED,
                COMMENT
            FROM {self._database}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{self._schema}'
            AND TABLE_NAME = '{self._table}'
        """
        results = self.execute_query(query)
        return results[0] if results else {}

    def get_query_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent query history for this table."""
        query = f"""
            SELECT
                QUERY_ID,
                QUERY_TEXT,
                USER_NAME,
                WAREHOUSE_NAME,
                EXECUTION_STATUS,
                START_TIME,
                END_TIME,
                TOTAL_ELAPSED_TIME,
                BYTES_SCANNED,
                CREDITS_USED_CLOUD_SERVICES
            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
            WHERE QUERY_TEXT ILIKE '%{self._table}%'
            ORDER BY START_TIME DESC
            LIMIT {limit}
        """
        return self.execute_query(query)

    def clone_table(self, target_table: str, schema: str | None = None) -> None:
        """Create a zero-copy clone of the table.

        Args:
            target_table: Name for the cloned table.
            schema: Target schema (default: same as source).
        """
        target_schema = schema or self._schema
        target_full = f'"{self._database}"."{target_schema}"."{target_table}"'

        query = f"CREATE TABLE {target_full} CLONE {self.full_table_name}"
        self.execute_query(query)

    def get_clustering_info(self) -> dict[str, Any] | None:
        """Get clustering key information for the table."""
        query = f"""
            SELECT
                CLUSTERING_KEY
            FROM {self._database}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{self._schema}'
            AND TABLE_NAME = '{self._table}'
        """
        results = self.execute_query(query)
        if results and results[0].get("CLUSTERING_KEY"):
            return {"clustering_key": results[0]["CLUSTERING_KEY"]}
        return None

    @classmethod
    def from_env(
        cls,
        table: str,
        schema: str = "PUBLIC",
        env_prefix: str = "SNOWFLAKE",
    ) -> "SnowflakeDataSource":
        """Create data source from environment variables.

        Reads: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
               SNOWFLAKE_DATABASE, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE

        Args:
            table: Table name.
            schema: Schema name.
            env_prefix: Environment variable prefix.

        Returns:
            SnowflakeDataSource configured from environment.
        """
        creds = load_credentials_from_env(env_prefix)

        required = ["account", "user", "database"]
        missing = [k for k in required if k not in creds]
        if missing:
            raise DataSourceError(
                f"Missing required environment variables: "
                f"{[f'{env_prefix}_{k.upper()}' for k in missing]}"
            )

        return cls(
            table=table,
            account=creds["account"],
            user=creds["user"],
            password=creds.get("password"),
            database=creds["database"],
            schema=schema,
            warehouse=creds.get("warehouse"),
            role=creds.get("role"),
        )
