"""Snowflake test backend implementation.

This module provides the test backend for Snowflake integration testing.

Features:
    - Multiple authentication methods (password, key pair, SSO, OAuth)
    - Warehouse management
    - Schema/database lifecycle management
    - Query profiling

Usage:
    >>> from tests.integration.cloud_dw.backends import SnowflakeTestBackend, SnowflakeCredentials
    >>>
    >>> credentials = SnowflakeCredentials(
    ...     account="my-account",
    ...     user="my-user",
    ...     password="my-password",
    ...     warehouse="COMPUTE_WH",
    ... )
    >>> backend = SnowflakeTestBackend(credentials)
    >>>
    >>> with backend:
    ...     dataset = backend.create_test_dataset()
    ...     # ... run tests
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, ClassVar

from tests.integration.cloud_dw.base import (
    BaseCredentials,
    CloudDWTestBackend,
    IntegrationTestConfig,
)

if TYPE_CHECKING:
    from snowflake.connector import SnowflakeConnection
    from truthound.datasources.sql.snowflake import SnowflakeDataSource

logger = logging.getLogger(__name__)


# =============================================================================
# Credentials
# =============================================================================


@dataclass
class SnowflakeCredentials(BaseCredentials):
    """Credentials for Snowflake.

    Supports:
        - Username/password authentication
        - Key pair authentication
        - External browser (SSO)
        - OAuth

    Attributes:
        account: Snowflake account identifier.
        user: Username.
        password: Password (for password auth).
        warehouse: Compute warehouse name.
        database: Default database.
        schema: Default schema.
        role: Role to use.
        private_key_path: Path to private key file (for key pair auth).
        private_key_passphrase: Passphrase for private key.
        authenticator: Authentication method.
    """

    account: str = ""
    user: str = ""
    password: str | None = None
    warehouse: str | None = None
    database: str | None = None
    schema: str = "PUBLIC"
    role: str | None = None
    private_key_path: str | None = None
    private_key_passphrase: str | None = None
    authenticator: str = "snowflake"  # snowflake, externalbrowser, oauth

    def __post_init__(self) -> None:
        # Try to get from environment if not set
        if not self.account:
            self.account = os.getenv("SNOWFLAKE_ACCOUNT", "")
        if not self.user:
            self.user = os.getenv("SNOWFLAKE_USER", "")
        if not self.password:
            self.password = os.getenv("SNOWFLAKE_PASSWORD")
        if not self.warehouse:
            self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
        if not self.database:
            self.database = os.getenv("SNOWFLAKE_DATABASE")
        if not self.role:
            self.role = os.getenv("SNOWFLAKE_ROLE")
        if not self.private_key_path:
            self.private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")

    def validate(self) -> bool:
        """Validate that credentials are properly configured."""
        if not self.account:
            logger.error("Snowflake account not specified")
            return False
        if not self.user:
            logger.error("Snowflake user not specified")
            return False

        # Check authentication method
        if self.authenticator == "snowflake":
            if not self.password and not self.private_key_path:
                logger.error("Snowflake password or private key required")
                return False
            if self.private_key_path and not os.path.exists(self.private_key_path):
                logger.error(f"Private key file not found: {self.private_key_path}")
                return False

        return True

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters."""
        params: dict[str, Any] = {
            "account": self.account,
            "user": self.user,
            "authenticator": self.authenticator,
        }

        if self.password:
            params["password"] = "***"  # Don't expose
        if self.warehouse:
            params["warehouse"] = self.warehouse
        if self.database:
            params["database"] = self.database
        if self.schema:
            params["schema"] = self.schema
        if self.role:
            params["role"] = self.role

        return params

    @property
    def is_service_account(self) -> bool:
        """Check if using service account (key pair) credentials."""
        return bool(self.private_key_path)

    def get_private_key(self) -> bytes | None:
        """Load and decrypt private key if configured."""
        if not self.private_key_path:
            return None

        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        with open(self.private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=(
                    self.private_key_passphrase.encode()
                    if self.private_key_passphrase
                    else None
                ),
                backend=default_backend(),
            )

        return private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )


# =============================================================================
# Backend Implementation
# =============================================================================


class SnowflakeTestBackend(CloudDWTestBackend[SnowflakeCredentials]):
    """Test backend for Snowflake.

    This backend provides:
        - Full query execution
        - Schema/database management
        - Warehouse suspend/resume (cost optimization)
        - Query history access
    """

    platform_name: ClassVar[str] = "snowflake"
    supports_dry_run: ClassVar[bool] = False  # Snowflake doesn't have dry run
    supports_cost_estimation: ClassVar[bool] = False
    default_quote_char: ClassVar[str] = '"'

    def __init__(
        self,
        credentials: SnowflakeCredentials,
        config: IntegrationTestConfig | None = None,
    ) -> None:
        super().__init__(credentials, config)
        self._conn: "SnowflakeConnection | None" = None
        self._cursor: Any = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def _create_connection(self) -> "SnowflakeConnection":
        """Create Snowflake connection."""
        import snowflake.connector

        connect_params: dict[str, Any] = {
            "account": self.credentials.account,
            "user": self.credentials.user,
            "authenticator": self.credentials.authenticator,
        }

        # Authentication
        if self.credentials.private_key_path:
            connect_params["private_key"] = self.credentials.get_private_key()
        elif self.credentials.password:
            connect_params["password"] = self.credentials.password

        # Optional parameters
        if self.credentials.warehouse:
            connect_params["warehouse"] = self.credentials.warehouse
        if self.credentials.database:
            connect_params["database"] = self.credentials.database
        if self.credentials.schema:
            connect_params["schema"] = self.credentials.schema
        if self.credentials.role:
            connect_params["role"] = self.credentials.role

        # Connection settings
        connect_params["login_timeout"] = min(self.config.timeout_seconds, 60)
        connect_params["network_timeout"] = self.config.timeout_seconds

        self._conn = snowflake.connector.connect(**connect_params)
        return self._conn

    def _close_connection(self) -> None:
        """Close Snowflake connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._conn:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def _execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute query on Snowflake."""
        cursor = self._conn.cursor()
        self._cursor = cursor

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch results
            rows = cursor.fetchall()

            return [dict(zip(columns, row)) for row in rows]

        finally:
            cursor.close()
            self._cursor = None

    # -------------------------------------------------------------------------
    # Dataset (Schema) Management
    # -------------------------------------------------------------------------

    def _create_dataset(self, name: str) -> None:
        """Create a Snowflake schema (within the configured database)."""
        # First ensure we have a database
        if not self.credentials.database:
            # Create a test database
            db_name = f"TRUTHOUND_TEST_{datetime.utcnow().strftime('%Y%m%d')}"
            self._execute_query(f"CREATE DATABASE IF NOT EXISTS {self.quote_identifier(db_name)}")
            self._execute_query(f"USE DATABASE {self.quote_identifier(db_name)}")

        # Create schema with comment for tracking
        self._execute_query(
            f"CREATE SCHEMA IF NOT EXISTS {self.quote_identifier(name)} "
            f"COMMENT = 'Truthound integration test - created {datetime.utcnow().isoformat()}'"
        )

    def _drop_dataset(self, name: str) -> None:
        """Drop a Snowflake schema."""
        self._execute_query(
            f"DROP SCHEMA IF EXISTS {self.quote_identifier(name)} CASCADE"
        )

    def _create_table(
        self,
        dataset: str,
        table: str,
        schema: dict[str, str],
    ) -> None:
        """Create a Snowflake table."""
        columns = ", ".join(
            f"{self.quote_identifier(name)} {dtype}"
            for name, dtype in schema.items()
        )
        full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        self._execute_query(f"CREATE TABLE IF NOT EXISTS {full_name} ({columns})")

    def _insert_data(
        self,
        dataset: str,
        table: str,
        data: list[dict[str, Any]],
    ) -> None:
        """Insert data into a Snowflake table."""
        if not data:
            return

        columns = list(data[0].keys())
        placeholders = ", ".join(["%s"] * len(columns))
        column_names = ", ".join(self.quote_identifier(c) for c in columns)
        full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"

        insert_sql = f"INSERT INTO {full_name} ({column_names}) VALUES ({placeholders})"

        cursor = self._conn.cursor()
        try:
            for row in data:
                values = tuple(row[c] for c in columns)
                cursor.execute(insert_sql, values)
        finally:
            cursor.close()

    def _find_stale_datasets(self, max_hours: int) -> list[str]:
        """Find stale test schemas."""
        cutoff = datetime.utcnow() - timedelta(hours=max_hours)

        # Query information_schema for schemas with our prefix
        result = self._execute_query(
            f"""
            SELECT SCHEMA_NAME, CREATED
            FROM INFORMATION_SCHEMA.SCHEMATA
            WHERE SCHEMA_NAME LIKE '{self.config.test_dataset_prefix.upper()}%'
            """
        )

        stale = []
        for row in result:
            created = row.get("CREATED")
            if created and created < cutoff:
                stale.append(row["SCHEMA_NAME"])

        return stale

    # -------------------------------------------------------------------------
    # Schema Operations
    # -------------------------------------------------------------------------

    def get_table_schema(
        self,
        dataset: str,
        table: str,
    ) -> dict[str, str]:
        """Get the schema of a Snowflake table."""
        result = self._execute_query(
            f"""
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{dataset.upper()}'
            AND TABLE_NAME = '{table.upper()}'
            ORDER BY ORDINAL_POSITION
            """
        )
        return {row["COLUMN_NAME"]: row["DATA_TYPE"] for row in result}

    def get_row_count(
        self,
        dataset: str,
        table: str,
    ) -> int:
        """Get the row count of a Snowflake table."""
        full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        result = self._execute_query(f"SELECT COUNT(*) AS cnt FROM {full_name}")
        return result[0]["CNT"] if result else 0

    # -------------------------------------------------------------------------
    # Truthound Integration
    # -------------------------------------------------------------------------

    def create_datasource(
        self,
        dataset: str,
        table: str,
    ) -> "SnowflakeDataSource":
        """Create a Truthound Snowflake DataSource."""
        from truthound.datasources.sql.snowflake import (
            SnowflakeDataSource,
            SnowflakeConfig,
        )

        config = SnowflakeConfig(
            account=self.credentials.account,
            user=self.credentials.user,
            password=self.credentials.password,
            warehouse=self.credentials.warehouse,
            database=self.credentials.database,
            role=self.credentials.role,
            authenticator=self.credentials.authenticator,
            private_key_path=self.credentials.private_key_path,
        )

        return SnowflakeDataSource(
            table=table,
            schema_name=dataset,
            config=config,
        )

    # -------------------------------------------------------------------------
    # Snowflake-specific methods
    # -------------------------------------------------------------------------

    def suspend_warehouse(self) -> None:
        """Suspend the warehouse to save costs."""
        if self.credentials.warehouse:
            self._execute_query(
                f"ALTER WAREHOUSE {self.quote_identifier(self.credentials.warehouse)} SUSPEND"
            )
            logger.info(f"[Snowflake] Suspended warehouse: {self.credentials.warehouse}")

    def resume_warehouse(self) -> None:
        """Resume the warehouse."""
        if self.credentials.warehouse:
            self._execute_query(
                f"ALTER WAREHOUSE {self.quote_identifier(self.credentials.warehouse)} RESUME"
            )
            logger.info(f"[Snowflake] Resumed warehouse: {self.credentials.warehouse}")

    def get_query_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent query history."""
        return self._execute_query(
            f"""
            SELECT QUERY_ID, QUERY_TEXT, START_TIME, END_TIME,
                   EXECUTION_STATUS, BYTES_SCANNED, ROWS_PRODUCED
            FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
            ORDER BY START_TIME DESC
            LIMIT {limit}
            """
        )
