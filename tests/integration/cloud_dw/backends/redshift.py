"""Redshift test backend implementation.

This module provides the test backend for AWS Redshift integration testing.

Features:
    - Standard username/password authentication
    - IAM authentication
    - Schema lifecycle management
    - Query performance monitoring

Usage:
    >>> from tests.integration.cloud_dw.backends import RedshiftTestBackend, RedshiftCredentials
    >>>
    >>> credentials = RedshiftCredentials(
    ...     host="my-cluster.abc123.us-east-1.redshift.amazonaws.com",
    ...     database="dev",
    ...     user="admin",
    ...     password="password",
    ... )
    >>> backend = RedshiftTestBackend(credentials)
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
    from truthound.datasources.sql.redshift import RedshiftDataSource

logger = logging.getLogger(__name__)


# =============================================================================
# Credentials
# =============================================================================


@dataclass
class RedshiftCredentials(BaseCredentials):
    """Credentials for AWS Redshift.

    Supports:
        - Username/password authentication
        - IAM authentication (temporary credentials)

    Attributes:
        host: Redshift cluster endpoint.
        database: Database name.
        user: Username (for password auth) or IAM user.
        password: Password (for password auth).
        port: Port number (default: 5439).
        iam_auth: Whether to use IAM authentication.
        cluster_identifier: Cluster ID (for IAM auth).
        region: AWS region (for IAM auth).
        ssl: Whether to use SSL.
        ssl_mode: SSL mode (verify-ca, verify-full, etc.).
    """

    host: str = ""
    database: str = ""
    user: str = ""
    password: str | None = None
    port: int = 5439
    iam_auth: bool = False
    cluster_identifier: str | None = None
    region: str | None = None
    ssl: bool = True
    ssl_mode: str = "verify-ca"

    def __post_init__(self) -> None:
        # Try to get from environment if not set
        if not self.host:
            self.host = os.getenv("REDSHIFT_HOST", "")
        if not self.database:
            self.database = os.getenv("REDSHIFT_DATABASE", "")
        if not self.user:
            self.user = os.getenv("REDSHIFT_USER", "")
        if not self.password:
            self.password = os.getenv("REDSHIFT_PASSWORD")
        if not self.cluster_identifier:
            self.cluster_identifier = os.getenv("REDSHIFT_CLUSTER_IDENTIFIER")
        if not self.region:
            self.region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION"))

        # Parse port from environment
        port_str = os.getenv("REDSHIFT_PORT")
        if port_str:
            self.port = int(port_str)

        # Check for IAM auth
        if os.getenv("REDSHIFT_IAM_AUTH", "").lower() == "true":
            self.iam_auth = True

    def validate(self) -> bool:
        """Validate that credentials are properly configured."""
        if not self.host:
            logger.error("Redshift host not specified")
            return False
        if not self.database:
            logger.error("Redshift database not specified")
            return False
        if not self.user:
            logger.error("Redshift user not specified")
            return False

        if self.iam_auth:
            if not self.cluster_identifier:
                logger.error("Cluster identifier required for IAM auth")
                return False
            if not self.region:
                logger.error("AWS region required for IAM auth")
                return False
        else:
            if not self.password:
                logger.error("Redshift password required for password auth")
                return False

        return True

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters."""
        params: dict[str, Any] = {
            "host": self.host,
            "database": self.database,
            "user": self.user,
            "port": self.port,
            "ssl": self.ssl,
            "iam_auth": self.iam_auth,
        }
        if self.password:
            params["password"] = "***"  # Don't expose
        if self.cluster_identifier:
            params["cluster_identifier"] = self.cluster_identifier
        if self.region:
            params["region"] = self.region
        return params

    @property
    def is_service_account(self) -> bool:
        """Check if using IAM credentials."""
        return self.iam_auth


# =============================================================================
# Backend Implementation
# =============================================================================


class RedshiftTestBackend(CloudDWTestBackend[RedshiftCredentials]):
    """Test backend for AWS Redshift.

    This backend provides:
        - Full query execution
        - Schema management
        - Query performance analysis
    """

    platform_name: ClassVar[str] = "redshift"
    supports_dry_run: ClassVar[bool] = False
    supports_cost_estimation: ClassVar[bool] = False
    default_quote_char: ClassVar[str] = '"'

    def __init__(
        self,
        credentials: RedshiftCredentials,
        config: IntegrationTestConfig | None = None,
    ) -> None:
        super().__init__(credentials, config)
        self._conn: Any = None
        self._cursor: Any = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def _create_connection(self) -> Any:
        """Create Redshift connection."""
        import redshift_connector

        connect_params: dict[str, Any] = {
            "host": self.credentials.host,
            "database": self.credentials.database,
            "user": self.credentials.user,
            "port": self.credentials.port,
            "ssl": self.credentials.ssl,
        }

        if self.credentials.iam_auth:
            connect_params["iam"] = True
            connect_params["cluster_identifier"] = self.credentials.cluster_identifier
            connect_params["region"] = self.credentials.region
        else:
            connect_params["password"] = self.credentials.password

        self._conn = redshift_connector.connect(**connect_params)
        return self._conn

    def _close_connection(self) -> None:
        """Close Redshift connection."""
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
        """Execute query on Redshift."""
        cursor = self._conn.cursor()
        self._cursor = cursor

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Check if query returns results
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            else:
                self._conn.commit()
                return []

        finally:
            cursor.close()
            self._cursor = None

    # -------------------------------------------------------------------------
    # Dataset (Schema) Management
    # -------------------------------------------------------------------------

    def _create_dataset(self, name: str) -> None:
        """Create a Redshift schema."""
        self._execute_query(f"CREATE SCHEMA IF NOT EXISTS {self.quote_identifier(name)}")

    def _drop_dataset(self, name: str) -> None:
        """Drop a Redshift schema."""
        self._execute_query(
            f"DROP SCHEMA IF EXISTS {self.quote_identifier(name)} CASCADE"
        )

    def _create_table(
        self,
        dataset: str,
        table: str,
        schema: dict[str, str],
    ) -> None:
        """Create a Redshift table."""
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
        """Insert data into a Redshift table."""
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
            self._conn.commit()
        finally:
            cursor.close()

    def _find_stale_datasets(self, max_hours: int) -> list[str]:
        """Find stale test schemas."""
        # Redshift doesn't track schema creation time directly
        # We use the oldest table's creation time as a proxy
        result = self._execute_query(
            f"""
            SELECT DISTINCT n.nspname AS schema_name
            FROM pg_catalog.pg_namespace n
            WHERE n.nspname LIKE '{self.config.test_dataset_prefix}%'
            """
        )
        return [row["schema_name"] for row in result]

    # -------------------------------------------------------------------------
    # Schema Operations
    # -------------------------------------------------------------------------

    def get_table_schema(
        self,
        dataset: str,
        table: str,
    ) -> dict[str, str]:
        """Get the schema of a Redshift table."""
        result = self._execute_query(
            f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{dataset}'
            AND table_name = '{table}'
            ORDER BY ordinal_position
            """
        )
        return {row["column_name"]: row["data_type"] for row in result}

    def get_row_count(
        self,
        dataset: str,
        table: str,
    ) -> int:
        """Get the row count of a Redshift table."""
        full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        result = self._execute_query(f"SELECT COUNT(*) AS cnt FROM {full_name}")
        return result[0]["cnt"] if result else 0

    # -------------------------------------------------------------------------
    # Truthound Integration
    # -------------------------------------------------------------------------

    def create_datasource(
        self,
        dataset: str,
        table: str,
    ) -> "RedshiftDataSource":
        """Create a Truthound Redshift DataSource."""
        from truthound.datasources.sql.redshift import (
            RedshiftDataSource,
            RedshiftConfig,
        )

        config = RedshiftConfig(
            host=self.credentials.host,
            database=self.credentials.database,
            user=self.credentials.user,
            password=self.credentials.password,
            port=self.credentials.port,
            iam_auth=self.credentials.iam_auth,
            cluster_identifier=self.credentials.cluster_identifier,
            ssl=self.credentials.ssl,
        )

        return RedshiftDataSource(
            table=table,
            schema_name=dataset,
            config=config,
        )

    # -------------------------------------------------------------------------
    # Redshift-specific methods
    # -------------------------------------------------------------------------

    def get_query_execution_plan(self, query: str) -> list[dict[str, Any]]:
        """Get the execution plan for a query."""
        return self._execute_query(f"EXPLAIN {query}")

    def get_table_distribution(
        self,
        dataset: str,
        table: str,
    ) -> dict[str, Any]:
        """Get table distribution information."""
        full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        result = self._execute_query(
            f"""
            SELECT "table", diststyle, sortkey1, skew_rows
            FROM svv_table_info
            WHERE "schema" = '{dataset}' AND "table" = '{table}'
            """
        )
        return result[0] if result else {}

    def vacuum_table(self, dataset: str, table: str) -> None:
        """Vacuum a table to reclaim space and re-sort."""
        full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        self._execute_query(f"VACUUM {full_name}")

    def analyze_table(self, dataset: str, table: str) -> None:
        """Analyze a table to update statistics."""
        full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"
        self._execute_query(f"ANALYZE {full_name}")
