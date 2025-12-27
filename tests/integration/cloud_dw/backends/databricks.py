"""Databricks test backend implementation.

This module provides the test backend for Databricks SQL integration testing.

Features:
    - Personal Access Token authentication
    - OAuth M2M authentication
    - Unity Catalog support
    - SQL warehouse management

Usage:
    >>> from tests.integration.cloud_dw.backends import DatabricksTestBackend, DatabricksCredentials
    >>>
    >>> credentials = DatabricksCredentials(
    ...     host="adb-12345.azuredatabricks.net",
    ...     http_path="/sql/1.0/warehouses/abc123",
    ...     access_token="dapi...",
    ... )
    >>> backend = DatabricksTestBackend(credentials)
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
    from truthound.datasources.sql.databricks import DatabricksDataSource

logger = logging.getLogger(__name__)


# =============================================================================
# Credentials
# =============================================================================


@dataclass
class DatabricksCredentials(BaseCredentials):
    """Credentials for Databricks SQL.

    Supports:
        - Personal Access Token (PAT) authentication
        - OAuth M2M (service principal) authentication
        - Azure AD authentication (for Azure Databricks)

    Attributes:
        host: Databricks workspace URL.
        http_path: SQL warehouse HTTP path.
        access_token: Personal access token.
        client_id: OAuth client ID (for M2M auth).
        client_secret: OAuth client secret (for M2M auth).
        catalog: Unity Catalog name.
        schema: Default schema.
    """

    host: str = ""
    http_path: str = ""
    access_token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    catalog: str | None = None
    schema: str = "default"

    def __post_init__(self) -> None:
        # Try to get from environment if not set
        if not self.host:
            self.host = os.getenv("DATABRICKS_HOST", "")
        if not self.http_path:
            self.http_path = os.getenv("DATABRICKS_HTTP_PATH", "")
        if not self.access_token:
            self.access_token = os.getenv("DATABRICKS_TOKEN")
        if not self.client_id:
            self.client_id = os.getenv("DATABRICKS_CLIENT_ID")
        if not self.client_secret:
            self.client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
        if not self.catalog:
            self.catalog = os.getenv("DATABRICKS_CATALOG")
        if not self.schema:
            self.schema = os.getenv("DATABRICKS_SCHEMA", "default")

    def validate(self) -> bool:
        """Validate that credentials are properly configured."""
        if not self.host:
            logger.error("Databricks host not specified")
            return False
        if not self.http_path:
            logger.error("Databricks HTTP path not specified")
            return False

        # Check authentication
        has_pat = bool(self.access_token)
        has_oauth = bool(self.client_id and self.client_secret)

        if not has_pat and not has_oauth:
            logger.error("Databricks authentication required (PAT or OAuth)")
            return False

        return True

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters."""
        params: dict[str, Any] = {
            "host": self.host,
            "http_path": self.http_path,
        }
        if self.access_token:
            params["access_token"] = "***"  # Don't expose
        if self.client_id:
            params["client_id"] = self.client_id
        if self.catalog:
            params["catalog"] = self.catalog
        if self.schema:
            params["schema"] = self.schema
        return params

    @property
    def is_service_account(self) -> bool:
        """Check if using OAuth M2M credentials."""
        return bool(self.client_id and self.client_secret)


# =============================================================================
# Backend Implementation
# =============================================================================


class DatabricksTestBackend(CloudDWTestBackend[DatabricksCredentials]):
    """Test backend for Databricks SQL.

    This backend provides:
        - Full query execution
        - Unity Catalog schema management
        - SQL warehouse integration
    """

    platform_name: ClassVar[str] = "databricks"
    supports_dry_run: ClassVar[bool] = False
    supports_cost_estimation: ClassVar[bool] = False
    default_quote_char: ClassVar[str] = "`"

    def __init__(
        self,
        credentials: DatabricksCredentials,
        config: IntegrationTestConfig | None = None,
    ) -> None:
        super().__init__(credentials, config)
        self._conn: Any = None
        self._cursor: Any = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def _create_connection(self) -> Any:
        """Create Databricks SQL connection."""
        from databricks import sql

        connect_params: dict[str, Any] = {
            "server_hostname": self.credentials.host,
            "http_path": self.credentials.http_path,
        }

        if self.credentials.access_token:
            connect_params["access_token"] = self.credentials.access_token
        elif self.credentials.client_id:
            # OAuth M2M authentication
            connect_params["auth_type"] = "oauth-m2m"
            connect_params["client_id"] = self.credentials.client_id
            connect_params["client_secret"] = self.credentials.client_secret

        if self.credentials.catalog:
            connect_params["catalog"] = self.credentials.catalog

        self._conn = sql.connect(**connect_params)
        return self._conn

    def _close_connection(self) -> None:
        """Close Databricks connection."""
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
        """Execute query on Databricks."""
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
                return []

        finally:
            cursor.close()
            self._cursor = None

    # -------------------------------------------------------------------------
    # Dataset (Schema) Management
    # -------------------------------------------------------------------------

    def _create_dataset(self, name: str) -> None:
        """Create a Databricks schema."""
        if self.credentials.catalog:
            # Unity Catalog
            self._execute_query(
                f"CREATE SCHEMA IF NOT EXISTS "
                f"{self.quote_identifier(self.credentials.catalog)}."
                f"{self.quote_identifier(name)}"
            )
        else:
            # Hive metastore
            self._execute_query(f"CREATE SCHEMA IF NOT EXISTS {self.quote_identifier(name)}")

    def _drop_dataset(self, name: str) -> None:
        """Drop a Databricks schema."""
        if self.credentials.catalog:
            self._execute_query(
                f"DROP SCHEMA IF EXISTS "
                f"{self.quote_identifier(self.credentials.catalog)}."
                f"{self.quote_identifier(name)} CASCADE"
            )
        else:
            self._execute_query(
                f"DROP SCHEMA IF EXISTS {self.quote_identifier(name)} CASCADE"
            )

    def _create_table(
        self,
        dataset: str,
        table: str,
        schema: dict[str, str],
    ) -> None:
        """Create a Databricks table."""
        columns = ", ".join(
            f"{self.quote_identifier(name)} {dtype}"
            for name, dtype in schema.items()
        )

        if self.credentials.catalog:
            full_name = (
                f"{self.quote_identifier(self.credentials.catalog)}."
                f"{self.quote_identifier(dataset)}."
                f"{self.quote_identifier(table)}"
            )
        else:
            full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"

        self._execute_query(f"CREATE TABLE IF NOT EXISTS {full_name} ({columns})")

    def _insert_data(
        self,
        dataset: str,
        table: str,
        data: list[dict[str, Any]],
    ) -> None:
        """Insert data into a Databricks table."""
        if not data:
            return

        columns = list(data[0].keys())
        column_names = ", ".join(self.quote_identifier(c) for c in columns)

        if self.credentials.catalog:
            full_name = (
                f"{self.quote_identifier(self.credentials.catalog)}."
                f"{self.quote_identifier(dataset)}."
                f"{self.quote_identifier(table)}"
            )
        else:
            full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"

        # Build VALUES clause
        def format_value(v: Any) -> str:
            if v is None:
                return "NULL"
            elif isinstance(v, str):
                return f"'{v.replace(chr(39), chr(39)+chr(39))}'"
            elif isinstance(v, bool):
                return "TRUE" if v else "FALSE"
            else:
                return str(v)

        values_list = []
        for row in data:
            values = ", ".join(format_value(row[c]) for c in columns)
            values_list.append(f"({values})")

        insert_sql = (
            f"INSERT INTO {full_name} ({column_names}) VALUES {', '.join(values_list)}"
        )
        self._execute_query(insert_sql)

    def _find_stale_datasets(self, max_hours: int) -> list[str]:
        """Find stale test schemas."""
        if self.credentials.catalog:
            result = self._execute_query(
                f"""
                SELECT schema_name
                FROM {self.quote_identifier(self.credentials.catalog)}.information_schema.schemata
                WHERE schema_name LIKE '{self.config.test_dataset_prefix}%'
                """
            )
        else:
            result = self._execute_query(
                f"""
                SHOW SCHEMAS LIKE '{self.config.test_dataset_prefix}*'
                """
            )
        return [row.get("schema_name") or row.get("databaseName", "") for row in result]

    # -------------------------------------------------------------------------
    # Schema Operations
    # -------------------------------------------------------------------------

    def get_table_schema(
        self,
        dataset: str,
        table: str,
    ) -> dict[str, str]:
        """Get the schema of a Databricks table."""
        if self.credentials.catalog:
            full_name = (
                f"{self.quote_identifier(self.credentials.catalog)}."
                f"{self.quote_identifier(dataset)}."
                f"{self.quote_identifier(table)}"
            )
        else:
            full_name = f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"

        result = self._execute_query(f"DESCRIBE TABLE {full_name}")
        return {row["col_name"]: row["data_type"] for row in result if row.get("col_name")}

    def get_row_count(
        self,
        dataset: str,
        table: str,
    ) -> int:
        """Get the row count of a Databricks table."""
        if self.credentials.catalog:
            full_name = (
                f"{self.quote_identifier(self.credentials.catalog)}."
                f"{self.quote_identifier(dataset)}."
                f"{self.quote_identifier(table)}"
            )
        else:
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
    ) -> "DatabricksDataSource":
        """Create a Truthound Databricks DataSource."""
        from truthound.datasources.sql.databricks import (
            DatabricksDataSource,
            DatabricksConfig,
        )

        config = DatabricksConfig(
            host=self.credentials.host,
            http_path=self.credentials.http_path,
            access_token=self.credentials.access_token,
            client_id=self.credentials.client_id,
            client_secret=self.credentials.client_secret,
            catalog=self.credentials.catalog,
        )

        return DatabricksDataSource(
            table=table,
            schema_name=dataset,
            config=config,
        )

    # -------------------------------------------------------------------------
    # Databricks-specific methods
    # -------------------------------------------------------------------------

    def get_full_table_name(self, dataset: str, table: str) -> str:
        """Get the fully qualified table name for Databricks."""
        if self.credentials.catalog:
            return (
                f"{self.quote_identifier(self.credentials.catalog)}."
                f"{self.quote_identifier(dataset)}."
                f"{self.quote_identifier(table)}"
            )
        return f"{self.quote_identifier(dataset)}.{self.quote_identifier(table)}"

    def optimize_table(self, dataset: str, table: str) -> None:
        """Run OPTIMIZE on a Delta table."""
        full_name = self.get_full_table_name(dataset, table)
        self._execute_query(f"OPTIMIZE {full_name}")

    def get_table_history(
        self,
        dataset: str,
        table: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get Delta table history."""
        full_name = self.get_full_table_name(dataset, table)
        return self._execute_query(f"DESCRIBE HISTORY {full_name} LIMIT {limit}")

    def get_warehouse_status(self) -> dict[str, Any]:
        """Get SQL warehouse status information."""
        # This would require using the Databricks REST API
        # Returning empty dict as placeholder
        return {}
