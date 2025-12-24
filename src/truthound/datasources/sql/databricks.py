"""Databricks data source implementation.

This module provides a data source for Databricks SQL,
supporting both SQL Warehouse and All-Purpose Clusters.

Requires: pip install databricks-sql-connector
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
    pass


def _check_databricks_available() -> None:
    """Check if Databricks SQL connector is available."""
    try:
        from databricks import sql  # noqa: F401
    except ImportError:
        raise ImportError(
            "databricks-sql-connector is required for DatabricksDataSource. "
            "Install with: pip install databricks-sql-connector"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DatabricksConfig(CloudDWConfig):
    """Configuration for Databricks data source.

    Attributes:
        host: Databricks workspace hostname.
        http_path: HTTP path for SQL warehouse or cluster.
        access_token: Personal access token.
        catalog: Unity Catalog name.
        schema_name: Schema/database name.
        use_cloud_fetch: Use cloud fetch for large results.
        max_download_threads: Threads for cloud fetch.
        client_id: OAuth client ID (for OAuth).
        client_secret: OAuth client secret.
        use_oauth: Use OAuth instead of PAT.
    """

    host: str | None = None
    http_path: str | None = None
    access_token: str | None = None
    catalog: str | None = None
    use_cloud_fetch: bool = True
    max_download_threads: int = 10
    client_id: str | None = None
    client_secret: str | None = None
    use_oauth: bool = False


# =============================================================================
# Databricks Data Source
# =============================================================================


class DatabricksDataSource(CloudDWDataSource):
    """Data source for Databricks SQL.

    Supports:
    - SQL Warehouses (recommended for BI/analytics)
    - All-Purpose Clusters
    - Unity Catalog
    - Personal Access Token authentication
    - OAuth M2M authentication

    Example:
        >>> # Using Personal Access Token
        >>> source = DatabricksDataSource(
        ...     table="users",
        ...     host="adb-12345.azuredatabricks.net",
        ...     http_path="/sql/1.0/warehouses/abc123",
        ...     access_token="dapi...",
        ...     catalog="main",
        ...     schema="default",
        ... )

        >>> # Using OAuth
        >>> source = DatabricksDataSource(
        ...     table="users",
        ...     host="adb-12345.azuredatabricks.net",
        ...     http_path="/sql/1.0/warehouses/abc123",
        ...     client_id="...",
        ...     client_secret="...",
        ...     use_oauth=True,
        ...     catalog="main",
        ...     schema="default",
        ... )

        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
    """

    source_type = "databricks"

    def __init__(
        self,
        table: str,
        host: str,
        http_path: str,
        access_token: str | None = None,
        catalog: str | None = None,
        schema: str = "default",
        client_id: str | None = None,
        client_secret: str | None = None,
        use_oauth: bool = False,
        config: DatabricksConfig | None = None,
    ) -> None:
        """Initialize Databricks data source.

        Args:
            table: Table name.
            host: Databricks workspace hostname.
            http_path: HTTP path for SQL warehouse or cluster.
            access_token: Personal access token.
            catalog: Unity Catalog name.
            schema: Schema name.
            client_id: OAuth client ID.
            client_secret: OAuth client secret.
            use_oauth: Use OAuth authentication.
            config: Optional configuration.
        """
        _check_databricks_available()

        if config is None:
            config = DatabricksConfig()

        config.host = host
        config.http_path = http_path
        config.access_token = access_token
        config.catalog = catalog
        config.schema_name = schema
        config.client_id = client_id
        config.client_secret = client_secret
        config.use_oauth = use_oauth

        # Build name
        catalog_part = f"{catalog}." if catalog else ""
        config.name = config.name or f"{catalog_part}{schema}.{table}"

        self._catalog = catalog
        self._schema = schema
        self._use_oauth = use_oauth

        super().__init__(table=table, config=config)

    @classmethod
    def _default_config(cls) -> DatabricksConfig:
        return DatabricksConfig()

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        if self._catalog:
            return f"`{self._catalog}`.`{self._schema}`.`{self._table}`"
        return f"`{self._schema}`.`{self._table}`"

    def _validate_credentials(self) -> bool:
        """Validate Databricks credentials."""
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            raise DataSourceConnectionError(
                source_type="databricks",
                details=f"Failed to authenticate: {e}",
            )

    def _create_connection(self) -> Any:
        """Create Databricks connection."""
        from databricks import sql

        conn_params = {
            "server_hostname": self._config.host,
            "http_path": self._config.http_path,
            "use_cloud_fetch": self._config.use_cloud_fetch,
            "max_download_threads": self._config.max_download_threads,
        }

        if self._catalog:
            conn_params["catalog"] = self._catalog
        conn_params["schema"] = self._schema

        if self._use_oauth:
            if not self._config.client_id or not self._config.client_secret:
                raise DataSourceError("client_id and client_secret required for OAuth")

            # OAuth M2M authentication
            from databricks.sdk.core import oauth_service_principal

            def credential_provider():
                return oauth_service_principal(
                    host=self._config.host,
                    client_id=self._config.client_id,
                    client_secret=self._config.client_secret,
                )

            conn_params["credentials_provider"] = credential_provider
        else:
            if not self._config.access_token:
                raise DataSourceError("access_token required for PAT authentication")
            conn_params["access_token"] = self._config.access_token

        return sql.connect(**conn_params)

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from Databricks."""
        query = f"DESCRIBE TABLE {self.full_table_name}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            # DESCRIBE returns (col_name, data_type, comment)
            return [(row[0], row[1]) for row in result if not row[0].startswith("#")]

    def _get_row_count_query(self) -> str:
        """Get Databricks row count query."""
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote Databricks identifier with backticks."""
        escaped = identifier.replace("`", "``")
        return f"`{escaped}`"

    def _get_cost_estimate(self, query: str) -> dict[str, Any] | None:
        """Databricks doesn't have direct cost estimation via dry run."""
        return None

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute Databricks query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return [dict(zip(columns, row)) for row in results]

    def execute_scalar(self, query: str) -> Any:
        """Execute Databricks query returning single value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None

    def to_polars_lazyframe(self):
        """Convert Databricks table to Polars LazyFrame."""
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

        df_dict = {col: [row[i] for row in data] for i, col in enumerate(columns)}
        return pl.DataFrame(df_dict).lazy()

    def validate_connection(self) -> bool:
        """Validate Databricks connection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Databricks-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> dict[str, Any]:
        """Get detailed table information."""
        query = f"DESCRIBE DETAIL {self.full_table_name}"
        results = self.execute_query(query)
        return results[0] if results else {}

    def get_table_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get table history (Delta Lake)."""
        query = f"DESCRIBE HISTORY {self.full_table_name} LIMIT {limit}"
        return self.execute_query(query)

    def get_table_properties(self) -> dict[str, str]:
        """Get table properties."""
        query = f"SHOW TBLPROPERTIES {self.full_table_name}"
        results = self.execute_query(query)
        return {r["key"]: r["value"] for r in results}

    def optimize(self, zorder_by: list[str] | None = None) -> None:
        """Optimize Delta table.

        Args:
            zorder_by: Columns to Z-order by for better query performance.
        """
        query = f"OPTIMIZE {self.full_table_name}"
        if zorder_by:
            query += f" ZORDER BY ({', '.join(zorder_by)})"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            cursor.close()

    def vacuum(self, retention_hours: int = 168) -> None:
        """Vacuum Delta table to remove old files.

        Args:
            retention_hours: Hours to retain files (default: 168 = 7 days).
        """
        query = f"VACUUM {self.full_table_name} RETAIN {retention_hours} HOURS"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            cursor.close()

    def time_travel_query(
        self,
        version: int | None = None,
        timestamp: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query historical version of table.

        Args:
            version: Version number to query.
            timestamp: Timestamp to query (e.g., '2023-01-01').

        Returns:
            Query results from historical version.
        """
        if version is not None:
            query = f"SELECT * FROM {self.full_table_name} VERSION AS OF {version}"
        elif timestamp is not None:
            query = f"SELECT * FROM {self.full_table_name} TIMESTAMP AS OF '{timestamp}'"
        else:
            raise DataSourceError("Either version or timestamp must be provided")

        if self._config.max_rows:
            query += f" LIMIT {self._config.max_rows}"

        return self.execute_query(query)

    def restore_table(self, version: int | None = None, timestamp: str | None = None) -> None:
        """Restore table to a previous version.

        Args:
            version: Version number to restore to.
            timestamp: Timestamp to restore to.
        """
        if version is not None:
            query = f"RESTORE TABLE {self.full_table_name} TO VERSION AS OF {version}"
        elif timestamp is not None:
            query = f"RESTORE TABLE {self.full_table_name} TO TIMESTAMP AS OF '{timestamp}'"
        else:
            raise DataSourceError("Either version or timestamp must be provided")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            cursor.close()

    def get_catalogs(self) -> list[str]:
        """List available catalogs."""
        results = self.execute_query("SHOW CATALOGS")
        return [r["catalog"] for r in results]

    def get_schemas(self, catalog: str | None = None) -> list[str]:
        """List schemas in a catalog."""
        cat = catalog or self._catalog
        if cat:
            query = f"SHOW SCHEMAS IN `{cat}`"
        else:
            query = "SHOW SCHEMAS"
        results = self.execute_query(query)
        return [r["databaseName"] for r in results]

    def get_tables(self, schema: str | None = None, catalog: str | None = None) -> list[str]:
        """List tables in a schema."""
        cat = catalog or self._catalog
        sch = schema or self._schema

        if cat:
            query = f"SHOW TABLES IN `{cat}`.`{sch}`"
        else:
            query = f"SHOW TABLES IN `{sch}`"

        results = self.execute_query(query)
        return [r["tableName"] for r in results]

    @classmethod
    def from_env(
        cls,
        table: str,
        schema: str = "default",
        catalog: str | None = None,
        env_prefix: str = "DATABRICKS",
    ) -> "DatabricksDataSource":
        """Create data source from environment variables.

        Reads: DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN, etc.

        Args:
            table: Table name.
            schema: Schema name.
            catalog: Catalog name.
            env_prefix: Environment variable prefix.

        Returns:
            DatabricksDataSource configured from environment.
        """
        creds = load_credentials_from_env(env_prefix)

        required = ["host"]
        missing = [k for k in required if k not in creds]
        if missing:
            raise DataSourceError(
                f"Missing required environment variables: "
                f"{[f'{env_prefix}_{k.upper()}' for k in missing]}"
            )

        # Check for HTTP_PATH in env
        import os
        http_path = creds.get("http_path") or os.environ.get(f"{env_prefix}_HTTP_PATH")
        if not http_path:
            raise DataSourceError(f"Missing {env_prefix}_HTTP_PATH environment variable")

        return cls(
            table=table,
            host=creds["host"],
            http_path=http_path,
            access_token=creds.get("token"),
            catalog=catalog,
            schema=schema,
        )


# Alias for convenience
DatabricksSQLDataSource = DatabricksDataSource
