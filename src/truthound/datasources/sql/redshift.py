"""Amazon Redshift data source implementation.

This module provides a data source for Amazon Redshift,
supporting both traditional authentication and IAM authentication.

Requires: pip install redshift-connector or psycopg2
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


def _check_redshift_available() -> tuple[str, Any]:
    """Check if Redshift driver is available.

    Returns:
        Tuple of (driver_name, driver_module).
    """
    # Try redshift-connector first (native driver)
    try:
        import redshift_connector
        return "redshift_connector", redshift_connector
    except ImportError:
        pass

    # Fall back to psycopg2 (PostgreSQL protocol compatible)
    try:
        import psycopg2
        return "psycopg2", psycopg2
    except ImportError:
        pass

    raise ImportError(
        "redshift-connector or psycopg2 is required for RedshiftDataSource. "
        "Install with: pip install redshift-connector  OR  pip install psycopg2-binary"
    )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RedshiftConfig(CloudDWConfig):
    """Configuration for Redshift data source.

    Attributes:
        host: Cluster endpoint.
        port: Cluster port (default: 5439).
        database: Database name.
        user: Username.
        password: Password.
        iam_auth: Use IAM authentication.
        cluster_identifier: Cluster identifier (for IAM auth).
        db_user: Database user (for IAM auth).
        access_key_id: AWS access key (for IAM auth).
        secret_access_key: AWS secret key (for IAM auth).
        session_token: AWS session token (for temporary credentials).
        ssl: Use SSL connection.
        ssl_mode: SSL mode (verify-ca, verify-full, etc.).
    """

    host: str | None = None
    port: int = 5439
    database: str | None = None
    user: str | None = None
    password: str | None = None
    iam_auth: bool = False
    cluster_identifier: str | None = None
    db_user: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    ssl: bool = True
    ssl_mode: str = "verify-ca"


# =============================================================================
# Redshift Data Source
# =============================================================================


class RedshiftDataSource(CloudDWDataSource):
    """Data source for Amazon Redshift.

    Supports:
    - Traditional username/password authentication
    - IAM authentication
    - Serverless Redshift
    - Both redshift-connector and psycopg2 drivers

    Example:
        >>> # Traditional authentication
        >>> source = RedshiftDataSource(
        ...     table="users",
        ...     host="cluster.abc123.us-east-1.redshift.amazonaws.com",
        ...     database="mydb",
        ...     user="admin",
        ...     password="password",
        ... )

        >>> # IAM authentication
        >>> source = RedshiftDataSource(
        ...     table="users",
        ...     host="cluster.abc123.us-east-1.redshift.amazonaws.com",
        ...     database="mydb",
        ...     cluster_identifier="my-cluster",
        ...     db_user="admin",
        ...     iam_auth=True,
        ... )

        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
    """

    source_type = "redshift"

    def __init__(
        self,
        table: str,
        host: str,
        database: str,
        user: str | None = None,
        password: str | None = None,
        port: int = 5439,
        schema: str = "public",
        iam_auth: bool = False,
        cluster_identifier: str | None = None,
        db_user: str | None = None,
        config: RedshiftConfig | None = None,
    ) -> None:
        """Initialize Redshift data source.

        Args:
            table: Table name.
            host: Cluster endpoint.
            database: Database name.
            user: Username.
            password: Password.
            port: Cluster port.
            schema: Schema name.
            iam_auth: Use IAM authentication.
            cluster_identifier: Cluster ID (for IAM).
            db_user: Database user (for IAM).
            config: Optional configuration.
        """
        self._driver_name, self._driver = _check_redshift_available()

        if config is None:
            config = RedshiftConfig()

        config.host = host
        config.port = port
        config.database = database
        config.user = user
        config.password = password
        config.schema_name = schema
        config.iam_auth = iam_auth
        config.cluster_identifier = cluster_identifier
        config.db_user = db_user
        config.name = config.name or f"{host}/{database}.{schema}.{table}"

        self._schema = schema
        self._iam_auth = iam_auth

        super().__init__(table=table, config=config)

    @classmethod
    def _default_config(cls) -> RedshiftConfig:
        return RedshiftConfig()

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f'"{self._schema}"."{self._table}"'

    def _validate_credentials(self) -> bool:
        """Validate Redshift credentials."""
        try:
            conn = self._create_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            raise DataSourceConnectionError(
                source_type="redshift",
                details=f"Failed to authenticate: {e}",
            )

    def _create_connection(self) -> Any:
        """Create Redshift connection."""
        if self._driver_name == "redshift_connector":
            return self._create_native_connection()
        else:
            return self._create_psycopg2_connection()

    def _create_native_connection(self) -> Any:
        """Create connection using redshift-connector."""
        import redshift_connector

        conn_params = {
            "host": self._config.host,
            "port": self._config.port,
            "database": self._config.database,
            "ssl": self._config.ssl,
        }

        if self._iam_auth:
            conn_params["iam"] = True
            conn_params["cluster_identifier"] = self._config.cluster_identifier
            conn_params["db_user"] = self._config.db_user or self._config.user

            if self._config.access_key_id:
                conn_params["access_key_id"] = self._config.access_key_id
                conn_params["secret_access_key"] = self._config.secret_access_key
                if self._config.session_token:
                    conn_params["session_token"] = self._config.session_token
        else:
            conn_params["user"] = self._config.user
            conn_params["password"] = self._config.password

        return redshift_connector.connect(**conn_params)

    def _create_psycopg2_connection(self) -> Any:
        """Create connection using psycopg2."""
        import psycopg2

        conn_params = {
            "host": self._config.host,
            "port": self._config.port,
            "dbname": self._config.database,
            "user": self._config.user,
            "password": self._config.password,
        }

        if self._config.ssl:
            conn_params["sslmode"] = self._config.ssl_mode

        return psycopg2.connect(**conn_params)

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from Redshift."""
        query = f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = '{self._schema}'
            AND table_name = '{self._table}'
            ORDER BY ordinal_position
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.close()
            return [(row[0], row[1]) for row in result]

    def _get_row_count_query(self) -> str:
        """Get Redshift row count query."""
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote Redshift identifier with double quotes."""
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _get_cost_estimate(self, query: str) -> dict[str, Any] | None:
        """Redshift doesn't have direct cost estimation."""
        # Could use EXPLAIN to get scan info
        return None

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute Redshift query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            return [dict(zip(columns, row)) for row in results]

    def execute_scalar(self, query: str) -> Any:
        """Execute Redshift query returning single value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else None

    def to_polars_lazyframe(self):
        """Convert Redshift table to Polars LazyFrame."""
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
        """Validate Redshift connection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Redshift-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> dict[str, Any]:
        """Get detailed table information."""
        query = f"""
            SELECT
                "schema" AS table_schema,
                "table" AS table_name,
                diststyle,
                sortkey1,
                size AS size_mb,
                pct_used,
                unsorted,
                stats_off,
                tbl_rows
            FROM svv_table_info
            WHERE "schema" = '{self._schema}'
            AND "table" = '{self._table}'
        """
        results = self.execute_query(query)
        return results[0] if results else {}

    def get_dist_style(self) -> str | None:
        """Get table distribution style."""
        info = self.get_table_info()
        return info.get("diststyle")

    def get_sort_keys(self) -> list[str]:
        """Get table sort keys."""
        query = f"""
            SELECT column_name
            FROM svv_table_info ti
            JOIN information_schema.columns c
                ON ti."table" = c.table_name
                AND ti."schema" = c.table_schema
            WHERE ti."schema" = '{self._schema}'
            AND ti."table" = '{self._table}'
            AND c.ordinal_position <= ti.sortkey_num
            ORDER BY c.ordinal_position
        """
        results = self.execute_query(query)
        return [r["column_name"] for r in results]

    def analyze(self) -> None:
        """Run ANALYZE on the table."""
        query = f"ANALYZE {self.full_table_name}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()

    def vacuum(self, full: bool = False, sort_only: bool = False) -> None:
        """Run VACUUM on the table.

        Args:
            full: Perform full vacuum.
            sort_only: Only sort, don't reclaim space.
        """
        vacuum_type = ""
        if full:
            vacuum_type = "FULL"
        elif sort_only:
            vacuum_type = "SORT ONLY"

        query = f"VACUUM {vacuum_type} {self.full_table_name}"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()

    def unload_to_s3(
        self,
        s3_path: str,
        iam_role: str,
        format: str = "PARQUET",
        partition_by: list[str] | None = None,
    ) -> None:
        """Unload table to S3.

        Args:
            s3_path: S3 destination path.
            iam_role: IAM role ARN for S3 access.
            format: Output format (PARQUET, CSV, JSON).
            partition_by: Columns to partition by.
        """
        partition_clause = ""
        if partition_by:
            partition_clause = f"PARTITION BY ({', '.join(partition_by)})"

        query = f"""
            UNLOAD ('SELECT * FROM {self.full_table_name}')
            TO '{s3_path}'
            IAM_ROLE '{iam_role}'
            FORMAT AS {format}
            {partition_clause}
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()

    @classmethod
    def from_env(
        cls,
        table: str,
        schema: str = "public",
        env_prefix: str = "REDSHIFT",
    ) -> "RedshiftDataSource":
        """Create data source from environment variables.

        Reads: REDSHIFT_HOST, REDSHIFT_DATABASE, REDSHIFT_USER, REDSHIFT_PASSWORD, etc.

        Args:
            table: Table name.
            schema: Schema name.
            env_prefix: Environment variable prefix.

        Returns:
            RedshiftDataSource configured from environment.
        """
        creds = load_credentials_from_env(env_prefix)

        required = ["host", "database"]
        missing = [k for k in required if k not in creds]
        if missing:
            raise DataSourceError(
                f"Missing required environment variables: "
                f"{[f'{env_prefix}_{k.upper()}' for k in missing]}"
            )

        return cls(
            table=table,
            host=creds["host"],
            database=creds["database"],
            user=creds.get("user"),
            password=creds.get("password"),
            schema=schema,
        )
