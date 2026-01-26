"""SQL data source implementations.

This subpackage provides data sources for SQL databases,
enabling validation directly on database tables with SQL pushdown
optimization where possible.

Supported databases:

Traditional RDBMS:
- SQLite (built-in)
- DuckDB (requires: pip install duckdb)
- PostgreSQL (requires: pip install psycopg2-binary)
- MySQL (requires: pip install pymysql)
- Oracle (requires: pip install oracledb)
- SQL Server (requires: pip install pyodbc or pymssql)

Cloud Data Warehouses:
- BigQuery (requires: pip install google-cloud-bigquery db-dtypes)
- Snowflake (requires: pip install snowflake-connector-python)
- Redshift (requires: pip install redshift-connector)
- Databricks (requires: pip install databricks-sql-connector)
"""

from truthound.datasources.sql.base import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
    SQLConnectionPool,
)
from truthound.datasources.sql.cloud_base import (
    CloudDWConfig,
    CloudDWDataSource,
    load_credentials_from_env,
    load_service_account_json,
)

# Core SQL databases (always available)
from truthound.datasources.sql.sqlite import SQLiteDataSource

# Optional imports with graceful fallback
try:
    from truthound.datasources.sql.duckdb import DuckDBDataSource, DuckDBDataSourceConfig
except ImportError:
    DuckDBDataSource = None  # type: ignore
    DuckDBDataSourceConfig = None  # type: ignore

try:
    from truthound.datasources.sql.postgresql import PostgreSQLDataSource
except ImportError:
    PostgreSQLDataSource = None  # type: ignore

try:
    from truthound.datasources.sql.mysql import MySQLDataSource
except ImportError:
    MySQLDataSource = None  # type: ignore

try:
    from truthound.datasources.sql.oracle import OracleDataSource, OracleConfig
except ImportError:
    OracleDataSource = None  # type: ignore
    OracleConfig = None  # type: ignore

try:
    from truthound.datasources.sql.sqlserver import SQLServerDataSource, SQLServerConfig
except ImportError:
    SQLServerDataSource = None  # type: ignore
    SQLServerConfig = None  # type: ignore

try:
    from truthound.datasources.sql.bigquery import BigQueryDataSource, BigQueryConfig
except ImportError:
    BigQueryDataSource = None  # type: ignore
    BigQueryConfig = None  # type: ignore

try:
    from truthound.datasources.sql.snowflake import SnowflakeDataSource, SnowflakeConfig
except ImportError:
    SnowflakeDataSource = None  # type: ignore
    SnowflakeConfig = None  # type: ignore

try:
    from truthound.datasources.sql.redshift import RedshiftDataSource, RedshiftConfig
except ImportError:
    RedshiftDataSource = None  # type: ignore
    RedshiftConfig = None  # type: ignore

try:
    from truthound.datasources.sql.databricks import (
        DatabricksDataSource,
        DatabricksConfig,
        DatabricksSQLDataSource,
    )
except ImportError:
    DatabricksDataSource = None  # type: ignore
    DatabricksConfig = None  # type: ignore
    DatabricksSQLDataSource = None  # type: ignore


def get_available_sources() -> dict[str, type | None]:
    """Get dictionary of available SQL data sources.

    Returns:
        Dictionary mapping source name to class (or None if not available).
    """
    return {
        "sqlite": SQLiteDataSource,
        "duckdb": DuckDBDataSource,
        "postgresql": PostgreSQLDataSource,
        "mysql": MySQLDataSource,
        "oracle": OracleDataSource,
        "sqlserver": SQLServerDataSource,
        "bigquery": BigQueryDataSource,
        "snowflake": SnowflakeDataSource,
        "redshift": RedshiftDataSource,
        "databricks": DatabricksDataSource,
    }


def check_source_available(source_type: str) -> bool:
    """Check if a specific SQL source type is available.

    Args:
        source_type: Source type name.

    Returns:
        True if the source is available.
    """
    sources = get_available_sources()
    return sources.get(source_type) is not None


__all__ = [
    # Base classes
    "BaseSQLDataSource",
    "SQLDataSourceConfig",
    "SQLConnectionPool",
    "CloudDWConfig",
    "CloudDWDataSource",
    # Utilities
    "load_credentials_from_env",
    "load_service_account_json",
    "get_available_sources",
    "check_source_available",
    # Core implementations (always available)
    "SQLiteDataSource",
    # Traditional RDBMS (optional)
    "DuckDBDataSource",
    "DuckDBDataSourceConfig",
    "PostgreSQLDataSource",
    "MySQLDataSource",
    "OracleDataSource",
    "OracleConfig",
    "SQLServerDataSource",
    "SQLServerConfig",
    # Cloud Data Warehouses (optional)
    "BigQueryDataSource",
    "BigQueryConfig",
    "SnowflakeDataSource",
    "SnowflakeConfig",
    "RedshiftDataSource",
    "RedshiftConfig",
    "DatabricksDataSource",
    "DatabricksConfig",
    "DatabricksSQLDataSource",
]
