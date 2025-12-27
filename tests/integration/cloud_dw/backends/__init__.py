"""Cloud DW test backend implementations.

This package contains concrete implementations of CloudDWTestBackend
for each supported cloud data warehouse platform.

Each backend provides:
    - Platform-specific connection handling
    - Credential management
    - SQL dialect translations
    - Cost estimation (where supported)
    - Dry-run capabilities (where supported)

Available Backends:
    - BigQueryTestBackend: Google BigQuery
    - SnowflakeTestBackend: Snowflake
    - RedshiftTestBackend: AWS Redshift
    - DatabricksTestBackend: Databricks SQL
    - OracleTestBackend: Oracle Database
    - SQLServerTestBackend: Microsoft SQL Server
"""

from tests.integration.cloud_dw.backends.bigquery import (
    BigQueryTestBackend,
    BigQueryCredentials,
)
from tests.integration.cloud_dw.backends.snowflake import (
    SnowflakeTestBackend,
    SnowflakeCredentials,
)
from tests.integration.cloud_dw.backends.redshift import (
    RedshiftTestBackend,
    RedshiftCredentials,
)
from tests.integration.cloud_dw.backends.databricks import (
    DatabricksTestBackend,
    DatabricksCredentials,
)
from tests.integration.cloud_dw.backends.registry import (
    BackendRegistry,
    get_backend,
    register_backend,
    get_available_backends,
)

__all__ = [
    # Backends
    "BigQueryTestBackend",
    "SnowflakeTestBackend",
    "RedshiftTestBackend",
    "DatabricksTestBackend",
    # Credentials
    "BigQueryCredentials",
    "SnowflakeCredentials",
    "RedshiftCredentials",
    "DatabricksCredentials",
    # Registry
    "BackendRegistry",
    "get_backend",
    "register_backend",
    "get_available_backends",
]
