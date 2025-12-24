"""Base classes for cloud data warehouse sources.

This module provides abstract base classes for cloud-based data warehouses
like BigQuery, Snowflake, Redshift, and Databricks.

These sources typically:
- Require specific authentication (OAuth, service accounts, etc.)
- Support warehouse-specific SQL dialects
- Have different connection patterns (REST API, JDBC, native drivers)
- May have compute/storage separation
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from truthound.datasources.sql.base import (
    BaseSQLDataSource,
    SQLDataSourceConfig,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Cloud DW Configuration
# =============================================================================


@dataclass
class CloudDWConfig(SQLDataSourceConfig):
    """Configuration for cloud data warehouse sources.

    Attributes:
        project: Project/Account identifier (BigQuery project, Snowflake account).
        warehouse: Compute warehouse name (Snowflake, Databricks).
        region: Cloud region for the service.
        role: Role to assume for access control.
        timeout: Query timeout in seconds.
        use_cache: Whether to use query result caching.
        credentials_path: Path to credentials file (service account JSON, etc.).
        credentials_dict: Credentials as dictionary (alternative to file).
    """

    project: str | None = None
    warehouse: str | None = None
    region: str | None = None
    role: str | None = None
    timeout: int = 300
    use_cache: bool = True
    credentials_path: str | None = None
    credentials_dict: dict[str, Any] | None = field(default_factory=dict)


# =============================================================================
# Abstract Cloud DW Base
# =============================================================================


class CloudDWDataSource(BaseSQLDataSource):
    """Abstract base class for cloud data warehouse sources.

    This class provides common functionality for cloud DW sources including:
    - Credential management
    - Query result caching
    - Warehouse-specific SQL dialect handling
    - Cost-aware query execution

    Subclasses must implement:
    - _create_connection(): Create connection to the DW
    - _fetch_schema(): Retrieve table schema
    - _get_row_count_query(): SQL for counting rows
    - _quote_identifier(): Quote identifiers per dialect
    """

    _config: CloudDWConfig  # Type hint for subclass config

    @classmethod
    def _default_config(cls) -> CloudDWConfig:
        return CloudDWConfig()

    @abstractmethod
    def _validate_credentials(self) -> bool:
        """Validate that credentials are properly configured.

        Returns:
            True if credentials are valid and usable.

        Raises:
            DataSourceConnectionError: If credentials are missing or invalid.
        """
        pass

    @abstractmethod
    def _get_cost_estimate(self, query: str) -> dict[str, Any] | None:
        """Estimate the cost of running a query (if supported).

        Args:
            query: SQL query to estimate.

        Returns:
            Dictionary with cost information or None if not supported.
            Example: {"bytes_processed": 1000000, "estimated_cost_usd": 0.005}
        """
        pass

    def execute_with_cost_check(
        self,
        query: str,
        max_bytes: int | None = None,
        max_cost_usd: float | None = None,
    ) -> list[dict[str, Any]]:
        """Execute query with optional cost/size limits.

        Args:
            query: SQL query to execute.
            max_bytes: Maximum bytes to process (abort if exceeded).
            max_cost_usd: Maximum estimated cost in USD.

        Returns:
            Query results as list of dictionaries.

        Raises:
            DataSourceError: If cost limits would be exceeded.
        """
        from truthound.datasources.base import DataSourceError

        if max_bytes is not None or max_cost_usd is not None:
            estimate = self._get_cost_estimate(query)
            if estimate:
                if max_bytes and estimate.get("bytes_processed", 0) > max_bytes:
                    raise DataSourceError(
                        f"Query would process {estimate['bytes_processed']:,} bytes, "
                        f"exceeding limit of {max_bytes:,} bytes"
                    )
                if max_cost_usd and estimate.get("estimated_cost_usd", 0) > max_cost_usd:
                    raise DataSourceError(
                        f"Query estimated cost ${estimate['estimated_cost_usd']:.4f} "
                        f"exceeds limit of ${max_cost_usd:.4f}"
                    )

        return self.execute_query(query)


# =============================================================================
# Authentication Helpers
# =============================================================================


def load_credentials_from_env(prefix: str) -> dict[str, str]:
    """Load credentials from environment variables with given prefix.

    Args:
        prefix: Environment variable prefix (e.g., "BIGQUERY", "SNOWFLAKE").

    Returns:
        Dictionary of credential key-value pairs.

    Example:
        >>> creds = load_credentials_from_env("SNOWFLAKE")
        # Reads SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, SNOWFLAKE_ACCOUNT, etc.
    """
    import os

    creds = {}
    prefix_upper = prefix.upper()

    common_keys = [
        "USER", "PASSWORD", "ACCOUNT", "PROJECT", "DATABASE",
        "WAREHOUSE", "ROLE", "SCHEMA", "REGION", "HOST",
        "PRIVATE_KEY_PATH", "PRIVATE_KEY", "TOKEN",
    ]

    for key in common_keys:
        env_var = f"{prefix_upper}_{key}"
        value = os.environ.get(env_var)
        if value:
            creds[key.lower()] = value

    return creds


def load_service_account_json(path: str) -> dict[str, Any]:
    """Load service account credentials from JSON file.

    Args:
        path: Path to service account JSON file.

    Returns:
        Parsed credentials dictionary.
    """
    import json
    from pathlib import Path

    with Path(path).open() as f:
        return json.load(f)
