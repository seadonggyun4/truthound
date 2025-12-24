"""Google BigQuery data source implementation.

This module provides a data source for Google BigQuery,
supporting both service account and application default credentials.

Requires: pip install google-cloud-bigquery db-dtypes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from truthound.datasources.sql.cloud_base import (
    CloudDWConfig,
    CloudDWDataSource,
    load_service_account_json,
)
from truthound.datasources.base import (
    DataSourceConnectionError,
    DataSourceError,
)

if TYPE_CHECKING:
    from google.cloud.bigquery import Client


def _check_bigquery_available() -> None:
    """Check if BigQuery client is available."""
    try:
        from google.cloud import bigquery  # noqa: F401
    except ImportError:
        raise ImportError(
            "google-cloud-bigquery is required for BigQueryDataSource. "
            "Install with: pip install google-cloud-bigquery db-dtypes"
        )


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BigQueryConfig(CloudDWConfig):
    """Configuration for BigQuery data source.

    Attributes:
        project: GCP project ID.
        dataset: BigQuery dataset name.
        location: BigQuery location (e.g., 'US', 'EU').
        use_legacy_sql: Whether to use legacy SQL (default: False).
        maximum_bytes_billed: Maximum bytes billed per query (cost control).
        job_timeout: Timeout for query jobs in seconds.
    """

    dataset: str | None = None
    location: str | None = None
    use_legacy_sql: bool = False
    maximum_bytes_billed: int | None = None
    job_timeout: int = 300


# =============================================================================
# BigQuery Data Source
# =============================================================================


class BigQueryDataSource(CloudDWDataSource):
    """Data source for Google BigQuery.

    Supports authentication via:
    - Service account JSON file
    - Application Default Credentials (ADC)
    - Explicit credentials dictionary

    Example:
        >>> # Using service account
        >>> source = BigQueryDataSource(
        ...     table="users",
        ...     project="my-project",
        ...     dataset="my_dataset",
        ...     credentials_path="/path/to/service-account.json",
        ... )

        >>> # Using Application Default Credentials
        >>> source = BigQueryDataSource(
        ...     table="users",
        ...     project="my-project",
        ...     dataset="my_dataset",
        ... )

        >>> engine = source.get_execution_engine()
        >>> print(engine.count_rows())
    """

    source_type = "bigquery"

    def __init__(
        self,
        table: str,
        project: str,
        dataset: str,
        credentials_path: str | None = None,
        location: str | None = None,
        config: BigQueryConfig | None = None,
    ) -> None:
        """Initialize BigQuery data source.

        Args:
            table: Table name.
            project: GCP project ID.
            dataset: BigQuery dataset name.
            credentials_path: Path to service account JSON (optional).
            location: BigQuery location (optional).
            config: Optional configuration.
        """
        _check_bigquery_available()

        if config is None:
            config = BigQueryConfig()

        config.project = project
        config.credentials_path = credentials_path
        config.name = config.name or f"{project}.{dataset}.{table}"

        self._project = project
        self._dataset = dataset
        self._location = location

        super().__init__(table=table, config=config)

        # Initialize client
        self._client: Client | None = None
        self._validate_credentials()

    @classmethod
    def _default_config(cls) -> BigQueryConfig:
        return BigQueryConfig()

    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f"`{self._project}.{self._dataset}.{self._table}`"

    @property
    def dataset(self) -> str:
        """Get dataset name."""
        return self._dataset

    @property
    def project(self) -> str:
        """Get project ID."""
        return self._project

    def _validate_credentials(self) -> bool:
        """Validate BigQuery credentials."""
        try:
            client = self._get_client()
            # Test connection by getting dataset info
            client.get_dataset(f"{self._project}.{self._dataset}")
            return True
        except Exception as e:
            raise DataSourceConnectionError(
                source_type="bigquery",
                details=f"Failed to authenticate: {e}",
            )

    def _get_client(self) -> "Client":
        """Get or create BigQuery client."""
        if self._client is None:
            from google.cloud import bigquery
            from google.oauth2 import service_account

            if self._config.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self._config.credentials_path
                )
                self._client = bigquery.Client(
                    project=self._project,
                    credentials=credentials,
                    location=self._location,
                )
            elif self._config.credentials_dict:
                credentials = service_account.Credentials.from_service_account_info(
                    self._config.credentials_dict
                )
                self._client = bigquery.Client(
                    project=self._project,
                    credentials=credentials,
                    location=self._location,
                )
            else:
                # Use Application Default Credentials
                self._client = bigquery.Client(
                    project=self._project,
                    location=self._location,
                )

        return self._client

    def _create_connection(self) -> Any:
        """BigQuery uses client-based access, not traditional connections."""
        return self._get_client()

    def _fetch_schema(self) -> list[tuple[str, str]]:
        """Fetch schema from BigQuery."""
        client = self._get_client()
        table_ref = f"{self._project}.{self._dataset}.{self._table}"
        table = client.get_table(table_ref)

        return [(field.name, field.field_type) for field in table.schema]

    def _get_row_count_query(self) -> str:
        """Get BigQuery row count query."""
        return f"SELECT COUNT(*) FROM {self.full_table_name}"

    def _quote_identifier(self, identifier: str) -> str:
        """Quote BigQuery identifier with backticks."""
        escaped = identifier.replace("`", "\\`")
        return f"`{escaped}`"

    def _get_cost_estimate(self, query: str) -> dict[str, Any] | None:
        """Estimate query cost using dry run."""
        from google.cloud import bigquery

        client = self._get_client()
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

        try:
            query_job = client.query(query, job_config=job_config)
            bytes_processed = query_job.total_bytes_processed

            # BigQuery pricing: $5 per TB (on-demand)
            cost_per_byte = 5.0 / (1024**4)
            estimated_cost = bytes_processed * cost_per_byte

            return {
                "bytes_processed": bytes_processed,
                "estimated_cost_usd": estimated_cost,
            }
        except Exception:
            return None

    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute BigQuery query."""
        from google.cloud import bigquery

        client = self._get_client()

        job_config = bigquery.QueryJobConfig(
            use_legacy_sql=self._config.use_legacy_sql,
            use_query_cache=self._config.use_cache,
        )

        if self._config.maximum_bytes_billed:
            job_config.maximum_bytes_billed = self._config.maximum_bytes_billed

        query_job = client.query(query, job_config=job_config)
        results = query_job.result(timeout=self._config.job_timeout)

        return [dict(row) for row in results]

    def execute_scalar(self, query: str) -> Any:
        """Execute BigQuery query returning single value."""
        results = self.execute_query(query)
        if results and len(results) > 0:
            first_row = results[0]
            return next(iter(first_row.values()))
        return None

    def to_polars_lazyframe(self):
        """Convert BigQuery table to Polars LazyFrame."""
        import polars as pl

        # Fetch data as pandas first (BigQuery has native pandas support)
        client = self._get_client()
        query = f"SELECT * FROM {self.full_table_name}"

        if self._config.max_rows:
            query += f" LIMIT {self._config.max_rows}"

        df = client.query(query).to_dataframe()
        return pl.from_pandas(df).lazy()

    def validate_connection(self) -> bool:
        """Validate BigQuery connection."""
        try:
            client = self._get_client()
            client.get_dataset(f"{self._project}.{self._dataset}")
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # BigQuery-specific Methods
    # -------------------------------------------------------------------------

    def get_table_info(self) -> dict[str, Any]:
        """Get detailed table information."""
        client = self._get_client()
        table_ref = f"{self._project}.{self._dataset}.{self._table}"
        table = client.get_table(table_ref)

        return {
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes,
            "created": table.created,
            "modified": table.modified,
            "description": table.description,
            "labels": dict(table.labels) if table.labels else {},
            "partitioning": str(table.time_partitioning) if table.time_partitioning else None,
            "clustering": table.clustering_fields,
        }

    def get_partition_info(self) -> list[dict[str, Any]]:
        """Get partition information for partitioned tables."""
        query = f"""
            SELECT
                partition_id,
                total_rows,
                total_logical_bytes,
                last_modified_time
            FROM `{self._project}.{self._dataset}.INFORMATION_SCHEMA.PARTITIONS`
            WHERE table_name = '{self._table}'
            ORDER BY partition_id
        """
        return self.execute_query(query)

    def export_to_gcs(self, destination_uri: str, format: str = "PARQUET") -> str:
        """Export table to Google Cloud Storage.

        Args:
            destination_uri: GCS URI (e.g., 'gs://bucket/path/file-*.parquet').
            format: Export format ('PARQUET', 'CSV', 'JSON', 'AVRO').

        Returns:
            Job ID of the export job.
        """
        from google.cloud import bigquery

        client = self._get_client()
        table_ref = f"{self._project}.{self._dataset}.{self._table}"

        job_config = bigquery.ExtractJobConfig()
        job_config.destination_format = getattr(
            bigquery.DestinationFormat, format.upper()
        )

        extract_job = client.extract_table(
            table_ref,
            destination_uri,
            job_config=job_config,
        )
        extract_job.result()  # Wait for completion

        return extract_job.job_id

    @classmethod
    def from_query(
        cls,
        query: str,
        project: str,
        dataset: str,
        table_name: str = "_query_result",
        credentials_path: str | None = None,
    ) -> "BigQueryDataSource":
        """Create data source from a BigQuery query.

        This creates a temporary view for the query results.

        Args:
            query: SQL query to execute.
            project: GCP project ID.
            dataset: BigQuery dataset name.
            table_name: Name for the query result (used internally).
            credentials_path: Path to service account JSON.

        Returns:
            BigQueryDataSource wrapping the query results.
        """
        _check_bigquery_available()
        from google.cloud import bigquery
        from google.oauth2 import service_account

        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            client = bigquery.Client(project=project, credentials=credentials)
        else:
            client = bigquery.Client(project=project)

        # Create a view for the query
        view_id = f"{project}.{dataset}.{table_name}"
        view = bigquery.Table(view_id)
        view.view_query = query

        try:
            client.delete_table(view_id, not_found_ok=True)
            client.create_table(view)
        except Exception as e:
            raise DataSourceError(f"Failed to create view from query: {e}")

        return cls(
            table=table_name,
            project=project,
            dataset=dataset,
            credentials_path=credentials_path,
        )
