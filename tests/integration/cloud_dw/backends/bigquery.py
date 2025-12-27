"""BigQuery test backend implementation.

This module provides the test backend for Google BigQuery integration testing.

Features:
    - Service account and application default credentials
    - Dry-run query validation
    - Cost estimation before execution
    - Dataset lifecycle management
    - Bytes processed tracking

Usage:
    >>> from tests.integration.cloud_dw.backends import BigQueryTestBackend, BigQueryCredentials
    >>>
    >>> credentials = BigQueryCredentials(
    ...     project="my-project",
    ...     location="US",
    ... )
    >>> backend = BigQueryTestBackend(credentials)
    >>>
    >>> with backend:
    ...     dataset = backend.create_test_dataset()
    ...     table = backend.create_test_table(
    ...         dataset,
    ...         "users",
    ...         {"id": "INT64", "name": "STRING"},
    ...         [{"id": 1, "name": "Alice"}],
    ...     )
    ...     result = backend.execute_query(
    ...         f"SELECT * FROM {backend.get_full_table_name(dataset.name, table.name)}"
    ...     )
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar

from tests.integration.cloud_dw.base import (
    BaseCredentials,
    CloudDWTestBackend,
    IntegrationTestConfig,
    TestDataset,
)

if TYPE_CHECKING:
    from google.cloud import bigquery
    from truthound.datasources.sql.bigquery import BigQueryDataSource

logger = logging.getLogger(__name__)


# =============================================================================
# Credentials
# =============================================================================


@dataclass
class BigQueryCredentials(BaseCredentials):
    """Credentials for BigQuery.

    Supports:
        - Service account JSON file (via credentials_path or GOOGLE_APPLICATION_CREDENTIALS)
        - Application Default Credentials (when running on GCP)
        - Explicit service account JSON content

    Attributes:
        project: GCP project ID.
        location: BigQuery location (e.g., "US", "EU").
        credentials_path: Path to service account JSON file.
        credentials_json: Service account JSON content (for CI/CD secrets).
        dataset: Default dataset for tests.
    """

    project: str = ""
    location: str = "US"
    credentials_path: str | None = None
    credentials_json: str | None = None
    dataset: str | None = None

    def __post_init__(self) -> None:
        # Try to get project from environment if not set
        if not self.project:
            self.project = os.getenv("BIGQUERY_PROJECT", "")

        # Try to get credentials path from environment
        if not self.credentials_path and not self.credentials_json:
            self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    def validate(self) -> bool:
        """Validate that credentials are properly configured."""
        if not self.project:
            logger.error("BigQuery project not specified")
            return False

        # Either credentials_path, credentials_json, or ADC should be available
        if self.credentials_path and not os.path.exists(self.credentials_path):
            logger.error(f"Credentials file not found: {self.credentials_path}")
            return False

        return True

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters."""
        params = {
            "project": self.project,
            "location": self.location,
        }
        if self.credentials_path:
            params["credentials_path"] = self.credentials_path
        if self.credentials_json:
            params["credentials_json"] = "***"  # Don't expose
        if self.dataset:
            params["dataset"] = self.dataset
        return params

    @property
    def is_service_account(self) -> bool:
        """Check if using service account credentials."""
        return bool(self.credentials_path or self.credentials_json)

    def get_credentials_object(self) -> Any:
        """Get the Google credentials object."""
        if self.credentials_json:
            from google.oauth2 import service_account
            info = json.loads(self.credentials_json)
            return service_account.Credentials.from_service_account_info(info)
        elif self.credentials_path:
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
        else:
            # Use Application Default Credentials
            import google.auth
            credentials, _ = google.auth.default()
            return credentials


# =============================================================================
# Backend Implementation
# =============================================================================


class BigQueryTestBackend(CloudDWTestBackend[BigQueryCredentials]):
    """Test backend for Google BigQuery.

    This backend provides:
        - Full query execution with cost tracking
        - Dry-run query validation
        - Cost estimation before execution
        - Dataset and table management
        - Stale resource cleanup
    """

    platform_name: ClassVar[str] = "bigquery"
    supports_dry_run: ClassVar[bool] = True
    supports_cost_estimation: ClassVar[bool] = True
    default_quote_char: ClassVar[str] = "`"

    # BigQuery pricing (as of 2024)
    BYTES_PER_TB: ClassVar[int] = 1024**4
    COST_PER_TB_USD: ClassVar[float] = 5.0  # On-demand pricing

    def __init__(
        self,
        credentials: BigQueryCredentials,
        config: IntegrationTestConfig | None = None,
    ) -> None:
        super().__init__(credentials, config)
        self._client: "bigquery.Client | None" = None
        self._last_job: Any = None

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def _create_connection(self) -> "bigquery.Client":
        """Create BigQuery client."""
        from google.cloud import bigquery

        client_options = {
            "project": self.credentials.project,
            "location": self.credentials.location,
        }

        if self.credentials.is_service_account:
            client_options["credentials"] = self.credentials.get_credentials_object()

        self._client = bigquery.Client(**client_options)
        return self._client

    def _close_connection(self) -> None:
        """Close BigQuery client."""
        if self._client:
            self._client.close()
            self._client = None

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def _execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute query on BigQuery."""
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig()

        if params:
            job_config.query_parameters = [
                bigquery.ScalarQueryParameter(name, "STRING", value)
                for name, value in params.items()
            ]

        # Set timeout
        job_config.job_timeout_ms = self.config.timeout_seconds * 1000

        query_job = self._client.query(query, job_config=job_config)
        self._last_job = query_job

        # Wait for completion
        result = query_job.result()

        # Convert to list of dicts
        return [dict(row) for row in result]

    def _dry_run_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a dry run of the query."""
        from google.cloud import bigquery

        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

        if params:
            job_config.query_parameters = [
                bigquery.ScalarQueryParameter(name, "STRING", value)
                for name, value in params.items()
            ]

        query_job = self._client.query(query, job_config=job_config)
        self._last_job = query_job

        logger.info(
            f"[BigQuery] Dry run: {query_job.total_bytes_processed:,} bytes would be processed"
        )
        return []

    def _estimate_query_cost(self, query: str) -> dict[str, Any] | None:
        """Estimate query cost using dry run."""
        try:
            self._dry_run_query(query)
            if self._last_job:
                bytes_processed = self._last_job.total_bytes_processed or 0
                cost_usd = (bytes_processed / self.BYTES_PER_TB) * self.COST_PER_TB_USD
                return {
                    "bytes": bytes_processed,
                    "cost_usd": cost_usd,
                }
        except Exception as e:
            logger.warning(f"[BigQuery] Cost estimation failed: {e}")
        return None

    def _get_last_query_bytes(self) -> int:
        """Get bytes processed by the last query."""
        if self._last_job and hasattr(self._last_job, "total_bytes_processed"):
            return self._last_job.total_bytes_processed or 0
        return 0

    def _get_last_query_cost(self) -> float:
        """Get estimated cost of the last query."""
        bytes_processed = self._get_last_query_bytes()
        return (bytes_processed / self.BYTES_PER_TB) * self.COST_PER_TB_USD

    # -------------------------------------------------------------------------
    # Dataset Management
    # -------------------------------------------------------------------------

    def _create_dataset(self, name: str) -> None:
        """Create a BigQuery dataset."""
        from google.cloud import bigquery

        dataset_ref = bigquery.DatasetReference(self.credentials.project, name)
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = self.credentials.location

        # Set labels for tracking
        dataset.labels = {
            "created_by": "truthound_integration_test",
            "created_at": datetime.utcnow().strftime("%Y%m%d"),
        }

        # Set default expiration for tables (24 hours)
        from datetime import timedelta
        dataset.default_table_expiration_ms = int(
            timedelta(hours=self.config.cleanup_after_hours).total_seconds() * 1000
        )

        self._client.create_dataset(dataset, exists_ok=True)

    def _drop_dataset(self, name: str) -> None:
        """Drop a BigQuery dataset."""
        from google.cloud import bigquery

        dataset_ref = bigquery.DatasetReference(self.credentials.project, name)
        self._client.delete_dataset(
            dataset_ref,
            delete_contents=True,
            not_found_ok=True,
        )

    def _create_table(
        self,
        dataset: str,
        table: str,
        schema: dict[str, str],
    ) -> None:
        """Create a BigQuery table."""
        from google.cloud import bigquery

        table_ref = self._client.dataset(dataset).table(table)

        bq_schema = [
            bigquery.SchemaField(name, dtype)
            for name, dtype in schema.items()
        ]

        bq_table = bigquery.Table(table_ref, schema=bq_schema)
        self._client.create_table(bq_table, exists_ok=True)

    def _insert_data(
        self,
        dataset: str,
        table: str,
        data: list[dict[str, Any]],
    ) -> None:
        """Insert data into a BigQuery table."""
        table_ref = self._client.dataset(dataset).table(table)

        errors = self._client.insert_rows_json(table_ref, data)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")

    def _find_stale_datasets(self, max_hours: int) -> list[str]:
        """Find stale test datasets."""
        from datetime import timedelta

        stale = []
        cutoff = datetime.utcnow() - timedelta(hours=max_hours)

        # List datasets with our prefix
        for dataset in self._client.list_datasets():
            if dataset.dataset_id.startswith(self.config.test_dataset_prefix):
                # Get full dataset to check labels
                full_dataset = self._client.get_dataset(dataset.reference)
                created_at_str = full_dataset.labels.get("created_at", "")

                if created_at_str:
                    try:
                        created_at = datetime.strptime(created_at_str, "%Y%m%d")
                        if created_at < cutoff:
                            stale.append(dataset.dataset_id)
                    except ValueError:
                        # If we can't parse, include in stale list for safety
                        stale.append(dataset.dataset_id)

        return stale

    # -------------------------------------------------------------------------
    # Schema Operations
    # -------------------------------------------------------------------------

    def get_table_schema(
        self,
        dataset: str,
        table: str,
    ) -> dict[str, str]:
        """Get the schema of a BigQuery table."""
        table_ref = self._client.dataset(dataset).table(table)
        bq_table = self._client.get_table(table_ref)

        return {
            field.name: field.field_type
            for field in bq_table.schema
        }

    def get_row_count(
        self,
        dataset: str,
        table: str,
    ) -> int:
        """Get the row count of a BigQuery table."""
        table_ref = self._client.dataset(dataset).table(table)
        bq_table = self._client.get_table(table_ref)
        return bq_table.num_rows or 0

    # -------------------------------------------------------------------------
    # Truthound Integration
    # -------------------------------------------------------------------------

    def create_datasource(
        self,
        dataset: str,
        table: str,
    ) -> "BigQueryDataSource":
        """Create a Truthound BigQuery DataSource."""
        from truthound.datasources.sql.bigquery import (
            BigQueryDataSource,
            BigQueryConfig,
        )

        config = BigQueryConfig(
            project=self.credentials.project,
            location=self.credentials.location,
            credentials_path=self.credentials.credentials_path,
        )

        return BigQueryDataSource(
            table=table,
            dataset=dataset,
            config=config,
        )

    # -------------------------------------------------------------------------
    # BigQuery-specific methods
    # -------------------------------------------------------------------------

    def get_full_table_name(self, dataset: str, table: str) -> str:
        """Get the fully qualified table name for BigQuery."""
        project = self.quote_identifier(self.credentials.project)
        dataset_q = self.quote_identifier(dataset)
        table_q = self.quote_identifier(table)
        return f"{project}.{dataset_q}.{table_q}"

    def get_job_statistics(self) -> dict[str, Any] | None:
        """Get statistics from the last executed job."""
        if not self._last_job:
            return None

        return {
            "job_id": self._last_job.job_id,
            "state": self._last_job.state,
            "total_bytes_processed": self._last_job.total_bytes_processed,
            "total_bytes_billed": getattr(self._last_job, "total_bytes_billed", None),
            "cache_hit": getattr(self._last_job, "cache_hit", None),
            "num_dml_affected_rows": getattr(
                self._last_job, "num_dml_affected_rows", None
            ),
            "slot_millis": getattr(self._last_job, "slot_millis", None),
        }
