"""Google Cloud Storage store backend.

This module provides a store implementation that persists data to Google Cloud Storage.
Requires the google-cloud-storage package.

Install with: pip install truthound[gcs]
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

# Lazy import to avoid ImportError when google-cloud-storage is not installed
try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound

    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    storage = None  # type: ignore
    NotFound = Exception  # type: ignore

if TYPE_CHECKING:
    from truthound.stores.backends._protocols import (
        GCSBucketProtocol,
        GCSClientProtocol,
    )

from truthound.stores.base import (
    StoreConfig,
    StoreConnectionError,
    StoreNotFoundError,
    StoreQuery,
    StoreReadError,
    StoreWriteError,
    ValidationStore,
)
from truthound.stores.results import ValidationResult


def _require_gcs() -> None:
    """Check if google-cloud-storage is available."""
    if not HAS_GCS:
        raise ImportError(
            "google-cloud-storage is required for GCSStore. "
            "Install with: pip install truthound[gcs]"
        )


@dataclass
class GCSConfig(StoreConfig):
    """Configuration for GCS store.

    Attributes:
        bucket: GCS bucket name.
        prefix: Object name prefix for all stored objects.
        project: Google Cloud project ID.
        credentials_path: Path to service account credentials JSON.
        use_compression: Whether to compress stored objects.
    """

    bucket: str = ""
    prefix: str = "truthound/"
    project: str | None = None
    credentials_path: str | None = None
    use_compression: bool = True

    def get_full_prefix(self) -> str:
        """Get the full key prefix including namespace."""
        parts = [p for p in [self.prefix.rstrip("/"), self.namespace] if p]
        return "/".join(parts) + "/"


class GCSStore(ValidationStore["GCSConfig"]):
    """Google Cloud Storage validation store.

    Stores validation results as JSON objects in a GCS bucket.

    Example:
        >>> store = GCSStore(
        ...     bucket="my-validation-bucket",
        ...     prefix="validations/",
        ...     project="my-gcp-project"
        ... )
        >>> result = ValidationResult.from_report(report, "customers.csv")
        >>> run_id = store.save(result)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "truthound/",
        project: str | None = None,
        credentials_path: str | None = None,
        namespace: str = "default",
        compression: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the GCS store.

        Args:
            bucket: GCS bucket name.
            prefix: Object name prefix for stored objects.
            project: Google Cloud project ID.
            credentials_path: Path to service account credentials JSON.
            namespace: Namespace for organizing data.
            compression: Whether to compress stored objects.
            **kwargs: Additional configuration options.

        Note:
            Dependency check is handled by the factory. Direct instantiation
            requires google-cloud-storage to be installed.
        """
        config = GCSConfig(
            bucket=bucket,
            prefix=prefix,
            project=project,
            credentials_path=credentials_path,
            namespace=namespace,
            use_compression=compression,
            **{k: v for k, v in kwargs.items() if hasattr(GCSConfig, k)},
        )
        super().__init__(config)
        self._client: GCSClientProtocol | None = None
        self._bucket: GCSBucketProtocol | None = None
        self._index: dict[str, dict[str, Any]] = {}

    @classmethod
    def _default_config(cls) -> "GCSConfig":
        """Create default configuration."""
        return GCSConfig()

    def _do_initialize(self) -> None:
        """Initialize the GCS client and load index."""
        try:
            client_kwargs: dict[str, Any] = {}

            if self._config.project:
                client_kwargs["project"] = self._config.project

            if self._config.credentials_path:
                self._client = storage.Client.from_service_account_json(
                    self._config.credentials_path,
                    **client_kwargs,
                )
            else:
                self._client = storage.Client(**client_kwargs)

            # Get bucket
            self._bucket = self._client.bucket(self._config.bucket)

            # Check if bucket exists
            if not self._bucket.exists():
                raise StoreConnectionError("GCS", f"Bucket not found: {self._config.bucket}")

            # Load index
            self._load_index()

        except Exception as e:
            if "NotFound" in str(type(e).__name__):
                raise StoreConnectionError("GCS", f"Bucket not found: {self._config.bucket}")
            raise StoreConnectionError("GCS", str(e))

    def _load_index(self) -> None:
        """Load the index from GCS."""
        index_name = f"{self._config.get_full_prefix()}_index.json"
        blob = self._bucket.blob(index_name)

        try:
            content = blob.download_as_bytes()
            self._index = json.loads(content.decode("utf-8"))
        except NotFound:
            self._index = {}
        except Exception as e:
            # Handle other GCS errors gracefully
            self._index = {}

    def _save_index(self) -> None:
        """Save the index to GCS."""
        index_name = f"{self._config.get_full_prefix()}_index.json"
        blob = self._bucket.blob(index_name)
        content = json.dumps(self._index, indent=2, default=str)

        try:
            blob.upload_from_string(content, content_type="application/json")
        except Exception as e:
            raise StoreWriteError(f"Failed to save index: {e}")

    def _get_blob_name(self, item_id: str) -> str:
        """Get the GCS blob name for an item."""
        ext = ".json.gz" if self._config.use_compression else ".json"
        return f"{self._config.get_full_prefix()}results/{item_id}{ext}"

    def _serialize(self, data: dict[str, Any]) -> bytes:
        """Serialize data to bytes."""
        json_str = json.dumps(data, indent=2, default=str)
        content = json_str.encode("utf-8")

        if self._config.use_compression:
            content = gzip.compress(content)

        return content

    def _deserialize(self, content: bytes) -> dict[str, Any]:
        """Deserialize bytes to data."""
        if self._config.use_compression:
            content = gzip.decompress(content)

        return json.loads(content.decode("utf-8"))

    def save(self, item: ValidationResult) -> str:
        """Save a validation result to GCS.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.

        Raises:
            StoreWriteError: If saving fails.
        """
        self.initialize()

        item_id = item.run_id
        blob_name = self._get_blob_name(item_id)
        content = self._serialize(item.to_dict())

        try:
            blob = self._bucket.blob(blob_name)

            content_type = "application/json"
            if self._config.use_compression:
                blob.content_encoding = "gzip"

            blob.upload_from_string(content, content_type=content_type)

            # Update index
            self._index[item_id] = {
                "data_asset": item.data_asset,
                "run_time": item.run_time.isoformat(),
                "status": item.status.value,
                "blob_name": blob_name,
                "tags": item.tags,
            }
            self._save_index()

            return item_id

        except Exception as e:
            raise StoreWriteError(f"Failed to write to GCS: {e}")

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result from GCS.

        Args:
            item_id: The run ID of the result to retrieve.

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist.
            StoreReadError: If reading fails.
        """
        self.initialize()

        blob_name = self._get_blob_name(item_id)
        blob = self._bucket.blob(blob_name)

        try:
            content = blob.download_as_bytes()
            data = self._deserialize(content)
            return ValidationResult.from_dict(data)

        except NotFound:
            raise StoreNotFoundError("ValidationResult", item_id)
        except Exception as e:
            raise StoreReadError(f"Failed to read from GCS: {e}")

    def exists(self, item_id: str) -> bool:
        """Check if a validation result exists.

        Args:
            item_id: The run ID to check.

        Returns:
            True if the result exists.
        """
        self.initialize()

        if item_id in self._index:
            return True

        blob_name = self._get_blob_name(item_id)
        blob = self._bucket.blob(blob_name)

        return blob.exists()

    def delete(self, item_id: str) -> bool:
        """Delete a validation result from GCS.

        Args:
            item_id: The run ID of the result to delete.

        Returns:
            True if the result was deleted, False if it didn't exist.

        Raises:
            StoreWriteError: If deletion fails.
        """
        self.initialize()

        if not self.exists(item_id):
            return False

        blob_name = self._get_blob_name(item_id)
        blob = self._bucket.blob(blob_name)

        try:
            blob.delete()

            if item_id in self._index:
                del self._index[item_id]
                self._save_index()

            return True

        except NotFound:
            return False
        except Exception as e:
            raise StoreWriteError(f"Failed to delete from GCS: {e}")

    def list_ids(self, query: StoreQuery | None = None) -> list[str]:
        """List validation result IDs matching the query.

        Args:
            query: Optional query to filter results.

        Returns:
            List of matching run IDs.
        """
        self.initialize()

        if not query:
            return list(self._index.keys())

        # Filter by query
        matching_ids: list[tuple[str, datetime]] = []

        for item_id, meta in self._index.items():
            if query.matches(meta):
                run_time = datetime.fromisoformat(meta["run_time"])
                matching_ids.append((item_id, run_time))

        # Sort
        reverse = not query.ascending
        matching_ids.sort(key=lambda x: x[1], reverse=reverse)

        # Apply offset and limit
        ids = [item_id for item_id, _ in matching_ids]

        if query.offset:
            ids = ids[query.offset :]
        if query.limit:
            ids = ids[: query.limit]

        return ids

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results from GCS.

        Args:
            query: Query parameters for filtering.

        Returns:
            List of matching validation results.
        """
        ids = self.list_ids(query)
        results: list[ValidationResult] = []

        for item_id in ids:
            try:
                result = self.get(item_id)
                results.append(result)
            except (StoreNotFoundError, StoreReadError):
                continue

        return results

    def close(self) -> None:
        """Close the GCS client."""
        self._client = None
        self._bucket = None
