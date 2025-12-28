"""Azure Blob Storage store backend.

This module provides a store implementation that persists data to Azure Blob Storage.
Requires the azure-storage-blob package.

Install with: pip install truthound[azure]
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

# Lazy import to avoid ImportError when azure-storage-blob is not installed
try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.core.exceptions import ResourceNotFoundError, AzureError

    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False
    BlobServiceClient = None  # type: ignore
    ContentSettings = None  # type: ignore
    ResourceNotFoundError = Exception  # type: ignore
    AzureError = Exception  # type: ignore

if TYPE_CHECKING:
    from truthound.stores.backends._protocols import (
        AzureBlobServiceClientProtocol,
        AzureContainerClientProtocol,
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


def _require_azure() -> None:
    """Check if azure-storage-blob is available."""
    if not HAS_AZURE:
        raise ImportError(
            "azure-storage-blob is required for AzureBlobStore. "
            "Install with: pip install truthound[azure]"
        )


@dataclass
class AzureBlobConfig(StoreConfig):
    """Configuration for Azure Blob store.

    Attributes:
        container: Azure Blob container name.
        prefix: Blob name prefix for all stored objects.
        connection_string: Azure Storage connection string.
        account_url: Azure Storage account URL (alternative to connection_string).
        account_name: Azure Storage account name.
        account_key: Azure Storage account key.
        sas_token: SAS token for authentication.
        use_compression: Whether to compress stored objects.
        content_type: Content type for stored blobs.
        access_tier: Access tier for blobs (Hot, Cool, Archive).
        metadata: Additional metadata to include with stored blobs.
    """

    container: str = ""
    prefix: str = "truthound/"
    connection_string: str | None = None
    account_url: str | None = None
    account_name: str | None = None
    account_key: str | None = None
    sas_token: str | None = None
    use_compression: bool = True
    content_type: str = "application/json"
    access_tier: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def get_full_prefix(self) -> str:
        """Get the full blob name prefix including namespace."""
        parts = [p for p in [self.prefix.rstrip("/"), self.namespace] if p]
        return "/".join(parts) + "/"


class AzureBlobStore(ValidationStore["AzureBlobConfig"]):
    """Azure Blob Storage validation store.

    Stores validation results as JSON objects in an Azure Blob container.

    Example:
        >>> store = AzureBlobStore(
        ...     container="my-validation-container",
        ...     connection_string="DefaultEndpointsProtocol=https;...",
        ...     prefix="validations/",
        ... )
        >>> result = ValidationResult.from_report(report, "customers.csv")
        >>> run_id = store.save(result)

    Authentication Methods:
        1. Connection string (recommended for development):
            AzureBlobStore(container="...", connection_string="...")

        2. Account URL with SAS token:
            AzureBlobStore(container="...", account_url="https://...", sas_token="...")

        3. Account name and key:
            AzureBlobStore(container="...", account_name="...", account_key="...")

        4. Default Azure credentials (for managed identity):
            AzureBlobStore(container="...", account_url="https://...")
    """

    def __init__(
        self,
        container: str,
        prefix: str = "truthound/",
        connection_string: str | None = None,
        account_url: str | None = None,
        account_name: str | None = None,
        account_key: str | None = None,
        sas_token: str | None = None,
        namespace: str = "default",
        compression: bool = True,
        access_tier: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Azure Blob store.

        Args:
            container: Azure Blob container name.
            prefix: Blob name prefix for stored objects.
            connection_string: Azure Storage connection string.
            account_url: Azure Storage account URL.
            account_name: Azure Storage account name.
            account_key: Azure Storage account key.
            sas_token: SAS token for authentication.
            namespace: Namespace for organizing data.
            compression: Whether to compress stored objects.
            access_tier: Access tier for blobs (Hot, Cool, Archive).
            **kwargs: Additional configuration options.

        Note:
            Dependency check is handled by the factory. Direct instantiation
            requires azure-storage-blob to be installed.
        """
        config = AzureBlobConfig(
            container=container,
            prefix=prefix,
            connection_string=connection_string,
            account_url=account_url,
            account_name=account_name,
            account_key=account_key,
            sas_token=sas_token,
            namespace=namespace,
            use_compression=compression,
            access_tier=access_tier,
            **{k: v for k, v in kwargs.items() if hasattr(AzureBlobConfig, k)},
        )
        super().__init__(config)
        self._client: AzureBlobServiceClientProtocol | None = None
        self._container_client: AzureContainerClientProtocol | None = None
        self._index: dict[str, dict[str, Any]] = {}

    @classmethod
    def _default_config(cls) -> "AzureBlobConfig":
        """Create default configuration."""
        return AzureBlobConfig()

    def _do_initialize(self) -> None:
        """Initialize the Azure Blob client and load index."""
        try:
            # Create client based on available credentials
            if self._config.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self._config.connection_string
                )
            elif self._config.account_url:
                credential = None
                if self._config.sas_token:
                    credential = self._config.sas_token
                elif self._config.account_key:
                    credential = self._config.account_key
                else:
                    # Try DefaultAzureCredential for managed identity
                    try:
                        from azure.identity import DefaultAzureCredential

                        credential = DefaultAzureCredential()
                    except ImportError:
                        pass

                self._client = BlobServiceClient(
                    account_url=self._config.account_url,
                    credential=credential,
                )
            elif self._config.account_name:
                account_url = f"https://{self._config.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=self._config.account_key,
                )
            else:
                raise StoreConnectionError(
                    "Azure Blob",
                    "No connection credentials provided. Provide connection_string, "
                    "account_url, or account_name.",
                )

            # Get container client
            self._container_client = self._client.get_container_client(
                self._config.container
            )

            # Check if container exists
            if not self._container_client.exists():
                raise StoreConnectionError(
                    "Azure Blob",
                    f"Container not found: {self._config.container}",
                )

            # Load index
            self._load_index()

        except AzureError as e:
            raise StoreConnectionError("Azure Blob", str(e))

    def _load_index(self) -> None:
        """Load the index from Azure Blob."""
        index_name = f"{self._config.get_full_prefix()}_index.json"
        blob_client = self._container_client.get_blob_client(index_name)

        try:
            if blob_client.exists():
                downloader = blob_client.download_blob()
                content = downloader.readall()
                self._index = json.loads(content.decode("utf-8"))
            else:
                self._index = {}
        except ResourceNotFoundError:
            self._index = {}
        except AzureError as e:
            raise StoreReadError(f"Failed to load index: {e}")

    def _save_index(self) -> None:
        """Save the index to Azure Blob."""
        index_name = f"{self._config.get_full_prefix()}_index.json"
        blob_client = self._container_client.get_blob_client(index_name)
        content = json.dumps(self._index, indent=2, default=str)

        try:
            content_settings = ContentSettings(content_type="application/json")
            blob_client.upload_blob(
                content.encode("utf-8"),
                overwrite=True,
                content_settings=content_settings,
            )
        except AzureError as e:
            raise StoreWriteError(f"Failed to save index: {e}")

    def _get_blob_name(self, item_id: str) -> str:
        """Get the Azure Blob name for an item."""
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
        """Save a validation result to Azure Blob.

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
            blob_client = self._container_client.get_blob_client(blob_name)

            # Set content settings
            content_encoding = "gzip" if self._config.use_compression else None
            content_settings = ContentSettings(
                content_type=self._config.content_type,
                content_encoding=content_encoding,
            )

            # Prepare upload kwargs
            upload_kwargs: dict[str, Any] = {
                "overwrite": True,
                "content_settings": content_settings,
            }

            # Add metadata
            metadata = {
                "data_asset": item.data_asset,
                "run_time": item.run_time.isoformat(),
                "status": item.status.value,
                **self._config.metadata,
            }
            upload_kwargs["metadata"] = metadata

            blob_client.upload_blob(content, **upload_kwargs)

            # Set access tier if specified
            if self._config.access_tier:
                try:
                    blob_client.set_standard_blob_tier(self._config.access_tier)
                except Exception:
                    # Tier setting may fail for certain storage types
                    pass

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

        except AzureError as e:
            raise StoreWriteError(f"Failed to write to Azure Blob: {e}")

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result from Azure Blob.

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
        blob_client = self._container_client.get_blob_client(blob_name)

        try:
            if not blob_client.exists():
                raise StoreNotFoundError("ValidationResult", item_id)

            downloader = blob_client.download_blob()
            content = downloader.readall()
            data = self._deserialize(content)
            return ValidationResult.from_dict(data)

        except ResourceNotFoundError:
            raise StoreNotFoundError("ValidationResult", item_id)
        except AzureError as e:
            raise StoreReadError(f"Failed to read from Azure Blob: {e}")

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
        blob_client = self._container_client.get_blob_client(blob_name)

        try:
            return blob_client.exists()
        except AzureError:
            return False

    def delete(self, item_id: str) -> bool:
        """Delete a validation result from Azure Blob.

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
        blob_client = self._container_client.get_blob_client(blob_name)

        try:
            blob_client.delete_blob()

            if item_id in self._index:
                del self._index[item_id]
                self._save_index()

            return True

        except ResourceNotFoundError:
            return False
        except AzureError as e:
            raise StoreWriteError(f"Failed to delete from Azure Blob: {e}")

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
            ids = ids[query.offset:]
        if query.limit:
            ids = ids[: query.limit]

        return ids

    def query(self, query: StoreQuery) -> list[ValidationResult]:
        """Query validation results from Azure Blob.

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
        """Close the Azure Blob client."""
        self._client = None
        self._container_client = None

    def set_access_tier(self, item_id: str, tier: str) -> bool:
        """Set the access tier for a stored result.

        Args:
            item_id: The run ID of the result.
            tier: The access tier (Hot, Cool, Archive).

        Returns:
            True if successful.

        Raises:
            StoreWriteError: If setting tier fails.
        """
        self.initialize()

        blob_name = self._get_blob_name(item_id)
        blob_client = self._container_client.get_blob_client(blob_name)

        try:
            blob_client.set_standard_blob_tier(tier)
            return True
        except AzureError as e:
            raise StoreWriteError(f"Failed to set access tier: {e}")
