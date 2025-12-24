"""AWS S3 store backend.

This module provides a store implementation that persists data to AWS S3.
Requires the boto3 package.

Install with: pip install truthound[s3]
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

# Lazy import to avoid ImportError when boto3 is not installed
try:
    import boto3
    from botocore.exceptions import ClientError

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore

if TYPE_CHECKING:
    from truthound.stores.backends._protocols import S3ClientProtocol

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


def _require_boto3() -> None:
    """Check if boto3 is available."""
    if not HAS_BOTO3:
        raise ImportError(
            "boto3 is required for S3Store. "
            "Install with: pip install truthound[s3]"
        )


@dataclass
class S3Config(StoreConfig):
    """Configuration for S3 store.

    Attributes:
        bucket: S3 bucket name.
        prefix: Key prefix for all stored objects.
        region: AWS region name.
        endpoint_url: Custom endpoint URL (for S3-compatible services).
        use_compression: Whether to compress stored objects.
        storage_class: S3 storage class for objects.
        server_side_encryption: Server-side encryption setting.
        kms_key_id: KMS key ID for encryption (if using aws:kms).
        tags: Tags to apply to stored objects.
    """

    bucket: str = ""
    prefix: str = "truthound/"
    region: str | None = None
    endpoint_url: str | None = None
    use_compression: bool = True
    storage_class: str = "STANDARD"
    server_side_encryption: str | None = None
    kms_key_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def get_full_prefix(self) -> str:
        """Get the full key prefix including namespace."""
        parts = [p for p in [self.prefix.rstrip("/"), self.namespace] if p]
        return "/".join(parts) + "/"


class S3Store(ValidationStore["S3Config"]):
    """AWS S3 validation store.

    Stores validation results as JSON objects in an S3 bucket.

    Example:
        >>> store = S3Store(
        ...     bucket="my-validation-bucket",
        ...     prefix="validations/",
        ...     region="us-east-1"
        ... )
        >>> result = ValidationResult.from_report(report, "customers.csv")
        >>> run_id = store.save(result)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "truthound/",
        region: str | None = None,
        endpoint_url: str | None = None,
        namespace: str = "default",
        compression: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the S3 store.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix for stored objects.
            region: AWS region name.
            endpoint_url: Custom endpoint URL (for MinIO, LocalStack, etc.).
            namespace: Namespace for organizing data.
            compression: Whether to compress stored objects.
            **kwargs: Additional configuration options.

        Note:
            Dependency check is handled by the factory. Direct instantiation
            requires boto3 to be installed.
        """
        config = S3Config(
            bucket=bucket,
            prefix=prefix,
            region=region,
            endpoint_url=endpoint_url,
            namespace=namespace,
            use_compression=compression,
            **{k: v for k, v in kwargs.items() if hasattr(S3Config, k)},
        )
        super().__init__(config)
        self._client: S3ClientProtocol | None = None
        self._index: dict[str, dict[str, Any]] = {}

    @classmethod
    def _default_config(cls) -> "S3Config":
        """Create default configuration."""
        return S3Config()

    def _do_initialize(self) -> None:
        """Initialize the S3 client and load index."""
        try:
            session_kwargs: dict[str, Any] = {}
            if self._config.region:
                session_kwargs["region_name"] = self._config.region

            client_kwargs: dict[str, Any] = {}
            if self._config.endpoint_url:
                client_kwargs["endpoint_url"] = self._config.endpoint_url

            self._client = boto3.client("s3", **session_kwargs, **client_kwargs)

            # Test connection by checking if bucket exists
            self._client.head_bucket(Bucket=self._config.bucket)

            # Load index
            self._load_index()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code in ("404", "NoSuchBucket"):
                raise StoreConnectionError("S3", f"Bucket not found: {self._config.bucket}")
            elif error_code in ("403", "AccessDenied"):
                raise StoreConnectionError("S3", f"Access denied to bucket: {self._config.bucket}")
            else:
                raise StoreConnectionError("S3", str(e))

    def _load_index(self) -> None:
        """Load the index from S3."""
        index_key = f"{self._config.get_full_prefix()}_index.json"

        try:
            response = self._client.get_object(
                Bucket=self._config.bucket,
                Key=index_key,
            )
            content = response["Body"].read()
            self._index = json.loads(content.decode("utf-8"))
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                self._index = {}
            else:
                raise StoreReadError(f"Failed to load index: {e}")

    def _save_index(self) -> None:
        """Save the index to S3."""
        index_key = f"{self._config.get_full_prefix()}_index.json"
        content = json.dumps(self._index, indent=2, default=str)

        try:
            self._client.put_object(
                Bucket=self._config.bucket,
                Key=index_key,
                Body=content.encode("utf-8"),
                ContentType="application/json",
            )
        except ClientError as e:
            raise StoreWriteError(f"Failed to save index: {e}")

    def _get_key(self, item_id: str) -> str:
        """Get the S3 key for an item."""
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
        """Save a validation result to S3.

        Args:
            item: The validation result to save.

        Returns:
            The run ID of the saved result.

        Raises:
            StoreWriteError: If saving fails.
        """
        self.initialize()

        item_id = item.run_id
        key = self._get_key(item_id)
        content = self._serialize(item.to_dict())

        try:
            put_kwargs: dict[str, Any] = {
                "Bucket": self._config.bucket,
                "Key": key,
                "Body": content,
                "ContentType": "application/json",
                "StorageClass": self._config.storage_class,
            }

            if self._config.use_compression:
                put_kwargs["ContentEncoding"] = "gzip"

            if self._config.server_side_encryption:
                put_kwargs["ServerSideEncryption"] = self._config.server_side_encryption
                if self._config.kms_key_id:
                    put_kwargs["SSEKMSKeyId"] = self._config.kms_key_id

            if self._config.tags:
                # S3 tags must be URL-encoded
                from urllib.parse import quote

                tag_str = "&".join(f"{quote(k)}={quote(v)}" for k, v in self._config.tags.items())
                put_kwargs["Tagging"] = tag_str

            self._client.put_object(**put_kwargs)

            # Update index
            self._index[item_id] = {
                "data_asset": item.data_asset,
                "run_time": item.run_time.isoformat(),
                "status": item.status.value,
                "key": key,
                "tags": item.tags,
            }
            self._save_index()

            return item_id

        except ClientError as e:
            raise StoreWriteError(f"Failed to write to S3: {e}")

    def get(self, item_id: str) -> ValidationResult:
        """Retrieve a validation result from S3.

        Args:
            item_id: The run ID of the result to retrieve.

        Returns:
            The validation result.

        Raises:
            StoreNotFoundError: If the result doesn't exist.
            StoreReadError: If reading fails.
        """
        self.initialize()

        key = self._get_key(item_id)

        try:
            response = self._client.get_object(
                Bucket=self._config.bucket,
                Key=key,
            )
            content = response["Body"].read()
            data = self._deserialize(content)
            return ValidationResult.from_dict(data)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                raise StoreNotFoundError("ValidationResult", item_id)
            raise StoreReadError(f"Failed to read from S3: {e}")

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

        key = self._get_key(item_id)

        try:
            self._client.head_object(
                Bucket=self._config.bucket,
                Key=key,
            )
            return True
        except ClientError:
            return False

    def delete(self, item_id: str) -> bool:
        """Delete a validation result from S3.

        Args:
            item_id: The run ID of the result to delete.

        Returns:
            True if the result was deleted, False if it didn't exist.

        Raises:
            StoreWriteError: If deletion fails.
        """
        self.initialize()

        key = self._get_key(item_id)

        if not self.exists(item_id):
            return False

        try:
            self._client.delete_object(
                Bucket=self._config.bucket,
                Key=key,
            )

            if item_id in self._index:
                del self._index[item_id]
                self._save_index()

            return True

        except ClientError as e:
            raise StoreWriteError(f"Failed to delete from S3: {e}")

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
        """Query validation results from S3.

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
        """Close the S3 client."""
        self._client = None
