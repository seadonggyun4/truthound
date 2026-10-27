"""AWS S3 store backend.

This module provides a store implementation that persists data to AWS S3.
Requires the boto3 package.

Install with: pip install truthound[s3]
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from threading import RLock
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

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

from truthound.stores.backends._object_store import ObjectValidationStore
from truthound.stores.base import (
    StoreConfig,
    StoreConnectionError,
)
from truthound.stores.compression import get_compressor


def _require_boto3() -> None:
    """Check if boto3 is available."""
    if not HAS_BOTO3:
        raise ImportError("boto3 is required for S3Store. Install with: pip install truthound[s3]")


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

    max_object_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.bucket, str) or not self.bucket:
            raise ValueError("S3 bucket must be a non-empty string")
        for value in (self.prefix, self.namespace):
            if (
                not isinstance(value, str)
                or "\\" in value
                or value.startswith("/")
                or any(part in {".", ".."} for part in value.split("/"))
            ):
                raise ValueError("S3 prefix and namespace must be relative object paths")
        if type(self.use_compression) is not bool or not isinstance(self.metadata, dict):
            raise ValueError("Invalid S3 compression flag or metadata")
        if type(self.max_object_bytes) is not int or self.max_object_bytes < 1:
            raise ValueError("max_object_bytes must be a positive integer")
        if self.serialization_format not in {"json", "yaml"}:
            raise ValueError("S3 records require JSON or safe YAML")
        if not isinstance(self.tags, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in self.tags.items()
        ):
            raise ValueError("S3 tags must map strings to strings")
        if self.compression is not None:
            self.compression = get_compressor(self.compression).algorithm.value
            self.use_compression = self.compression != "none"
        elif self.use_compression:
            self.compression = "gzip"

    def get_full_prefix(self) -> str:
        parts = [p.strip("/") for p in (self.prefix, self.namespace) if p.strip("/")]
        return "/".join(parts) + ("/" if parts else "")


class S3Store(ObjectValidationStore[S3Config]):
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

    _provider_errors = (ClientError,)

    @classmethod
    def _default_config(cls) -> S3Config:
        raise ValueError("S3Store requires an explicit bucket configuration")

    def __init__(
        self,
        bucket: str = "",
        prefix: str = "truthound/",
        region: str | None = None,
        endpoint_url: str | None = None,
        namespace: str = "default",
        compression: bool | str = True,
        *,
        config: S3Config | None = None,
        **kwargs: Any,
    ) -> None:
        _require_boto3()
        if config is not None:
            if not isinstance(config, S3Config):
                raise TypeError("config must be S3Config")
            if (
                bucket
                or prefix != "truthound/"
                or region is not None
                or endpoint_url is not None
                or namespace != "default"
                or compression is not True
                or kwargs
            ):
                raise ValueError("Select a config object or individual S3 settings")
            selected = replace(deepcopy(config))
        else:
            unknown = set(kwargs) - {item.name for item in fields(S3Config)}
            if unknown:
                raise TypeError("Unknown S3 store setting")
            options = deepcopy(kwargs)
            if type(compression) is bool:
                options.setdefault("use_compression", compression)
            elif isinstance(compression, str):
                options["compression"] = compression
            else:
                raise ValueError("Invalid S3 compression setting")
            selected = S3Config(
                bucket=bucket,
                prefix=prefix,
                region=region,
                endpoint_url=endpoint_url,
                namespace=namespace,
                **options,
            )
        super().__init__(selected)
        self._client: S3ClientProtocol | None = None
        self._index = {}
        self._index_etag = None
        self._needs_reconcile = False
        self._io_guard = None
        self._lock = RLock()

    def _connect(self):
        self._guard()
        if self._client is not None:
            return
        arguments = {}
        if self._config.region:
            arguments["region_name"] = self._config.region
        if self._config.endpoint_url:
            arguments["endpoint_url"] = self._config.endpoint_url
        client = boto3.client("s3", **arguments)
        try:
            self._guard()
            client.head_bucket(Bucket=self._config.bucket)
        except BaseException as original:
            try:
                client.close()
            except BaseException as cleanup:
                self._client = client
                raise BaseExceptionGroup(
                    "S3 connection and cleanup failed", [original, cleanup]
                ) from None
            if isinstance(original, ClientError):
                code = original.response.get("Error", {}).get("Code")
                message = (
                    "Bucket not found"
                    if code in {"404", "NoSuchBucket"}
                    else "Access denied"
                    if code in {"403", "AccessDenied"}
                    else "Connection failed"
                )
                raise StoreConnectionError("S3", message) from None
            raise
        self._client = client

    @staticmethod
    def _missing(error):
        return isinstance(error, ClientError) and error.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }

    @staticmethod
    def _etag(response):
        etag = response.get("ETag")
        if not isinstance(etag, str) or not etag:
            raise RuntimeError("S3 publication has no confirmed ETag receipt")
        return etag

    def _read_bytes(self, key, *, etag=None):
        self._guard()
        response = self._client.get_object(
            Bucket=self._config.bucket, Key=key, **({"IfMatch": etag} if etag else {})
        )
        body = response["Body"]
        try:
            if response.get("ContentLength", 0) > self._config.max_object_bytes:
                raise ValueError("S3 object exceeds configured bound")
            content = body.read(self._config.max_object_bytes + 1)
        finally:
            body.close()
        self._guard()
        if len(content) > self._config.max_object_bytes:
            raise ValueError("S3 object exceeds configured bound")
        if "ContentLength" in response and len(content) != response["ContentLength"]:
            raise ValueError("S3 object length differs from its receipt")
        observed = self._etag(response)
        if etag is not None and observed != etag:
            raise ValueError("S3 object receipt changed")
        return content, observed

    def close(self):
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except BaseException:
                    self._cleanup_uncertain = True
                    self._initialized = False
                    raise
                self._client = None
            self._cleanup_uncertain = False
            self._initialized = False

    def _head(self, key, *, etag=None):
        self._guard()
        response = self._client.head_object(
            Bucket=self._config.bucket, Key=key, **({"IfMatch": etag} if etag else {})
        )
        return self._etag(response)

    def _put(self, key, content, *, etag=None, is_index=False, item=None):
        arguments = {
            "Bucket": self._config.bucket,
            "Key": key,
            "Body": content,
            "ContentType": "application/json"
            if is_index or self._config.serialization_format == "json"
            else "application/yaml",
        }
        arguments.update({"IfMatch": etag} if etag else {"IfNoneMatch": "*"})
        if not is_index:
            arguments["StorageClass"] = self._config.storage_class
            if self._config.use_compression and self._config.compression == "gzip":
                arguments["ContentEncoding"] = "gzip"
            if self._config.server_side_encryption:
                arguments["ServerSideEncryption"] = self._config.server_side_encryption
                if self._config.kms_key_id:
                    arguments["SSEKMSKeyId"] = self._config.kms_key_id
            if self._config.tags:
                arguments["Tagging"] = urlencode(self._config.tags)
        self._guard()
        return self._etag(self._client.put_object(**arguments))

    def _delete(self, key, etag):
        self._guard()
        response = self._client.delete_object(Bucket=self._config.bucket, Key=key, IfMatch=etag)
        if response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 204:
            raise RuntimeError("S3 deletion has no confirmed receipt")

    def _keys(self, prefix):
        for page in self._client.get_paginator("list_objects_v2").paginate(
            Bucket=self._config.bucket, Prefix=prefix
        ):
            for row in page.get("Contents", []):
                yield row["Key"]
