"""Protocol definitions for optional dependency clients.

This module defines structural typing protocols for external library clients,
allowing type-safe code without requiring type stubs packages.

These protocols define only the methods actually used by Truthound,
providing minimal but sufficient type coverage.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# =============================================================================
# S3 Client Protocol (boto3)
# =============================================================================


class S3ResponseBody(Protocol):
    """Protocol for S3 response body stream."""

    def read(self, amt: int | None = None) -> bytes:
        """Read all bytes from the response body."""
        ...

    def close(self) -> None:
        """Release the response connection."""
        ...


class S3GetObjectResponse(Protocol):
    """Protocol for S3 GetObject response."""

    def __getitem__(self, key: str) -> S3ResponseBody:
        """Get response field by key."""
        ...


@runtime_checkable
class S3ClientProtocol(Protocol):
    """Protocol for boto3 S3 client.

    Defines the minimal interface used by S3Store.
    """

    def head_bucket(self, *, Bucket: str) -> dict[str, Any]:
        """Check if a bucket exists and is accessible."""
        ...

    def get_object(self, *, Bucket: str, Key: str, IfMatch: str = ...) -> dict[str, Any]:
        """Retrieve an object from S3."""
        ...

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str = ...,
        ContentEncoding: str = ...,
        StorageClass: str = ...,
        ServerSideEncryption: str = ...,
        SSEKMSKeyId: str = ...,
        Tagging: str = ...,
        IfMatch: str = ...,
        IfNoneMatch: str = ...,
    ) -> dict[str, Any]:
        """Upload an object to S3."""
        ...

    def head_object(self, *, Bucket: str, Key: str, IfMatch: str = ...) -> dict[str, Any]:
        """Check if an object exists."""
        ...

    def delete_object(self, *, Bucket: str, Key: str, IfMatch: str = ...) -> dict[str, Any]:
        """Delete an object from S3."""
        ...

    def get_paginator(self, operation_name: str) -> Any:
        """Return a paginator for bounded object inventory pages."""
        ...

    def close(self) -> None:
        """Release the owned native client."""
        ...


# =============================================================================
# GCS Client Protocol (google-cloud-storage)
# =============================================================================


@runtime_checkable
class GCSBlobProtocol(Protocol):
    """Protocol for GCS Blob object."""

    content_encoding: str | None
    generation: int | str | None
    size: int | None
    name: str

    def reload(self, *, if_generation_match: int = ...) -> None:
        """Read provider metadata under an optional generation condition."""
        ...

    def download_as_bytes(
        self, *, raw_download: bool = ..., if_generation_match: int = ...
    ) -> bytes:
        """Download blob contents as bytes."""
        ...

    def upload_from_string(
        self,
        data: str | bytes,
        content_type: str = ...,
        if_generation_match: int = ...,
    ) -> None:
        """Upload data to the blob."""
        ...

    def exists(self) -> bool:
        """Check if the blob exists."""
        ...

    def delete(self, *, if_generation_match: int = ...) -> None:
        """Delete the blob under its confirmed generation."""
        ...


@runtime_checkable
class GCSBucketProtocol(Protocol):
    """Protocol for GCS Bucket object."""

    def list_blobs(self, *, prefix: str = ...) -> Any:
        """Iterate the selected namespace inventory."""
        ...

    def blob(self, blob_name: str) -> GCSBlobProtocol:
        """Get a blob reference."""
        ...

    def exists(self) -> bool:
        """Check if the bucket exists."""
        ...


@runtime_checkable
class GCSClientProtocol(Protocol):
    """Protocol for GCS Client.

    Defines the minimal interface used by GCSStore.
    """

    def close(self) -> None:
        """Release the owned Google client."""
        ...

    def bucket(self, bucket_name: str) -> GCSBucketProtocol:
        """Get a bucket reference."""
        ...

    @classmethod
    def from_service_account_json(
        cls,
        json_credentials_path: str,
        **kwargs: Any,
    ) -> GCSClientProtocol:
        """Create client from service account credentials."""
        ...


# =============================================================================
# SQLAlchemy Protocol
# =============================================================================


@runtime_checkable
class SQLConnectionProtocol(Protocol):
    """Protocol for SQLAlchemy connection."""

    def execute(self, statement: Any) -> Any:
        """Execute a SQL statement."""
        ...

    def __enter__(self) -> SQLConnectionProtocol:
        """Enter context manager."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        ...


@runtime_checkable
class SQLEngineProtocol(Protocol):
    """Protocol for SQLAlchemy Engine.

    Defines the minimal interface used by DatabaseStore.
    """

    def connect(self) -> SQLConnectionProtocol:
        """Create a new connection."""
        ...

    def dispose(self) -> None:
        """Dispose of the connection pool."""
        ...


@runtime_checkable
class SQLSessionProtocol(Protocol):
    """Protocol for SQLAlchemy Session."""

    def query(self, *entities: Any) -> Any:
        """Create a query."""
        ...

    def add(self, instance: Any) -> None:
        """Add an object to the session."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def __enter__(self) -> SQLSessionProtocol:
        """Enter context manager."""
        ...

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        ...


class SessionFactoryProtocol(Protocol):
    """Protocol for SQLAlchemy session factory."""

    def __call__(self) -> SQLSessionProtocol:
        """Create a new session."""
        ...


# =============================================================================
# Jinja2 Protocol (for HTMLReporter)
# =============================================================================


@runtime_checkable
class Jinja2TemplateProtocol(Protocol):
    """Protocol for Jinja2 Template."""

    def render(self, **context: Any) -> str:
        """Render the template with given context."""
        ...


@runtime_checkable
class Jinja2EnvironmentProtocol(Protocol):
    """Protocol for Jinja2 Environment.

    Defines the minimal interface used by HTMLReporter.
    """

    def get_template(self, name: str) -> Jinja2TemplateProtocol:
        """Get a template by name."""
        ...


# =============================================================================
# Azure Blob Storage Protocol (azure-storage-blob)
# =============================================================================


class AzureBlobPropertiesProtocol(Protocol):
    """Protocol for Azure Blob Properties."""

    content_settings: Any
    last_modified: Any
    creation_time: Any
    etag: str | None
    content_length: int

    def get(self, key: str, default: Any = None) -> Any:
        """Read SDK property keys such as size, etag and blob_tier."""
        ...


@runtime_checkable
class AzureBlobClientProtocol(Protocol):
    """Protocol for Azure Blob Client.

    Defines the minimal interface used by AzureBlobStore.
    """

    def upload_blob(
        self,
        data: bytes | str,
        *,
        overwrite: bool = False,
        content_settings: Any = None,
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Upload data to the blob."""
        ...

    def download_blob(
        self, *, decompress: bool = ..., **kwargs: Any
    ) -> AzureStorageStreamDownloaderProtocol:
        """Download the blob content."""
        ...

    def exists(self) -> bool:
        """Check if the blob exists."""
        ...

    def delete_blob(self, **kwargs: Any) -> None:
        """Delete the blob."""
        ...

    def set_standard_blob_tier(self, tier: Any, **kwargs: Any) -> None:
        """Request a tier change under ETag conditions and inspect the native response."""
        ...

    def get_blob_properties(self, **kwargs: Any) -> AzureBlobPropertiesProtocol:
        """Get blob properties."""
        ...


class AzureStorageStreamDownloaderProtocol(Protocol):
    """Protocol for Azure Storage Stream Downloader."""

    def readall(self) -> bytes:
        """Read all bytes from the blob."""
        ...


@runtime_checkable
class AzureContainerClientProtocol(Protocol):
    """Protocol for Azure Container Client."""

    def get_blob_client(self, blob: str) -> AzureBlobClientProtocol:
        """Get a blob client."""
        ...

    def exists(self) -> bool:
        """Check if the container exists."""
        ...

    def list_blobs(
        self,
        name_starts_with: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """List blobs in the container."""
        ...


@runtime_checkable
class AzureBlobServiceClientProtocol(Protocol):
    """Protocol for Azure Blob Service Client.

    Defines the minimal interface used by AzureBlobStore.
    """

    def close(self) -> None:
        """Release the owned Azure client pipeline."""
        ...

    def get_container_client(self, container: str) -> AzureContainerClientProtocol:
        """Get a container client."""
        ...

    @classmethod
    def from_connection_string(
        cls,
        conn_str: str,
        **kwargs: Any,
    ) -> AzureBlobServiceClientProtocol:
        """Create client from connection string."""
        ...
