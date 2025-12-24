"""Protocol definitions for optional dependency clients.

This module defines structural typing protocols for external library clients,
allowing type-safe code without requiring type stubs packages.

These protocols define only the methods actually used by Truthound,
providing minimal but sufficient type coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# S3 Client Protocol (boto3)
# =============================================================================


class S3ResponseBody(Protocol):
    """Protocol for S3 response body stream."""

    def read(self) -> bytes:
        """Read all bytes from the response body."""
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

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        """Upload an object to S3."""
        ...

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        """Check if an object exists."""
        ...

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        """Delete an object from S3."""
        ...


# =============================================================================
# GCS Client Protocol (google-cloud-storage)
# =============================================================================


@runtime_checkable
class GCSBlobProtocol(Protocol):
    """Protocol for GCS Blob object."""

    content_encoding: str | None

    def download_as_bytes(self) -> bytes:
        """Download blob contents as bytes."""
        ...

    def upload_from_string(
        self,
        data: str | bytes,
        content_type: str = ...,
    ) -> None:
        """Upload data to the blob."""
        ...

    def exists(self) -> bool:
        """Check if the blob exists."""
        ...

    def delete(self) -> None:
        """Delete the blob."""
        ...


@runtime_checkable
class GCSBucketProtocol(Protocol):
    """Protocol for GCS Bucket object."""

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

    def bucket(self, bucket_name: str) -> GCSBucketProtocol:
        """Get a bucket reference."""
        ...

    @classmethod
    def from_service_account_json(
        cls,
        json_credentials_path: str,
        **kwargs: Any,
    ) -> "GCSClientProtocol":
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

    def __enter__(self) -> "SQLConnectionProtocol":
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

    def __enter__(self) -> "SQLSessionProtocol":
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
