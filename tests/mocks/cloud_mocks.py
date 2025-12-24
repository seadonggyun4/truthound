"""Mock implementations for cloud storage backends.

These mocks simulate S3 and GCS behavior using in-memory storage,
matching the Protocol definitions for type safety.
"""

from __future__ import annotations

import gzip
import io
import json
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# S3 Mock Implementation
# =============================================================================


class MockS3ClientError(Exception):
    """Mock for botocore.exceptions.ClientError."""

    def __init__(self, error_code: str, message: str = ""):
        self.response = {"Error": {"Code": error_code, "Message": message}}
        super().__init__(message)


@dataclass
class MockS3Object:
    """In-memory S3 object."""

    key: str
    body: bytes
    content_type: str = "application/octet-stream"
    content_encoding: str | None = None
    storage_class: str = "STANDARD"
    metadata: dict[str, str] = field(default_factory=dict)


class MockS3ResponseBody:
    """Mock for S3 response body stream."""

    def __init__(self, data: bytes):
        self._data = data
        self._stream = io.BytesIO(data)

    def read(self) -> bytes:
        return self._data

    def iter_chunks(self, chunk_size: int = 1024) -> Any:
        while True:
            chunk = self._stream.read(chunk_size)
            if not chunk:
                break
            yield chunk


class MockS3Client:
    """Mock S3 client with in-memory storage.

    Simulates boto3 S3 client behavior for testing.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, MockS3Object]] = {}
        self._bucket_exists: set[str] = set()

    def create_bucket(self, Bucket: str, **kwargs: Any) -> dict[str, Any]:
        """Create a bucket."""
        self._buckets[Bucket] = {}
        self._bucket_exists.add(Bucket)
        return {"Location": f"/{Bucket}"}

    def head_bucket(self, *, Bucket: str) -> dict[str, Any]:
        """Check if bucket exists."""
        if Bucket not in self._bucket_exists:
            raise MockS3ClientError("404", "NoSuchBucket")
        return {}

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str = "application/octet-stream",
        ContentEncoding: str | None = None,
        StorageClass: str = "STANDARD",
        ServerSideEncryption: str | None = None,
        SSEKMSKeyId: str | None = None,
        Tagging: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Upload an object."""
        if Bucket not in self._bucket_exists:
            raise MockS3ClientError("404", "NoSuchBucket")

        self._buckets.setdefault(Bucket, {})[Key] = MockS3Object(
            key=Key,
            body=Body,
            content_type=ContentType,
            content_encoding=ContentEncoding,
            storage_class=StorageClass,
        )
        return {"ETag": f'"{hash(Body)}"'}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        """Retrieve an object."""
        if Bucket not in self._bucket_exists:
            raise MockS3ClientError("404", "NoSuchBucket")

        bucket = self._buckets.get(Bucket, {})
        if Key not in bucket:
            raise MockS3ClientError("NoSuchKey", f"Key not found: {Key}")

        obj = bucket[Key]
        return {
            "Body": MockS3ResponseBody(obj.body),
            "ContentType": obj.content_type,
            "ContentEncoding": obj.content_encoding,
            "StorageClass": obj.storage_class,
        }

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        """Check if object exists."""
        if Bucket not in self._bucket_exists:
            raise MockS3ClientError("404", "NoSuchBucket")

        bucket = self._buckets.get(Bucket, {})
        if Key not in bucket:
            raise MockS3ClientError("404", "Not Found")

        obj = bucket[Key]
        return {
            "ContentLength": len(obj.body),
            "ContentType": obj.content_type,
        }

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        """Delete an object."""
        if Bucket not in self._bucket_exists:
            raise MockS3ClientError("404", "NoSuchBucket")

        bucket = self._buckets.get(Bucket, {})
        if Key in bucket:
            del bucket[Key]

        return {}

    def list_objects_v2(
        self,
        *,
        Bucket: str,
        Prefix: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """List objects in bucket."""
        if Bucket not in self._bucket_exists:
            raise MockS3ClientError("404", "NoSuchBucket")

        bucket = self._buckets.get(Bucket, {})
        contents = [
            {"Key": key, "Size": len(obj.body)}
            for key, obj in bucket.items()
            if key.startswith(Prefix)
        ]

        return {
            "Contents": contents,
            "KeyCount": len(contents),
            "IsTruncated": False,
        }


def create_mock_s3_client(with_bucket: str | None = None) -> MockS3Client:
    """Create a mock S3 client, optionally with a pre-created bucket."""
    client = MockS3Client()
    if with_bucket:
        client.create_bucket(Bucket=with_bucket)
    return client


# =============================================================================
# GCS Mock Implementation
# =============================================================================


class MockGCSNotFound(Exception):
    """Mock for google.cloud.exceptions.NotFound."""

    pass


@dataclass
class MockGCSBlobData:
    """In-memory GCS blob data."""

    name: str
    data: bytes
    content_type: str = "application/octet-stream"
    content_encoding: str | None = None


class MockGCSBlob:
    """Mock GCS Blob object."""

    def __init__(self, name: str, bucket: "MockGCSBucket"):
        self.name = name
        self._bucket = bucket
        self.content_encoding: str | None = None

    def download_as_bytes(self) -> bytes:
        """Download blob contents."""
        data = self._bucket._get_blob_data(self.name)
        if data is None:
            raise MockGCSNotFound(f"Blob not found: {self.name}")
        return data.data

    def upload_from_string(
        self,
        data: str | bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload data to blob."""
        if isinstance(data, str):
            data = data.encode("utf-8")

        self._bucket._set_blob_data(
            self.name,
            MockGCSBlobData(
                name=self.name,
                data=data,
                content_type=content_type,
                content_encoding=self.content_encoding,
            ),
        )

    def exists(self) -> bool:
        """Check if blob exists."""
        return self._bucket._get_blob_data(self.name) is not None

    def delete(self) -> None:
        """Delete the blob."""
        if not self.exists():
            raise MockGCSNotFound(f"Blob not found: {self.name}")
        self._bucket._delete_blob(self.name)


class MockGCSBucket:
    """Mock GCS Bucket object."""

    def __init__(self, name: str, exists: bool = True):
        self.name = name
        self._exists = exists
        self._blobs: dict[str, MockGCSBlobData] = {}

    def blob(self, blob_name: str) -> MockGCSBlob:
        """Get a blob reference."""
        return MockGCSBlob(blob_name, self)

    def exists(self) -> bool:
        """Check if bucket exists."""
        return self._exists

    def _get_blob_data(self, name: str) -> MockGCSBlobData | None:
        return self._blobs.get(name)

    def _set_blob_data(self, name: str, data: MockGCSBlobData) -> None:
        self._blobs[name] = data

    def _delete_blob(self, name: str) -> None:
        if name in self._blobs:
            del self._blobs[name]

    def list_blobs(self, prefix: str = "") -> list[MockGCSBlob]:
        """List blobs with prefix."""
        return [
            MockGCSBlob(name, self)
            for name in self._blobs
            if name.startswith(prefix)
        ]


class MockGCSClient:
    """Mock GCS Client."""

    def __init__(self, project: str | None = None):
        self.project = project
        self._buckets: dict[str, MockGCSBucket] = {}

    def bucket(self, bucket_name: str) -> MockGCSBucket:
        """Get a bucket reference."""
        if bucket_name not in self._buckets:
            # Return a bucket that doesn't exist yet
            self._buckets[bucket_name] = MockGCSBucket(bucket_name, exists=False)
        return self._buckets[bucket_name]

    def create_bucket(self, bucket_name: str) -> MockGCSBucket:
        """Create a bucket."""
        bucket = MockGCSBucket(bucket_name, exists=True)
        self._buckets[bucket_name] = bucket
        return bucket

    @classmethod
    def from_service_account_json(
        cls,
        json_credentials_path: str,
        **kwargs: Any,
    ) -> "MockGCSClient":
        """Create client from service account (mock)."""
        return cls(**kwargs)


def create_mock_gcs_client(with_bucket: str | None = None) -> MockGCSClient:
    """Create a mock GCS client, optionally with a pre-created bucket."""
    client = MockGCSClient()
    if with_bucket:
        client.create_bucket(with_bucket)
    return client
