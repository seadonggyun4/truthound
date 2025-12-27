"""Streaming S3 store implementation with multipart uploads.

This module provides a streaming-capable S3 store that uses multipart uploads
for efficient handling of large validation results.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator
from uuid import uuid4

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingMetrics,
    StreamingValidationStore,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.reader import (
    AsyncStreamReader,
    BaseStreamReader,
    ChunkLoader,
    ChunkedResultReader,
    get_decompressor,
    get_deserializer,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
    BaseStreamWriter,
    get_compressor,
    get_serializer,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class StreamingS3Config(StreamingConfig):
    """Configuration for streaming S3 store.

    Attributes:
        bucket: S3 bucket name.
        prefix: Key prefix for all objects.
        region: AWS region.
        endpoint_url: Custom endpoint URL (for S3-compatible services).
        storage_class: S3 storage class.
        server_side_encryption: SSE type (AES256, aws:kms).
        kms_key_id: KMS key ID for SSE-KMS.
        multipart_threshold: Size threshold for multipart uploads (bytes).
        multipart_part_size: Size of each part in multipart upload (bytes).
        max_concurrency: Maximum concurrent upload threads.
    """

    bucket: str = ""
    prefix: str = "truthound/streaming/"
    region: str | None = None
    endpoint_url: str | None = None
    storage_class: str = "STANDARD"
    server_side_encryption: str | None = None
    kms_key_id: str | None = None
    multipart_threshold: int = 8 * 1024 * 1024  # 8MB
    multipart_part_size: int = 8 * 1024 * 1024  # 8MB
    max_concurrency: int = 10

    def validate(self) -> None:
        """Validate configuration."""
        super().validate()
        if not self.bucket:
            raise ValueError("S3 bucket name is required")
        if self.multipart_part_size < 5 * 1024 * 1024:
            raise ValueError("multipart_part_size must be at least 5MB")


# =============================================================================
# S3 Streaming Writer
# =============================================================================


class S3StreamWriter(BaseStreamWriter):
    """S3 streaming writer with multipart upload support.

    Uses S3 multipart uploads for efficient streaming of large results.
    """

    def __init__(
        self,
        session: StreamSession,
        config: StreamingS3Config,
        s3_client: Any,
    ):
        """Initialize the S3 writer.

        Args:
            session: The streaming session.
            config: S3 streaming configuration.
            s3_client: Boto3 S3 client.
        """
        super().__init__(session, config)
        self.s3_config = config
        self._s3 = s3_client
        self._multipart_uploads: dict[str, dict[str, Any]] = {}

    def _get_chunk_key(self, chunk_info: ChunkInfo) -> str:
        """Get the S3 key for a chunk."""
        ext = {
            StreamingFormat.JSONL: ".jsonl",
            StreamingFormat.NDJSON: ".ndjson",
            StreamingFormat.CSV: ".csv",
        }.get(self.config.format, ".jsonl")

        ext += self.compressor.get_extension()

        prefix = self.s3_config.prefix.rstrip("/")
        return f"{prefix}/{self.session.run_id}/{chunk_info.chunk_id}{ext}"

    def _get_manifest_key(self) -> str:
        """Get the S3 key for the manifest."""
        prefix = self.s3_config.prefix.rstrip("/")
        return f"{prefix}/{self.session.run_id}/_manifest.json"

    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write a chunk to S3."""
        key = self._get_chunk_key(chunk_info)
        chunk_info.path = f"s3://{self.s3_config.bucket}/{key}"

        extra_args: dict[str, Any] = {
            "StorageClass": self.s3_config.storage_class,
        }

        if self.s3_config.server_side_encryption:
            extra_args["ServerSideEncryption"] = self.s3_config.server_side_encryption
            if self.s3_config.kms_key_id:
                extra_args["SSEKMSKeyId"] = self.s3_config.kms_key_id

        # Use multipart upload for large chunks
        if len(data) >= self.s3_config.multipart_threshold:
            self._multipart_upload(key, data, extra_args)
        else:
            self._s3.put_object(
                Bucket=self.s3_config.bucket,
                Key=key,
                Body=data,
                ContentType=self.serializer.get_content_type(),
                **extra_args,
            )

    def _multipart_upload(
        self,
        key: str,
        data: bytes,
        extra_args: dict[str, Any],
    ) -> None:
        """Perform multipart upload for large data."""
        # Initiate multipart upload
        response = self._s3.create_multipart_upload(
            Bucket=self.s3_config.bucket,
            Key=key,
            ContentType=self.serializer.get_content_type(),
            **extra_args,
        )
        upload_id = response["UploadId"]

        parts: list[dict[str, Any]] = []
        part_size = self.s3_config.multipart_part_size
        part_number = 1

        try:
            # Upload parts
            for i in range(0, len(data), part_size):
                part_data = data[i : i + part_size]
                response = self._s3.upload_part(
                    Bucket=self.s3_config.bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=part_data,
                )
                parts.append({"PartNumber": part_number, "ETag": response["ETag"]})
                part_number += 1

            # Complete multipart upload
            self._s3.complete_multipart_upload(
                Bucket=self.s3_config.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )
        except Exception:
            # Abort on failure
            self._s3.abort_multipart_upload(
                Bucket=self.s3_config.bucket,
                Key=key,
                UploadId=upload_id,
            )
            raise

    def _write_session_state(self) -> None:
        """Write session state to S3."""
        key = self._get_manifest_key()
        manifest_data = self.session.to_dict()
        manifest_data["config"] = {
            "format": self.config.format.value,
            "compression": self.config.compression.value,
            "chunk_size": self.config.chunk_size,
            "buffer_size": self.config.buffer_size,
        }

        self._s3.put_object(
            Bucket=self.s3_config.bucket,
            Key=key,
            Body=json.dumps(manifest_data, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )

    def _finalize(self) -> None:
        """Finalize the stream."""
        # Manifest already written in _write_session_state
        pass


# =============================================================================
# S3 Streaming Reader
# =============================================================================


class S3ChunkLoader(ChunkLoader):
    """S3 chunk loader."""

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
    ):
        self._s3 = s3_client
        self._bucket = bucket

    def load(self, chunk_info: ChunkInfo) -> bytes:
        """Load chunk from S3."""
        # Extract key from path
        path = chunk_info.path
        if path.startswith("s3://"):
            # s3://bucket/key -> key
            parts = path.replace("s3://", "").split("/", 1)
            key = parts[1] if len(parts) > 1 else ""
        else:
            key = path

        response = self._s3.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()


class S3StreamReader(BaseStreamReader):
    """S3 streaming reader."""

    def __init__(
        self,
        run_id: str,
        s3_client: Any,
        config: StreamingS3Config,
    ):
        """Initialize the S3 reader.

        Args:
            run_id: The run ID to read.
            s3_client: Boto3 S3 client.
            config: S3 streaming configuration.
        """
        self._run_id = run_id
        self._s3 = s3_client
        self._s3_config = config
        self._chunks: list[ChunkInfo] = []
        self._session: StreamSession | None = None

        # Load manifest
        self._load_manifest()

        super().__init__(config)

    def _load_manifest(self) -> None:
        """Load manifest from S3."""
        prefix = self._s3_config.prefix.rstrip("/")
        manifest_key = f"{prefix}/{self._run_id}/_manifest.json"

        try:
            response = self._s3.get_object(
                Bucket=self._s3_config.bucket,
                Key=manifest_key,
            )
            data = json.loads(response["Body"].read().decode("utf-8"))
            self._session = StreamSession.from_dict(data)
            self._chunks = self._session.chunks
        except self._s3.exceptions.NoSuchKey:
            # No manifest - discover chunks
            self._chunks = self._discover_chunks()

    def _discover_chunks(self) -> list[ChunkInfo]:
        """Discover chunks by listing S3 objects."""
        prefix = f"{self._s3_config.prefix.rstrip('/')}/{self._run_id}/"
        chunks: list[ChunkInfo] = []

        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._s3_config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_manifest.json"):
                    continue

                chunk_id = key.split("/")[-1].split(".")[0]
                chunk_info = ChunkInfo(
                    chunk_id=chunk_id,
                    chunk_index=len(chunks),
                    record_count=0,
                    byte_size=obj["Size"],
                    start_offset=0,
                    end_offset=0,
                    path=f"s3://{self._s3_config.bucket}/{key}",
                )
                chunks.append(chunk_info)

        # Sort by chunk ID
        chunks.sort(key=lambda c: c.chunk_id)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _get_chunks(self) -> list[ChunkInfo]:
        """Get list of chunks."""
        return self._chunks

    def _read_chunk(self, chunk_info: ChunkInfo) -> bytes:
        """Read a chunk from S3."""
        path = chunk_info.path
        if path.startswith("s3://"):
            parts = path.replace("s3://", "").split("/", 1)
            key = parts[1] if len(parts) > 1 else ""
        else:
            key = path

        response = self._s3.get_object(Bucket=self._s3_config.bucket, Key=key)
        return response["Body"].read()


# =============================================================================
# Streaming S3 Store
# =============================================================================


class StreamingS3Store(StreamingValidationStore[StreamingS3Config]):
    """Streaming S3 store with multipart upload support.

    This store is optimized for handling large validation results on S3:

    - Multipart uploads for large chunks
    - Streaming reads without full download
    - Server-side encryption support
    - Custom storage classes

    Example:
        >>> store = StreamingS3Store(
        ...     bucket="my-bucket",
        ...     prefix="validations/",
        ...     region="us-east-1",
        ... )
        >>>
        >>> session = store.create_session("run_001", "large_dataset.csv")
        >>> with store.create_writer(session) as writer:
        ...     for result in validation_results:
        ...         writer.write_result(result)
        >>>
        >>> final_result = store.finalize_result(session)
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "truthound/streaming/",
        region: str | None = None,
        endpoint_url: str | None = None,
        format: StreamingFormat = StreamingFormat.JSONL,
        compression: CompressionType = CompressionType.GZIP,
        chunk_size: int = 10000,
        **kwargs: Any,
    ):
        """Initialize the streaming S3 store.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix for all objects.
            region: AWS region.
            endpoint_url: Custom endpoint URL.
            format: Output format.
            compression: Compression algorithm.
            chunk_size: Records per chunk.
            **kwargs: Additional configuration options.
        """
        config = StreamingS3Config(
            bucket=bucket,
            prefix=prefix,
            region=region,
            endpoint_url=endpoint_url,
            format=format,
            compression=compression,
            chunk_size=chunk_size,
            **{k: v for k, v in kwargs.items() if hasattr(StreamingS3Config, k)},
        )
        super().__init__(config)
        self._s3_client: Any = None

    @classmethod
    def _default_config(cls) -> StreamingS3Config:
        """Create default configuration."""
        return StreamingS3Config()

    def _get_s3_client(self) -> Any:
        """Get or create S3 client."""
        if self._s3_client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 library required for S3 streaming store")

            client_kwargs: dict[str, Any] = {}
            if self._config.region:
                client_kwargs["region_name"] = self._config.region
            if self._config.endpoint_url:
                client_kwargs["endpoint_url"] = self._config.endpoint_url

            self._s3_client = boto3.client("s3", **client_kwargs)

        return self._s3_client

    def _do_initialize(self) -> None:
        """Initialize the store and verify bucket access."""
        s3 = self._get_s3_client()
        # Verify bucket exists
        s3.head_bucket(Bucket=self._config.bucket)

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_session(
        self,
        run_id: str,
        data_asset: str,
        metadata: dict[str, Any] | None = None,
    ) -> StreamSession:
        """Create a new streaming session."""
        self.initialize()

        session_id = f"{run_id}_{uuid4().hex[:8]}"
        session = StreamSession(
            session_id=session_id,
            run_id=run_id,
            data_asset=data_asset,
            status=StreamStatus.PENDING,
            config=self._config,
            metadata=metadata or {},
        )

        # Write initial manifest
        self._write_manifest(session)

        self._active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> StreamSession | None:
        """Get an existing session."""
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        # Try to load from S3
        s3 = self._get_s3_client()
        prefix = self._config.prefix.rstrip("/")

        # List potential run directories
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self._config.bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            for common_prefix in page.get("CommonPrefixes", []):
                run_prefix = common_prefix["Prefix"]
                manifest_key = f"{run_prefix}_manifest.json"

                try:
                    response = s3.get_object(
                        Bucket=self._config.bucket,
                        Key=manifest_key,
                    )
                    data = json.loads(response["Body"].read().decode("utf-8"))
                    session = StreamSession.from_dict(data)
                    if session.session_id == session_id:
                        return session
                except Exception:
                    continue

        return None

    def resume_session(self, session_id: str) -> StreamSession:
        """Resume an interrupted session."""
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if session.status == StreamStatus.COMPLETED:
            raise ValueError(f"Session already completed: {session_id}")

        session.status = StreamStatus.ACTIVE
        session.updated_at = datetime.now()

        self._active_sessions[session_id] = session
        return session

    def _close_session(self, session: StreamSession) -> None:
        """Close and finalize a session."""
        if session.session_id in self._active_sessions:
            del self._active_sessions[session.session_id]
        session.status = StreamStatus.COMPLETED
        session.updated_at = datetime.now()
        self._write_manifest(session)

    def _write_manifest(self, session: StreamSession) -> None:
        """Write session manifest to S3."""
        s3 = self._get_s3_client()
        prefix = self._config.prefix.rstrip("/")
        key = f"{prefix}/{session.run_id}/_manifest.json"

        manifest_data = session.to_dict()
        manifest_data["config"] = {
            "format": self._config.format.value,
            "compression": self._config.compression.value,
            "chunk_size": self._config.chunk_size,
            "buffer_size": self._config.buffer_size,
        }

        s3.put_object(
            Bucket=self._config.bucket,
            Key=key,
            Body=json.dumps(manifest_data, indent=2, default=str).encode("utf-8"),
            ContentType="application/json",
        )

    # -------------------------------------------------------------------------
    # Writer Operations
    # -------------------------------------------------------------------------

    def create_writer(self, session: StreamSession) -> S3StreamWriter:
        """Create a writer for the session."""
        self.initialize()
        return S3StreamWriter(
            session=session,
            config=self._config,
            s3_client=self._get_s3_client(),
        )

    async def create_async_writer(self, session: StreamSession) -> AsyncStreamWriter:
        """Create an async writer for the session."""
        writer = self.create_writer(session)
        return AsyncStreamWriter(writer)

    # -------------------------------------------------------------------------
    # Reader Operations
    # -------------------------------------------------------------------------

    def create_reader(self, run_id: str) -> S3StreamReader:
        """Create a reader for a run's results."""
        self.initialize()
        return S3StreamReader(
            run_id=run_id,
            s3_client=self._get_s3_client(),
            config=self._config,
        )

    async def create_async_reader(self, run_id: str) -> AsyncStreamReader:
        """Create an async reader for a run's results."""
        reader = self.create_reader(run_id)
        return AsyncStreamReader(reader)

    def iter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> Iterator["ValidatorResult"]:
        """Iterate over results for a run."""
        reader = self.create_reader(run_id)
        with reader:
            yield from reader.iter_results()

    async def aiter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> AsyncIterator["ValidatorResult"]:
        """Async iterate over results for a run."""
        reader = await self.create_async_reader(run_id)
        async with reader:
            async for result in reader.aiter_results():
                yield result

    # -------------------------------------------------------------------------
    # Chunk Management
    # -------------------------------------------------------------------------

    def list_chunks(self, run_id: str) -> list[ChunkInfo]:
        """List all chunks for a run."""
        self.initialize()
        reader = self.create_reader(run_id)
        return reader._chunks

    def get_chunk(self, chunk_info: ChunkInfo) -> list["ValidatorResult"]:
        """Get records from a specific chunk."""
        from truthound.stores.results import ValidatorResult

        s3 = self._get_s3_client()

        path = chunk_info.path
        if path.startswith("s3://"):
            parts = path.replace("s3://", "").split("/", 1)
            key = parts[1] if len(parts) > 1 else ""
        else:
            key = path

        response = s3.get_object(Bucket=self._config.bucket, Key=key)
        compressed_data = response["Body"].read()

        decompressor = get_decompressor(self._config.compression)
        deserializer = get_deserializer(self._config.format)

        data = decompressor.decompress(compressed_data)
        records = list(deserializer.deserialize(data))

        return [ValidatorResult.from_dict(r) for r in records]

    def delete_chunks(self, run_id: str) -> int:
        """Delete all chunks for a run."""
        self.initialize()

        s3 = self._get_s3_client()
        prefix = f"{self._config.prefix.rstrip('/')}/{run_id}/"

        # List all objects with this prefix
        objects_to_delete: list[dict[str, str]] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._config.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                objects_to_delete.append({"Key": obj["Key"]})

        if not objects_to_delete:
            return 0

        # Delete objects in batches of 1000
        deleted = 0
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i : i + 1000]
            s3.delete_objects(
                Bucket=self._config.bucket,
                Delete={"Objects": batch},
            )
            deleted += len(batch)

        return deleted

    # -------------------------------------------------------------------------
    # Validation Result Operations
    # -------------------------------------------------------------------------

    def stream_write_result(
        self,
        session: StreamSession,
        result: "ValidatorResult",
    ) -> None:
        """Write a single validator result to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_result(result)

    def stream_write_batch(
        self,
        session: StreamSession,
        results: list["ValidatorResult"],
    ) -> None:
        """Write a batch of validator results to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_results(results)

    def _get_or_create_writer(self, session: StreamSession) -> S3StreamWriter:
        """Get or create a writer for a session."""
        writer_key = f"_writer_{session.session_id}"
        if not hasattr(self, writer_key):
            writer = self.create_writer(session)
            setattr(self, writer_key, writer)
        return getattr(self, writer_key)

    def finalize_result(
        self,
        session: StreamSession,
        additional_metadata: dict[str, Any] | None = None,
    ) -> "ValidationResult":
        """Finalize the streaming session and create a ValidationResult."""
        from truthound.stores.results import (
            ResultStatistics,
            ResultStatus,
            ValidationResult,
        )

        # Close any active writer
        writer_key = f"_writer_{session.session_id}"
        if hasattr(self, writer_key):
            writer = getattr(self, writer_key)
            writer.close()
            delattr(self, writer_key)

        # Aggregate statistics
        total_validators = 0
        passed_validators = 0
        failed_validators = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        all_results: list["ValidatorResult"] = []
        for result in self.iter_results(session.run_id):
            all_results.append(result)
            total_validators += 1
            if result.success:
                passed_validators += 1
            else:
                failed_validators += 1
                if result.severity and result.severity in severity_counts:
                    severity_counts[result.severity] += 1

        # Determine status
        if severity_counts["critical"] > 0:
            status = ResultStatus.FAILURE
        elif failed_validators > 0:
            status = ResultStatus.WARNING
        else:
            status = ResultStatus.SUCCESS

        statistics = ResultStatistics(
            total_validators=total_validators,
            passed_validators=passed_validators,
            failed_validators=failed_validators,
            total_issues=failed_validators,
            critical_issues=severity_counts["critical"],
            high_issues=severity_counts["high"],
            medium_issues=severity_counts["medium"],
            low_issues=severity_counts["low"],
        )

        metadata = session.metadata.copy()
        if additional_metadata:
            metadata.update(additional_metadata)
        metadata["streaming"] = {
            "storage": "s3",
            "bucket": self._config.bucket,
            "chunks": len(session.chunks),
            "total_records": session.metrics.records_written,
        }

        result = ValidationResult(
            run_id=session.run_id,
            run_time=session.started_at,
            data_asset=session.data_asset,
            status=status,
            results=all_results,
            statistics=statistics,
            metadata=metadata,
        )

        self._close_session(session)
        return result

    def get_streaming_stats(self, run_id: str) -> dict[str, Any]:
        """Get statistics about a streaming run."""
        self.initialize()

        s3 = self._get_s3_client()
        prefix = self._config.prefix.rstrip("/")
        manifest_key = f"{prefix}/{run_id}/_manifest.json"

        try:
            response = s3.get_object(Bucket=self._config.bucket, Key=manifest_key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            session = StreamSession.from_dict(data)

            return {
                "run_id": run_id,
                "data_asset": session.data_asset,
                "status": session.status.value,
                "chunks": len(session.chunks),
                "total_records": session.metrics.records_written,
                "bytes_written": session.metrics.bytes_written,
                "storage": "s3",
                "bucket": self._config.bucket,
            }
        except Exception:
            return {}

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def list_runs(self) -> list[str]:
        """List all run IDs in the store."""
        self.initialize()

        s3 = self._get_s3_client()
        prefix = self._config.prefix.rstrip("/") + "/"
        runs: list[str] = []

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self._config.bucket,
            Prefix=prefix,
            Delimiter="/",
        ):
            for common_prefix in page.get("CommonPrefixes", []):
                run_id = common_prefix["Prefix"].rstrip("/").split("/")[-1]
                runs.append(run_id)

        return sorted(runs)
