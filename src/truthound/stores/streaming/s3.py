"""Streaming S3 store implementation with multipart uploads.

This module provides a streaming-capable S3 store that uses multipart uploads
for efficient handling of large validation results.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator  # noqa: TC003 - public annotation introspection
from concurrent.futures import FIRST_COMPLETED, CancelledError, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime
from numbers import Integral
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingValidationStore,
    StreamSession,
    StreamStatus,
)
from truthound.stores.streaming.reader import (
    AsyncStreamReader,
    BaseStreamReader,
    ChunkLoader,
    _validator_result,
)
from truthound.stores.streaming.writer import (
    AsyncStreamWriter,
    BaseStreamWriter,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidationResult, ValidatorResult


def _etag_receipt(response):
    etag = response.get("ETag")
    if not isinstance(etag, str) or not etag:
        raise RuntimeError("Native S3 publication has no confirmed ETag receipt")
    return etag


def _publish_manifest(client, config, session, guard, *, recovery_document=None):
    from truthound.stores.streaming.s3_multipart import document

    guard()
    previous = getattr(session, "_manifest_etag", None)
    response = client.put_object(
        Bucket=config.bucket,
        Key=f"{config.prefix.rstrip('/')}/{session.run_id}/_manifest.json",
        Body=json.dumps(
            document(session, recovery_document=recovery_document), indent=2, default=str
        ).encode("utf-8"),
        ContentType="application/json",
        **({"IfMatch": previous} if previous is not None else {"IfNoneMatch": "*"}),
    )
    session._manifest_etag = _etag_receipt(response)
    guard()


def _read_manifest(client, config, run_id, guard, *, allow_recovery=False):
    from truthound.stores.streaming._filesystem_io import component

    component(run_id)
    guard()
    response = client.get_object(
        Bucket=config.bucket,
        Key=f"{config.prefix.rstrip('/')}/{run_id}/_manifest.json",
    )
    body = response["Body"]
    try:
        data = body.read(config.max_memory_mb * 1024**2 + 1)
    finally:
        body.close()
    if len(data) > config.max_memory_mb * 1024**2:
        raise RuntimeError("Native S3 manifest exceeds its inspection bound")
    document = json.loads(data)
    if "_truthound_recovery_v1" in document and not allow_recovery:
        raise RuntimeError("Native S3 reconciliation is pending; explicit recovery is required")
    session = StreamSession.from_dict(document)
    if session.run_id != run_id:
        raise RuntimeError("Native S3 stream identity requires reconciliation")
    from truthound.stores.streaming.s3_multipart import capture, require_clear

    capture(session, document, config)
    if not allow_recovery:
        require_clear(session)
    session._manifest_etag = _etag_receipt(response)
    return session, document, data


def _validate_inventory(client, config, session, guard):
    from truthound.stores.streaming.s3_multipart import require_clear

    require_clear(session)
    prefix = f"{config.prefix.rstrip('/')}/{session.run_id}/"
    expected = {}
    for chunk in session.chunks:
        path = f"s3://{config.bucket}/"
        if not chunk.path.startswith(path):
            raise RuntimeError("Native S3 chunk identity requires reconciliation")
        key = chunk.path[len(path) :]
        if not key.startswith(prefix) or "/" in key[len(prefix) :] or key in expected:
            raise RuntimeError("Native S3 chunk scope requires reconciliation")
        expected[key] = chunk.byte_size
    actual = {}
    guard()
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=config.bucket, Prefix=prefix
    ):
        guard()
        for row in page.get("Contents", []):
            if row["Key"] != prefix + "_manifest.json":
                actual[row["Key"]] = row["Size"]
    if actual != expected:
        raise RuntimeError("Unconfirmed or inconsistent native S3 chunks require reconciliation")


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
        for name in ("multipart_threshold", "multipart_part_size", "max_concurrency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
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
        self._multipart_failed = False
        if not getattr(session, "_manifest_etag", None):
            raise ValueError("A native S3 writer requires its store-owned manifest receipt")

    def _get_chunk_key(self, chunk_info: ChunkInfo) -> str:
        """Get the S3 key for a chunk."""
        ext = {
            StreamingFormat.JSONL: ".jsonl",
            StreamingFormat.NDJSON: ".ndjson",
            StreamingFormat.CSV: ".csv",
            StreamingFormat.PARQUET: ".parquet",
        }.get(self.config.format, ".jsonl")

        ext += self.compressor.get_extension()

        prefix = self.s3_config.prefix.rstrip("/")
        return f"{prefix}/{self.session.run_id}/{chunk_info.chunk_id}{ext}"

    def _get_manifest_key(self) -> str:
        """Get the S3 key for the manifest."""
        prefix = self.s3_config.prefix.rstrip("/")
        return f"{prefix}/{self.session.run_id}/_manifest.json"

    def _check_manifest_receipt(self) -> None:
        self._guard_io()
        expected = self.session._manifest_etag
        response = self._s3.head_object(
            Bucket=self.s3_config.bucket,
            Key=self._get_manifest_key(),
            IfMatch=expected,
        )
        if response.get("ETag") != expected:
            raise RuntimeError("Native S3 writer manifest receipt is stale")

    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write a chunk to S3."""
        from truthound.stores.streaming.s3_multipart import require_clear

        if self._multipart_failed:
            raise RuntimeError("Native S3 multipart write requires explicit reconciliation")
        require_clear(self.session)
        self._check_manifest_receipt()
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
            self._guard_io()
            response = self._s3.put_object(
                Bucket=self.s3_config.bucket,
                Key=key,
                Body=data,
                ContentType=self.serializer.get_content_type(),
                IfNoneMatch="*",
                **extra_args,
            )
            _etag_receipt(response)

    def _multipart_upload(
        self,
        key: str,
        data: bytes,
        extra_args: dict[str, Any],
    ) -> None:
        """Perform multipart upload for large data."""
        from truthound.stores.streaming.s3_multipart import pending, require_clear

        require_clear(self.session)
        self._multipart_failed = True
        intent = uuid4().hex
        self.session._multipart_intents = {intent: {"key": key, "upload_id": None}}
        self._write_session_state()
        # Initiate multipart upload
        self._guard_io()
        response = self._s3.create_multipart_upload(
            Bucket=self.s3_config.bucket,
            Key=key,
            ContentType=self.serializer.get_content_type(),
            **extra_args,
        )
        upload_id = response["UploadId"]
        if not isinstance(upload_id, str) or not upload_id:
            raise RuntimeError("Native S3 multipart creation has no confirmed upload receipt")
        self._multipart_uploads[upload_id] = {"key": key, "intent": intent}
        pending(self.session)[intent]["upload_id"] = upload_id

        parts: list[dict[str, Any]] = []
        part_size = self.s3_config.multipart_part_size
        try:
            self._write_session_state()

            def upload_part(part_number):
                self._guard_io()
                self._check_manifest_receipt()
                start = (part_number - 1) * part_size
                self._guard_io()
                response = self._s3.upload_part(
                    Bucket=self.s3_config.bucket,
                    Key=key,
                    UploadId=upload_id,
                    PartNumber=part_number,
                    Body=data[start : start + part_size],
                )
                return {"PartNumber": part_number, "ETag": _etag_receipt(response)}

            count = (len(data) + part_size - 1) // part_size
            if count > 10000:
                raise ValueError("Native S3 multipart upload exceeds 10000 parts")
            numbers = iter(range(1, count + 1))
            workers = min(self.s3_config.max_concurrency, max(1, count))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = set()
                failures = []
                while True:
                    while len(pending) < workers and not failures:
                        number = next(numbers, None)
                        if number is None:
                            break
                        pending.add(pool.submit(upload_part, number))
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            parts.append(future.result())
                        except CancelledError:
                            if not failures:
                                raise
                        except Exception as error:
                            failures.append(error)
                    if failures:
                        for future in pending:
                            future.cancel()
                        # Finish every started SDK call before aborting the
                        # owned upload; no parts may race the abort request.
                        for future in pending:
                            try:
                                future.result()
                            except CancelledError:
                                pass
                            except Exception as error:
                                failures.append(error)
                        if len(failures) == 1:
                            raise failures[0]
                        raise ExceptionGroup("Native multipart parts failed", failures)
            parts.sort(key=lambda part: part["PartNumber"])

            # Complete multipart upload
            self._guard_io()
            self._check_manifest_receipt()
            self._guard_io()
            response = self._s3.complete_multipart_upload(
                Bucket=self.s3_config.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
            _etag_receipt(response)
            self._clear_multipart_intent(intent)
            del self._multipart_uploads[upload_id]
            self._multipart_failed = False
        except Exception as original:
            try:
                self._abort_upload(upload_id)
            except Exception as cleanup:
                self.metrics.record_error("multipart_abort:" + type(cleanup).__name__)
                raise ExceptionGroup(
                    "Native multipart execution and cleanup failed", [original, cleanup]
                ) from None
            raise

    def _abort_upload(self, identity):
        from truthound.stores.streaming.s3_multipart import abort

        entry = self._multipart_uploads[identity]
        abort(self._s3, self.s3_config, entry["key"], identity, self._guard_io)
        self._clear_multipart_intent(entry["intent"])
        del self._multipart_uploads[identity]

    def _clear_multipart_intent(self, identity):
        from truthound.stores.streaming.s3_multipart import pending

        previous = pending(self.session)
        remaining = dict(previous)
        del remaining[identity]
        self.session._multipart_intents = remaining
        try:
            self._write_session_state()
        except BaseException:
            self.session._multipart_intents = previous
            raise

    def _abort(self) -> None:
        failures = []
        for identity in list(self._multipart_uploads):
            try:
                self._abort_upload(identity)
            except Exception as error:
                self.metrics.record_error("multipart_abort:" + type(error).__name__)
                failures.append(error)
        if failures:
            raise ExceptionGroup("Native multipart cleanup failed", failures)

    def _write_session_state(self) -> None:
        """Write session state to S3."""
        _publish_manifest(self._s3, self.s3_config, self.session, self._guard_io)

    def _finalize(self) -> None:
        """Finalize the stream."""
        _validate_inventory(self._s3, self.s3_config, self.session, self._guard_io)


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
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()


class S3StreamReader(BaseStreamReader):
    """S3 streaming reader."""

    def __init__(
        self,
        run_id: str,
        s3_client: Any,
        config: StreamingS3Config,
        *,
        session: StreamSession | None = None,
        io_guard=None,
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
        self._session = session
        self._opening_guard = io_guard or (lambda: None)

        # Load manifest
        if session is None:
            self._load_manifest()
        else:
            if session.run_id != run_id:
                raise ValueError("Native S3 reader session identity differs")
            self._chunks = session.chunks

        settings = (
            replace(
                config,
                **{
                    key: getattr(self._session.config, key)
                    for key in self._session.config.to_dict()
                },
            )
            if self._session is not None
            else config
        )
        super().__init__(settings)
        self.set_io_guard(self._opening_guard)

    def _load_manifest(self) -> None:
        """Load manifest from S3."""
        try:
            self._session, _, _ = _read_manifest(
                self._s3,
                self._s3_config,
                self._run_id,
                self._opening_guard,
            )
            self._chunks = self._session.chunks
        except self._s3.exceptions.NoSuchKey:
            raise RuntimeError(
                "Native S3 stream manifest is missing; reconciliation is required"
            ) from None

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

        expected = f"{self._s3_config.prefix.rstrip('/')}/{self._run_id}/"
        if not key.startswith(expected) or "/" in key[len(expected) :]:
            raise ValueError("Native stream chunk must remain in its selected S3 run")
        self._guard_io()
        self._s3.head_object(
            Bucket=self._s3_config.bucket,
            Key=f"{self._s3_config.prefix.rstrip('/')}/{self._run_id}/_manifest.json",
            IfMatch=self._session._manifest_etag,
        )
        self._guard_io()
        response = self._s3.get_object(Bucket=self._s3_config.bucket, Key=key)
        body = response["Body"]
        try:
            return body.read(self.config.max_memory_mb * 1024**2 + 1)
        finally:
            body.close()


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
                raise ImportError("boto3 library required for S3 streaming store") from None

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

    def close(self) -> None:
        try:
            super().close()
        finally:
            if self._s3_client is not None:
                self._s3_client.close()
                self._s3_client = None
                self._initialized = False

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
        from truthound.stores.streaming._filesystem_io import component

        component(run_id)
        self.initialize()
        if self.get_session_for_run(run_id) is not None:
            raise ValueError("A native S3 stream run already exists")

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
            cached = self._active_sessions[session_id]
            self._guard_io()
            response = self._get_s3_client().head_object(
                Bucket=self._config.bucket,
                Key=f"{self._config.prefix.rstrip('/')}/{cached.run_id}/_manifest.json",
                IfMatch=cached._manifest_etag,
            )
            if _etag_receipt(response) != cached._manifest_etag:
                raise RuntimeError("Native S3 cached session receipt is stale")
            return cached

        for run_id in self.list_runs():
            session = self.get_session_for_run(run_id)
            if session is not None and session.session_id == session_id:
                return session
        return None

    def get_session_for_run(self, run_id: str) -> StreamSession | None:
        from truthound.stores.streaming._filesystem_io import component

        component(run_id)
        self._guard_io()
        client = self._get_s3_client()
        try:
            session, _, _ = _read_manifest(client, self._config, run_id, self._guard_io)
        except client.exceptions.NoSuchKey:
            return None
        return session

    def recovery_status(self, run_id: str) -> dict[str, Any]:
        from truthound.stores.streaming.s3_recovery import status

        self.initialize()
        return status(self, run_id)

    def reconcile_multipart(
        self, run_id: str, *, expected_fingerprint: str, receipts: dict[str, str] | None = None
    ) -> dict[str, Any]:
        from truthound.stores.streaming.s3_multipart import recover

        return recover(self, run_id, expected_fingerprint=expected_fingerprint, receipts=receipts)

    def reconcile_stream(
        self, run_id: str, *, resolution: str, expected_fingerprint: str
    ) -> dict[str, Any]:
        from truthound.stores.streaming.s3_recovery import reconcile

        self.initialize()
        return reconcile(
            self, run_id, resolution=resolution, expected_fingerprint=expected_fingerprint
        )

    def list_sessions(self, limit: int = 100, offset: int = 0) -> list[StreamSession]:
        if limit < 1 or offset < 0:
            raise ValueError("Invalid native streaming session page")
        result = []
        for run_id in self.list_runs()[offset : offset + limit]:
            session = self.get_session_for_run(run_id)
            if session is None:
                raise RuntimeError(
                    "Native S3 stream manifest is missing; reconciliation is required"
                )
            result.append(session)
        return result

    def resume_session(self, session_id: str) -> StreamSession:
        """Resume an interrupted session."""
        session = self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        if session.status in {StreamStatus.COMPLETED, StreamStatus.ABORTED}:
            raise ValueError("A terminal native S3 stream cannot be resumed")
        self._guard_io()
        _validate_inventory(self._get_s3_client(), self._config, session, self._guard_io)

        session.status = StreamStatus.ACTIVE
        session.updated_at = datetime.now()

        self._active_sessions[session_id] = session
        return session

    def _close_session(self, session: StreamSession) -> None:
        """Close and finalize a session."""
        if session.session_id in self._active_sessions:
            del self._active_sessions[session.session_id]
        if session.status not in {StreamStatus.FAILED, StreamStatus.ABORTED}:
            session.status = StreamStatus.COMPLETED
        session.updated_at = datetime.now()
        self._write_manifest(session)

    def _write_manifest(self, session: StreamSession) -> None:
        """Write session manifest to S3."""
        _publish_manifest(self._get_s3_client(), self._config, session, self._guard_io)

    # -------------------------------------------------------------------------
    # Writer Operations
    # -------------------------------------------------------------------------

    def create_writer(self, session: StreamSession) -> S3StreamWriter:
        """Create a writer for the session."""
        self.initialize()
        self._assert_writer_available(session)
        current = self.get_session_for_run(session.run_id)
        if (
            current is None
            or current.session_id != session.session_id
            or current.status in {StreamStatus.COMPLETED, StreamStatus.ABORTED}
            or current._manifest_etag != getattr(session, "_manifest_etag", None)
            or [chunk.to_dict() for chunk in current.chunks]
            != [chunk.to_dict() for chunk in session.chunks]
        ):
            raise RuntimeError("Native S3 writer session is stale")
        _validate_inventory(self._get_s3_client(), self._config, session, self._guard_io)
        settings = replace(
            self._config, **{key: getattr(session.config, key) for key in session.config.to_dict()}
        )
        return self._track_writer(
            session,
            S3StreamWriter(
                session=session,
                config=settings,
                s3_client=self._get_s3_client(),
            ),
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
        self._guard_io()
        session = self.get_session_for_run(run_id)
        if session is None:
            raise RuntimeError("Native S3 stream manifest is missing; reconciliation is required")
        _validate_inventory(self._get_s3_client(), self._config, session, self._guard_io)
        return self._track_reader(
            S3StreamReader(
                run_id=run_id,
                s3_client=self._get_s3_client(),
                config=self._config,
                session=session,
                io_guard=self._guard_io,
            )
        )

    async def create_async_reader(self, run_id: str) -> AsyncStreamReader:
        """Create an async reader for a run's results."""
        reader = self.create_reader(run_id)
        return AsyncStreamReader(reader)

    def iter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> Iterator[ValidatorResult]:
        """Iterate over results for a run."""
        reader = self.create_reader(run_id)
        with reader:
            yield from reader.iter_results()

    async def aiter_results(
        self,
        run_id: str,
        batch_size: int = 1000,
    ) -> AsyncIterator[ValidatorResult]:
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
        try:
            return reader._get_chunks()
        finally:
            reader.close()

    def get_chunk(self, chunk_info: ChunkInfo) -> list[ValidatorResult]:
        """Read only a confirmed chunk using its persisted run codec."""
        from truthound.stores.streaming._filesystem_io import component

        self._guard_io()
        prefix = f"s3://{self._config.bucket}/{self._config.prefix.rstrip('/')}/"
        if not chunk_info.path.startswith(prefix):
            raise RuntimeError("Native S3 chunk scope requires reconciliation")
        tail = chunk_info.path[len(prefix) :]
        if tail.count("/") != 1:
            raise RuntimeError("Native S3 chunk scope requires reconciliation")
        run_id, _ = tail.split("/", 1)
        component(run_id)
        session = self.get_session_for_run(run_id)
        if session is None or chunk_info.to_dict() not in [
            chunk.to_dict() for chunk in session.chunks
        ]:
            raise RuntimeError("Native S3 stream chunk is not confirmed by its manifest")
        reader = self.create_reader(run_id)
        try:
            return [_validator_result(record) for record in reader._load_chunk(chunk_info)]
        finally:
            reader.close()

    def delete_chunks(self, run_id: str) -> int:
        """Conditionally delete inspected current objects, including the manifest.

        Versioned buckets retain historical versions/delete markers. Partial or
        uncertain deletion leaves a blocking journal and never returns a count.
        """
        from truthound.stores.streaming._filesystem_io import component
        from truthound.stores.streaming.s3_recovery import delete

        component(run_id)
        self.initialize()
        return delete(self, run_id)

    # -------------------------------------------------------------------------
    # Validation Result Operations
    # -------------------------------------------------------------------------

    def stream_write_result(
        self,
        session: StreamSession,
        result: ValidatorResult,
    ) -> None:
        """Write a single validator result to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_result(result)

    def stream_write_batch(
        self,
        session: StreamSession,
        results: list[ValidatorResult],
    ) -> None:
        """Write a batch of validator results to the stream."""
        if session.session_id not in self._active_sessions:
            raise ValueError(f"Session not active: {session.session_id}")

        writer = self._get_or_create_writer(session)
        writer.write_results(results)

    def _get_or_create_writer(self, session: StreamSession) -> S3StreamWriter:
        """Get or create a writer for a session."""
        writer = self._writers.get(session.session_id)
        if writer is None or writer._closed:
            writer = self.create_writer(session)
        return writer

    def finalize_result(
        self,
        session: StreamSession,
        additional_metadata: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Finalize the streaming session and create a ValidationResult."""
        from truthound.stores.results import (
            ResultStatistics,
            ResultStatus,
            ValidationResult,
        )

        if session.status in {StreamStatus.FAILED, StreamStatus.ABORTED}:
            raise RuntimeError("A failed native stream cannot produce a completed result")
        writer = self._writers.get(session.session_id)
        if writer is not None:
            writer.close()

        # Aggregate statistics
        total_validators = 0
        passed_validators = 0
        failed_validators = 0
        total_issues = 0
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        all_results: list[ValidatorResult] = []
        for result in self.iter_results(session.run_id):
            all_results.append(result)
            total_validators += 1
            if result.success:
                passed_validators += 1
            else:
                failed_validators += 1
                total_issues += result.count
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
            total_issues=total_issues,
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
