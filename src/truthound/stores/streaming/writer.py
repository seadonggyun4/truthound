"""Streaming writers for incremental result storage.

This module provides writers that can incrementally write validation results
to storage without holding all results in memory.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingMetrics,
    StreamSession,
    StreamStatus,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidatorResult


# =============================================================================
# Exceptions
# =============================================================================


class StreamWriteError(Exception):
    """Error during streaming write operation."""

    pass


class StreamBufferOverflowError(StreamWriteError):
    """Buffer exceeded maximum size."""

    pass


class StreamFlushError(StreamWriteError):
    """Error flushing buffer to storage."""

    pass


# =============================================================================
# Serializers
# =============================================================================


class RecordSerializer(ABC):
    """Abstract record serializer."""

    @abstractmethod
    def serialize(self, record: dict[str, Any]) -> bytes:
        """Serialize a single record."""
        pass

    @abstractmethod
    def serialize_batch(self, records: list[dict[str, Any]]) -> bytes:
        """Serialize a batch of records."""
        pass

    @abstractmethod
    def get_content_type(self) -> str:
        """Get the content type for this format."""
        pass


class JSONLSerializer(RecordSerializer):
    """JSON Lines serializer."""

    def serialize(self, record: dict[str, Any]) -> bytes:
        """Serialize a single record to JSONL format."""
        return (json.dumps(record, default=str) + "\n").encode("utf-8")

    def serialize_batch(self, records: list[dict[str, Any]]) -> bytes:
        """Serialize a batch of records to JSONL format."""
        lines = [json.dumps(r, default=str) for r in records]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def get_content_type(self) -> str:
        return "application/x-ndjson"


class CSVSerializer(RecordSerializer):
    """Readable CSV columns plus a versioned JSON record column for exact types."""

    RECORD_COLUMN = "__truthound_record_v1__"

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self._explicit_columns = columns is not None
        self._header_written = False
        self._record_column = self.RECORD_COLUMN

    def serialize(self, record: dict[str, Any]) -> bytes:
        return self.serialize_batch([record])

    def serialize_batch(self, records: list[dict[str, Any]]) -> bytes:
        import csv

        if not records:
            return b""
        if self.columns is None:
            self.columns = list(dict.fromkeys(key for record in records for key in record))
        while self._record_column in self.columns:
            self._record_column += "_"
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=[*self.columns, self._record_column], extrasaction="ignore")
        if not self._header_written:
            writer.writeheader()
            self._header_written = True
        for record in records:
            original = {key: record.get(key) for key in self.columns} if self._explicit_columns else record
            writer.writerow({**record, self._record_column: json.dumps(original, ensure_ascii=False, default=str)})
        return output.getvalue().encode("utf-8")

    def get_content_type(self) -> str:
        return "text/csv"


class ParquetSerializer(RecordSerializer):
    """Independent Parquet chunks with JSON records, including empty structures."""

    def serialize(self, record: dict[str, Any]) -> bytes:
        return self.serialize_batch([record])

    def serialize_batch(self, records: list[dict[str, Any]]) -> bytes:
        from io import BytesIO

        import polars as pl

        output = BytesIO()
        pl.DataFrame({"__truthound_record_v1__": [json.dumps(record, ensure_ascii=False, default=str) for record in records]},
                     schema={"__truthound_record_v1__": pl.String}).write_parquet(output, compression="uncompressed")
        return output.getvalue()

    def get_content_type(self) -> str:
        return "application/vnd.apache.parquet"


def get_serializer(format: StreamingFormat, **kwargs: Any) -> RecordSerializer:
    """Get a serializer for the specified format."""
    if format == StreamingFormat.JSONL or format == StreamingFormat.NDJSON:
        return JSONLSerializer()
    elif format == StreamingFormat.CSV:
        return CSVSerializer(columns=kwargs.get("columns"))
    elif format == StreamingFormat.PARQUET:
        return ParquetSerializer()
    else:
        raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Compressors
# =============================================================================


class Compressor(ABC):
    """Abstract compressor."""

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Get file extension for compressed files."""
        pass


class NoCompressor(Compressor):
    """No compression."""

    def compress(self, data: bytes) -> bytes:
        return data

    def get_extension(self) -> str:
        return ""


class GzipCompressor(Compressor):
    """Gzip compression."""

    def __init__(self, level: int = 6):
        self.level = level

    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.level)

    def get_extension(self) -> str:
        return ".gz"


class ZstdCompressor(Compressor):
    """Zstandard compression."""

    def __init__(self, level: int = 3):
        self.level = level
        self._compressor = None

    def _get_compressor(self) -> Any:
        if self._compressor is None:
            try:
                import zstandard as zstd

                self._compressor = zstd.ZstdCompressor(level=self.level)
            except ImportError:
                raise ImportError("zstandard library required for zstd compression") from None
        return self._compressor

    def compress(self, data: bytes) -> bytes:
        compressor = self._get_compressor()
        return compressor.compress(data)

    def get_extension(self) -> str:
        return ".zst"


class LZ4Compressor(Compressor):
    """LZ4 compression."""

    def compress(self, data: bytes) -> bytes:
        try:
            import lz4.frame

            return lz4.frame.compress(data)
        except ImportError:
            raise ImportError("lz4 library required for lz4 compression") from None

    def get_extension(self) -> str:
        return ".lz4"


class SnappyCompressor(Compressor):
    """Reuse Core's native Snappy provider and standard raw block format."""

    def compress(self, data: bytes) -> bytes:
        from truthound.stores.compression.providers import SnappyCompressor as NativeSnappy

        return NativeSnappy().compress(data)

    def get_extension(self) -> str:
        return ".snappy"


def get_compressor(compression: CompressionType, **kwargs: Any) -> Compressor:
    """Get a compressor for the specified type."""
    if compression == CompressionType.NONE:
        return NoCompressor()
    elif compression == CompressionType.GZIP:
        return GzipCompressor(level=kwargs.get("level", 6))
    elif compression == CompressionType.ZSTD:
        return ZstdCompressor(level=kwargs.get("level", 3))
    elif compression == CompressionType.LZ4:
        return LZ4Compressor()
    elif compression == CompressionType.SNAPPY:
        return SnappyCompressor()
    else:
        raise ValueError(f"Unsupported compression: {compression}")


# =============================================================================
# Buffer Management
# =============================================================================


@dataclass
class WriteBuffer:
    """In-memory buffer for batching writes.

    Attributes:
        max_records: Maximum records before flush.
        max_bytes: Maximum bytes before flush.
        records: Buffered records.
        byte_size: Current buffer size in bytes.
    """

    max_records: int = 1000
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    records: list[dict[str, Any]] = field(default_factory=list)
    byte_size: int = 0

    def add(self, record: dict[str, Any]) -> bool:
        """Add a record to the buffer.

        Returns:
            True if buffer should be flushed.
        """
        record_size = len(json.dumps(record, default=str).encode("utf-8"))

        self.records.append(record)
        self.byte_size += record_size

        return self.should_flush()

    def add_batch(self, records: list[dict[str, Any]]) -> bool:
        """Add multiple records to the buffer.

        Returns:
            True if buffer should be flushed.
        """
        for record in records:
            record_size = len(json.dumps(record, default=str).encode("utf-8"))
            self.records.append(record)
            self.byte_size += record_size

        return self.should_flush()

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return len(self.records) >= self.max_records or self.byte_size >= self.max_bytes

    def clear(self) -> list[dict[str, Any]]:
        """Clear and return buffered records."""
        records = self.records
        self.records = []
        self.byte_size = 0
        return records

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.records) == 0


# =============================================================================
# Base Writer
# =============================================================================


class BaseStreamWriter(ABC):
    """Base class for streaming writers.

    Handles buffering, serialization, and compression.
    """

    def __init__(
        self,
        session: StreamSession,
        config: StreamingConfig,
        serializer: RecordSerializer | None = None,
        compressor: Compressor | None = None,
    ):
        """Initialize the writer.

        Args:
            session: The streaming session.
            config: Streaming configuration.
            serializer: Record serializer (auto-selected if None).
            compressor: Data compressor (auto-selected if None).
        """
        config.validate()
        self.session = session
        self.session.config = config
        self.config = config
        self.serializer = serializer or get_serializer(config.format)
        self.compressor = compressor or get_compressor(config.compression)

        self.buffer = WriteBuffer(
            max_records=min(config.buffer_size, config.chunk_size),
            max_bytes=config.max_memory_mb * 1024 * 1024 // 4,  # 25% of max memory
        )

        self.metrics = session.metrics
        self._chunk_index = len(session.chunks)
        self._record_offset = sum(c.record_count for c in session.chunks)
        self._closed = False
        self._closing = False
        self._failure = None
        self._io_guard = None
        self._ownership_lease = None
        self._lock = threading.RLock()

        # Auto-flush timer
        self._last_flush_time = time.time()
        self._flush_timer: threading.Timer | None = None


    def set_io_guard(self, guard) -> None:
        self._io_guard = guard

    def _guard_io(self) -> None:
        if self._io_guard is not None:
            self._io_guard()

    def _ensure_writable(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise StreamWriteError("Writer is closed")
        self._guard_io()

    def _fail(self, error, stage="flush"):
        if self._failure is None:
            self._failure = StreamFlushError(f"Native streaming {stage} failed ({type(error).__name__})")
            self.session.status = StreamStatus.FAILED
            self.metrics.record_error(f"{stage}:{type(error).__name__}")
        return self._failure

    def _start_flush_timer(self) -> None:
        if self._closed or self._closing or self._failure is not None or self.config.flush_interval_seconds <= 0 or self.buffer.is_empty():
            return
        if self._flush_timer is not None:
            return
        self._flush_timer = threading.Timer(self.config.flush_interval_seconds, self._auto_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _auto_flush(self) -> None:
        self._flush_timer = None
        if self._closed or self._closing or self._failure is not None:
            return
        try:
            if not self.buffer.is_empty():
                self.flush()
        except Exception as error:
            self._fail(error, "background_flush")
        finally:
            self._start_flush_timer()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._ensure_writable()
            if self.buffer.add(record):
                self.flush()
            self._start_flush_timer()

    def write_result(self, result: ValidatorResult) -> None:
        """Write a ValidatorResult.

        Args:
            result: The validator result to write.
        """
        self.write(result.to_dict())

    def write_batch(self, records: list[dict[str, Any]]) -> None:
        # Apply the native buffer/chunk boundary to each record, not one huge batch.
        with self._lock:
            self._ensure_writable()
            for record in records:
                self.write(record)

    def write_results(self, results: list[ValidatorResult]) -> None:
        """Write a batch of ValidatorResults.

        Args:
            results: The validator results to write.
        """
        self.write_batch([r.to_dict() for r in results])

    def flush(self) -> ChunkInfo:
        with self._lock:
            self._ensure_writable()
            records = list(self.buffer.records)
            if not records:
                return ChunkInfo("", -1, 0, 0, self._record_offset, self._record_offset)
            try:
                if isinstance(self.serializer, CSVSerializer) and getattr(self, "independent_chunks", True):
                    self.serializer._header_written = False
                data = self.serializer.serialize_batch(records)
                maximum = self.config.max_memory_mb * 1024 ** 2
                if len(data) > maximum:
                    raise StreamBufferOverflowError("Native serialized chunk exceeds its memory bound")
                checksum = hashlib.md5(data).hexdigest()
                compressed_data = self.compressor.compress(data)
                if len(compressed_data) > maximum:
                    raise StreamBufferOverflowError("Native encoded chunk exceeds its memory bound")
                chunk_info = ChunkInfo(
                    chunk_id=f"{self.session.run_id}_chunk_{self._chunk_index:06d}",
                    chunk_index=self._chunk_index, record_count=len(records),
                    byte_size=len(compressed_data), start_offset=self._record_offset,
                    end_offset=self._record_offset + len(records), checksum=checksum,
                )
                # Retry requires a backend's explicit idempotent-write contract.
                retries = self.config.max_retries if getattr(self, "idempotent_chunk_write", False) else 0
                for attempt in range(retries + 1):
                    self._guard_io()
                    try:
                        self._write_chunk(chunk_info, compressed_data)
                        break
                    except Exception:
                        if attempt == retries:
                            raise
                        self.metrics.retry_count += 1
                        time.sleep(self.config.retry_delay_seconds * 2 ** attempt)
                self.buffer.clear()
                self.session.chunks.append(chunk_info)
                self._chunk_index += 1
                self._record_offset += len(records)
                self._last_flush_time = time.time()
                self.metrics.record_write(len(records), len(compressed_data))
                self.metrics.record_chunk(is_write=True)
                self.metrics.flush_count += 1
                if self.config.enable_checkpoints and self._record_offset - self.session.checkpoint_offset >= self.config.checkpoint_interval:
                    self._guard_io()
                    self._write_checkpoint()
                return chunk_info
            except Exception as error:
                raise self._fail(error) from None

    @abstractmethod
    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write a chunk to storage.

        Args:
            chunk_info: Chunk metadata.
            data: Compressed chunk data.
        """
        pass

    def _abort(self) -> None:
        """Release resources without publishing a completed stream."""
        return None

    def _write_checkpoint(self) -> None:
        """Write a checkpoint for recovery."""
        self.session.checkpoint_offset = self._record_offset
        self.session.updated_at = datetime.now()
        self._write_session_state()

    @abstractmethod
    def _write_session_state(self) -> None:
        """Write session state for recovery."""
        pass

    def close(self) -> None:
        try:
            self._close_impl()
        finally:
            if self._closed and self._ownership_lease is not None:
                self._ownership_lease.close()
                self._ownership_lease = None

    def abort(self) -> None:
        """Stop without flushing pending records; retain terminal failure evidence."""
        self._closing = True
        timer, self._flush_timer = self._flush_timer, None
        if timer is not None:
            timer.cancel()
            if timer is not threading.current_thread():
                timer.join(timeout=1)
                if timer.is_alive():
                    raise self._fail(StreamWriteError("Native background flush is still running"), "abort")
        with self._lock:
            if self._closed:
                return
            try:
                self._abort()
                self._closed = True
                if self.session.status != StreamStatus.FAILED:
                    self.session.status = StreamStatus.ABORTED
                self.metrics.finish()
                self.session.updated_at = datetime.now()
                self._guard_io()
                self._write_session_state()
            except Exception as error:
                raise self._fail(error, "abort") from None
            finally:
                if self._closed and self._ownership_lease is not None:
                    self._ownership_lease.close()
                    self._ownership_lease = None

    def _close_impl(self) -> None:
        self._closing = True
        timer, self._flush_timer = self._flush_timer, None
        if timer is not None:
            timer.cancel()
            if timer is not threading.current_thread():
                timer.join(timeout=1)
                if timer.is_alive():
                    raise self._fail(StreamWriteError("Native background flush is still running"), "close")
        with self._lock:
            if self._closed:
                if self._failure is not None:
                    raise self._failure
                return
            if self._failure is not None or self.session.status == StreamStatus.FAILED:
                self.session.status = StreamStatus.FAILED
                self.metrics.finish()
                try:
                    self._abort()
                except Exception as error:
                    self.metrics.record_error("abort:" + type(error).__name__)
                else:
                    self._closed = True
                with suppress(Exception):
                    self._guard_io()
                    self._write_session_state()
                raise self._failure or self._fail(StreamWriteError("Native session failed"), "close")
            try:
                if not self.buffer.is_empty():
                    self.flush()
                self._guard_io()
                self.session.updated_at = datetime.now()
                self.metrics.finish()
                self._finalize()
                self.session.status = StreamStatus.COMPLETED
                self._write_session_state()
                self._closed = True
            except Exception as error:
                failure = self._fail(error, "close")
                with suppress(Exception):
                    self._abort()
                    self._closed = True
                with suppress(Exception):
                    self._guard_io()
                    self._write_session_state()
                raise failure from None

    @abstractmethod
    def _finalize(self) -> None:
        """Finalize the stream (e.g., create manifest)."""
        pass

    def __enter__(self) -> BaseStreamWriter:
        """Context manager entry."""
        self.metrics.start()
        self.session.status = StreamStatus.ACTIVE
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self._fail(exc_val, "context")
            with suppress(StreamFlushError):
                self.close()
            return
        self.close()


# =============================================================================
# Concrete Writers
# =============================================================================


class StreamingResultWriter(BaseStreamWriter):
    """Filesystem-based streaming writer.

    Writes records to JSONL files on the local filesystem.
    """

    def __init__(
        self,
        session: StreamSession,
        config: StreamingConfig,
        base_path: Path | str,
        serializer: RecordSerializer | None = None,
        compressor: Compressor | None = None,
        manifest_filename: str = "_manifest.json",
        require_manifest: bool = False,
    ):
        """Initialize the filesystem writer.

        Args:
            session: The streaming session.
            config: Streaming configuration.
            base_path: Base directory for writing chunks.
            serializer: Record serializer.
            compressor: Data compressor.
        """
        from truthound.stores.streaming._filesystem_io import component, contained_path

        component(session.run_id)
        component(manifest_filename)
        selected = Path(base_path).resolve()
        run_path = contained_path(selected, selected / session.run_id)
        super().__init__(session, config, serializer, compressor)
        self.base_path = Path(base_path)
        self.manifest_filename = manifest_filename
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create run directory
        self.run_path = run_path
        from truthound.stores.streaming._filesystem_io import StreamFileLease, run_lease_path

        self._ownership_lease = StreamFileLease(run_lease_path(self.base_path, session.run_id))
        try:
            if require_manifest and not self.run_path.is_dir():
                raise RuntimeError("Native stream run no longer exists")
            self.run_path.mkdir(parents=True, exist_ok=True)
            manifest = contained_path(self.run_path, self.manifest_filename)
            if require_manifest and not manifest.exists():
                raise RuntimeError("Native stream manifest is missing; reconciliation is required")
            if manifest.exists():
                with manifest.open(encoding="utf-8") as stream:
                    stored = StreamSession.from_dict(json.load(stream))
                if stored.session_id != session.session_id or stored.status in {
                    StreamStatus.COMPLETED, StreamStatus.ABORTED,
                } or [chunk.to_dict() for chunk in stored.chunks] != [chunk.to_dict() for chunk in session.chunks]:
                    raise RuntimeError("Stale native stream state requires reconciliation before writing")
        except BaseException:
            self._ownership_lease.close()
            self._ownership_lease = None
            raise

    def _get_chunk_path(self, chunk_info: ChunkInfo) -> Path:
        """Get the file path for a chunk."""
        ext = {
            StreamingFormat.JSONL: ".jsonl",
            StreamingFormat.NDJSON: ".ndjson",
            StreamingFormat.CSV: ".csv",
            StreamingFormat.PARQUET: ".parquet",
        }.get(self.config.format, ".jsonl")

        ext += self.compressor.get_extension()
        from truthound.stores.streaming._filesystem_io import component, contained_path

        return contained_path(self.run_path, component(chunk_info.chunk_id) + ext)

    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write a chunk to the filesystem."""
        chunk_path = self._get_chunk_path(chunk_info)
        chunk_info.path = str(chunk_path)

        from truthound.stores.streaming._filesystem_io import publish

        publish(chunk_path, data, replace=False)

    def _write_session_state(self) -> None:
        """Write session state to a manifest file."""
        from truthound.stores.streaming._filesystem_io import contained_path, publish

        manifest_path = contained_path(self.run_path, self.manifest_filename)
        publish(manifest_path, json.dumps(self.session.to_dict(), indent=2, default=str).encode("utf-8"))

    def _finalize(self) -> None:
        """Create final manifest."""
        # Already handled in _write_session_state
        pass


class BufferedStreamWriter(BaseStreamWriter):
    """In-memory buffered writer for testing or small results.

    Accumulates all records in memory and writes on close.
    """

    independent_chunks = True

    def __init__(
        self,
        session: StreamSession,
        config: StreamingConfig,
        output: BinaryIO | None = None,
    ):
        """Initialize the buffered writer.

        Args:
            session: The streaming session.
            config: Streaming configuration.
            output: Output stream (BytesIO created if None).
        """
        super().__init__(session, config)
        self.output = output or BytesIO()
        if not self.output.seekable():
            raise ValueError("A native buffered stream requires a seekable binary output")
        self._all_records: list[dict[str, Any]] = []

    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Publish one valid document, rather than concatenate codec/file frames."""
        from truthound.stores.streaming.reader import get_decompressor, get_deserializer

        incoming = list(get_deserializer(self.config.format).deserialize(
            get_decompressor(self.config.compression).decompress(data)
        ))
        records = self._all_records + incoming
        serialized = get_serializer(self.config.format).serialize_batch(records)
        if len(serialized) > self.config.max_memory_mb * 1024 ** 2:
            raise StreamBufferOverflowError("Native buffered output exceeds its memory bound")
        content = self.compressor.compress(serialized)
        self._guard_io()
        self.output.seek(0)
        written = self.output.write(content)
        if written != len(content):
            raise StreamWriteError("Native buffered output acknowledged a short write")
        self.output.truncate()
        self._all_records = records

    def _write_session_state(self) -> None:
        """No-op for buffered writer."""
        pass

    def _finalize(self) -> None:
        """Flush the caller's output; ownership remains with the caller."""
        self.output.flush()

    def get_output(self) -> bytes:
        """Get the accumulated output."""
        if isinstance(self.output, BytesIO):
            return self.output.getvalue()
        raise TypeError("Native get_output requires an in-memory output")


class AsyncStreamWriter:
    """Async wrapper for streaming writers.

    Provides async interface for any BaseStreamWriter.
    """

    def __init__(self, writer: BaseStreamWriter):
        """Initialize the async writer.

        Args:
            writer: The underlying synchronous writer.
        """
        self._writer = writer

    def _get_loop(self):
        return asyncio.get_running_loop()

    async def write(self, record: dict[str, Any]) -> None:
        """Write a single record asynchronously."""
        await self._get_loop().run_in_executor(None, self._writer.write, record)

    async def write_result(self, result: ValidatorResult) -> None:
        """Write a ValidatorResult asynchronously."""
        await self._get_loop().run_in_executor(None, self._writer.write_result, result)

    async def write_batch(self, records: list[dict[str, Any]]) -> None:
        """Write a batch of records asynchronously."""
        await self._get_loop().run_in_executor(None, self._writer.write_batch, records)

    async def write_results(self, results: list[ValidatorResult]) -> None:
        """Write a batch of ValidatorResults asynchronously."""
        await self._get_loop().run_in_executor(None, self._writer.write_results, results)

    async def flush(self) -> ChunkInfo:
        """Flush buffered records asynchronously."""
        return await self._get_loop().run_in_executor(None, self._writer.flush)

    async def close(self) -> None:
        """Close the writer asynchronously."""
        await self._get_loop().run_in_executor(None, self._writer.close)

    async def abort(self) -> None:
        await self._get_loop().run_in_executor(None, self._writer.abort)

    async def __aenter__(self) -> AsyncStreamWriter:
        """Async context manager entry."""
        self._writer.metrics.start()
        self._writer.session.status = StreamStatus.ACTIVE
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._get_loop().run_in_executor(None, self._writer.__exit__, exc_type, exc_val, exc_tb)

    @property
    def session(self) -> StreamSession:
        """Get the streaming session."""
        return self._writer.session

    @property
    def metrics(self) -> StreamingMetrics:
        """Get the streaming metrics."""
        return self._writer.metrics


# =============================================================================
# Factory Functions
# =============================================================================


def create_stream_writer(
    session: StreamSession,
    config: StreamingConfig,
    backend: str = "filesystem",
    **kwargs: Any,
) -> BaseStreamWriter:
    """Create a streaming writer for the specified backend.

    Args:
        session: The streaming session.
        config: Streaming configuration.
        backend: Storage backend ("filesystem", "memory", "s3", "gcs", "database").
        **kwargs: Backend-specific options.

    Returns:
        A streaming writer instance.
    """
    if backend == "filesystem":
        base_path = kwargs.get("base_path", ".truthound/streaming")
        return StreamingResultWriter(session, config, base_path)
    elif backend == "memory":
        output = kwargs.get("output")
        return BufferedStreamWriter(session, config, output)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
