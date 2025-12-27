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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Iterator, TextIO
from uuid import uuid4

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
    """CSV serializer."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns
        self._header_written = False

    def serialize(self, record: dict[str, Any]) -> bytes:
        """Serialize a single record to CSV format."""
        import csv
        from io import StringIO

        output = StringIO()
        if self.columns is None:
            self.columns = list(record.keys())

        writer = csv.DictWriter(output, fieldnames=self.columns, extrasaction="ignore")

        if not self._header_written:
            writer.writeheader()
            self._header_written = True

        writer.writerow(record)
        return output.getvalue().encode("utf-8")

    def serialize_batch(self, records: list[dict[str, Any]]) -> bytes:
        """Serialize a batch of records to CSV format."""
        import csv
        from io import StringIO

        if not records:
            return b""

        output = StringIO()
        if self.columns is None:
            self.columns = list(records[0].keys())

        writer = csv.DictWriter(output, fieldnames=self.columns, extrasaction="ignore")

        if not self._header_written:
            writer.writeheader()
            self._header_written = True

        for record in records:
            writer.writerow(record)

        return output.getvalue().encode("utf-8")

    def get_content_type(self) -> str:
        return "text/csv"


def get_serializer(format: StreamingFormat, **kwargs: Any) -> RecordSerializer:
    """Get a serializer for the specified format."""
    if format == StreamingFormat.JSONL or format == StreamingFormat.NDJSON:
        return JSONLSerializer()
    elif format == StreamingFormat.CSV:
        return CSVSerializer(columns=kwargs.get("columns"))
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
                raise ImportError("zstandard library required for zstd compression")
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
            raise ImportError("lz4 library required for lz4 compression")

    def get_extension(self) -> str:
        return ".lz4"


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
        self.session = session
        self.config = config
        self.serializer = serializer or get_serializer(config.format)
        self.compressor = compressor or get_compressor(config.compression)

        self.buffer = WriteBuffer(
            max_records=config.buffer_size,
            max_bytes=config.max_memory_mb * 1024 * 1024 // 4,  # 25% of max memory
        )

        self.metrics = session.metrics
        self._chunk_index = len(session.chunks)
        self._record_offset = sum(c.record_count for c in session.chunks)
        self._closed = False
        self._lock = threading.RLock()

        # Auto-flush timer
        self._last_flush_time = time.time()
        self._flush_timer: threading.Timer | None = None

        if config.flush_interval_seconds > 0:
            self._start_flush_timer()

    def _start_flush_timer(self) -> None:
        """Start the auto-flush timer."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()

        self._flush_timer = threading.Timer(
            self.config.flush_interval_seconds,
            self._auto_flush,
        )
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _auto_flush(self) -> None:
        """Auto-flush callback."""
        if not self._closed and not self.buffer.is_empty():
            try:
                self.flush()
            except Exception:
                pass  # Ignore auto-flush errors
        if not self._closed:
            self._start_flush_timer()

    def write(self, record: dict[str, Any]) -> None:
        """Write a single record.

        Args:
            record: The record to write.
        """
        if self._closed:
            raise StreamWriteError("Writer is closed")

        with self._lock:
            if self.buffer.add(record):
                self.flush()

    def write_result(self, result: "ValidatorResult") -> None:
        """Write a ValidatorResult.

        Args:
            result: The validator result to write.
        """
        self.write(result.to_dict())

    def write_batch(self, records: list[dict[str, Any]]) -> None:
        """Write a batch of records.

        Args:
            records: The records to write.
        """
        if self._closed:
            raise StreamWriteError("Writer is closed")

        with self._lock:
            if self.buffer.add_batch(records):
                self.flush()

    def write_results(self, results: list["ValidatorResult"]) -> None:
        """Write a batch of ValidatorResults.

        Args:
            results: The validator results to write.
        """
        self.write_batch([r.to_dict() for r in results])

    def flush(self) -> ChunkInfo:
        """Flush buffered records to storage.

        Returns:
            Information about the written chunk.
        """
        if self._closed:
            raise StreamWriteError("Writer is closed")

        with self._lock:
            records = self.buffer.clear()
            if not records:
                return ChunkInfo(
                    chunk_id="",
                    chunk_index=-1,
                    record_count=0,
                    byte_size=0,
                    start_offset=self._record_offset,
                    end_offset=self._record_offset,
                )

            # Serialize
            data = self.serializer.serialize_batch(records)

            # Compute checksum before compression
            checksum = hashlib.md5(data).hexdigest()

            # Compress
            compressed_data = self.compressor.compress(data)

            # Create chunk info
            chunk_id = f"{self.session.run_id}_chunk_{self._chunk_index:06d}"
            chunk_info = ChunkInfo(
                chunk_id=chunk_id,
                chunk_index=self._chunk_index,
                record_count=len(records),
                byte_size=len(compressed_data),
                start_offset=self._record_offset,
                end_offset=self._record_offset + len(records),
                checksum=checksum,
            )

            # Write to storage
            try:
                self._write_chunk(chunk_info, compressed_data)
            except Exception as e:
                # Retry logic
                for attempt in range(self.config.max_retries):
                    try:
                        time.sleep(self.config.retry_delay_seconds * (2**attempt))
                        self._write_chunk(chunk_info, compressed_data)
                        self.metrics.retry_count += 1
                        break
                    except Exception:
                        if attempt == self.config.max_retries - 1:
                            self.metrics.record_error(str(e))
                            raise StreamFlushError(f"Failed to write chunk: {e}")

            # Update state
            self.session.chunks.append(chunk_info)
            self._chunk_index += 1
            self._record_offset += len(records)
            self._last_flush_time = time.time()

            # Update metrics
            self.metrics.record_write(len(records), len(compressed_data))
            self.metrics.record_chunk(is_write=True)
            self.metrics.flush_count += 1

            # Checkpoint if needed
            if (
                self.config.enable_checkpoints
                and self._record_offset - self.session.checkpoint_offset
                >= self.config.checkpoint_interval
            ):
                self._write_checkpoint()

            return chunk_info

    @abstractmethod
    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write a chunk to storage.

        Args:
            chunk_info: Chunk metadata.
            data: Compressed chunk data.
        """
        pass

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
        """Close the writer and finalize."""
        if self._closed:
            return

        with self._lock:
            # Stop flush timer
            if self._flush_timer is not None:
                self._flush_timer.cancel()
                self._flush_timer = None

            # Flush remaining records
            if not self.buffer.is_empty():
                self.flush()

            # Update session
            self.session.status = StreamStatus.COMPLETED
            self.session.updated_at = datetime.now()
            self.metrics.finish()

            # Write final state
            self._write_session_state()
            self._finalize()

            self._closed = True

    @abstractmethod
    def _finalize(self) -> None:
        """Finalize the stream (e.g., create manifest)."""
        pass

    def __enter__(self) -> "BaseStreamWriter":
        """Context manager entry."""
        self.metrics.start()
        self.session.status = StreamStatus.ACTIVE
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.session.status = StreamStatus.FAILED
            self.metrics.record_error(str(exc_val))
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
    ):
        """Initialize the filesystem writer.

        Args:
            session: The streaming session.
            config: Streaming configuration.
            base_path: Base directory for writing chunks.
            serializer: Record serializer.
            compressor: Data compressor.
        """
        super().__init__(session, config, serializer, compressor)
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create run directory
        self.run_path = self.base_path / session.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)

    def _get_chunk_path(self, chunk_info: ChunkInfo) -> Path:
        """Get the file path for a chunk."""
        ext = {
            StreamingFormat.JSONL: ".jsonl",
            StreamingFormat.NDJSON: ".ndjson",
            StreamingFormat.CSV: ".csv",
            StreamingFormat.PARQUET: ".parquet",
        }.get(self.config.format, ".jsonl")

        ext += self.compressor.get_extension()
        return self.run_path / f"{chunk_info.chunk_id}{ext}"

    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Write a chunk to the filesystem."""
        chunk_path = self._get_chunk_path(chunk_info)
        chunk_info.path = str(chunk_path)

        # Atomic write: write to temp file then rename
        temp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
        try:
            with open(temp_path, "wb") as f:
                f.write(data)
            temp_path.rename(chunk_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _write_session_state(self) -> None:
        """Write session state to a manifest file."""
        manifest_path = self.run_path / "_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.session.to_dict(), f, indent=2, default=str)

    def _finalize(self) -> None:
        """Create final manifest."""
        # Already handled in _write_session_state
        pass


class BufferedStreamWriter(BaseStreamWriter):
    """In-memory buffered writer for testing or small results.

    Accumulates all records in memory and writes on close.
    """

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
        self._all_records: list[dict[str, Any]] = []

    def _write_chunk(self, chunk_info: ChunkInfo, data: bytes) -> None:
        """Accumulate chunk data in memory."""
        self.output.write(data)

    def _write_session_state(self) -> None:
        """No-op for buffered writer."""
        pass

    def _finalize(self) -> None:
        """No-op for buffered writer."""
        pass

    def get_output(self) -> bytes:
        """Get the accumulated output."""
        if isinstance(self.output, BytesIO):
            return self.output.getvalue()
        return b""


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
        self._loop = asyncio.get_event_loop()

    async def write(self, record: dict[str, Any]) -> None:
        """Write a single record asynchronously."""
        await self._loop.run_in_executor(None, self._writer.write, record)

    async def write_result(self, result: "ValidatorResult") -> None:
        """Write a ValidatorResult asynchronously."""
        await self._loop.run_in_executor(None, self._writer.write_result, result)

    async def write_batch(self, records: list[dict[str, Any]]) -> None:
        """Write a batch of records asynchronously."""
        await self._loop.run_in_executor(None, self._writer.write_batch, records)

    async def write_results(self, results: list["ValidatorResult"]) -> None:
        """Write a batch of ValidatorResults asynchronously."""
        await self._loop.run_in_executor(None, self._writer.write_results, results)

    async def flush(self) -> ChunkInfo:
        """Flush buffered records asynchronously."""
        return await self._loop.run_in_executor(None, self._writer.flush)

    async def close(self) -> None:
        """Close the writer asynchronously."""
        await self._loop.run_in_executor(None, self._writer.close)

    async def __aenter__(self) -> "AsyncStreamWriter":
        """Async context manager entry."""
        self._writer.metrics.start()
        self._writer.session.status = StreamStatus.ACTIVE
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if exc_type is not None:
            self._writer.session.status = StreamStatus.FAILED
            self._writer.metrics.record_error(str(exc_val))
        await self.close()

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
