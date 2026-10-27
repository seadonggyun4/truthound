"""Streaming readers for memory-efficient result retrieval.

This module provides readers that can iterate over stored validation results
without loading everything into memory.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator  # noqa: TC003 - public annotation introspection
from pathlib import Path
from typing import TYPE_CHECKING, Any

from truthound.stores.streaming.base import (
    ChunkInfo,
    CompressionType,
    StreamingConfig,
    StreamingFormat,
    StreamingMetrics,
    StreamSession,
)

if TYPE_CHECKING:
    from truthound.stores.results import ValidatorResult


# =============================================================================
# Exceptions
# =============================================================================


class StreamReadError(Exception):
    """Error during streaming read operation."""

    pass


class ChunkNotFoundError(StreamReadError):
    """Chunk not found in storage."""

    pass


def _validator_result(record):
    """Typed verdict iteration must not interpret a legacy 'False' as truthy."""
    from truthound.stores.results import ValidatorResult

    if (
        not isinstance(record, dict) or not isinstance(record.get("success"), bool)
        or not isinstance(record.get("validator_name"), str) or not record["validator_name"]
        or not isinstance(record.get("count", 0), int) or isinstance(record.get("count", 0), bool)
        or record.get("count", 0) < 0
    ):
        raise StreamReadError("Native stream contains an untyped validation verdict")
    return ValidatorResult.from_dict(record)


class CorruptedChunkError(StreamReadError):
    """Chunk data is corrupted."""

    pass


# =============================================================================
# Deserializers
# =============================================================================


class RecordDeserializer(ABC):
    """Abstract record deserializer."""

    @abstractmethod
    def deserialize(self, data: bytes) -> Iterator[dict[str, Any]]:
        """Deserialize data into records."""
        pass


class JSONLDeserializer(RecordDeserializer):
    """JSON Lines deserializer."""

    def deserialize(self, data: bytes) -> Iterator[dict[str, Any]]:
        """Deserialize JSONL data into records."""
        text = data.decode("utf-8")
        for line in text.strip().split("\n"):
            if line.strip():
                yield json.loads(line)


class CSVDeserializer(RecordDeserializer):
    """CSV deserializer."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def deserialize(self, data: bytes) -> Iterator[dict[str, Any]]:
        """Deserialize CSV data into records."""
        import csv
        from io import StringIO

        text = data.decode("utf-8")
        reader = csv.DictReader(StringIO(text), fieldnames=self.columns)
        columns = reader.fieldnames or []
        record_column = columns[-1] if columns and columns[-1].startswith("__truthound_record_v1__") else None
        for row in reader:
            if record_column is not None:
                value = json.loads(row[record_column])
                if not isinstance(value, dict):
                    raise StreamReadError("Invalid native CSV record payload")
                yield value
            else:
                yield dict(row)


class ParquetDeserializer(RecordDeserializer):
    def deserialize(self, data: bytes) -> Iterator[dict[str, Any]]:
        from io import BytesIO

        import polars as pl

        frame = pl.read_parquet(BytesIO(data))
        if frame.columns == ["__truthound_record_v1__"]:
            for item in frame.get_column("__truthound_record_v1__"):
                value = json.loads(item)
                if not isinstance(value, dict):
                    raise StreamReadError("Invalid native Parquet record payload")
                yield value
        else:
            yield from frame.iter_rows(named=True)


def get_deserializer(format: StreamingFormat, **kwargs: Any) -> RecordDeserializer:
    """Get a deserializer for the specified format."""
    if format == StreamingFormat.JSONL or format == StreamingFormat.NDJSON:
        return JSONLDeserializer()
    elif format == StreamingFormat.CSV:
        return CSVDeserializer(columns=kwargs.get("columns"))
    elif format == StreamingFormat.PARQUET:
        return ParquetDeserializer()
    else:
        raise ValueError(f"Unsupported format: {format}")


# =============================================================================
# Decompressors
# =============================================================================


class Decompressor(ABC):
    """Abstract decompressor."""

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass


class NoDecompressor(Decompressor):
    """No decompression."""

    def decompress(self, data: bytes) -> bytes:
        return data


class GzipDecompressor(Decompressor):
    """Gzip decompression."""

    def decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


class ZstdDecompressor(Decompressor):
    """Zstandard decompression."""

    def decompress(self, data: bytes) -> bytes:
        try:
            import zstandard as zstd

            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        except ImportError:
            raise ImportError("zstandard library required for zstd decompression") from None


class LZ4Decompressor(Decompressor):
    """LZ4 decompression."""

    def decompress(self, data: bytes) -> bytes:
        try:
            import lz4.frame

            return lz4.frame.decompress(data)
        except ImportError:
            raise ImportError("lz4 library required for lz4 decompression") from None


class SnappyDecompressor(Decompressor):
    def decompress(self, data: bytes) -> bytes:
        from truthound.stores.compression.providers import SnappyCompressor

        return SnappyCompressor().decompress(data)


def get_decompressor(compression: CompressionType) -> Decompressor:
    """Get a decompressor for the specified type."""
    if compression == CompressionType.NONE:
        return NoDecompressor()
    elif compression == CompressionType.GZIP:
        return GzipDecompressor()
    elif compression == CompressionType.ZSTD:
        return ZstdDecompressor()
    elif compression == CompressionType.LZ4:
        return LZ4Decompressor()
    elif compression == CompressionType.SNAPPY:
        return SnappyDecompressor()
    else:
        raise ValueError(f"Unsupported compression: {compression}")


# =============================================================================
# Base Reader
# =============================================================================


class BaseStreamReader(ABC):
    """Base class for streaming readers.

    Handles deserialization, decompression, and chunked reading.
    """

    def __init__(
        self,
        config: StreamingConfig,
        deserializer: RecordDeserializer | None = None,
        decompressor: Decompressor | None = None,
    ):
        """Initialize the reader.

        Args:
            config: Streaming configuration.
            deserializer: Record deserializer (auto-selected if None).
            decompressor: Data decompressor (auto-selected if None).
        """
        self.config = config
        self.deserializer = deserializer or get_deserializer(config.format)
        self.decompressor = decompressor or get_decompressor(config.compression)

        self.metrics = StreamingMetrics()
        self._io_guard = None
        self._closed = False
        self._current_chunk_index = 0
        self._current_record_index = 0
        self._current_chunk_records: list[dict[str, Any]] = []

    def set_io_guard(self, guard) -> None:
        self._io_guard = guard

    def _guard_io(self) -> None:
        if self._io_guard is not None:
            self._io_guard()

    @abstractmethod
    def _get_chunks(self) -> list[ChunkInfo]:
        """Get list of chunks to read."""
        pass

    @abstractmethod
    def _read_chunk(self, chunk_info: ChunkInfo) -> bytes:
        """Read a chunk from storage.

        Args:
            chunk_info: Chunk metadata.

        Returns:
            Compressed chunk data.
        """
        pass

    def _load_chunk(self, chunk_info: ChunkInfo) -> list[dict[str, Any]]:
        """Load and deserialize a chunk.

        Args:
            chunk_info: Chunk metadata.

        Returns:
            List of records in the chunk.
        """
        self._guard_io()
        # Read compressed data
        compressed_data = self._read_chunk(chunk_info)
        self.metrics.bytes_read += len(compressed_data)
        maximum = self.config.max_memory_mb * 1024 ** 2
        if len(compressed_data) > maximum or chunk_info.byte_size and len(compressed_data) != chunk_info.byte_size:
            raise StreamReadError("Native stream chunk size is inconsistent")

        # Decompress
        data = self.decompressor.decompress(compressed_data)
        if len(data) > maximum:
            raise StreamReadError("Native stream decoded chunk exceeds its memory bound")
        if chunk_info.checksum and hashlib.md5(data).hexdigest() != chunk_info.checksum:
            raise StreamReadError("Native stream chunk checksum changed")

        # Deserialize
        records = list(self.deserializer.deserialize(data))
        if chunk_info.checksum and len(records) != chunk_info.record_count:
            raise StreamReadError("Native stream chunk record count changed")

        self.metrics.record_read(len(records))
        self.metrics.record_chunk(is_write=False)

        return records

    def read(self) -> dict[str, Any] | None:
        """Read a single record.

        Returns:
            The next record, or None if no more records.
        """
        if self._closed:
            raise StreamReadError("Reader is closed")

        self._guard_io()
        chunks = self._get_chunks()

        # Load next chunk if needed
        while self._current_record_index >= len(self._current_chunk_records):
            if self._current_chunk_index >= len(chunks):
                return None  # No more records

            chunk_info = chunks[self._current_chunk_index]
            self._current_chunk_records = self._load_chunk(chunk_info)
            self._current_chunk_index += 1
            self._current_record_index = 0

        # Return next record
        record = self._current_chunk_records[self._current_record_index]
        self._current_record_index += 1
        return record

    def read_batch(self, size: int) -> list[dict[str, Any]]:
        """Read a batch of records.

        Args:
            size: Maximum number of records to read.

        Returns:
            List of records (may be less than size at end).
        """
        if self._closed:
            raise StreamReadError("Reader is closed")

        if size < 1:
            raise ValueError("Native stream read batch size must be positive")

        records: list[dict[str, Any]] = []
        while len(records) < size:
            record = self.read()
            if record is None:
                break
            records.append(record)
        return records

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all records."""
        self.metrics.start()
        try:
            while True:
                record = self.read()
                if record is None:
                    break
                yield record
        finally:
            self.metrics.finish()

    def iter_results(self) -> Iterator[ValidatorResult]:
        """Iterate over records as ValidatorResult objects."""

        for record in self:
            yield _validator_result(record)

    def close(self) -> None:
        """Close the reader."""
        self._closed = True
        self._current_chunk_records = []

    def __enter__(self) -> BaseStreamReader:
        """Context manager entry."""
        self.metrics.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.metrics.finish()
        self.close()

    def reset(self) -> None:
        """Reset reader to beginning."""
        self._current_chunk_index = 0
        self._current_record_index = 0
        self._current_chunk_records = []


# =============================================================================
# Concrete Readers
# =============================================================================


class StreamingResultReader(BaseStreamReader):
    """Filesystem-based streaming reader.

    Reads records from JSONL files on the local filesystem.
    """

    def __init__(
        self,
        run_path: Path | str,
        config: StreamingConfig | None = None,
        *, manifest_filename: str = "_manifest.json",
    ):
        """Initialize the filesystem reader.

        Args:
            run_path: Path to the run directory containing chunks.
            config: Streaming configuration (loaded from manifest if None).
        """
        from truthound.stores.streaming._filesystem_io import component, contained_path

        self.run_path = Path(run_path).absolute()
        if self.run_path.is_symlink():
            raise ValueError("Native stream directory cannot be a symlink")

        # Load manifest
        manifest_path = contained_path(self.run_path, component(manifest_filename))
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            self.session = StreamSession.from_dict(manifest_data)
            self._chunks = self.session.chunks
            if config is None:
                # Use config from manifest
                config = self.session.config
        else:
            raise StreamReadError("Native stream manifest is missing; explicit reconciliation is required")

        super().__init__(config)

    def _get_chunks(self) -> list[ChunkInfo]:
        """Get list of chunks."""
        return self._chunks

    def _read_chunk(self, chunk_info: ChunkInfo) -> bytes:
        """Read a chunk from the filesystem."""
        from truthound.stores.streaming._filesystem_io import contained_path

        chunk_path = contained_path(self.run_path, chunk_info.path) if chunk_info.path else None

        if chunk_path is None or not chunk_path.exists():
            # Try to find the chunk
            chunk_path = self._find_chunk_path(chunk_info)

        if not chunk_path.exists():
            raise ChunkNotFoundError(f"Chunk not found: {chunk_info.chunk_id}")

        with open(chunk_path, "rb") as f:
            return f.read(self.config.max_memory_mb * 1024 ** 2 + 1)

    def _find_chunk_path(self, chunk_info: ChunkInfo) -> Path:
        """Find the path for a chunk."""
        from truthound.stores.streaming._filesystem_io import component, contained_path

        component(chunk_info.chunk_id)
        extensions = [".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz", ".csv", ".csv.gz", ".jsonl.zst", ".jsonl.lz4"]
        for ext in extensions:
            path = contained_path(self.run_path, f"{chunk_info.chunk_id}{ext}")
            if path.exists():
                return path
        return self.run_path / f"{chunk_info.chunk_id}.jsonl"


class ChunkedResultReader(BaseStreamReader):
    """Reader that processes results in chunks.

    Optimized for memory-efficient processing of large result sets.
    """

    def __init__(
        self,
        chunks: list[ChunkInfo],
        chunk_loader: ChunkLoader,
        config: StreamingConfig | None = None,
    ):
        """Initialize the chunked reader.

        Args:
            chunks: List of chunk metadata.
            chunk_loader: Callable to load chunk data.
            config: Streaming configuration.
        """
        super().__init__(config or StreamingConfig())
        self._chunks = chunks
        self._chunk_loader = chunk_loader

    def _get_chunks(self) -> list[ChunkInfo]:
        """Get list of chunks."""
        return self._chunks

    def _read_chunk(self, chunk_info: ChunkInfo) -> bytes:
        """Read a chunk using the loader."""
        return self._chunk_loader.load(chunk_info)


class ChunkLoader(ABC):
    """Abstract chunk loader."""

    @abstractmethod
    def load(self, chunk_info: ChunkInfo) -> bytes:
        """Load chunk data."""
        pass


class FileSystemChunkLoader(ChunkLoader):
    """Filesystem chunk loader."""

    def __init__(self, base_path: Path | str):
        self.base_path = Path(base_path)

    def load(self, chunk_info: ChunkInfo) -> bytes:
        """Load chunk from filesystem."""
        path = Path(chunk_info.path) if chunk_info.path else self.base_path / chunk_info.chunk_id
        with open(path, "rb") as f:
            return f.read()


class MemoryChunkLoader(ChunkLoader):
    """In-memory chunk loader for testing."""

    def __init__(self, chunks: dict[str, bytes] | None = None):
        self.chunks = chunks or {}

    def add_chunk(self, chunk_id: str, data: bytes) -> None:
        """Add a chunk to memory."""
        self.chunks[chunk_id] = data

    def load(self, chunk_info: ChunkInfo) -> bytes:
        """Load chunk from memory."""
        if chunk_info.chunk_id not in self.chunks:
            raise ChunkNotFoundError(f"Chunk not found: {chunk_info.chunk_id}")
        return self.chunks[chunk_info.chunk_id]


# =============================================================================
# Async Reader
# =============================================================================


class AsyncStreamReader:
    """Async wrapper for streaming readers.

    Provides async interface for any BaseStreamReader.
    """

    def __init__(self, reader: BaseStreamReader):
        """Initialize the async reader.

        Args:
            reader: The underlying synchronous reader.
        """
        self._reader = reader
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        return asyncio.get_running_loop()

    async def read(self) -> dict[str, Any] | None:
        """Read a single record asynchronously."""
        loop = self._get_loop()
        return await loop.run_in_executor(None, self._reader.read)

    async def read_batch(self, size: int) -> list[dict[str, Any]]:
        """Read a batch of records asynchronously."""
        loop = self._get_loop()
        return await loop.run_in_executor(None, self._reader.read_batch, size)

    async def read_result(self) -> ValidatorResult | None:
        """Read a single ValidatorResult asynchronously."""

        record = await self.read()
        if record is None:
            return None
        return _validator_result(record)

    async def close(self) -> None:
        """Close the reader asynchronously."""
        loop = self._get_loop()
        await loop.run_in_executor(None, self._reader.close)

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        """Async iterate over records."""
        return self

    async def __anext__(self) -> dict[str, Any]:
        """Get next record."""
        record = await self.read()
        if record is None:
            raise StopAsyncIteration
        return record

    async def aiter_results(self) -> AsyncIterator[ValidatorResult]:
        """Async iterate over ValidatorResults."""

        async for record in self:
            yield _validator_result(record)

    async def __aenter__(self) -> AsyncStreamReader:
        """Async context manager entry."""
        self._reader.metrics.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self._reader.metrics.finish()
        await self.close()

    @property
    def metrics(self) -> StreamingMetrics:
        """Get streaming metrics."""
        return self._reader.metrics


# =============================================================================
# Factory Functions
# =============================================================================


def create_stream_reader(
    run_id: str,
    backend: str = "filesystem",
    **kwargs: Any,
) -> BaseStreamReader:
    """Create a streaming reader for the specified backend.

    Args:
        run_id: The run ID to read.
        backend: Storage backend ("filesystem", "memory").
        **kwargs: Backend-specific options.

    Returns:
        A streaming reader instance.
    """
    if backend == "filesystem":
        base_path = Path(kwargs.get("base_path", ".truthound/streaming"))
        run_path = base_path / run_id
        config = kwargs.get("config")
        return StreamingResultReader(run_path, config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
