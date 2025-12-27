"""Streaming readers for memory-efficient result retrieval.

This module provides readers that can iterate over stored validation results
without loading everything into memory.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, BinaryIO, Iterator

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
        for row in reader:
            yield dict(row)


def get_deserializer(format: StreamingFormat, **kwargs: Any) -> RecordDeserializer:
    """Get a deserializer for the specified format."""
    if format == StreamingFormat.JSONL or format == StreamingFormat.NDJSON:
        return JSONLDeserializer()
    elif format == StreamingFormat.CSV:
        return CSVDeserializer(columns=kwargs.get("columns"))
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
            raise ImportError("zstandard library required for zstd decompression")


class LZ4Decompressor(Decompressor):
    """LZ4 decompression."""

    def decompress(self, data: bytes) -> bytes:
        try:
            import lz4.frame

            return lz4.frame.decompress(data)
        except ImportError:
            raise ImportError("lz4 library required for lz4 decompression")


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
        self._closed = False
        self._current_chunk_index = 0
        self._current_record_index = 0
        self._current_chunk_records: list[dict[str, Any]] = []

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
        # Read compressed data
        compressed_data = self._read_chunk(chunk_info)
        self.metrics.bytes_read += len(compressed_data)

        # Decompress
        data = self.decompressor.decompress(compressed_data)

        # Deserialize
        records = list(self.deserializer.deserialize(data))

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

    def iter_results(self) -> Iterator["ValidatorResult"]:
        """Iterate over records as ValidatorResult objects."""
        from truthound.stores.results import ValidatorResult

        for record in self:
            yield ValidatorResult.from_dict(record)

    def close(self) -> None:
        """Close the reader."""
        self._closed = True
        self._current_chunk_records = []

    def __enter__(self) -> "BaseStreamReader":
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
    ):
        """Initialize the filesystem reader.

        Args:
            run_path: Path to the run directory containing chunks.
            config: Streaming configuration (loaded from manifest if None).
        """
        self.run_path = Path(run_path)

        # Load manifest
        manifest_path = self.run_path / "_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest_data = json.load(f)
            self.session = StreamSession.from_dict(manifest_data)
            self._chunks = self.session.chunks
            if config is None:
                # Use config from manifest
                config = StreamingConfig(
                    format=StreamingFormat(
                        manifest_data.get("config", {}).get("format", "jsonl")
                    ),
                    compression=CompressionType(
                        manifest_data.get("config", {}).get("compression", "none")
                    ),
                )
        else:
            self.session = None
            self._chunks = self._discover_chunks()
            if config is None:
                config = StreamingConfig()

        super().__init__(config)

    def _discover_chunks(self) -> list[ChunkInfo]:
        """Discover chunks by scanning the directory."""
        chunks: list[ChunkInfo] = []

        # Look for chunk files
        patterns = ["*.jsonl", "*.jsonl.gz", "*.ndjson", "*.ndjson.gz", "*.csv", "*.csv.gz"]
        chunk_files: list[Path] = []
        for pattern in patterns:
            chunk_files.extend(self.run_path.glob(pattern))

        # Filter out manifest and sort by name
        chunk_files = [f for f in chunk_files if not f.name.startswith("_")]
        chunk_files.sort(key=lambda p: p.name)

        for idx, path in enumerate(chunk_files):
            # Estimate record count from file size (rough)
            byte_size = path.stat().st_size
            estimated_records = max(1, byte_size // 200)  # ~200 bytes per record

            chunk_info = ChunkInfo(
                chunk_id=path.stem.replace(".jsonl", "").replace(".gz", ""),
                chunk_index=idx,
                record_count=estimated_records,
                byte_size=byte_size,
                start_offset=0,
                end_offset=estimated_records,
                path=str(path),
            )
            chunks.append(chunk_info)

        return chunks

    def _get_chunks(self) -> list[ChunkInfo]:
        """Get list of chunks."""
        return self._chunks

    def _read_chunk(self, chunk_info: ChunkInfo) -> bytes:
        """Read a chunk from the filesystem."""
        chunk_path = Path(chunk_info.path) if chunk_info.path else None

        if chunk_path is None or not chunk_path.exists():
            # Try to find the chunk
            chunk_path = self._find_chunk_path(chunk_info)

        if not chunk_path.exists():
            raise ChunkNotFoundError(f"Chunk not found: {chunk_info.chunk_id}")

        with open(chunk_path, "rb") as f:
            return f.read()

    def _find_chunk_path(self, chunk_info: ChunkInfo) -> Path:
        """Find the path for a chunk."""
        extensions = [".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz"]
        for ext in extensions:
            path = self.run_path / f"{chunk_info.chunk_id}{ext}"
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
        chunk_loader: "ChunkLoader",
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
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.get_event_loop()
        return self._loop

    async def read(self) -> dict[str, Any] | None:
        """Read a single record asynchronously."""
        loop = self._get_loop()
        return await loop.run_in_executor(None, self._reader.read)

    async def read_batch(self, size: int) -> list[dict[str, Any]]:
        """Read a batch of records asynchronously."""
        loop = self._get_loop()
        return await loop.run_in_executor(None, self._reader.read_batch, size)

    async def read_result(self) -> "ValidatorResult | None":
        """Read a single ValidatorResult asynchronously."""
        from truthound.stores.results import ValidatorResult

        record = await self.read()
        if record is None:
            return None
        return ValidatorResult.from_dict(record)

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

    async def aiter_results(self) -> AsyncIterator["ValidatorResult"]:
        """Async iterate over ValidatorResults."""
        from truthound.stores.results import ValidatorResult

        async for record in self:
            yield ValidatorResult.from_dict(record)

    async def __aenter__(self) -> "AsyncStreamReader":
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
