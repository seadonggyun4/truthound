"""Streaming compression support for memory-efficient processing.

This module provides streaming compression capabilities for processing
large data without loading everything into memory.

Example:
    >>> from truthound.stores.compression import (
    ...     StreamingCompressor,
    ...     ChunkedCompressor,
    ... )
    >>>
    >>> # Stream compression
    >>> with StreamingCompressor(output_file) as compressor:
    ...     for chunk in data_generator():
    ...         compressor.write(chunk)
    >>>
    >>> # Chunked compression for parallel processing
    >>> chunked = ChunkedCompressor(chunk_size=1024*1024)
    >>> compressed_chunks = chunked.compress_chunks(data)
"""

from __future__ import annotations

import gzip
import io
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Generator, Iterator

from truthound.stores.compression.base import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionError,
    CompressionLevel,
    CompressionMetrics,
    DecompressionError,
)
from truthound.stores.compression.providers import get_compressor


# =============================================================================
# Streaming Protocol
# =============================================================================


class StreamWriter(ABC):
    """Abstract base for streaming compression writers."""

    @abstractmethod
    def write(self, data: bytes) -> int:
        """Write data to compression stream.

        Args:
            data: Data chunk to write.

        Returns:
            Number of bytes written.
        """
        ...

    @abstractmethod
    def flush(self) -> bytes:
        """Flush buffered data.

        Returns:
            Flushed compressed data.
        """
        ...

    @abstractmethod
    def close(self) -> bytes:
        """Close stream and finalize.

        Returns:
            Final compressed data.
        """
        ...


class StreamReader(ABC):
    """Abstract base for streaming decompression readers."""

    @abstractmethod
    def read(self, size: int = -1) -> bytes:
        """Read decompressed data.

        Args:
            size: Number of bytes to read (-1 for all).

        Returns:
            Decompressed data.
        """
        ...

    @abstractmethod
    def read_chunk(self, compressed_chunk: bytes) -> bytes:
        """Read and decompress a chunk.

        Args:
            compressed_chunk: Compressed data chunk.

        Returns:
            Decompressed data.
        """
        ...


# =============================================================================
# Streaming Metrics
# =============================================================================


@dataclass
class StreamingMetrics:
    """Metrics for streaming compression.

    Attributes:
        bytes_in: Total input bytes.
        bytes_out: Total output bytes.
        chunks_processed: Number of chunks processed.
        compression_ratio: Overall compression ratio.
        start_time: Start time of streaming.
        end_time: End time of streaming.
        errors: Number of errors encountered.
    """

    bytes_in: int = 0
    bytes_out: int = 0
    chunks_processed: int = 0
    compression_ratio: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    errors: int = 0

    def update_ratio(self) -> None:
        """Update compression ratio."""
        if self.bytes_out > 0:
            self.compression_ratio = self.bytes_in / self.bytes_out

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.end_time > self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    @property
    def throughput_mbps(self) -> float:
        """Get throughput in MB/s."""
        duration_s = self.duration_ms / 1000
        if duration_s > 0:
            return (self.bytes_in / 1024 / 1024) / duration_s
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "chunks_processed": self.chunks_processed,
            "compression_ratio": round(self.compression_ratio, 2),
            "duration_ms": round(self.duration_ms, 2),
            "throughput_mbps": round(self.throughput_mbps, 2),
            "errors": self.errors,
        }


# =============================================================================
# Gzip Streaming
# =============================================================================


class GzipStreamWriter(StreamWriter):
    """Streaming gzip compression writer."""

    def __init__(
        self,
        output: BinaryIO | None = None,
        level: int = 6,
        buffer_size: int = 64 * 1024,
    ) -> None:
        """Initialize gzip stream writer.

        Args:
            output: Output file or buffer.
            level: Compression level (1-9).
            buffer_size: Internal buffer size.
        """
        self._buffer = io.BytesIO()
        self._output = output
        self._level = level
        self._gzip = gzip.GzipFile(mode="wb", fileobj=self._buffer, compresslevel=level)
        self._closed = False
        self._bytes_written = 0

    def write(self, data: bytes) -> int:
        """Write data to gzip stream."""
        if self._closed:
            raise CompressionError("Stream is closed")

        self._gzip.write(data)
        self._bytes_written += len(data)
        return len(data)

    def flush(self) -> bytes:
        """Flush buffered data."""
        self._gzip.flush()
        data = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()

        if self._output:
            self._output.write(data)
            return b""

        return data

    def close(self) -> bytes:
        """Close and finalize stream."""
        if self._closed:
            return b""

        self._gzip.close()
        self._closed = True

        data = self._buffer.getvalue()
        if self._output:
            self._output.write(data)
            return b""

        return data

    @property
    def bytes_written(self) -> int:
        """Get total bytes written."""
        return self._bytes_written


class GzipStreamReader(StreamReader):
    """Streaming gzip decompression reader."""

    def __init__(self, input_data: BinaryIO | bytes) -> None:
        """Initialize gzip stream reader.

        Args:
            input_data: Compressed input data or file.
        """
        if isinstance(input_data, bytes):
            self._buffer = io.BytesIO(input_data)
        else:
            self._buffer = input_data

        self._gzip = gzip.GzipFile(mode="rb", fileobj=self._buffer)

    def read(self, size: int = -1) -> bytes:
        """Read decompressed data."""
        return self._gzip.read(size)

    def read_chunk(self, compressed_chunk: bytes) -> bytes:
        """Read and decompress a chunk."""
        return gzip.decompress(compressed_chunk)

    def close(self) -> None:
        """Close the reader."""
        self._gzip.close()


# =============================================================================
# Generic Streaming Compressor
# =============================================================================


class StreamingCompressor:
    """Generic streaming compressor supporting multiple algorithms.

    Context manager that handles streaming compression with configurable
    chunk sizes and flush intervals.

    Example:
        >>> with StreamingCompressor(algorithm=CompressionAlgorithm.GZIP) as compressor:
        ...     for chunk in data_generator():
        ...         compressed = compressor.write(chunk)
        ...         if compressed:
        ...             output.write(compressed)
        ...     final = compressor.finalize()
        ...     output.write(final)
    """

    def __init__(
        self,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
        level: CompressionLevel = CompressionLevel.BALANCED,
        buffer_size: int = 64 * 1024,
        flush_threshold: int = 1024 * 1024,
    ) -> None:
        """Initialize streaming compressor.

        Args:
            algorithm: Compression algorithm.
            level: Compression level.
            buffer_size: Internal buffer size.
            flush_threshold: Threshold for automatic flush.
        """
        self.algorithm = algorithm
        self.level = level
        self.buffer_size = buffer_size
        self.flush_threshold = flush_threshold

        self._buffer = io.BytesIO()
        self._output_buffer = io.BytesIO()
        self._compressor = get_compressor(algorithm, level=level)
        self._metrics = StreamingMetrics()
        self._finalized = False

    def __enter__(self) -> "StreamingCompressor":
        """Enter context."""
        self._metrics.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        if not self._finalized:
            self.finalize()

    def write(self, data: bytes) -> bytes:
        """Write data to compression stream.

        Args:
            data: Data chunk to compress.

        Returns:
            Compressed data if flush threshold reached, else empty.
        """
        self._buffer.write(data)
        self._metrics.bytes_in += len(data)
        self._metrics.chunks_processed += 1

        # Check if we should flush
        if self._buffer.tell() >= self.flush_threshold:
            return self.flush()

        return b""

    def flush(self) -> bytes:
        """Flush buffered data.

        Returns:
            Compressed data.
        """
        data = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate()

        if not data:
            return b""

        compressed = self._compressor.compress(data)
        self._metrics.bytes_out += len(compressed)

        return compressed

    def finalize(self) -> bytes:
        """Finalize compression and get remaining data.

        Returns:
            Final compressed data.
        """
        if self._finalized:
            return b""

        self._finalized = True
        self._metrics.end_time = time.time()

        # Flush remaining data
        remaining = self.flush()
        self._metrics.update_ratio()

        return remaining

    @property
    def metrics(self) -> StreamingMetrics:
        """Get compression metrics."""
        return self._metrics


class StreamingDecompressor:
    """Generic streaming decompressor.

    Example:
        >>> with StreamingDecompressor(algorithm=CompressionAlgorithm.GZIP) as decompressor:
        ...     for chunk in compressed_generator():
        ...         decompressed = decompressor.write(chunk)
        ...         process(decompressed)
    """

    def __init__(
        self,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
    ) -> None:
        """Initialize streaming decompressor.

        Args:
            algorithm: Compression algorithm.
        """
        self.algorithm = algorithm
        self._compressor = get_compressor(algorithm)
        self._metrics = StreamingMetrics()
        self._finalized = False

    def __enter__(self) -> "StreamingDecompressor":
        """Enter context."""
        self._metrics.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context."""
        self._metrics.end_time = time.time()
        self._metrics.update_ratio()

    def write(self, data: bytes) -> bytes:
        """Write compressed data and get decompressed output.

        Args:
            data: Compressed data chunk.

        Returns:
            Decompressed data.
        """
        self._metrics.bytes_in += len(data)
        self._metrics.chunks_processed += 1

        try:
            decompressed = self._compressor.decompress(data)
            self._metrics.bytes_out += len(decompressed)
            return decompressed
        except Exception as e:
            self._metrics.errors += 1
            raise DecompressionError(f"Stream decompression failed: {e}")

    @property
    def metrics(self) -> StreamingMetrics:
        """Get decompression metrics."""
        return self._metrics


# =============================================================================
# Chunked Compression
# =============================================================================


@dataclass
class ChunkInfo:
    """Information about a compressed chunk.

    Attributes:
        index: Chunk index.
        original_size: Original size before compression.
        compressed_size: Size after compression.
        offset: Offset in output stream.
        checksum: Data checksum.
    """

    index: int
    original_size: int
    compressed_size: int
    offset: int = 0
    checksum: bytes = b""


@dataclass
class ChunkIndex:
    """Index of all chunks in a chunked compression.

    Attributes:
        algorithm: Compression algorithm used.
        chunk_size: Target chunk size.
        chunks: List of chunk information.
        total_original_size: Total original size.
        total_compressed_size: Total compressed size.
    """

    algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    chunk_size: int = 0
    chunks: list[ChunkInfo] = field(default_factory=list)
    total_original_size: int = 0
    total_compressed_size: int = 0

    def to_bytes(self) -> bytes:
        """Serialize index to bytes."""
        output = io.BytesIO()

        # Write header
        output.write(b"CIDX")  # Magic
        output.write(struct.pack("<BIQ", list(CompressionAlgorithm).index(self.algorithm), self.chunk_size, len(self.chunks)))

        # Write chunks
        for chunk in self.chunks:
            output.write(
                struct.pack(
                    "<IIIIQ",
                    chunk.index,
                    chunk.original_size,
                    chunk.compressed_size,
                    chunk.offset,
                    len(chunk.checksum),
                )
            )
            output.write(chunk.checksum)

        return output.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ChunkIndex":
        """Deserialize index from bytes."""
        if len(data) < 13 or data[:4] != b"CIDX":
            raise CompressionError("Invalid chunk index format")

        algo_idx, chunk_size, num_chunks = struct.unpack("<BIQ", data[4:17])
        algorithm = list(CompressionAlgorithm)[algo_idx]

        index = cls(algorithm=algorithm, chunk_size=chunk_size)
        offset = 17

        for _ in range(num_chunks):
            idx, orig_size, comp_size, chunk_offset, checksum_len = struct.unpack("<IIIIQ", data[offset : offset + 24])
            offset += 24

            checksum = data[offset : offset + checksum_len]
            offset += checksum_len

            index.chunks.append(
                ChunkInfo(
                    index=idx,
                    original_size=orig_size,
                    compressed_size=comp_size,
                    offset=chunk_offset,
                    checksum=checksum,
                )
            )

            index.total_original_size += orig_size
            index.total_compressed_size += comp_size

        return index


class ChunkedCompressor:
    """Chunked compression for parallel processing and random access.

    Splits data into chunks that can be compressed independently,
    enabling parallel compression and random access decompression.

    Example:
        >>> chunked = ChunkedCompressor(chunk_size=1024*1024)
        >>>
        >>> # Compress in chunks
        >>> compressed_data, index = chunked.compress(large_data)
        >>>
        >>> # Decompress specific chunk
        >>> chunk_data = chunked.decompress_chunk(compressed_data, index, chunk_index=5)
        >>>
        >>> # Iterator for large data
        >>> for chunk in chunked.compress_iter(data_generator()):
        ...     store(chunk)
    """

    # Header: magic (4) + version (1) + algorithm (1) + chunk_size (4) + num_chunks (4) + index_offset (8)
    HEADER_FORMAT = "<4sBBIIQ"
    HEADER_SIZE = 22
    MAGIC = b"CHNK"
    VERSION = 1

    def __init__(
        self,
        chunk_size: int = 1024 * 1024,  # 1MB
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> None:
        """Initialize chunked compressor.

        Args:
            chunk_size: Size of each chunk before compression.
            algorithm: Compression algorithm.
            level: Compression level.
        """
        self.chunk_size = chunk_size
        self.algorithm = algorithm
        self.level = level
        self._compressor = get_compressor(algorithm, level=level)

    def compress(self, data: bytes) -> tuple[bytes, ChunkIndex]:
        """Compress data in chunks.

        Args:
            data: Data to compress.

        Returns:
            Tuple of (compressed_data, chunk_index).
        """
        index = ChunkIndex(algorithm=self.algorithm, chunk_size=self.chunk_size)
        output = io.BytesIO()
        current_offset = 0

        # Split into chunks and compress
        for i, start in enumerate(range(0, len(data), self.chunk_size)):
            chunk = data[start : start + self.chunk_size]
            compressed = self._compressor.compress(chunk)

            # Calculate checksum
            import hashlib

            checksum = hashlib.md5(chunk).digest()

            chunk_info = ChunkInfo(
                index=i,
                original_size=len(chunk),
                compressed_size=len(compressed),
                offset=current_offset,
                checksum=checksum,
            )
            index.chunks.append(chunk_info)
            index.total_original_size += len(chunk)
            index.total_compressed_size += len(compressed)

            output.write(compressed)
            current_offset += len(compressed)

        return output.getvalue(), index

    def compress_with_header(self, data: bytes) -> bytes:
        """Compress with embedded header and index.

        Args:
            data: Data to compress.

        Returns:
            Compressed data with header and index.
        """
        compressed, index = self.compress(data)
        index_bytes = index.to_bytes()

        # Build output: header + compressed_chunks + index
        output = io.BytesIO()

        # Calculate index offset
        index_offset = self.HEADER_SIZE + len(compressed)

        # Write header
        header = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            self.VERSION,
            list(CompressionAlgorithm).index(self.algorithm),
            self.chunk_size,
            len(index.chunks),
            index_offset,
        )
        output.write(header)
        output.write(compressed)
        output.write(index_bytes)

        return output.getvalue()

    def decompress(self, data: bytes, index: ChunkIndex | None = None) -> bytes:
        """Decompress chunked data.

        Args:
            data: Compressed data.
            index: Chunk index (if not embedded).

        Returns:
            Decompressed data.
        """
        # Check for embedded header
        if data[: len(self.MAGIC)] == self.MAGIC:
            return self._decompress_with_header(data)

        if index is None:
            raise CompressionError("Chunk index required for decompression")

        output = io.BytesIO()
        for chunk_info in index.chunks:
            chunk_data = data[chunk_info.offset : chunk_info.offset + chunk_info.compressed_size]
            decompressed = self._compressor.decompress(chunk_data)

            # Verify checksum
            if chunk_info.checksum:
                import hashlib

                actual = hashlib.md5(decompressed).digest()
                if actual != chunk_info.checksum:
                    raise DecompressionError(f"Checksum mismatch for chunk {chunk_info.index}")

            output.write(decompressed)

        return output.getvalue()

    def _decompress_with_header(self, data: bytes) -> bytes:
        """Decompress data with embedded header."""
        # Parse header
        if len(data) < self.HEADER_SIZE:
            raise DecompressionError("Data too short")

        magic, version, algo_idx, chunk_size, num_chunks, index_offset = struct.unpack(self.HEADER_FORMAT, data[: self.HEADER_SIZE])

        if magic != self.MAGIC:
            raise DecompressionError("Invalid magic")

        # Parse index
        index = ChunkIndex.from_bytes(data[index_offset:])

        # Decompress chunks
        compressed_data = data[self.HEADER_SIZE : index_offset]
        return self.decompress(compressed_data, index)

    def decompress_chunk(self, data: bytes, index: ChunkIndex, chunk_index: int) -> bytes:
        """Decompress a specific chunk.

        Args:
            data: Compressed data.
            index: Chunk index.
            chunk_index: Index of chunk to decompress.

        Returns:
            Decompressed chunk data.
        """
        if chunk_index >= len(index.chunks):
            raise CompressionError(f"Chunk index {chunk_index} out of range")

        chunk_info = index.chunks[chunk_index]
        chunk_data = data[chunk_info.offset : chunk_info.offset + chunk_info.compressed_size]
        return self._compressor.decompress(chunk_data)

    def compress_iter(self, data_iter: Iterator[bytes]) -> Generator[tuple[bytes, ChunkInfo], None, ChunkIndex]:
        """Compress data from iterator, yielding chunks.

        Args:
            data_iter: Iterator yielding data chunks.

        Yields:
            Tuples of (compressed_chunk, chunk_info).

        Returns:
            Final chunk index.
        """
        index = ChunkIndex(algorithm=self.algorithm, chunk_size=self.chunk_size)
        buffer = io.BytesIO()
        chunk_idx = 0
        current_offset = 0

        import hashlib

        for data in data_iter:
            buffer.write(data)

            # Process complete chunks
            while buffer.tell() >= self.chunk_size:
                buffer.seek(0)
                chunk = buffer.read(self.chunk_size)
                remaining = buffer.read()
                buffer.seek(0)
                buffer.truncate()
                buffer.write(remaining)

                compressed = self._compressor.compress(chunk)
                checksum = hashlib.md5(chunk).digest()

                chunk_info = ChunkInfo(
                    index=chunk_idx,
                    original_size=len(chunk),
                    compressed_size=len(compressed),
                    offset=current_offset,
                    checksum=checksum,
                )
                index.chunks.append(chunk_info)
                index.total_original_size += len(chunk)
                index.total_compressed_size += len(compressed)

                yield compressed, chunk_info

                current_offset += len(compressed)
                chunk_idx += 1

        # Process remaining data
        remaining = buffer.getvalue()
        if remaining:
            compressed = self._compressor.compress(remaining)
            checksum = hashlib.md5(remaining).digest()

            chunk_info = ChunkInfo(
                index=chunk_idx,
                original_size=len(remaining),
                compressed_size=len(compressed),
                offset=current_offset,
                checksum=checksum,
            )
            index.chunks.append(chunk_info)
            index.total_original_size += len(remaining)
            index.total_compressed_size += len(compressed)

            yield compressed, chunk_info

        return index

    def decompress_iter(self, chunks_iter: Iterator[bytes], index: ChunkIndex) -> Generator[bytes, None, None]:
        """Decompress chunks from iterator.

        Args:
            chunks_iter: Iterator yielding compressed chunks.
            index: Chunk index.

        Yields:
            Decompressed data chunks.
        """
        import hashlib

        for i, chunk_data in enumerate(chunks_iter):
            if i >= len(index.chunks):
                break

            chunk_info = index.chunks[i]
            decompressed = self._compressor.decompress(chunk_data)

            # Verify checksum
            if chunk_info.checksum:
                actual = hashlib.md5(decompressed).digest()
                if actual != chunk_info.checksum:
                    raise DecompressionError(f"Checksum mismatch for chunk {i}")

            yield decompressed
