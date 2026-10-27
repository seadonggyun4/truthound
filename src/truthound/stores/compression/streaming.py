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
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

from truthound.stores.compression.base import (
    CompressionAlgorithm,
    CompressionError,
    CompressionLevel,
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
    """Emit independent complete native frames at each flush boundary.

    The paired StreamingDecompressor accepts one complete returned frame per
    write. For transport fragments use a native codec stream reader instead.
    """

    def __init__(
        self,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
        level: CompressionLevel = CompressionLevel.BALANCED,
        buffer_size: int = 64 * 1024,
        flush_threshold: int = 1024 * 1024,
    ) -> None:
        for value in (buffer_size, flush_threshold):
            if type(value) is not int or value < 1:
                raise ValueError("Stream buffer sizes must be positive integers")
        self.algorithm, self.level = algorithm, level
        self.buffer_size, self.flush_threshold = buffer_size, flush_threshold
        self._buffer = io.BytesIO()
        self._compressor = get_compressor(algorithm, level=level)
        self._metrics = StreamingMetrics()
        self._finalized = self._failed = False
        self._execution_guard = None

    def set_execution_guard(self, guard) -> None:
        self._execution_guard = guard

    def _check_guard(self):
        if self._execution_guard is not None:
            self._execution_guard()

    def _require_open(self):
        self._check_guard()
        if self._finalized or self._failed:
            raise CompressionError("Compression stream is no longer writable")

    def __enter__(self) -> StreamingCompressor:
        self._metrics.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None or self._failed:
            self.abort()
        elif not self._finalized:
            self.finalize()

    def write(self, data: bytes) -> bytes:
        self._require_open()
        if not isinstance(data, bytes):
            raise ValueError("Compression input must be bytes")
        try:
            for offset in range(0, len(data), self.buffer_size):
                self._check_guard()
                self._buffer.write(data[offset : offset + self.buffer_size])
            self._metrics.bytes_in += len(data)
            self._metrics.chunks_processed += 1
            return self.flush() if self._buffer.tell() >= self.flush_threshold else b""
        except BaseException:
            self._failed = True
            raise

    def flush(self) -> bytes:
        self._require_open()
        data = self._buffer.getvalue()
        if not data:
            return b""
        try:
            compressed = self._compressor.compress(data)
            self._check_guard()
            self._buffer.seek(0)
            self._buffer.truncate()
            self._metrics.bytes_out += len(compressed)
            return compressed
        except BaseException:
            self._metrics.errors += 1
            self._failed = True
            raise

    def finalize(self) -> bytes:
        if self._finalized:
            return b""
        self._require_open()
        remaining = self.flush()
        self._finalized = True
        self._metrics.end_time = time.time()
        self._metrics.update_ratio()
        return remaining

    def abort(self) -> None:
        self._failed = True
        self._buffer.seek(0)
        self._buffer.truncate()
        self._metrics.end_time = time.time()

    @property
    def buffered_bytes(self) -> int:
        return self._buffer.tell()

    @property
    def metrics(self) -> StreamingMetrics:
        return self._metrics


class StreamingDecompressor:
    """Decode one complete native frame per write; failure is terminal."""

    def __init__(self, algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP) -> None:
        self.algorithm = algorithm
        self._compressor = get_compressor(algorithm)
        self._metrics = StreamingMetrics()
        self._finalized = self._failed = False
        self._execution_guard = None

    def set_execution_guard(self, guard) -> None:
        self._execution_guard = guard

    def _check_guard(self):
        if self._execution_guard is not None:
            self._execution_guard()

    def __enter__(self) -> StreamingDecompressor:
        self._metrics.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None or self._failed:
            self.abort()
        else:
            self.finalize()

    def write(self, data: bytes) -> bytes:
        self._check_guard()
        if self._finalized or self._failed:
            raise DecompressionError("Compression stream is no longer readable")
        if not isinstance(data, bytes):
            raise ValueError("Compressed input must be bytes")
        try:
            decompressed = self._compressor.decompress(data)
            self._check_guard()
            self._metrics.bytes_in += len(data)
            self._metrics.bytes_out += len(decompressed)
            self._metrics.chunks_processed += 1
            return decompressed
        except BaseException:
            self._failed = True
            self._metrics.errors += 1
            raise

    def finalize(self) -> bytes:
        self._check_guard()
        if self._failed:
            raise DecompressionError("Compression stream failed")
        self._finalized = True
        self._metrics.end_time = time.time()
        self._metrics.update_ratio()
        return b""

    def abort(self) -> None:
        self._failed = True
        self._metrics.end_time = time.time()

    @property
    def buffered_bytes(self) -> int:
        return 0

    @property
    def metrics(self) -> StreamingMetrics:
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
        output.write(
            struct.pack(
                "<BIQ",
                list(CompressionAlgorithm).index(self.algorithm),
                self.chunk_size,
                len(self.chunks),
            )
        )

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
    def from_bytes(cls, data: bytes) -> ChunkIndex:
        try:
            if len(data) < 17 or data[:4] != b"CIDX":
                raise DecompressionError("Invalid chunk index format")
            algo_idx, chunk_size, num_chunks = struct.unpack("<BIQ", data[4:17])
            if algo_idx >= len(CompressionAlgorithm) or num_chunks > (len(data) - 17) // 24:
                raise DecompressionError("Invalid chunk index bounds")
            index = cls(algorithm=list(CompressionAlgorithm)[algo_idx], chunk_size=chunk_size)
            offset = 17
            for _ in range(num_chunks):
                number, original, compressed, start, checksum_size = struct.unpack(
                    "<IIIIQ", data[offset : offset + 24]
                )
                offset += 24
                if checksum_size not in {0, 16} or offset + checksum_size > len(data):
                    raise DecompressionError("Invalid chunk checksum metadata")
                index.chunks.append(
                    ChunkInfo(
                        index=number,
                        original_size=original,
                        compressed_size=compressed,
                        offset=start,
                        checksum=data[offset : offset + checksum_size],
                    )
                )
                offset += checksum_size
                index.total_original_size += original
                index.total_compressed_size += compressed
            if offset != len(data):
                raise DecompressionError("Trailing chunk index bytes")
            return index
        except (struct.error, IndexError, ValueError) as error:
            raise DecompressionError("Invalid chunk index") from error


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
        if type(chunk_size) is not int or not 1 <= chunk_size <= 16 * 1024 * 1024:
            raise ValueError("Invalid compression chunk size")
        self._execution_guard = None
        self.chunk_size = chunk_size
        self.algorithm = algorithm
        self.level = level
        self._compressor = get_compressor(algorithm, level=level)

    def set_execution_guard(self, guard) -> None:
        self._execution_guard = guard

    def _check_guard(self):
        if self._execution_guard is not None:
            self._execution_guard()

    def _validate_index(self, index, payload_size=None):
        if index.algorithm != self.algorithm or index.chunk_size != self.chunk_size:
            raise DecompressionError("Chunk index does not match selected codec")
        offset = original = 0
        for number, chunk in enumerate(index.chunks):
            if (
                type(chunk.index) is not int
                or chunk.index != number
                or type(chunk.offset) is not int
                or chunk.offset != offset
                or type(chunk.original_size) is not int
                or not 1 <= chunk.original_size <= self.chunk_size
                or type(chunk.compressed_size) is not int
                or chunk.compressed_size < 1
                or len(chunk.checksum) not in {0, 16}
                or (number < len(index.chunks) - 1 and chunk.original_size != self.chunk_size)
            ):
                raise DecompressionError("Invalid compression chunk metadata")
            offset += chunk.compressed_size
            original += chunk.original_size
        if (
            index.total_original_size != original
            or index.total_compressed_size != offset
            or (payload_size is not None and payload_size != offset)
        ):
            raise DecompressionError("Chunk index size mismatch")

    @staticmethod
    def _check_output_limit(size, maximum):
        if maximum is not None:
            if type(maximum) is not int or maximum < 0:
                raise ValueError("Invalid output size limit")
            if size > maximum:
                raise DecompressionError("Chunk output size limit exceeded")

    def _decode_chunk(self, data, info):
        import hashlib

        self._check_guard()
        if len(data) != info.compressed_size:
            raise DecompressionError("Compressed chunk size mismatch")
        result = self._compressor.decompress(data)
        self._check_guard()
        if len(result) != info.original_size or (
            info.checksum and hashlib.md5(result).digest() != info.checksum
        ):
            raise DecompressionError("Compression chunk checksum or size mismatch")
        return result

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
            self._check_guard()
            compressed = self._compressor.compress(chunk)
            self._check_guard()

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

    def decompress(
        self, data: bytes, index: ChunkIndex | None = None, *, max_output_bytes: int | None = None
    ) -> bytes:
        """Decompress chunked data.

        Args:
            data: Compressed data.
            index: Chunk index (if not embedded).

        Returns:
            Decompressed data.
        """
        self._check_guard()
        if index is None:
            if data[: len(self.MAGIC)] == self.MAGIC:
                return self._decompress_with_header(data, max_output_bytes=max_output_bytes)
            raise CompressionError("Chunk index required for decompression")
        self._validate_index(index, len(data))
        self._check_output_limit(index.total_original_size, max_output_bytes)
        return b"".join(
            self._decode_chunk(data[info.offset : info.offset + info.compressed_size], info)
            for info in index.chunks
        )

    def _decompress_with_header(self, data: bytes, *, max_output_bytes: int | None = None) -> bytes:
        if len(data) < self.HEADER_SIZE:
            raise DecompressionError("Incomplete chunk header")
        magic, version, algorithm, chunk_size, count, offset = struct.unpack(
            self.HEADER_FORMAT, data[: self.HEADER_SIZE]
        )
        if (
            magic != self.MAGIC
            or version != self.VERSION
            or algorithm != list(CompressionAlgorithm).index(self.algorithm)
            or chunk_size != self.chunk_size
            or not self.HEADER_SIZE <= offset <= len(data)
        ):
            raise DecompressionError("Chunk header does not match selected codec")
        index = ChunkIndex.from_bytes(data[offset:])
        if count != len(index.chunks):
            raise DecompressionError("Chunk header count mismatch")
        return self.decompress(
            data[self.HEADER_SIZE : offset], index, max_output_bytes=max_output_bytes
        )

    def decompress_chunk(self, data: bytes, index: ChunkIndex, chunk_index: int) -> bytes:
        self._check_guard()
        self._validate_index(index, len(data))
        if type(chunk_index) is not int or not 0 <= chunk_index < len(index.chunks):
            raise DecompressionError("Chunk index out of range")
        info = index.chunks[chunk_index]
        return self._decode_chunk(data[info.offset : info.offset + info.compressed_size], info)

    def decompress_range(
        self, data: bytes, start: int, end: int, *, max_output_bytes: int | None = None
    ) -> bytes:
        """Read a plaintext range from a native embedded-index bundle."""
        self._check_guard()
        if len(data) < self.HEADER_SIZE:
            raise DecompressionError("Incomplete chunk header")
        magic, version, algorithm, chunk_size, count, offset = struct.unpack(
            self.HEADER_FORMAT, data[: self.HEADER_SIZE]
        )
        if (
            magic != self.MAGIC
            or version != self.VERSION
            or algorithm != list(CompressionAlgorithm).index(self.algorithm)
            or chunk_size != self.chunk_size
            or not self.HEADER_SIZE <= offset <= len(data)
        ):
            raise DecompressionError("Chunk header does not match selected codec")
        index = ChunkIndex.from_bytes(data[offset:])
        payload = data[self.HEADER_SIZE : offset]
        self._validate_index(index, len(payload))
        if (
            count != len(index.chunks)
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= index.total_original_size
        ):
            raise DecompressionError("Invalid compression range or chunk count")
        self._check_output_limit(end - start, max_output_bytes)
        if start == end:
            return b""
        first = start // self.chunk_size
        last = (end - 1) // self.chunk_size
        output = b"".join(
            self._decode_chunk(payload[info.offset : info.offset + info.compressed_size], info)
            for info in index.chunks[first : last + 1]
        )
        relative = start - first * self.chunk_size
        return output[relative : relative + end - start]

    def compress_iter(
        self, data_iter: Iterator[bytes]
    ) -> Generator[tuple[bytes, ChunkInfo], None, ChunkIndex]:
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

                self._check_guard()
                compressed = self._compressor.compress(chunk)
                self._check_guard()
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
            self._check_guard()
            compressed = self._compressor.compress(remaining)
            self._check_guard()
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

    def decompress_iter(
        self, chunks_iter: Iterator[bytes], index: ChunkIndex
    ) -> Generator[bytes, None, None]:
        """Decode every indexed chunk, rejecting missing and extra input frames."""
        self._check_guard()
        self._validate_index(index)
        iterator = iter(chunks_iter)
        for info in index.chunks:
            self._check_guard()
            try:
                data = next(iterator)
            except StopIteration as error:
                raise DecompressionError("Missing compressed chunk") from error
            yield self._decode_chunk(data, info)
        self._check_guard()
        try:
            next(iterator)
        except StopIteration:
            return
        raise DecompressionError("Extra compressed chunk")
