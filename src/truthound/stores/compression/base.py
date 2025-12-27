"""Base classes, protocols, and types for compression system.

This module defines the core abstractions that all compression implementations
must follow. It uses Protocol-based structural typing for maximum flexibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    BinaryIO,
    Callable,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)


# =============================================================================
# Exceptions
# =============================================================================


class CompressionError(Exception):
    """Base exception for compression errors."""

    def __init__(self, message: str, algorithm: str | None = None) -> None:
        self.algorithm = algorithm
        super().__init__(f"[{algorithm}] {message}" if algorithm else message)


class DecompressionError(CompressionError):
    """Error during decompression."""

    pass


class UnsupportedAlgorithmError(CompressionError):
    """Requested algorithm is not available."""

    def __init__(self, algorithm: str, available: list[str] | None = None) -> None:
        self.available = available or []
        msg = f"Algorithm '{algorithm}' is not supported"
        if self.available:
            msg += f". Available: {', '.join(self.available)}"
        super().__init__(msg, algorithm)


class CompressionConfigError(CompressionError):
    """Invalid compression configuration."""

    pass


# =============================================================================
# Enums
# =============================================================================


class CompressionAlgorithm(str, Enum):
    """Supported compression algorithms."""

    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"
    SNAPPY = "snappy"
    BROTLI = "brotli"
    DEFLATE = "deflate"
    LZMA = "lzma"
    BZ2 = "bz2"

    @classmethod
    def from_extension(cls, ext: str) -> "CompressionAlgorithm":
        """Get algorithm from file extension."""
        ext_map = {
            ".gz": cls.GZIP,
            ".gzip": cls.GZIP,
            ".zst": cls.ZSTD,
            ".zstd": cls.ZSTD,
            ".lz4": cls.LZ4,
            ".snappy": cls.SNAPPY,
            ".br": cls.BROTLI,
            ".xz": cls.LZMA,
            ".lzma": cls.LZMA,
            ".bz2": cls.BZ2,
        }
        return ext_map.get(ext.lower(), cls.NONE)

    @property
    def extension(self) -> str:
        """Get file extension for this algorithm."""
        ext_map = {
            self.NONE: "",
            self.GZIP: ".gz",
            self.ZSTD: ".zst",
            self.LZ4: ".lz4",
            self.SNAPPY: ".snappy",
            self.BROTLI: ".br",
            self.DEFLATE: ".deflate",
            self.LZMA: ".xz",
            self.BZ2: ".bz2",
        }
        return ext_map.get(self, "")

    @property
    def content_encoding(self) -> str:
        """Get HTTP Content-Encoding header value."""
        encoding_map = {
            self.NONE: "identity",
            self.GZIP: "gzip",
            self.ZSTD: "zstd",
            self.LZ4: "lz4",
            self.SNAPPY: "snappy",
            self.BROTLI: "br",
            self.DEFLATE: "deflate",
        }
        return encoding_map.get(self, "identity")


class CompressionLevel(Enum):
    """Compression level presets."""

    FASTEST = auto()  # Prioritize speed over compression ratio
    FAST = auto()  # Good balance, slightly faster
    BALANCED = auto()  # Default balanced mode
    HIGH = auto()  # Better compression, slower
    MAXIMUM = auto()  # Best compression, slowest

    def get_level(self, algorithm: CompressionAlgorithm) -> int:
        """Get numeric level for a specific algorithm.

        Different algorithms have different level ranges:
        - gzip: 1-9 (default 6)
        - zstd: 1-22 (default 3)
        - lz4: 1-12 (default 0 = fast)
        - brotli: 0-11 (default 4)
        """
        level_map = {
            CompressionAlgorithm.GZIP: {
                self.FASTEST: 1,
                self.FAST: 3,
                self.BALANCED: 6,
                self.HIGH: 8,
                self.MAXIMUM: 9,
            },
            CompressionAlgorithm.ZSTD: {
                self.FASTEST: 1,
                self.FAST: 3,
                self.BALANCED: 6,
                self.HIGH: 12,
                self.MAXIMUM: 19,
            },
            CompressionAlgorithm.LZ4: {
                self.FASTEST: 0,
                self.FAST: 3,
                self.BALANCED: 6,
                self.HIGH: 9,
                self.MAXIMUM: 12,
            },
            CompressionAlgorithm.BROTLI: {
                self.FASTEST: 0,
                self.FAST: 2,
                self.BALANCED: 4,
                self.HIGH: 8,
                self.MAXIMUM: 11,
            },
            CompressionAlgorithm.LZMA: {
                self.FASTEST: 0,
                self.FAST: 2,
                self.BALANCED: 5,
                self.HIGH: 7,
                self.MAXIMUM: 9,
            },
            CompressionAlgorithm.BZ2: {
                self.FASTEST: 1,
                self.FAST: 3,
                self.BALANCED: 6,
                self.HIGH: 8,
                self.MAXIMUM: 9,
            },
        }
        algo_levels = level_map.get(algorithm, {})
        return algo_levels.get(self, 6)


class CompressionMode(Enum):
    """Compression operation mode."""

    COMPRESS = auto()
    DECOMPRESS = auto()
    BOTH = auto()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CompressionConfig:
    """Configuration for compression operations.

    Attributes:
        algorithm: Compression algorithm to use.
        level: Compression level preset.
        custom_level: Custom numeric level (overrides level preset).
        window_size: Window size for algorithms that support it.
        dictionary: Pre-trained dictionary for better compression.
        threads: Number of threads for parallel compression (0 = auto).
        chunk_size: Size of chunks for streaming compression.
        verify_checksum: Verify data integrity on decompression.
        include_header: Include metadata header in compressed output.
    """

    algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP
    level: CompressionLevel = CompressionLevel.BALANCED
    custom_level: int | None = None
    window_size: int | None = None
    dictionary: bytes | None = None
    threads: int = 0
    chunk_size: int = 64 * 1024  # 64KB
    verify_checksum: bool = True
    include_header: bool = False

    def get_effective_level(self) -> int:
        """Get the effective compression level."""
        if self.custom_level is not None:
            return self.custom_level
        return self.level.get_level(self.algorithm)

    def validate(self) -> None:
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise CompressionConfigError("chunk_size must be positive")
        if self.threads < 0:
            raise CompressionConfigError("threads must be non-negative")


@dataclass
class CompressionMetrics:
    """Metrics from a compression operation.

    Attributes:
        original_size: Size of original data in bytes.
        compressed_size: Size of compressed data in bytes.
        compression_ratio: Ratio of original to compressed size.
        compression_time_ms: Time taken to compress in milliseconds.
        decompression_time_ms: Time taken to decompress in milliseconds.
        algorithm: Algorithm used.
        level: Compression level used.
    """

    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    algorithm: CompressionAlgorithm = CompressionAlgorithm.NONE
    level: int = 0

    def update_ratio(self) -> None:
        """Update compression ratio from sizes."""
        if self.compressed_size > 0:
            self.compression_ratio = self.original_size / self.compressed_size
        else:
            self.compression_ratio = 0.0

    @property
    def space_savings(self) -> float:
        """Calculate space savings percentage."""
        if self.original_size == 0:
            return 0.0
        return (1 - self.compressed_size / self.original_size) * 100

    @property
    def throughput_compress_mbps(self) -> float:
        """Calculate compression throughput in MB/s."""
        if self.compression_time_ms == 0:
            return 0.0
        return (self.original_size / 1024 / 1024) / (self.compression_time_ms / 1000)

    @property
    def throughput_decompress_mbps(self) -> float:
        """Calculate decompression throughput in MB/s."""
        if self.decompression_time_ms == 0:
            return 0.0
        return (self.original_size / 1024 / 1024) / (self.decompression_time_ms / 1000)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": round(self.compression_ratio, 2),
            "space_savings_percent": round(self.space_savings, 2),
            "compression_time_ms": round(self.compression_time_ms, 2),
            "decompression_time_ms": round(self.decompression_time_ms, 2),
            "throughput_compress_mbps": round(self.throughput_compress_mbps, 2),
            "throughput_decompress_mbps": round(self.throughput_decompress_mbps, 2),
            "algorithm": self.algorithm.value,
            "level": self.level,
        }


@dataclass
class CompressionResult:
    """Result of a compression operation.

    Attributes:
        data: Compressed data bytes.
        metrics: Compression metrics.
        header: Optional metadata header.
        checksum: Data integrity checksum.
    """

    data: bytes
    metrics: CompressionMetrics
    header: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None

    def to_bytes(self) -> bytes:
        """Get compressed data bytes."""
        return self.data


@dataclass
class CompressionStats:
    """Aggregated compression statistics across multiple operations.

    Attributes:
        total_operations: Number of compression operations.
        total_original_bytes: Total bytes before compression.
        total_compressed_bytes: Total bytes after compression.
        total_compression_time_ms: Total compression time.
        total_decompression_time_ms: Total decompression time.
        algorithm_usage: Count of each algorithm used.
        errors: Number of errors encountered.
    """

    total_operations: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    total_compression_time_ms: float = 0.0
    total_decompression_time_ms: float = 0.0
    algorithm_usage: dict[str, int] = field(default_factory=dict)
    errors: int = 0

    def record(self, metrics: CompressionMetrics) -> None:
        """Record metrics from a compression operation."""
        self.total_operations += 1
        self.total_original_bytes += metrics.original_size
        self.total_compressed_bytes += metrics.compressed_size
        self.total_compression_time_ms += metrics.compression_time_ms
        self.total_decompression_time_ms += metrics.decompression_time_ms

        algo = metrics.algorithm.value
        self.algorithm_usage[algo] = self.algorithm_usage.get(algo, 0) + 1

    @property
    def average_ratio(self) -> float:
        """Calculate average compression ratio."""
        if self.total_compressed_bytes == 0:
            return 0.0
        return self.total_original_bytes / self.total_compressed_bytes

    @property
    def average_space_savings(self) -> float:
        """Calculate average space savings percentage."""
        if self.total_original_bytes == 0:
            return 0.0
        return (1 - self.total_compressed_bytes / self.total_original_bytes) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_operations": self.total_operations,
            "total_original_bytes": self.total_original_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "average_ratio": round(self.average_ratio, 2),
            "average_space_savings_percent": round(self.average_space_savings, 2),
            "total_compression_time_ms": round(self.total_compression_time_ms, 2),
            "total_decompression_time_ms": round(self.total_decompression_time_ms, 2),
            "algorithm_usage": self.algorithm_usage,
            "errors": self.errors,
        }


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class Compressor(Protocol):
    """Protocol for compression implementations."""

    @property
    def algorithm(self) -> CompressionAlgorithm:
        """Get the compression algorithm."""
        ...

    def compress(self, data: bytes) -> bytes:
        """Compress data.

        Args:
            data: Uncompressed data.

        Returns:
            Compressed data.
        """
        ...

    def compress_with_metrics(self, data: bytes) -> CompressionResult:
        """Compress data and return metrics.

        Args:
            data: Uncompressed data.

        Returns:
            Compression result with metrics.
        """
        ...


@runtime_checkable
class Decompressor(Protocol):
    """Protocol for decompression implementations."""

    @property
    def algorithm(self) -> CompressionAlgorithm:
        """Get the compression algorithm."""
        ...

    def decompress(self, data: bytes) -> bytes:
        """Decompress data.

        Args:
            data: Compressed data.

        Returns:
            Decompressed data.
        """
        ...


@runtime_checkable
class StreamingCompressor(Protocol):
    """Protocol for streaming compression."""

    def write(self, data: bytes) -> int:
        """Write data to compression stream.

        Args:
            data: Data chunk to compress.

        Returns:
            Number of bytes written.
        """
        ...

    def flush(self) -> bytes:
        """Flush buffered data.

        Returns:
            Compressed data from buffer.
        """
        ...

    def finalize(self) -> bytes:
        """Finalize compression and get remaining data.

        Returns:
            Final compressed data.
        """
        ...


@runtime_checkable
class StreamingDecompressor(Protocol):
    """Protocol for streaming decompression."""

    def write(self, data: bytes) -> bytes:
        """Write compressed data and get decompressed output.

        Args:
            data: Compressed data chunk.

        Returns:
            Decompressed data.
        """
        ...

    def flush(self) -> bytes:
        """Flush buffered data.

        Returns:
            Decompressed data from buffer.
        """
        ...


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
CompressorT = TypeVar("CompressorT", bound=Compressor)
DecompressorT = TypeVar("DecompressorT", bound=Decompressor)
