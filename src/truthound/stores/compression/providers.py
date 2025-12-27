"""Compression provider implementations.

This module provides concrete implementations of compression algorithms
with a unified interface and lazy loading of optional dependencies.
"""

from __future__ import annotations

import gzip
import hashlib
import time
import zlib
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Callable, Type

from truthound.stores.compression.base import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionError,
    CompressionLevel,
    CompressionMetrics,
    CompressionResult,
    Compressor,
    DecompressionError,
    Decompressor,
    UnsupportedAlgorithmError,
)


# =============================================================================
# Base Compressor
# =============================================================================


class BaseCompressor(ABC):
    """Abstract base class for compressor implementations.

    Provides common functionality for all compressors including
    metrics collection, checksum verification, and configuration handling.
    """

    def __init__(self, config: CompressionConfig | None = None) -> None:
        """Initialize the compressor.

        Args:
            config: Compression configuration.
        """
        self._config = config or CompressionConfig(algorithm=self.algorithm)
        self._config.validate()

    @property
    @abstractmethod
    def algorithm(self) -> CompressionAlgorithm:
        """Get the compression algorithm."""
        pass

    @property
    def config(self) -> CompressionConfig:
        """Get the configuration."""
        return self._config

    @property
    def level(self) -> int:
        """Get the effective compression level."""
        return self._config.get_effective_level()

    @abstractmethod
    def _do_compress(self, data: bytes) -> bytes:
        """Perform actual compression. Override in subclasses."""
        pass

    @abstractmethod
    def _do_decompress(self, data: bytes) -> bytes:
        """Perform actual decompression. Override in subclasses."""
        pass

    def compress(self, data: bytes) -> bytes:
        """Compress data.

        Args:
            data: Uncompressed data.

        Returns:
            Compressed data.

        Raises:
            CompressionError: If compression fails.
        """
        try:
            return self._do_compress(data)
        except Exception as e:
            raise CompressionError(
                f"Compression failed: {e}", self.algorithm.value
            ) from e

    def decompress(self, data: bytes) -> bytes:
        """Decompress data.

        Args:
            data: Compressed data.

        Returns:
            Decompressed data.

        Raises:
            DecompressionError: If decompression fails.
        """
        try:
            return self._do_decompress(data)
        except Exception as e:
            raise DecompressionError(
                f"Decompression failed: {e}", self.algorithm.value
            ) from e

    def compress_with_metrics(self, data: bytes) -> CompressionResult:
        """Compress data and return metrics.

        Args:
            data: Uncompressed data.

        Returns:
            Compression result with metrics.
        """
        original_size = len(data)

        # Measure compression time
        start_time = time.perf_counter()
        compressed = self.compress(data)
        compression_time = (time.perf_counter() - start_time) * 1000

        compressed_size = len(compressed)

        # Calculate checksum if enabled
        checksum = None
        if self._config.verify_checksum:
            checksum = hashlib.md5(data).hexdigest()

        metrics = CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_time_ms=compression_time,
            algorithm=self.algorithm,
            level=self.level,
        )
        metrics.update_ratio()

        return CompressionResult(
            data=compressed,
            metrics=metrics,
            checksum=checksum,
            header={
                "algorithm": self.algorithm.value,
                "level": self.level,
                "original_size": original_size,
            } if self._config.include_header else {},
        )

    def decompress_with_metrics(
        self,
        data: bytes,
        expected_checksum: str | None = None,
    ) -> tuple[bytes, CompressionMetrics]:
        """Decompress data and return metrics.

        Args:
            data: Compressed data.
            expected_checksum: Optional checksum to verify.

        Returns:
            Tuple of (decompressed data, metrics).
        """
        compressed_size = len(data)

        # Measure decompression time
        start_time = time.perf_counter()
        decompressed = self.decompress(data)
        decompression_time = (time.perf_counter() - start_time) * 1000

        original_size = len(decompressed)

        # Verify checksum if provided
        if expected_checksum and self._config.verify_checksum:
            actual_checksum = hashlib.md5(decompressed).hexdigest()
            if actual_checksum != expected_checksum:
                raise DecompressionError(
                    f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}",
                    self.algorithm.value,
                )

        metrics = CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            decompression_time_ms=decompression_time,
            algorithm=self.algorithm,
            level=self.level,
        )
        metrics.update_ratio()

        return decompressed, metrics

    def get_extension(self) -> str:
        """Get file extension for compressed files."""
        return self.algorithm.extension


# =============================================================================
# No-op Compressor
# =============================================================================


class NoopCompressor(BaseCompressor):
    """No-operation compressor (passthrough)."""

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.NONE

    def _do_compress(self, data: bytes) -> bytes:
        return data

    def _do_decompress(self, data: bytes) -> bytes:
        return data


# =============================================================================
# Gzip Compressor
# =============================================================================


class GzipCompressor(BaseCompressor):
    """Gzip compression using Python's built-in gzip module.

    Gzip is widely supported and provides good compression for text data.
    It's the default choice for web content and general-purpose compression.
    """

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.GZIP

    def _do_compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=self.level)

    def _do_decompress(self, data: bytes) -> bytes:
        return gzip.decompress(data)


# =============================================================================
# Zstandard Compressor
# =============================================================================


class ZstdCompressor(BaseCompressor):
    """Zstandard compression using the zstandard library.

    Zstd provides excellent compression ratio and speed balance.
    It's particularly good for structured data like JSON.
    """

    _zstd = None

    @classmethod
    def _get_zstd(cls):
        """Lazy import zstandard."""
        if cls._zstd is None:
            try:
                import zstandard
                cls._zstd = zstandard
            except ImportError:
                raise UnsupportedAlgorithmError(
                    "zstd",
                    ["gzip", "lz4"] + (["snappy"] if _is_snappy_available() else []),
                )
        return cls._zstd

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.ZSTD

    def _get_compressor(self):
        """Get zstd compressor instance."""
        zstd = self._get_zstd()
        return zstd.ZstdCompressor(
            level=self.level,
            threads=self._config.threads if self._config.threads > 0 else -1,
        )

    def _get_decompressor(self):
        """Get zstd decompressor instance."""
        zstd = self._get_zstd()
        return zstd.ZstdDecompressor()

    def _do_compress(self, data: bytes) -> bytes:
        compressor = self._get_compressor()
        return compressor.compress(data)

    def _do_decompress(self, data: bytes) -> bytes:
        decompressor = self._get_decompressor()
        return decompressor.decompress(data)


# =============================================================================
# LZ4 Compressor
# =============================================================================


class LZ4Compressor(BaseCompressor):
    """LZ4 compression using the lz4 library.

    LZ4 is optimized for speed and is one of the fastest compression
    algorithms available. Good for real-time compression needs.
    """

    _lz4 = None

    @classmethod
    def _get_lz4(cls):
        """Lazy import lz4."""
        if cls._lz4 is None:
            try:
                import lz4.frame
                cls._lz4 = lz4.frame
            except ImportError:
                raise UnsupportedAlgorithmError(
                    "lz4",
                    ["gzip", "zstd"] + (["snappy"] if _is_snappy_available() else []),
                )
        return cls._lz4

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.LZ4

    def _do_compress(self, data: bytes) -> bytes:
        lz4 = self._get_lz4()
        return lz4.compress(
            data,
            compression_level=self.level,
        )

    def _do_decompress(self, data: bytes) -> bytes:
        lz4 = self._get_lz4()
        return lz4.decompress(data)


# =============================================================================
# Snappy Compressor
# =============================================================================


def _is_snappy_available() -> bool:
    """Check if snappy is available."""
    try:
        import snappy
        return True
    except ImportError:
        return False


class SnappyCompressor(BaseCompressor):
    """Snappy compression using the python-snappy library.

    Snappy is designed for speed rather than maximum compression.
    Commonly used in distributed systems like Hadoop and Kafka.
    """

    _snappy = None

    @classmethod
    def _get_snappy(cls):
        """Lazy import snappy."""
        if cls._snappy is None:
            try:
                import snappy
                cls._snappy = snappy
            except ImportError:
                raise UnsupportedAlgorithmError(
                    "snappy",
                    ["gzip", "lz4", "zstd"],
                )
        return cls._snappy

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.SNAPPY

    def _do_compress(self, data: bytes) -> bytes:
        snappy = self._get_snappy()
        return snappy.compress(data)

    def _do_decompress(self, data: bytes) -> bytes:
        snappy = self._get_snappy()
        return snappy.decompress(data)


# =============================================================================
# Brotli Compressor
# =============================================================================


class BrotliCompressor(BaseCompressor):
    """Brotli compression using the brotli library.

    Brotli provides excellent compression for web content,
    particularly for text-based formats. It's slower but achieves
    better compression ratios than gzip.
    """

    _brotli = None

    @classmethod
    def _get_brotli(cls):
        """Lazy import brotli."""
        if cls._brotli is None:
            try:
                import brotli
                cls._brotli = brotli
            except ImportError:
                raise UnsupportedAlgorithmError(
                    "brotli",
                    ["gzip", "zstd", "lz4"],
                )
        return cls._brotli

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.BROTLI

    def _do_compress(self, data: bytes) -> bytes:
        brotli = self._get_brotli()
        return brotli.compress(data, quality=self.level)

    def _do_decompress(self, data: bytes) -> bytes:
        brotli = self._get_brotli()
        return brotli.decompress(data)


# =============================================================================
# LZMA Compressor
# =============================================================================


class LzmaCompressor(BaseCompressor):
    """LZMA compression using Python's built-in lzma module.

    LZMA provides very high compression ratios but is slower.
    Good for archival purposes where compression ratio matters more than speed.
    """

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.LZMA

    def _do_compress(self, data: bytes) -> bytes:
        import lzma
        return lzma.compress(data, preset=self.level)

    def _do_decompress(self, data: bytes) -> bytes:
        import lzma
        return lzma.decompress(data)


# =============================================================================
# BZ2 Compressor
# =============================================================================


class Bz2Compressor(BaseCompressor):
    """BZ2 compression using Python's built-in bz2 module.

    BZ2 provides good compression ratios, better than gzip but slower.
    It's a good middle ground between gzip and LZMA.
    """

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.BZ2

    def _do_compress(self, data: bytes) -> bytes:
        import bz2
        return bz2.compress(data, compresslevel=self.level)

    def _do_decompress(self, data: bytes) -> bytes:
        import bz2
        return bz2.decompress(data)


# =============================================================================
# Deflate Compressor
# =============================================================================


class DeflateCompressor(BaseCompressor):
    """Raw Deflate compression using Python's built-in zlib module.

    Deflate is the underlying algorithm for gzip and zip.
    This provides raw deflate without gzip headers.
    """

    @property
    def algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.DEFLATE

    def _do_compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=self.level)

    def _do_decompress(self, data: bytes) -> bytes:
        return zlib.decompress(data)


# =============================================================================
# Registry and Factory
# =============================================================================


_COMPRESSOR_REGISTRY: dict[CompressionAlgorithm, Type[BaseCompressor]] = {
    CompressionAlgorithm.NONE: NoopCompressor,
    CompressionAlgorithm.GZIP: GzipCompressor,
    CompressionAlgorithm.ZSTD: ZstdCompressor,
    CompressionAlgorithm.LZ4: LZ4Compressor,
    CompressionAlgorithm.SNAPPY: SnappyCompressor,
    CompressionAlgorithm.BROTLI: BrotliCompressor,
    CompressionAlgorithm.LZMA: LzmaCompressor,
    CompressionAlgorithm.BZ2: Bz2Compressor,
    CompressionAlgorithm.DEFLATE: DeflateCompressor,
}


def register_compressor(
    algorithm: CompressionAlgorithm,
    compressor_class: Type[BaseCompressor],
) -> None:
    """Register a custom compressor implementation.

    Args:
        algorithm: Algorithm identifier.
        compressor_class: Compressor class to register.
    """
    _COMPRESSOR_REGISTRY[algorithm] = compressor_class


def get_compressor(
    algorithm: str | CompressionAlgorithm,
    level: CompressionLevel | int = CompressionLevel.BALANCED,
    **kwargs: Any,
) -> BaseCompressor:
    """Create a compressor for the specified algorithm.

    Args:
        algorithm: Compression algorithm name or enum.
        level: Compression level.
        **kwargs: Additional configuration options.

    Returns:
        Configured compressor instance.

    Raises:
        UnsupportedAlgorithmError: If algorithm is not available.
    """
    # Convert string to enum
    if isinstance(algorithm, str):
        try:
            algorithm = CompressionAlgorithm(algorithm.lower())
        except ValueError:
            raise UnsupportedAlgorithmError(
                algorithm,
                [a.value for a in _COMPRESSOR_REGISTRY.keys()],
            )

    # Get compressor class
    compressor_class = _COMPRESSOR_REGISTRY.get(algorithm)
    if compressor_class is None:
        raise UnsupportedAlgorithmError(
            algorithm.value,
            [a.value for a in _COMPRESSOR_REGISTRY.keys()],
        )

    # Create config
    custom_level = level if isinstance(level, int) else None
    level_preset = level if isinstance(level, CompressionLevel) else CompressionLevel.BALANCED

    config = CompressionConfig(
        algorithm=algorithm,
        level=level_preset,
        custom_level=custom_level,
        **{k: v for k, v in kwargs.items() if hasattr(CompressionConfig, k)},
    )

    return compressor_class(config)


def get_decompressor(
    algorithm: str | CompressionAlgorithm,
    **kwargs: Any,
) -> BaseCompressor:
    """Create a decompressor for the specified algorithm.

    Args:
        algorithm: Compression algorithm name or enum.
        **kwargs: Additional configuration options.

    Returns:
        Configured decompressor instance.
    """
    return get_compressor(algorithm, **kwargs)


def list_available_algorithms() -> list[CompressionAlgorithm]:
    """List all available compression algorithms.

    Returns:
        List of available algorithms.
    """
    available = []
    for algorithm in _COMPRESSOR_REGISTRY.keys():
        if is_algorithm_available(algorithm):
            available.append(algorithm)
    return available


def is_algorithm_available(algorithm: str | CompressionAlgorithm) -> bool:
    """Check if a compression algorithm is available.

    Args:
        algorithm: Algorithm to check.

    Returns:
        True if algorithm is available.
    """
    if isinstance(algorithm, str):
        try:
            algorithm = CompressionAlgorithm(algorithm.lower())
        except ValueError:
            return False

    # Built-in algorithms are always available
    if algorithm in (
        CompressionAlgorithm.NONE,
        CompressionAlgorithm.GZIP,
        CompressionAlgorithm.DEFLATE,
        CompressionAlgorithm.LZMA,
        CompressionAlgorithm.BZ2,
    ):
        return True

    # Check optional dependencies
    if algorithm == CompressionAlgorithm.ZSTD:
        try:
            import zstandard
            return True
        except ImportError:
            return False

    if algorithm == CompressionAlgorithm.LZ4:
        try:
            import lz4.frame
            return True
        except ImportError:
            return False

    if algorithm == CompressionAlgorithm.SNAPPY:
        return _is_snappy_available()

    if algorithm == CompressionAlgorithm.BROTLI:
        try:
            import brotli
            return True
        except ImportError:
            return False

    return False
