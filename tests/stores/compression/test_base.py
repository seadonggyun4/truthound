"""Tests for compression base classes and types."""

import pytest

from truthound.stores.compression import (
    CompressionAlgorithm,
    CompressionConfig,
    CompressionLevel,
    CompressionMetrics,
    CompressionMode,
    CompressionResult,
    CompressionStats,
    CompressionError,
    DecompressionError,
    UnsupportedAlgorithmError,
)


class TestCompressionAlgorithm:
    """Tests for CompressionAlgorithm enum."""

    def test_algorithm_values(self):
        """Test algorithm enum values."""
        assert CompressionAlgorithm.NONE.value == "none"
        assert CompressionAlgorithm.GZIP.value == "gzip"
        assert CompressionAlgorithm.ZSTD.value == "zstd"
        assert CompressionAlgorithm.LZ4.value == "lz4"
        assert CompressionAlgorithm.SNAPPY.value == "snappy"
        assert CompressionAlgorithm.BROTLI.value == "brotli"
        assert CompressionAlgorithm.LZMA.value == "lzma"
        assert CompressionAlgorithm.BZ2.value == "bz2"

    def test_from_extension(self):
        """Test algorithm detection from file extension."""
        assert CompressionAlgorithm.from_extension(".gz") == CompressionAlgorithm.GZIP
        assert CompressionAlgorithm.from_extension(".gzip") == CompressionAlgorithm.GZIP
        assert CompressionAlgorithm.from_extension(".zst") == CompressionAlgorithm.ZSTD
        assert CompressionAlgorithm.from_extension(".lz4") == CompressionAlgorithm.LZ4
        assert CompressionAlgorithm.from_extension(".br") == CompressionAlgorithm.BROTLI
        assert CompressionAlgorithm.from_extension(".xz") == CompressionAlgorithm.LZMA
        assert CompressionAlgorithm.from_extension(".bz2") == CompressionAlgorithm.BZ2
        assert CompressionAlgorithm.from_extension(".txt") == CompressionAlgorithm.NONE

    def test_extension_property(self):
        """Test file extension property."""
        assert CompressionAlgorithm.GZIP.extension == ".gz"
        assert CompressionAlgorithm.ZSTD.extension == ".zst"
        assert CompressionAlgorithm.LZ4.extension == ".lz4"
        assert CompressionAlgorithm.BROTLI.extension == ".br"
        assert CompressionAlgorithm.NONE.extension == ""

    def test_content_encoding(self):
        """Test HTTP content encoding."""
        assert CompressionAlgorithm.GZIP.content_encoding == "gzip"
        assert CompressionAlgorithm.ZSTD.content_encoding == "zstd"
        assert CompressionAlgorithm.BROTLI.content_encoding == "br"
        assert CompressionAlgorithm.NONE.content_encoding == "identity"


class TestCompressionLevel:
    """Tests for CompressionLevel enum."""

    def test_level_values(self):
        """Test compression level enum values."""
        assert CompressionLevel.FASTEST is not None
        assert CompressionLevel.FAST is not None
        assert CompressionLevel.BALANCED is not None
        assert CompressionLevel.HIGH is not None
        assert CompressionLevel.MAXIMUM is not None

    def test_get_level_gzip(self):
        """Test level mapping for gzip."""
        assert CompressionLevel.FASTEST.get_level(CompressionAlgorithm.GZIP) == 1
        assert CompressionLevel.FAST.get_level(CompressionAlgorithm.GZIP) == 3
        assert CompressionLevel.BALANCED.get_level(CompressionAlgorithm.GZIP) == 6
        assert CompressionLevel.HIGH.get_level(CompressionAlgorithm.GZIP) == 8
        assert CompressionLevel.MAXIMUM.get_level(CompressionAlgorithm.GZIP) == 9

    def test_get_level_zstd(self):
        """Test level mapping for zstd."""
        assert CompressionLevel.FASTEST.get_level(CompressionAlgorithm.ZSTD) == 1
        assert CompressionLevel.BALANCED.get_level(CompressionAlgorithm.ZSTD) == 6
        assert CompressionLevel.MAXIMUM.get_level(CompressionAlgorithm.ZSTD) == 19

    def test_get_level_lz4(self):
        """Test level mapping for lz4."""
        assert CompressionLevel.FASTEST.get_level(CompressionAlgorithm.LZ4) == 0
        assert CompressionLevel.MAXIMUM.get_level(CompressionAlgorithm.LZ4) == 12


class TestCompressionConfig:
    """Tests for CompressionConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = CompressionConfig()

        assert config.algorithm == CompressionAlgorithm.GZIP
        assert config.level == CompressionLevel.BALANCED
        assert config.custom_level is None
        assert config.chunk_size == 64 * 1024
        assert config.verify_checksum is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.ZSTD,
            level=CompressionLevel.HIGH,
            chunk_size=128 * 1024,
        )

        assert config.algorithm == CompressionAlgorithm.ZSTD
        assert config.level == CompressionLevel.HIGH
        assert config.chunk_size == 128 * 1024

    def test_get_effective_level(self):
        """Test effective level calculation."""
        config = CompressionConfig(
            algorithm=CompressionAlgorithm.GZIP,
            level=CompressionLevel.BALANCED,
        )
        assert config.get_effective_level() == 6

        config.custom_level = 9
        assert config.get_effective_level() == 9

    def test_validate_success(self):
        """Test validation with valid config."""
        config = CompressionConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_chunk_size(self):
        """Test validation with invalid chunk size."""
        config = CompressionConfig(chunk_size=0)
        with pytest.raises(Exception):
            config.validate()

    def test_validate_invalid_threads(self):
        """Test validation with invalid threads."""
        config = CompressionConfig(threads=-1)
        with pytest.raises(Exception):
            config.validate()


class TestCompressionMetrics:
    """Tests for CompressionMetrics."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = CompressionMetrics()

        assert metrics.original_size == 0
        assert metrics.compressed_size == 0
        assert metrics.compression_ratio == 0.0
        assert metrics.compression_time_ms == 0.0

    def test_update_ratio(self):
        """Test ratio calculation."""
        metrics = CompressionMetrics(
            original_size=1000,
            compressed_size=250,
        )
        metrics.update_ratio()

        assert metrics.compression_ratio == 4.0

    def test_update_ratio_zero_compressed(self):
        """Test ratio with zero compressed size."""
        metrics = CompressionMetrics(
            original_size=1000,
            compressed_size=0,
        )
        metrics.update_ratio()

        assert metrics.compression_ratio == 0.0

    def test_space_savings(self):
        """Test space savings calculation."""
        metrics = CompressionMetrics(
            original_size=1000,
            compressed_size=250,
        )

        assert metrics.space_savings == 75.0

    def test_space_savings_zero_original(self):
        """Test space savings with zero original."""
        metrics = CompressionMetrics(
            original_size=0,
            compressed_size=0,
        )

        assert metrics.space_savings == 0.0

    def test_throughput_compress(self):
        """Test compression throughput."""
        metrics = CompressionMetrics(
            original_size=1024 * 1024,  # 1MB
            compression_time_ms=1000,  # 1 second
        )

        assert metrics.throughput_compress_mbps == 1.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = CompressionMetrics(
            original_size=1000,
            compressed_size=250,
            compression_time_ms=10.5,
            algorithm=CompressionAlgorithm.GZIP,
            level=6,
        )
        metrics.update_ratio()

        data = metrics.to_dict()

        assert data["original_size"] == 1000
        assert data["compressed_size"] == 250
        assert data["compression_ratio"] == 4.0
        assert data["algorithm"] == "gzip"


class TestCompressionResult:
    """Tests for CompressionResult."""

    def test_creation(self):
        """Test result creation."""
        metrics = CompressionMetrics(
            original_size=100,
            compressed_size=50,
        )
        result = CompressionResult(
            data=b"compressed",
            metrics=metrics,
        )

        assert result.data == b"compressed"
        assert result.metrics.original_size == 100

    def test_to_bytes(self):
        """Test data extraction."""
        result = CompressionResult(
            data=b"test",
            metrics=CompressionMetrics(),
        )

        assert result.to_bytes() == b"test"


class TestCompressionStats:
    """Tests for CompressionStats."""

    def test_initial_values(self):
        """Test initial stats."""
        stats = CompressionStats()

        assert stats.total_operations == 0
        assert stats.total_original_bytes == 0
        assert stats.errors == 0

    def test_record(self):
        """Test recording metrics."""
        stats = CompressionStats()

        metrics = CompressionMetrics(
            original_size=1000,
            compressed_size=250,
            compression_time_ms=10,
            algorithm=CompressionAlgorithm.GZIP,
        )

        stats.record(metrics)

        assert stats.total_operations == 1
        assert stats.total_original_bytes == 1000
        assert stats.total_compressed_bytes == 250
        assert stats.algorithm_usage["gzip"] == 1

    def test_average_ratio(self):
        """Test average ratio calculation."""
        stats = CompressionStats()
        stats.total_original_bytes = 2000
        stats.total_compressed_bytes = 500

        assert stats.average_ratio == 4.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = CompressionStats()
        stats.total_operations = 5
        stats.total_original_bytes = 5000
        stats.total_compressed_bytes = 1000

        data = stats.to_dict()

        assert data["total_operations"] == 5
        assert data["average_ratio"] == 5.0


class TestExceptions:
    """Tests for exception classes."""

    def test_compression_error(self):
        """Test CompressionError."""
        error = CompressionError("Test error", algorithm="gzip")

        assert "gzip" in str(error)
        assert "Test error" in str(error)
        assert error.algorithm == "gzip"

    def test_decompression_error(self):
        """Test DecompressionError."""
        error = DecompressionError("Invalid data", algorithm="zstd")

        assert isinstance(error, CompressionError)
        assert "Invalid data" in str(error)

    def test_unsupported_algorithm_error(self):
        """Test UnsupportedAlgorithmError."""
        error = UnsupportedAlgorithmError("foo", available=["gzip", "zstd"])

        assert "foo" in str(error)
        assert "gzip" in str(error)
        assert error.available == ["gzip", "zstd"]
