"""Tests for compression providers."""

import pytest

from truthound.stores.compression import (
    CompressionAlgorithm,
    CompressionLevel,
    GzipCompressor,
    NoopCompressor,
    get_compressor,
    get_decompressor,
    list_available_algorithms,
    is_algorithm_available,
    register_compressor,
    BaseCompressor,
    CompressionError,
)


class TestNoopCompressor:
    """Tests for NoopCompressor."""

    def test_compress_returns_original(self):
        """Test that noop returns original data."""
        compressor = NoopCompressor()
        data = b"test data"

        result = compressor.compress(data)
        assert result == data

    def test_decompress_returns_original(self):
        """Test that noop returns original data."""
        compressor = NoopCompressor()
        data = b"test data"

        result = compressor.decompress(data)
        assert result == data

    def test_algorithm_property(self):
        """Test algorithm property."""
        compressor = NoopCompressor()
        assert compressor.algorithm == CompressionAlgorithm.NONE

    def test_compress_with_metrics(self):
        """Test compression with metrics."""
        compressor = NoopCompressor()
        data = b"test data"

        result = compressor.compress_with_metrics(data)

        assert result.data == data
        assert result.metrics.original_size == len(data)
        assert result.metrics.compressed_size == len(data)
        assert result.metrics.compression_ratio == 1.0


class TestGzipCompressor:
    """Tests for GzipCompressor."""

    def test_compress_decompress(self):
        """Test basic compression and decompression."""
        compressor = GzipCompressor()
        data = b"Hello, World! " * 100

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data
        assert len(compressed) < len(data)

    def test_algorithm_property(self):
        """Test algorithm property."""
        compressor = GzipCompressor()
        assert compressor.algorithm == CompressionAlgorithm.GZIP

    def test_compression_levels(self):
        """Test different compression levels."""
        data = b"Compressible data " * 1000

        # Use factory function to create with different levels
        fast = get_compressor(CompressionAlgorithm.GZIP, level=CompressionLevel.FAST)
        maximum = get_compressor(CompressionAlgorithm.GZIP, level=CompressionLevel.MAXIMUM)

        fast_result = fast.compress(data)
        max_result = maximum.compress(data)

        # Maximum should be smaller (better compression)
        assert len(max_result) <= len(fast_result)

        # Both should decompress correctly
        assert fast.decompress(fast_result) == data
        assert maximum.decompress(max_result) == data

    def test_compress_with_metrics(self):
        """Test compression with metrics."""
        compressor = GzipCompressor()
        data = b"Test data " * 100

        result = compressor.compress_with_metrics(data)

        assert result.metrics.original_size == len(data)
        assert result.metrics.compressed_size == len(result.data)
        assert result.metrics.compression_ratio > 1.0
        assert result.metrics.algorithm == CompressionAlgorithm.GZIP

    def test_empty_data(self):
        """Test handling empty data."""
        compressor = GzipCompressor()
        data = b""

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_compressor_by_string(self):
        """Test getting compressor by string name."""
        compressor = get_compressor("gzip")
        assert compressor.algorithm == CompressionAlgorithm.GZIP

    def test_get_compressor_by_enum(self):
        """Test getting compressor by enum."""
        compressor = get_compressor(CompressionAlgorithm.GZIP)
        assert compressor.algorithm == CompressionAlgorithm.GZIP

    def test_get_compressor_with_level(self):
        """Test getting compressor with level."""
        compressor = get_compressor("gzip", level=CompressionLevel.HIGH)
        assert compressor.algorithm == CompressionAlgorithm.GZIP

    def test_get_compressor_invalid(self):
        """Test getting invalid compressor."""
        with pytest.raises(Exception):
            get_compressor("invalid_algorithm")

    def test_get_decompressor(self):
        """Test getting decompressor."""
        decompressor = get_decompressor(CompressionAlgorithm.GZIP)
        assert decompressor.algorithm == CompressionAlgorithm.GZIP

    def test_list_available_algorithms(self):
        """Test listing available algorithms."""
        algorithms = list_available_algorithms()

        # At minimum, gzip should always be available
        assert CompressionAlgorithm.GZIP in algorithms
        assert CompressionAlgorithm.NONE in algorithms

    def test_is_algorithm_available(self):
        """Test checking algorithm availability."""
        assert is_algorithm_available(CompressionAlgorithm.GZIP)
        assert is_algorithm_available(CompressionAlgorithm.NONE)


class TestCustomCompressor:
    """Tests for custom compressor registration."""

    def test_register_custom_compressor(self):
        """Test registering a custom compressor."""

        class CustomCompressor(BaseCompressor):
            @property
            def algorithm(self):
                return CompressionAlgorithm.DEFLATE

            def _do_compress(self, data):
                return b"custom:" + data

            def _do_decompress(self, data):
                if data.startswith(b"custom:"):
                    return data[7:]
                return data

        # Register using the algorithm enum
        register_compressor(CompressionAlgorithm.DEFLATE, CustomCompressor)

        # Should be able to get it by enum
        compressor = get_compressor(CompressionAlgorithm.DEFLATE)
        assert compressor.compress(b"test") == b"custom:test"


class TestOptionalCompressors:
    """Tests for optional compressor availability."""

    def test_zstd_availability(self):
        """Test zstd availability detection."""
        available = is_algorithm_available(CompressionAlgorithm.ZSTD)
        # Just check it returns a boolean
        assert isinstance(available, bool)

        if available:
            compressor = get_compressor(CompressionAlgorithm.ZSTD)
            data = b"test data " * 100
            compressed = compressor.compress(data)
            assert compressor.decompress(compressed) == data

    def test_lz4_availability(self):
        """Test lz4 availability detection."""
        available = is_algorithm_available(CompressionAlgorithm.LZ4)
        assert isinstance(available, bool)

        if available:
            compressor = get_compressor(CompressionAlgorithm.LZ4)
            data = b"test data " * 100
            compressed = compressor.compress(data)
            assert compressor.decompress(compressed) == data

    def test_snappy_availability(self):
        """Test snappy availability detection."""
        available = is_algorithm_available(CompressionAlgorithm.SNAPPY)
        assert isinstance(available, bool)

    def test_brotli_availability(self):
        """Test brotli availability detection."""
        available = is_algorithm_available(CompressionAlgorithm.BROTLI)
        assert isinstance(available, bool)

        if available:
            compressor = get_compressor(CompressionAlgorithm.BROTLI)
            data = b"test data " * 100
            compressed = compressor.compress(data)
            assert compressor.decompress(compressed) == data


class TestCompressDecompressRoundtrip:
    """Test roundtrip for all available algorithms."""

    @pytest.fixture
    def test_data(self):
        """Generate test data."""
        return b"Hello, this is test data for compression! " * 100

    def test_roundtrip_all_available(self, test_data):
        """Test roundtrip for all available algorithms."""
        for algo in list_available_algorithms():
            if algo == CompressionAlgorithm.NONE:
                continue

            compressor = get_compressor(algo)
            compressed = compressor.compress(test_data)
            decompressed = compressor.decompress(compressed)

            assert decompressed == test_data, f"Roundtrip failed for {algo}"

    def test_empty_data_all_algorithms(self):
        """Test empty data handling for all algorithms."""
        for algo in list_available_algorithms():
            compressor = get_compressor(algo)
            compressed = compressor.compress(b"")
            decompressed = compressor.decompress(compressed)

            assert decompressed == b"", f"Empty data failed for {algo}"

    def test_single_byte_all_algorithms(self):
        """Test single byte for all algorithms."""
        for algo in list_available_algorithms():
            compressor = get_compressor(algo)
            data = b"X"
            compressed = compressor.compress(data)
            decompressed = compressor.decompress(compressed)

            assert decompressed == data, f"Single byte failed for {algo}"


class TestCompressionQuality:
    """Tests for compression quality."""

    def test_compressible_data(self):
        """Test that compressible data actually compresses."""
        compressor = GzipCompressor()
        data = b"AAAAAAAAAA" * 1000  # Highly compressible

        compressed = compressor.compress(data)

        # Should achieve significant compression
        assert len(compressed) < len(data) * 0.2

    def test_random_data_not_much_compression(self):
        """Test that random data doesn't compress much."""
        import random

        compressor = GzipCompressor()
        data = bytes(random.randint(0, 255) for _ in range(1000))

        compressed = compressor.compress(data)

        # Random data shouldn't compress much
        # Compressed might even be larger due to header overhead
        assert len(compressed) > len(data) * 0.9


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_compression(self):
        """Test concurrent compression."""
        import concurrent.futures

        compressor = GzipCompressor()
        data = b"Test data for concurrent compression " * 100

        def compress_task(task_id):
            compressed = compressor.compress(data)
            decompressed = compressor.decompress(compressed)
            return decompressed == data

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compress_task, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert all(results)
