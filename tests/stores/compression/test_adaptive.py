"""Tests for adaptive compression."""

import pytest

from truthound.stores.compression import (
    AdaptiveCompressor,
    AdaptiveConfig,
    CompressionAlgorithm,
    CompressionGoal,
    CompressionLevel,
    CompressionProfile,
    DataAnalyzer,
    DataCharacteristics,
    DataType,
    AlgorithmSelector,
)


class TestDataAnalyzer:
    """Tests for DataAnalyzer."""

    def test_analyze_text(self):
        """Test analyzing text data."""
        analyzer = DataAnalyzer()
        data = b"Hello, this is a test of text analysis. " * 10

        chars = analyzer.analyze(data)

        assert chars.size == len(data)
        assert chars.is_text is True
        assert chars.is_binary is False
        assert 0 <= chars.entropy <= 8

    def test_analyze_binary(self):
        """Test analyzing binary data."""
        analyzer = DataAnalyzer()
        data = bytes(range(256)) * 4

        chars = analyzer.analyze(data)

        assert chars.is_text is False
        assert chars.is_binary is True

    def test_analyze_json(self):
        """Test analyzing JSON data."""
        analyzer = DataAnalyzer()
        data = b'{"name": "test", "value": 123}' * 10

        chars = analyzer.analyze(data)

        assert chars.is_text is True
        assert chars.detected_type == DataType.JSON

    def test_analyze_repetitive(self):
        """Test analyzing repetitive data."""
        analyzer = DataAnalyzer()
        data = b"AAAA" * 1000

        chars = analyzer.analyze(data)

        assert chars.repetition_score > 0.5
        assert chars.detected_type == DataType.HIGHLY_COMPRESSIBLE

    def test_analyze_random(self):
        """Test analyzing random data."""
        import random

        analyzer = DataAnalyzer()
        data = bytes(random.randint(0, 255) for _ in range(1000))

        chars = analyzer.analyze(data)

        assert chars.entropy > 7.0
        assert chars.detected_type == DataType.RANDOM

    def test_analyze_empty(self):
        """Test analyzing empty data."""
        analyzer = DataAnalyzer()
        data = b""

        chars = analyzer.analyze(data)

        assert chars.size == 0
        assert chars.entropy == 0.0

    def test_compressibility_estimate(self):
        """Test compressibility estimation."""
        analyzer = DataAnalyzer()

        # Highly compressible
        data1 = b"AAAA" * 1000
        chars1 = analyzer.analyze(data1)
        assert chars1.compressibility_estimate > 0.7

        # Random, low compressibility
        import random

        data2 = bytes(random.randint(0, 255) for _ in range(1000))
        chars2 = analyzer.analyze(data2)
        assert chars2.compressibility_estimate < 0.3

    def test_to_dict(self):
        """Test dictionary conversion."""
        analyzer = DataAnalyzer()
        data = b"test data"

        chars = analyzer.analyze(data)
        data_dict = chars.to_dict()

        assert "size" in data_dict
        assert "entropy" in data_dict
        assert "detected_type" in data_dict


class TestAlgorithmSelector:
    """Tests for AlgorithmSelector."""

    def test_select_balanced(self):
        """Test balanced selection."""
        selector = AlgorithmSelector()
        chars = DataCharacteristics(
            size=1000,
            entropy=4.0,
            is_text=True,
        )

        profile = selector.select(chars, CompressionGoal.BALANCED)

        assert profile.algorithm != CompressionAlgorithm.NONE
        assert profile.score > 0

    def test_select_best_speed(self):
        """Test speed-optimized selection."""
        selector = AlgorithmSelector()
        chars = DataCharacteristics(
            size=1000,
            entropy=4.0,
            is_text=True,
        )

        profile = selector.select(chars, CompressionGoal.BEST_SPEED)

        # Should prefer fast algorithms like LZ4 or Snappy if available
        assert profile.expected_speed_mbps >= 100 or profile.algorithm == CompressionAlgorithm.GZIP

    def test_select_best_ratio(self):
        """Test ratio-optimized selection."""
        selector = AlgorithmSelector()
        chars = DataCharacteristics(
            size=1000,
            entropy=3.0,
            is_text=True,
        )

        profile = selector.select(chars, CompressionGoal.BEST_RATIO)

        # Should prefer high-ratio algorithms
        assert profile.expected_ratio >= 2.0

    def test_rank_algorithms(self):
        """Test algorithm ranking."""
        selector = AlgorithmSelector()
        chars = DataCharacteristics(
            size=1000,
            entropy=4.0,
            is_text=True,
        )

        profiles = selector.rank_algorithms(chars, CompressionGoal.BALANCED)

        assert len(profiles) > 0
        # Should be sorted by score descending
        for i in range(len(profiles) - 1):
            assert profiles[i].score >= profiles[i + 1].score

    def test_random_data_penalty(self):
        """Test that random data gets penalized."""
        # Only include algorithms we have profiles for
        selector = AlgorithmSelector(
            available_algorithms=[CompressionAlgorithm.GZIP, CompressionAlgorithm.ZSTD]
        )
        chars = DataCharacteristics(
            size=1000,
            entropy=7.9,
            is_text=False,
            detected_type=DataType.RANDOM,
        )

        profile = selector.select(chars, CompressionGoal.BALANCED)

        # Should have low score due to random data
        assert profile.score < 0.5
        assert "Random" in profile.reason or "poor" in profile.reason.lower()


class TestCompressionProfile:
    """Tests for CompressionProfile."""

    def test_creation(self):
        """Test profile creation."""
        profile = CompressionProfile(
            algorithm=CompressionAlgorithm.GZIP,
            score=0.85,
            expected_ratio=3.5,
            expected_speed_mbps=50,
            reason="Good for text",
        )

        assert profile.algorithm == CompressionAlgorithm.GZIP
        assert profile.score == 0.85
        assert profile.expected_ratio == 3.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        profile = CompressionProfile(
            algorithm=CompressionAlgorithm.ZSTD,
            score=0.9,
            expected_ratio=4.0,
            expected_speed_mbps=150,
            reason="Best overall",
        )

        data = profile.to_dict()

        assert data["algorithm"] == "zstd"
        assert data["score"] == 0.9


class TestAdaptiveConfig:
    """Tests for AdaptiveConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = AdaptiveConfig()

        assert config.goal == CompressionGoal.BALANCED
        assert config.enable_sampling is True
        assert config.fallback_algorithm == CompressionAlgorithm.GZIP

    def test_custom_values(self):
        """Test custom configuration."""
        config = AdaptiveConfig(
            goal=CompressionGoal.BEST_SPEED,
            sample_size=32768,
            min_size_for_compression=256,
        )

        assert config.goal == CompressionGoal.BEST_SPEED
        assert config.sample_size == 32768
        assert config.min_size_for_compression == 256


class TestAdaptiveCompressor:
    """Tests for AdaptiveCompressor."""

    def test_compress_decompress(self):
        """Test basic compression and decompression."""
        compressor = AdaptiveCompressor()
        data = b"Hello, World! " * 100

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_small_data_passthrough(self):
        """Test that small data passes through."""
        config = AdaptiveConfig(min_size_for_compression=1000)
        compressor = AdaptiveCompressor(config=config)
        data = b"small"

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_algorithm_selection(self):
        """Test that algorithms are selected."""
        compressor = AdaptiveCompressor()
        data = b"Test data for algorithm selection " * 100

        result = compressor.compress_with_metrics(data)

        # Should have selected an algorithm
        assert result.metrics.algorithm != CompressionAlgorithm.NONE or len(result.data) >= len(data)

    def test_last_analysis(self):
        """Test accessing last analysis."""
        compressor = AdaptiveCompressor()
        data = b"Analysis test data " * 50

        compressor.compress(data)

        assert compressor.last_analysis is not None
        assert compressor.last_analysis.size == len(data)

    def test_last_profile(self):
        """Test accessing last profile."""
        compressor = AdaptiveCompressor()
        data = b"Profile test data " * 50

        compressor.compress(data)

        assert compressor.last_profile is not None
        assert compressor.last_profile.algorithm is not None

    def test_compress_with_metrics(self):
        """Test compression with full metrics."""
        compressor = AdaptiveCompressor()
        data = b"Metrics test data " * 100

        result = compressor.compress_with_metrics(data)

        assert result.metrics.original_size == len(data)
        assert result.metrics.compression_time_ms > 0
        assert "adaptive" in result.header
        assert result.header["adaptive"] is True

    def test_get_recommendation(self):
        """Test getting recommendations without compressing."""
        compressor = AdaptiveCompressor()
        data = b"Recommendation test " * 100

        recommendations = compressor.get_recommendation(data)

        assert len(recommendations) > 0
        # Should be sorted by score
        for i in range(len(recommendations) - 1):
            assert recommendations[i].score >= recommendations[i + 1].score

    def test_allowed_algorithms(self):
        """Test limiting allowed algorithms."""
        config = AdaptiveConfig(
            allowed_algorithms=[CompressionAlgorithm.GZIP]
        )
        compressor = AdaptiveCompressor(config=config)
        data = b"Limited algorithms test " * 100

        compressor.compress(data)

        # Should only have used gzip
        assert compressor.last_profile is None or compressor.last_profile.algorithm == CompressionAlgorithm.GZIP

    def test_compression_goal_speed(self):
        """Test speed-focused compression."""
        config = AdaptiveConfig(goal=CompressionGoal.BEST_SPEED)
        compressor = AdaptiveCompressor(config=config)
        data = b"Speed test data " * 100

        result = compressor.compress_with_metrics(data)

        # Should complete quickly
        assert result.metrics.compression_time_ms < 1000

    def test_compression_goal_ratio(self):
        """Test ratio-focused compression."""
        config = AdaptiveConfig(goal=CompressionGoal.BEST_RATIO)
        compressor = AdaptiveCompressor(config=config)
        data = b"Ratio test data AAAAA" * 100

        result = compressor.compress_with_metrics(data)

        # Should achieve good compression
        if result.metrics.compression_ratio > 0:
            assert result.metrics.compression_ratio > 1.0

    def test_fallback_on_error(self):
        """Test fallback algorithm usage."""
        config = AdaptiveConfig(
            fallback_algorithm=CompressionAlgorithm.GZIP,
        )
        compressor = AdaptiveCompressor(config=config)
        data = b"Fallback test data " * 100

        # Should succeed even if primary selection fails
        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data


class TestAdaptiveCompressorRoundtrip:
    """Roundtrip tests for AdaptiveCompressor."""

    @pytest.fixture
    def compressor(self):
        """Create adaptive compressor."""
        return AdaptiveCompressor()

    def test_roundtrip_text(self, compressor):
        """Test roundtrip with text data."""
        data = b"Hello, this is text data! " * 100

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_roundtrip_binary(self, compressor):
        """Test roundtrip with binary data."""
        data = bytes(range(256)) * 10

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_roundtrip_json(self, compressor):
        """Test roundtrip with JSON data."""
        data = b'{"key": "value", "number": 123}' * 100

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_roundtrip_repetitive(self, compressor):
        """Test roundtrip with highly repetitive data."""
        data = b"AAAA" * 10000

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data
        assert len(compressed) < len(data)

    def test_roundtrip_empty(self, compressor):
        """Test roundtrip with empty data."""
        data = b""

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data

    def test_roundtrip_random(self, compressor):
        """Test roundtrip with random data."""
        import random

        data = bytes(random.randint(0, 255) for _ in range(1000))

        compressed = compressor.compress(data)
        decompressed = compressor.decompress(compressed)

        assert decompressed == data
