"""Tests for probabilistic data structures (sketches).

This module tests:
    - HyperLogLog cardinality estimation
    - Count-Min Sketch frequency estimation
    - Bloom Filter membership testing
    - Sketch merging for distributed processing
    - Factory functions and presets
"""

import math
import pytest

from truthound.profiler.sketches import (
    # Core implementations
    HyperLogLog,
    CountMinSketch,
    BloomFilter,
    # Factory
    create_sketch,
    SketchType,
    SketchFactory,
)
from truthound.profiler.sketches.protocols import (
    HyperLogLogConfig,
    CountMinSketchConfig,
    BloomFilterConfig,
    SketchMetrics,
)


# ============================================================================
# HyperLogLog Tests
# ============================================================================


class TestHyperLogLog:
    """Tests for HyperLogLog cardinality estimator."""

    def test_basic_cardinality(self):
        """Test basic cardinality estimation."""
        hll = HyperLogLog(HyperLogLogConfig(precision=12))

        # Add 10,000 distinct values
        for i in range(10000):
            hll.add(f"value_{i}")

        estimate = hll.estimate()

        # Should be within 5% of true count (2 standard errors)
        error_margin = 10000 * hll.standard_error() * 2
        assert abs(estimate - 10000) < error_margin

    def test_duplicate_handling(self):
        """Test that duplicates don't affect cardinality."""
        hll = HyperLogLog(HyperLogLogConfig(precision=12))

        # Add same value 1000 times
        for _ in range(1000):
            hll.add("same_value")

        # Should estimate ~1 distinct value
        assert hll.estimate() == 1

    def test_batch_add(self):
        """Test batch add operation."""
        hll1 = HyperLogLog(HyperLogLogConfig(precision=12, seed=42))
        hll2 = HyperLogLog(HyperLogLogConfig(precision=12, seed=42))

        values = [f"val_{i}" for i in range(5000)]

        # Add one by one
        for v in values:
            hll1.add(v)

        # Add in batch
        hll2.add_batch(values)

        # Should produce same result
        assert hll1.estimate() == hll2.estimate()

    def test_merge(self):
        """Test merging two HyperLogLogs."""
        hll1 = HyperLogLog(HyperLogLogConfig(precision=12))
        hll2 = HyperLogLog(HyperLogLogConfig(precision=12))

        # Add distinct values to each
        for i in range(5000):
            hll1.add(f"a_{i}")
        for i in range(5000):
            hll2.add(f"b_{i}")

        # Merge
        merged = hll1.merge(hll2)

        # Should estimate ~10,000
        estimate = merged.estimate()
        # Use 5x standard error to account for probabilistic variance
        error_margin = 10000 * merged.standard_error() * 5
        assert abs(estimate - 10000) < error_margin

    def test_merge_with_overlap(self):
        """Test merging HyperLogLogs with overlapping data."""
        hll1 = HyperLogLog(HyperLogLogConfig(precision=12))
        hll2 = HyperLogLog(HyperLogLogConfig(precision=12))

        # Add overlapping values
        for i in range(7000):
            hll1.add(f"val_{i}")  # 0-6999
        for i in range(3000, 10000):
            hll2.add(f"val_{i}")  # 3000-9999

        # Merged should have ~10,000 distinct (0-9999)
        merged = hll1.merge(hll2)
        estimate = merged.estimate()
        # Use 5x standard error to account for probabilistic variance
        error_margin = 10000 * merged.standard_error() * 5
        assert abs(estimate - 10000) < error_margin

    def test_merge_incompatible_precision(self):
        """Test that merging different precisions fails."""
        hll1 = HyperLogLog(HyperLogLogConfig(precision=10))
        hll2 = HyperLogLog(HyperLogLogConfig(precision=12))

        with pytest.raises(ValueError, match="different precision"):
            hll1.merge(hll2)

    def test_precision_levels(self):
        """Test different precision levels."""
        for precision in [10, 12, 14, 16]:
            hll = HyperLogLog(HyperLogLogConfig(precision=precision))

            # Higher precision = lower error
            expected_error = 1.04 / (2**precision) ** 0.5
            assert abs(hll.standard_error() - expected_error) < 0.001

    def test_memory_usage(self):
        """Test memory usage reporting."""
        hll = HyperLogLog(HyperLogLogConfig(precision=12))
        assert hll.memory_bytes() == 4096  # 2^12 registers

    def test_clear(self):
        """Test clearing the sketch."""
        hll = HyperLogLog(HyperLogLogConfig(precision=12))

        for i in range(1000):
            hll.add(i)

        assert hll.estimate() > 0

        hll.clear()
        assert hll.estimate() == 0

    def test_config_for_error_rate(self):
        """Test creating config from target error rate."""
        # Target 1% error
        config = HyperLogLogConfig.for_error_rate(0.01)

        # Should select precision that achieves target error
        assert config.expected_error <= 0.015  # Allow some slack

    def test_metrics(self):
        """Test metrics collection."""
        hll = HyperLogLog(HyperLogLogConfig(precision=12))

        for i in range(1000):
            hll.add(i)

        metrics = hll.metrics()
        assert isinstance(metrics, SketchMetrics)
        assert metrics.elements_added == 1000
        assert metrics.memory_bytes == 4096
        assert 0 < metrics.fill_ratio < 1


# ============================================================================
# Count-Min Sketch Tests
# ============================================================================


class TestCountMinSketch:
    """Tests for Count-Min Sketch frequency estimator."""

    def test_basic_frequency(self):
        """Test basic frequency estimation."""
        cms = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        # Add items with known frequencies
        for _ in range(100):
            cms.add("frequent")
        for _ in range(10):
            cms.add("rare")

        # Check estimates
        freq_estimate = cms.estimate_frequency("frequent")
        rare_estimate = cms.estimate_frequency("rare")

        # Should be >= true count (CMS never underestimates)
        assert freq_estimate >= 100
        assert rare_estimate >= 10

        # Should not massively overestimate
        assert freq_estimate < 150
        assert rare_estimate < 50

    def test_add_with_count(self):
        """Test adding with explicit count."""
        cms = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        cms.add("item", count=50)

        assert cms.estimate_frequency("item") >= 50

    def test_batch_add(self):
        """Test batch add operation."""
        cms = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        values = ["a"] * 100 + ["b"] * 50 + ["c"] * 10
        cms.add_batch(values)

        assert cms.estimate_frequency("a") >= 100
        assert cms.estimate_frequency("b") >= 50
        assert cms.estimate_frequency("c") >= 10

    def test_heavy_hitters(self):
        """Test heavy hitter detection."""
        cms = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        # Add items: some frequent, some rare
        for _ in range(500):
            cms.add("very_frequent")
        for _ in range(100):
            cms.add("frequent")
        for _ in range(10):
            cms.add("rare")

        # Get heavy hitters (> 10% of total)
        heavy = cms.get_heavy_hitters(threshold=0.10)

        # Should include very_frequent (500/610 = 82%)
        heavy_items = [item for item, _ in heavy]
        assert "very_frequent" in heavy_items

    def test_merge(self):
        """Test merging two Count-Min Sketches."""
        cms1 = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))
        cms2 = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        for _ in range(100):
            cms1.add("item")
        for _ in range(50):
            cms2.add("item")

        merged = cms1.merge(cms2)

        # Merged should have sum of counts
        assert merged.estimate_frequency("item") >= 150

    def test_merge_incompatible(self):
        """Test that merging different dimensions fails."""
        cms1 = CountMinSketch(CountMinSketchConfig(width=1000, depth=5))
        cms2 = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        with pytest.raises(ValueError, match="different dimensions"):
            cms1.merge(cms2)

    def test_config_for_error_and_confidence(self):
        """Test creating config from error and confidence targets."""
        config = CountMinSketchConfig.for_error_and_confidence(
            epsilon=0.001,  # 0.1% error
            delta=0.01,  # 99% confidence
        )

        # Check bounds
        assert config.expected_error <= 0.002
        assert config.confidence >= 0.95

    def test_total_count(self):
        """Test total count tracking."""
        cms = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        cms.add("a", count=10)
        cms.add("b", count=20)
        cms.add("c", count=30)

        assert cms.total_count == 60

    def test_clear(self):
        """Test clearing the sketch."""
        cms = CountMinSketch(CountMinSketchConfig(width=2000, depth=5))

        cms.add("item", count=100)
        assert cms.total_count == 100

        cms.clear()
        assert cms.total_count == 0
        assert cms.estimate_frequency("item") == 0


# ============================================================================
# Bloom Filter Tests
# ============================================================================


class TestBloomFilter:
    """Tests for Bloom Filter membership tester."""

    def test_basic_membership(self):
        """Test basic membership testing."""
        bf = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01))

        # Add some items
        for i in range(1000):
            bf.add(f"item_{i}")

        # Should find all added items
        for i in range(1000):
            assert bf.contains(f"item_{i}")

        # Should not find most non-added items
        false_positives = 0
        for i in range(1000, 2000):
            if bf.contains(f"item_{i}"):
                false_positives += 1

        # False positive rate should be close to configured rate
        actual_fp_rate = false_positives / 1000
        assert actual_fp_rate < 0.05  # Allow some slack

    def test_no_false_negatives(self):
        """Test that there are no false negatives."""
        bf = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01))

        items = [f"item_{i}" for i in range(5000)]
        bf.add_batch(items)

        # Every added item must be found
        for item in items:
            assert bf.contains(item), f"False negative for {item}"

    def test_batch_add(self):
        """Test batch add operation."""
        bf1 = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01, seed=42))
        bf2 = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01, seed=42))

        items = [f"item_{i}" for i in range(1000)]

        for item in items:
            bf1.add(item)

        bf2.add_batch(items)

        # Both should contain all items
        for item in items:
            assert bf1.contains(item)
            assert bf2.contains(item)

    def test_merge(self):
        """Test merging two Bloom Filters."""
        bf1 = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01, seed=42))
        bf2 = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01, seed=42))

        for i in range(500):
            bf1.add(f"a_{i}")
        for i in range(500):
            bf2.add(f"b_{i}")

        merged = bf1.merge(bf2)

        # Should contain all items from both
        for i in range(500):
            assert merged.contains(f"a_{i}")
            assert merged.contains(f"b_{i}")

    def test_in_operator(self):
        """Test 'in' operator support."""
        bf = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01))

        bf.add("test_item")

        assert "test_item" in bf
        # Note: "other_item" might be in bf due to false positives

    def test_false_positive_rate(self):
        """Test false positive rate calculation."""
        bf = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01))

        # Empty filter should have 0% FP rate
        assert bf.false_positive_rate() == 0.0

        # Add items and check FP rate increases
        for i in range(5000):
            bf.add(f"item_{i}")

        fp_rate = bf.false_positive_rate()
        assert 0 < fp_rate < 0.05

    def test_fill_ratio(self):
        """Test fill ratio tracking."""
        bf = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01))

        assert bf.fill_ratio == 0.0

        for i in range(1000):
            bf.add(i)

        assert 0 < bf.fill_ratio < 1.0

    def test_clear(self):
        """Test clearing the filter."""
        bf = BloomFilter(BloomFilterConfig(capacity=10000, error_rate=0.01))

        bf.add("item")
        assert bf.contains("item")

        bf.clear()
        # After clear, the item might still "match" due to filter reset
        # but the internal state should be cleared
        assert bf.fill_ratio == 0.0


# ============================================================================
# Factory Tests
# ============================================================================


class TestSketchFactory:
    """Tests for sketch factory functions."""

    def test_create_hyperloglog(self):
        """Test creating HyperLogLog via factory."""
        factory = SketchFactory()

        hll = factory.create_hyperloglog(precision=14)
        assert isinstance(hll, HyperLogLog)
        assert hll.config.precision == 14

    def test_create_hyperloglog_by_error(self):
        """Test creating HyperLogLog with target error."""
        factory = SketchFactory()

        hll = factory.create_hyperloglog(target_error=0.01)
        assert hll.standard_error() <= 0.015

    def test_create_countmin(self):
        """Test creating CountMinSketch via factory."""
        factory = SketchFactory()

        cms = factory.create_countmin(width=5000, depth=7)
        assert isinstance(cms, CountMinSketch)
        assert cms.config.width == 5000
        assert cms.config.depth == 7

    def test_create_countmin_by_error(self):
        """Test creating CountMinSketch with target error."""
        factory = SketchFactory()

        cms = factory.create_countmin(epsilon=0.001, delta=0.01)
        assert cms.config.expected_error <= 0.002

    def test_create_bloom(self):
        """Test creating BloomFilter via factory."""
        factory = SketchFactory()

        bf = factory.create_bloom(capacity=1_000_000, error_rate=0.001)
        assert isinstance(bf, BloomFilter)
        assert bf.config.capacity == 1_000_000
        assert bf.config.error_rate == 0.001

    def test_create_sketch_string_type(self):
        """Test creating sketch with string type."""
        hll = create_sketch("hyperloglog", precision=10)
        assert isinstance(hll, HyperLogLog)

        cms = create_sketch("countmin", width=1000, depth=3)
        assert isinstance(cms, CountMinSketch)

        bf = create_sketch("bloom", capacity=100000)
        assert isinstance(bf, BloomFilter)

    def test_create_sketch_enum_type(self):
        """Test creating sketch with enum type."""
        hll = create_sketch(SketchType.HYPERLOGLOG, precision=10)
        assert isinstance(hll, HyperLogLog)

    def test_for_cardinality(self):
        """Test factory for cardinality use case."""
        factory = SketchFactory()

        hll = factory.for_cardinality(error_rate=0.005)
        assert isinstance(hll, HyperLogLog)
        assert hll.standard_error() <= 0.01

    def test_for_frequency(self):
        """Test factory for frequency use case."""
        factory = SketchFactory()

        cms = factory.for_frequency(epsilon=0.0001, delta=0.001)
        assert isinstance(cms, CountMinSketch)

    def test_for_membership(self):
        """Test factory for membership use case."""
        factory = SketchFactory()

        bf = factory.for_membership(capacity=500_000, error_rate=0.001)
        assert isinstance(bf, BloomFilter)


# ============================================================================
# Integration Tests
# ============================================================================


class TestSketchIntegration:
    """Integration tests for sketch usage patterns."""

    def test_distributed_cardinality(self):
        """Test distributed cardinality estimation with merge."""
        # Simulate distributed processing
        partitions = [
            [f"user_{i}" for i in range(j * 1000, (j + 1) * 1000)] for j in range(10)
        ]

        # Process each partition independently
        hlls = []
        for partition in partitions:
            hll = HyperLogLog(HyperLogLogConfig(precision=14))
            hll.add_batch(partition)
            hlls.append(hll)

        # Merge all partitions
        merged = hlls[0]
        for hll in hlls[1:]:
            merged = merged.merge(hll)

        # Should estimate ~10,000 distinct users
        estimate = merged.estimate()
        assert 9000 < estimate < 11000

    def test_streaming_heavy_hitters(self):
        """Test finding heavy hitters in a stream."""
        cms = CountMinSketch(
            CountMinSketchConfig.for_error_and_confidence(epsilon=0.001, delta=0.01)
        )

        # Simulate a stream with skewed distribution
        # Zipf-like: item_0 appears most frequently, item_1 less, etc.
        import random

        random.seed(42)
        for _ in range(100000):
            # Pick item with probability proportional to 1/(rank+1)
            rank = random.choices(
                range(100), weights=[1 / (i + 1) for i in range(100)]
            )[0]
            cms.add(f"item_{rank}")

        # Get heavy hitters (>1% of stream)
        heavy = cms.get_heavy_hitters(threshold=0.01)

        # item_0 should be among heavy hitters
        heavy_items = [item for item, _ in heavy]
        assert "item_0" in heavy_items

    def test_bloom_filter_deduplication(self):
        """Test using Bloom filter for stream deduplication."""
        bf = BloomFilter(BloomFilterConfig(capacity=100000, error_rate=0.001))

        unique_count = 0
        duplicates_detected = 0

        # Simulate stream with some duplicates
        items = [f"item_{i % 1000}" for i in range(10000)]  # 10x duplicates

        for item in items:
            if item in bf:
                duplicates_detected += 1
            else:
                bf.add(item)
                unique_count += 1

        # Should detect ~9000 duplicates (10000 - 1000 unique)
        assert duplicates_detected >= 8500
        assert unique_count <= 1100  # ~1000 unique + some false positives


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_sketch_estimate(self):
        """Test estimates on empty sketches."""
        hll = HyperLogLog()
        assert hll.estimate() == 0

        cms = CountMinSketch()
        assert cms.estimate_frequency("anything") == 0

        bf = BloomFilter()
        assert not bf.contains("anything")

    def test_null_handling(self):
        """Test handling of None values."""
        hll = HyperLogLog()
        hll.add_batch([None, "a", None, "b", None])
        # Should only count non-null values
        assert hll.estimate() <= 4  # At most a, b + some error

    def test_large_values(self):
        """Test with very large string values."""
        hll = HyperLogLog()

        large_value = "x" * 10000
        hll.add(large_value)
        assert hll.estimate() == 1

    def test_numeric_values(self):
        """Test with numeric values."""
        hll = HyperLogLog()

        for i in range(1000):
            hll.add(i)
            hll.add(float(i))

        # Integers and floats hash differently
        estimate = hll.estimate()
        assert 1500 < estimate < 2500

    def test_invalid_precision(self):
        """Test that invalid precision raises error."""
        with pytest.raises(ValueError):
            HyperLogLogConfig(precision=3)  # Too low

        with pytest.raises(ValueError):
            HyperLogLogConfig(precision=19)  # Too high

    def test_invalid_bloom_config(self):
        """Test that invalid Bloom config raises error."""
        with pytest.raises(ValueError):
            BloomFilterConfig(capacity=0)

        with pytest.raises(ValueError):
            BloomFilterConfig(error_rate=1.5)

    def test_invalid_cms_config(self):
        """Test that invalid CMS config raises error."""
        with pytest.raises(ValueError):
            CountMinSketchConfig(width=0)

        with pytest.raises(ValueError):
            CountMinSketchConfig(depth=-1)

    def test_heavy_hitters_threshold_validation(self):
        """Test heavy hitters threshold validation."""
        cms = CountMinSketch()
        cms.add("item")

        with pytest.raises(ValueError):
            cms.get_heavy_hitters(threshold=0)

        with pytest.raises(ValueError):
            cms.get_heavy_hitters(threshold=1.5)
