"""Tests for reference data caching module.

Tests cover:
1. NumericStatistics creation and serialization
2. CategoricalStatistics creation and serialization
3. MultiColumnStatistics creation
4. LRU cache functionality
5. Cache eviction and TTL
6. Global cache management
7. Integration with drift validators
"""

import time
import pytest
import polars as pl
import numpy as np

from truthound.validators.cache import (
    CacheConfig,
    NumericStatistics,
    CategoricalStatistics,
    MultiColumnStatistics,
    ReferenceCache,
    get_global_cache,
    clear_global_cache,
    reset_global_cache,
    make_cache_key,
    hash_dataframe,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def numeric_series():
    """Create a numeric series for testing."""
    np.random.seed(42)
    return pl.Series("value", np.random.normal(100, 15, 10000))


@pytest.fixture
def categorical_series():
    """Create a categorical series for testing."""
    categories = ["A", "B", "C", "D", "E"]
    np.random.seed(42)
    values = np.random.choice(categories, size=5000, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    return pl.Series("category", values)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pl.DataFrame({
        "col1": np.random.normal(50, 10, 1000),
        "col2": np.random.normal(100, 20, 1000),
        "col3": np.random.exponential(5, 1000),
    })


@pytest.fixture
def large_dataframe():
    """Create a large DataFrame for memory testing."""
    np.random.seed(42)
    return pl.DataFrame({
        f"col_{i}": np.random.normal(i * 10, i + 1, 100000)
        for i in range(10)
    })


# ============================================================================
# NumericStatistics Tests
# ============================================================================

class TestNumericStatistics:
    """Tests for NumericStatistics class."""

    def test_from_series_basic(self, numeric_series):
        """Test basic statistics creation from series."""
        stats = NumericStatistics.from_series(numeric_series)

        assert stats.count == len(numeric_series)
        assert stats.null_count == 0
        assert 95 < stats.mean < 105  # ~100
        assert 13 < stats.std < 17  # ~15
        assert len(stats.quantiles) > 0
        assert len(stats.histogram_edges) > 0
        assert len(stats.histogram_counts) > 0

    def test_from_series_with_nulls(self):
        """Test statistics with null values."""
        series = pl.Series("val", [1.0, 2.0, None, 4.0, None, 6.0])
        stats = NumericStatistics.from_series(series)

        assert stats.count == 4  # Non-null count
        assert stats.null_count == 2

    def test_from_series_empty(self):
        """Test with empty series."""
        series = pl.Series("val", [], dtype=pl.Float64)
        stats = NumericStatistics.from_series(series)

        assert stats.count == 0
        assert stats.mean == 0.0
        assert len(stats.histogram_edges) == 0

    def test_quantiles_accuracy(self, numeric_series):
        """Test quantile calculations."""
        stats = NumericStatistics.from_series(
            numeric_series,
            quantiles=(0.25, 0.5, 0.75),
        )

        # Compare with numpy
        arr = numeric_series.to_numpy()
        np_q25, np_q50, np_q75 = np.percentile(arr, [25, 50, 75])

        assert abs(stats.quantiles[0.25] - np_q25) < 0.01
        assert abs(stats.quantiles[0.5] - np_q50) < 0.01
        assert abs(stats.quantiles[0.75] - np_q75) < 0.01

    def test_histogram_sums_to_one(self, numeric_series):
        """Test histogram frequencies sum to 1."""
        stats = NumericStatistics.from_series(numeric_series, n_bins=20)

        total = sum(stats.histogram_counts)
        assert abs(total - 1.0) < 0.001

    def test_from_lazyframe(self, sample_dataframe):
        """Test creation from LazyFrame."""
        lf = sample_dataframe.lazy()
        stats = NumericStatistics.from_lazyframe(lf, "col1")

        assert stats.count == 1000
        assert 45 < stats.mean < 55  # ~50

    def test_memory_estimation(self, numeric_series):
        """Test memory usage estimation."""
        stats = NumericStatistics.from_series(numeric_series)
        memory = stats.estimate_memory_bytes()

        # Should be small (< 10KB for typical stats)
        assert memory > 0
        assert memory < 10 * 1024


# ============================================================================
# CategoricalStatistics Tests
# ============================================================================

class TestCategoricalStatistics:
    """Tests for CategoricalStatistics class."""

    def test_from_series_basic(self, categorical_series):
        """Test basic categorical statistics."""
        stats = CategoricalStatistics.from_series(categorical_series)

        assert stats.count == 5000
        assert stats.unique_count == 5
        assert len(stats.frequencies) == 5
        assert abs(sum(stats.frequencies.values()) - 1.0) < 0.001

    def test_frequency_order(self, categorical_series):
        """Test top categories are ordered by frequency."""
        stats = CategoricalStatistics.from_series(categorical_series)

        # Should be ordered by frequency descending
        for i in range(len(stats.top_categories) - 1):
            assert stats.top_categories[i][1] >= stats.top_categories[i + 1][1]

    def test_from_series_with_nulls(self):
        """Test with null values."""
        series = pl.Series("cat", ["A", "B", None, "A", None, "C"])
        stats = CategoricalStatistics.from_series(series)

        assert stats.count == 4
        assert stats.null_count == 2
        assert stats.unique_count == 3

    def test_memory_estimation(self, categorical_series):
        """Test memory usage estimation."""
        stats = CategoricalStatistics.from_series(categorical_series)
        memory = stats.estimate_memory_bytes()

        assert memory > 0
        assert memory < 10 * 1024


# ============================================================================
# MultiColumnStatistics Tests
# ============================================================================

class TestMultiColumnStatistics:
    """Tests for MultiColumnStatistics class."""

    def test_from_lazyframe(self, sample_dataframe):
        """Test creation from LazyFrame."""
        lf = sample_dataframe.lazy()
        stats = MultiColumnStatistics.from_lazyframe(lf, ["col1", "col2", "col3"])

        assert len(stats.column_stats) == 3
        assert len(stats.correlation_matrix) > 0
        assert len(stats.medians) == 3
        assert len(stats.iqrs) == 3

    def test_correlation_matrix(self, sample_dataframe):
        """Test correlation matrix."""
        lf = sample_dataframe.lazy()
        stats = MultiColumnStatistics.from_lazyframe(lf, ["col1", "col2"])

        # Self-correlation should be 1
        assert abs(stats.correlation_matrix[("col1", "col1")] - 1.0) < 0.01
        assert abs(stats.correlation_matrix[("col2", "col2")] - 1.0) < 0.01

    def test_normalization_params(self, sample_dataframe):
        """Test median and IQR for normalization."""
        lf = sample_dataframe.lazy()
        stats = MultiColumnStatistics.from_lazyframe(lf, ["col1", "col2"])

        # Medians should be close to means for normal data
        assert 45 < stats.medians["col1"] < 55
        assert 90 < stats.medians["col2"] < 110

        # IQRs should be approximately 1.35 * std for normal data
        assert stats.iqrs["col1"] > 0
        assert stats.iqrs["col2"] > 0


# ============================================================================
# ReferenceCache Tests
# ============================================================================

class TestReferenceCache:
    """Tests for ReferenceCache class."""

    def test_put_and_get(self, numeric_series):
        """Test basic put and get operations."""
        cache = ReferenceCache(CacheConfig(max_entries=10))
        stats = NumericStatistics.from_series(numeric_series)

        cache.put("test_key", stats)
        retrieved = cache.get("test_key")

        assert retrieved is not None
        assert retrieved.mean == stats.mean
        assert retrieved.std == stats.std

    def test_get_missing_key(self):
        """Test getting non-existent key."""
        cache = ReferenceCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_lru_eviction(self):
        """Test LRU eviction when max entries exceeded."""
        cache = ReferenceCache(CacheConfig(max_entries=3))

        # Add 4 entries
        for i in range(4):
            stats = NumericStatistics(
                count=100, null_count=0, mean=float(i),
                std=1.0, variance=1.0, min_value=0.0,
                max_value=100.0, sum_value=1000.0,
                quantiles={}, histogram_edges=[], histogram_counts=[],
            )
            cache.put(f"key_{i}", stats)

        # First entry should be evicted
        assert cache.get("key_0") is None
        assert cache.get("key_1") is not None
        assert cache.get("key_3") is not None

    def test_memory_limit_eviction(self):
        """Test eviction based on memory limit."""
        # Very low memory limit
        cache = ReferenceCache(CacheConfig(max_entries=1000, max_memory_mb=0.001))

        stats = NumericStatistics(
            count=100, null_count=0, mean=50.0,
            std=10.0, variance=100.0, min_value=0.0,
            max_value=100.0, sum_value=5000.0,
            quantiles={0.5: 50.0}, histogram_edges=[0.0, 100.0],
            histogram_counts=[1.0],
        )

        # Adding multiple entries should trigger eviction
        for i in range(5):
            cache.put(f"key_{i}", stats)

        # Due to memory limit, most entries should be evicted
        assert len(cache) <= 2

    def test_ttl_expiration(self):
        """Test time-to-live expiration."""
        cache = ReferenceCache(CacheConfig(ttl_seconds=0.1))

        stats = NumericStatistics(
            count=100, null_count=0, mean=50.0,
            std=10.0, variance=100.0, min_value=0.0,
            max_value=100.0, sum_value=5000.0,
            quantiles={}, histogram_edges=[], histogram_counts=[],
        )

        cache.put("expiring_key", stats)
        assert cache.get("expiring_key") is not None

        # Wait for expiration
        time.sleep(0.15)
        assert cache.get("expiring_key") is None

    def test_remove(self):
        """Test removing entries."""
        cache = ReferenceCache()
        stats = NumericStatistics(
            count=100, null_count=0, mean=50.0,
            std=10.0, variance=100.0, min_value=0.0,
            max_value=100.0, sum_value=5000.0,
            quantiles={}, histogram_edges=[], histogram_counts=[],
        )

        cache.put("to_remove", stats)
        assert cache.remove("to_remove") is True
        assert cache.remove("to_remove") is False
        assert cache.get("to_remove") is None

    def test_clear(self):
        """Test clearing cache."""
        cache = ReferenceCache()
        stats = NumericStatistics(
            count=100, null_count=0, mean=50.0,
            std=10.0, variance=100.0, min_value=0.0,
            max_value=100.0, sum_value=5000.0,
            quantiles={}, histogram_edges=[], histogram_counts=[],
        )

        for i in range(5):
            cache.put(f"key_{i}", stats)

        cache.clear()
        assert len(cache) == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ReferenceCache()
        stats = NumericStatistics(
            count=100, null_count=0, mean=50.0,
            std=10.0, variance=100.0, min_value=0.0,
            max_value=100.0, sum_value=5000.0,
            quantiles={}, histogram_edges=[], histogram_counts=[],
        )

        cache.put("key1", stats)
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        cache_stats = cache.get_stats()
        assert cache_stats["entries"] == 1
        assert cache_stats["hits"] == 2
        assert cache_stats["misses"] == 1
        assert cache_stats["hit_rate"] == 2 / 3

    def test_contains(self):
        """Test __contains__ method."""
        cache = ReferenceCache()
        stats = NumericStatistics(
            count=100, null_count=0, mean=50.0,
            std=10.0, variance=100.0, min_value=0.0,
            max_value=100.0, sum_value=5000.0,
            quantiles={}, histogram_edges=[], histogram_counts=[],
        )

        cache.put("exists", stats)
        assert "exists" in cache
        assert "not_exists" not in cache


# ============================================================================
# Global Cache Tests
# ============================================================================

class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_global_cache(self):
        """Test getting global cache instance."""
        clear_global_cache()
        cache1 = get_global_cache()
        cache2 = get_global_cache()
        assert cache1 is cache2

    def test_clear_global_cache(self):
        """Test clearing global cache."""
        cache = get_global_cache()
        stats = NumericStatistics(
            count=100, null_count=0, mean=50.0,
            std=10.0, variance=100.0, min_value=0.0,
            max_value=100.0, sum_value=5000.0,
            quantiles={}, histogram_edges=[], histogram_counts=[],
        )

        cache.put("global_key", stats)
        clear_global_cache()
        assert cache.get("global_key") is None

    def test_reset_global_cache(self):
        """Test resetting global cache with new config."""
        cache1 = get_global_cache()
        cache2 = reset_global_cache(CacheConfig(max_entries=50))

        assert cache1 is not cache2
        assert cache2.config.max_entries == 50


# ============================================================================
# Cache Key Utilities Tests
# ============================================================================

class TestCacheKeyUtilities:
    """Tests for cache key utilities."""

    def test_make_cache_key_basic(self):
        """Test basic cache key generation."""
        key = make_cache_key("psi", "price")
        assert "psi" in key
        assert "price" in key

    def test_make_cache_key_with_list(self):
        """Test cache key with list of columns."""
        key = make_cache_key("isolation_forest", ["col1", "col2", "col3"])
        assert "col1" in key
        assert "col2" in key
        assert "col3" in key

    def test_make_cache_key_with_version(self):
        """Test cache key with version."""
        key_v1 = make_cache_key("psi", "price", version="v1")
        key_v2 = make_cache_key("psi", "price", version="v2")
        assert key_v1 != key_v2

    def test_hash_dataframe(self, sample_dataframe):
        """Test dataframe hashing."""
        lf = sample_dataframe.lazy()
        hash1 = hash_dataframe(lf)
        hash2 = hash_dataframe(lf)

        assert hash1 == hash2
        assert len(hash1) == 16  # MD5 prefix


# ============================================================================
# Integration Tests
# ============================================================================

class TestCacheIntegration:
    """Integration tests for cache with validators."""

    def test_drift_validator_with_cache(self, sample_dataframe):
        """Test drift validator using cache."""
        from truthound.validators.drift.psi import PSIValidator

        lf = sample_dataframe.lazy()

        # Create validator with caching enabled
        validator = PSIValidator(
            column="col1",
            reference_data=lf,
            cache_reference=True,
        )

        # Cache and release
        validator.cache_and_release()

        # Verify statistics are cached
        assert validator.is_statistics_cached()

        # Validate should still work with cached stats
        current_data = pl.DataFrame({
            "col1": np.random.normal(55, 10, 500),  # Slight drift
        }).lazy()

        issues = validator.validate(current_data)
        # Should be able to calculate PSI using cached histogram

    def test_memory_savings(self, large_dataframe):
        """Test that caching reduces memory footprint."""
        lf = large_dataframe.lazy()

        # Compute statistics
        stats = NumericStatistics.from_lazyframe(lf, "col_0")

        # Statistics object should be much smaller than raw data
        stats_size = stats.estimate_memory_bytes()
        raw_data_size = large_dataframe.estimated_size()

        # Stats should be < 1% of raw data size
        assert stats_size < raw_data_size * 0.01
