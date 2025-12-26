"""Tests for memory-efficient validation modules.

This module tests the memory optimization abstractions:
- MemoryEfficientMixin
- ApproximateKNNMixin
- SGDOnlineMixin
- StreamingECDFMixin
"""

import pytest
import numpy as np
import polars as pl

# Check for optional dependencies
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

requires_sklearn = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="scikit-learn not installed"
)

from truthound.validators.memory.base import (
    MemoryEfficientMixin,
    MemoryConfig,
    MemoryStrategy,
    DataChunker,
    estimate_memory_usage,
    estimate_algorithm_memory,
    get_available_memory,
)
from truthound.validators.memory.approximate_knn import (
    ApproximateKNNMixin,
    KNNBackend,
    ApproximateNeighborResult,
)
from truthound.validators.memory.sgd_online import (
    SGDOnlineMixin,
    OnlineLearnerConfig,
    OnlineStatistics,
    OnlineScaler,
    SGDOneClassSVM,
    IncrementalMahalanobis,
)
from truthound.validators.memory.streaming_ecdf import (
    StreamingECDFMixin,
    StreamingECDF,
    StreamingStatistics,
    TDigest,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # Normal data with a few outliers
    data = np.random.randn(n_samples, n_features)
    # Add some outliers
    data[-10:] = np.random.randn(10, n_features) * 5 + 10

    return data


@pytest.fixture
def sample_lazyframe(sample_data):
    """Generate sample LazyFrame."""
    df = pl.DataFrame({
        f"col{i}": sample_data[:, i] for i in range(sample_data.shape[1])
    })
    return df.lazy()


@pytest.fixture
def large_lazyframe():
    """Generate larger LazyFrame for streaming tests."""
    np.random.seed(42)
    n_samples = 10000

    df = pl.DataFrame({
        "value": np.random.randn(n_samples) * 100 + 500,
        "category": np.random.choice(["A", "B", "C"], n_samples),
    })
    return df.lazy()


# ============================================================================
# MemoryConfig Tests
# ============================================================================

class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()

        assert config.max_memory_mb == 512.0
        assert config.batch_size == 50000
        assert config.chunk_size == 100000
        assert config.random_state == 42
        assert config.reserve_factor == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = MemoryConfig(
            max_memory_mb=1024,
            sample_size=10000,
            batch_size=5000,
        )

        assert config.max_memory_mb == 1024
        assert config.sample_size == 10000
        assert config.batch_size == 5000

    def test_invalid_config(self):
        """Test validation of invalid config values."""
        with pytest.raises(ValueError):
            MemoryConfig(max_memory_mb=-100)

        with pytest.raises(ValueError):
            MemoryConfig(batch_size=0)

        with pytest.raises(ValueError):
            MemoryConfig(reserve_factor=1.5)


# ============================================================================
# Memory Estimation Tests
# ============================================================================

class TestMemoryEstimation:
    """Tests for memory estimation functions."""

    def test_estimate_memory_usage(self):
        """Test basic memory estimation."""
        # 1000 rows × 10 cols × 8 bytes = 80,000 bytes = ~0.076 MB
        mb = estimate_memory_usage(1000, 10)
        assert 0.07 < mb < 0.08

    def test_estimate_memory_with_multiplier(self):
        """Test memory estimation with algorithm multiplier."""
        base_mb = estimate_memory_usage(1000, 10)
        doubled_mb = estimate_memory_usage(1000, 10, multiplier=2.0)

        assert abs(doubled_mb - base_mb * 2) < 0.001

    def test_estimate_algorithm_memory(self):
        """Test algorithm-specific memory estimation."""
        n_samples, n_features = 10000, 10

        # Different algorithms should have different multipliers
        lof_mb = estimate_algorithm_memory(n_samples, n_features, "lof")
        svm_mb = estimate_algorithm_memory(n_samples, n_features, "svm")
        iso_mb = estimate_algorithm_memory(n_samples, n_features, "isolation_forest")

        # SVM should be highest (O(n²) kernel matrix)
        assert svm_mb > lof_mb
        assert svm_mb > iso_mb

    def test_get_available_memory(self):
        """Test available memory detection."""
        available = get_available_memory()

        # Should return positive value
        assert available > 0

        # Should be reasonable (at least 100 MB, less than 1 TB)
        assert 100 < available < 1024 * 1024


# ============================================================================
# DataChunker Tests
# ============================================================================

class TestDataChunker:
    """Tests for DataChunker utility."""

    def test_chunker_iteration(self, large_lazyframe):
        """Test basic chunk iteration."""
        chunker = DataChunker(chunk_size=1000)

        chunks = list(chunker.iterate(large_lazyframe))

        assert len(chunks) == 10  # 10000 / 1000
        assert all(isinstance(c, pl.DataFrame) for c in chunks)

    def test_chunker_with_columns(self, large_lazyframe):
        """Test chunking with specific columns."""
        chunker = DataChunker(chunk_size=1000, columns=["value"])

        chunks = list(chunker.iterate(large_lazyframe))

        # Should only have one column
        for chunk in chunks:
            assert chunk.columns == ["value"]

    def test_chunker_as_numpy(self, large_lazyframe):
        """Test chunking with numpy output."""
        chunker = DataChunker(chunk_size=1000, columns=["value"])

        chunks = list(chunker.iterate(large_lazyframe, as_numpy=True))

        assert all(isinstance(c, np.ndarray) for c in chunks)

    def test_chunker_with_index(self, large_lazyframe):
        """Test chunking with index tracking."""
        chunker = DataChunker(chunk_size=1000)

        indices = list(chunker.iterate_with_index(large_lazyframe))

        # Check start indices
        starts = [idx[0] for idx in indices]
        assert starts[0] == 0
        assert starts[1] == 1000

    def test_get_total_rows(self, large_lazyframe):
        """Test row counting."""
        chunker = DataChunker()

        total = chunker.get_total_rows(large_lazyframe)
        assert total == 10000


# ============================================================================
# ApproximateKNNMixin Tests
# ============================================================================

@requires_sklearn
class TestApproximateKNNMixin:
    """Tests for approximate k-NN functionality."""

    def test_build_balltree_index(self, sample_data):
        """Test BallTree index building."""
        mixin = ApproximateKNNMixin()

        mixin.build_approximate_index(sample_data, backend=KNNBackend.BALLTREE)

        assert mixin._knn_index is not None
        assert mixin._knn_backend == KNNBackend.BALLTREE

    def test_build_kdtree_index(self, sample_data):
        """Test KDTree index building."""
        mixin = ApproximateKNNMixin()

        mixin.build_approximate_index(sample_data, backend=KNNBackend.KDTREE)

        assert mixin._knn_index is not None
        assert mixin._knn_backend == KNNBackend.KDTREE

    def test_find_neighbors(self, sample_data):
        """Test neighbor finding."""
        mixin = ApproximateKNNMixin()
        mixin.build_approximate_index(sample_data, backend=KNNBackend.BALLTREE)

        query = sample_data[:5]
        result = mixin.find_approximate_neighbors(query, k=10)

        assert isinstance(result, ApproximateNeighborResult)
        assert result.indices.shape == (5, 10)
        assert result.distances.shape == (5, 10)

    def test_single_query(self, sample_data):
        """Test single point query."""
        mixin = ApproximateKNNMixin()
        mixin.build_approximate_index(sample_data, backend=KNNBackend.BALLTREE)

        query = sample_data[0]  # Single point
        result = mixin.find_approximate_neighbors(query, k=5)

        assert result.indices.shape == (1, 5)

    def test_compute_lof_scores(self, sample_data):
        """Test LOF score computation."""
        mixin = ApproximateKNNMixin()
        mixin.build_approximate_index(sample_data)

        lof_scores = mixin.compute_local_outlier_factor(sample_data, k=20)

        assert len(lof_scores) == len(sample_data)
        assert np.all(lof_scores >= 0)

        # Outliers (last 10 points) should have higher LOF scores
        normal_mean = lof_scores[:-10].mean()
        outlier_mean = lof_scores[-10:].mean()
        assert outlier_mean > normal_mean

    def test_clear_index(self, sample_data):
        """Test index clearing."""
        mixin = ApproximateKNNMixin()
        mixin.build_approximate_index(sample_data)

        mixin.clear_index()

        assert mixin._knn_index is None
        assert mixin._knn_data is None

    def test_get_best_backend(self, sample_data):
        """Test automatic backend selection."""
        mixin = ApproximateKNNMixin()

        # Small dataset should prefer exact for small datasets
        backend = mixin.get_best_backend(1000, 10, prefer_exact=True)
        assert backend == KNNBackend.EXACT

        # Large dataset should use approximate
        backend = mixin.get_best_backend(100000, 10, prefer_exact=False)
        # Should be one of the tree-based methods
        assert backend in (KNNBackend.BALLTREE, KNNBackend.KDTREE, KNNBackend.ANNOY, KNNBackend.HNSW, KNNBackend.FAISS)


# ============================================================================
# SGDOnlineMixin Tests
# ============================================================================

class TestOnlineStatistics:
    """Tests for online statistics computation."""

    def test_single_updates(self):
        """Test updating with single values."""
        stats = OnlineStatistics(n_features=3)

        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])

        for row in data:
            stats.update(row.reshape(1, -1))

        expected_mean = data.mean(axis=0)
        np.testing.assert_array_almost_equal(stats.mean, expected_mean)

    def test_batch_update(self):
        """Test batch updating."""
        stats = OnlineStatistics(n_features=3)

        data = np.random.randn(100, 3)
        stats.update_batch(data)

        np.testing.assert_array_almost_equal(stats.mean, data.mean(axis=0), decimal=5)
        np.testing.assert_array_almost_equal(stats.variance, data.var(axis=0, ddof=1), decimal=5)

    def test_incremental_batch_update(self):
        """Test that incremental updates match single batch."""
        np.random.seed(42)
        data = np.random.randn(1000, 5)

        # Single batch
        stats1 = OnlineStatistics(n_features=5)
        stats1.update_batch(data)

        # Multiple batches
        stats2 = OnlineStatistics(n_features=5)
        for i in range(0, 1000, 100):
            stats2.update_batch(data[i:i+100])

        np.testing.assert_array_almost_equal(stats1.mean, stats2.mean, decimal=10)
        np.testing.assert_array_almost_equal(stats1.variance, stats2.variance, decimal=5)


class TestOnlineScaler:
    """Tests for online scaler."""

    def test_partial_fit_transform(self):
        """Test incremental fitting and transformation."""
        scaler = OnlineScaler()

        data = np.random.randn(100, 3) * 10 + 5

        scaler.partial_fit(data)
        transformed = scaler.transform(data)

        # Transformed data should have mean ~0 and std ~1
        assert np.abs(transformed.mean(axis=0)).max() < 0.1
        assert np.abs(transformed.std(axis=0) - 1).max() < 0.1

    def test_incremental_fit(self):
        """Test incremental fitting matches batch fitting."""
        np.random.seed(42)
        data = np.random.randn(1000, 5) * 10 + 5

        # Batch fit
        scaler1 = OnlineScaler()
        scaler1.partial_fit(data)

        # Incremental fit
        scaler2 = OnlineScaler()
        for i in range(0, 1000, 100):
            scaler2.partial_fit(data[i:i+100])

        np.testing.assert_array_almost_equal(scaler1.mean_, scaler2.mean_, decimal=10)


@requires_sklearn
class TestSGDOneClassSVM:
    """Tests for SGD-based One-Class SVM."""

    def test_partial_fit(self, sample_data):
        """Test incremental fitting."""
        model = SGDOneClassSVM(nu=0.1, n_components=50)

        model.partial_fit(sample_data)

        assert model._is_fitted

    def test_predict(self, sample_data):
        """Test prediction after fitting."""
        model = SGDOneClassSVM(nu=0.1, n_components=50)

        model.partial_fit(sample_data)
        predictions = model.predict(sample_data)

        assert len(predictions) == len(sample_data)
        assert set(predictions).issubset({-1, 1})

    def test_decision_function(self, sample_data):
        """Test decision function."""
        model = SGDOneClassSVM(nu=0.1, n_components=50)

        model.partial_fit(sample_data)
        scores = model.decision_function(sample_data)

        assert len(scores) == len(sample_data)

    def test_outlier_detection(self, sample_data):
        """Test that outliers are detected."""
        model = SGDOneClassSVM(nu=0.1, n_components=50)

        model.partial_fit(sample_data)
        predictions = model.predict(sample_data)

        # Should detect some outliers (last 10 points)
        outlier_predictions = predictions[-10:]
        assert (outlier_predictions == -1).sum() > 0  # At least some outliers detected


class TestIncrementalMahalanobis:
    """Tests for incremental Mahalanobis distance."""

    def test_partial_fit(self):
        """Test incremental fitting."""
        detector = IncrementalMahalanobis()

        data = np.random.randn(100, 3)
        detector.partial_fit(data)

        assert detector._n_samples == 100

    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance computation."""
        np.random.seed(42)
        detector = IncrementalMahalanobis()

        # Fit on normal data
        normal_data = np.random.randn(1000, 3)
        detector.partial_fit(normal_data)

        # Test on normal and outlier points
        test_normal = np.random.randn(10, 3)
        test_outlier = np.random.randn(10, 3) * 5 + 10

        dist_normal = detector.mahalanobis(test_normal)
        dist_outlier = detector.mahalanobis(test_outlier)

        # Outliers should have higher distances
        assert dist_outlier.mean() > dist_normal.mean()

    def test_predict(self):
        """Test outlier prediction."""
        np.random.seed(42)
        detector = IncrementalMahalanobis()

        normal_data = np.random.randn(1000, 3)
        detector.partial_fit(normal_data)

        outlier = np.array([[10.0, 10.0, 10.0]])
        prediction = detector.predict(outlier, threshold=3.0)

        assert prediction[0] == -1


# ============================================================================
# StreamingECDF Tests
# ============================================================================

class TestTDigest:
    """Tests for T-Digest data structure."""

    def test_update_and_quantile(self):
        """Test basic update and quantile computation."""
        digest = TDigest(compression=100)

        data = np.random.randn(10000)
        digest.update(data)

        # Test median
        estimated_median = digest.quantile(0.5)
        actual_median = np.median(data)

        assert abs(estimated_median - actual_median) < 0.1

    def test_cdf(self):
        """Test CDF computation."""
        digest = TDigest(compression=100)

        data = np.random.randn(10000)
        digest.update(data)

        # CDF at 0 should be ~0.5 for standard normal
        cdf_at_zero = digest.cdf(0.0)
        assert 0.45 < cdf_at_zero < 0.55

    def test_extreme_quantiles(self):
        """Test quantiles at distribution tails."""
        digest = TDigest(compression=200)

        data = np.random.randn(10000)
        digest.update(data)

        # Test tail quantiles
        p01 = digest.quantile(0.01)
        p99 = digest.quantile(0.99)

        # For standard normal, these should be around -2.33 and 2.33
        assert -3 < p01 < -1.5
        assert 1.5 < p99 < 3

    def test_merge(self):
        """Test merging two T-Digests."""
        np.random.seed(42)

        digest1 = TDigest(compression=100)
        digest2 = TDigest(compression=100)

        data1 = np.random.randn(5000)
        data2 = np.random.randn(5000)

        digest1.update(data1)
        digest2.update(data2)

        merged = digest1.merge(digest2)

        # Merged count should be sum
        assert merged.count == 10000

        # Median should be close to combined data median
        combined_median = np.median(np.concatenate([data1, data2]))
        merged_median = merged.quantile(0.5)
        assert abs(merged_median - combined_median) < 0.1


class TestStreamingECDF:
    """Tests for StreamingECDF."""

    def test_streaming_ecdf(self):
        """Test streaming ECDF computation."""
        ecdf = StreamingECDF(compression=200)

        data = np.random.randn(10000)
        ecdf.update(data)

        assert ecdf.count == 10000
        assert ecdf.min < ecdf.max

    def test_cdf_values(self):
        """Test CDF value computation."""
        ecdf = StreamingECDF(compression=200)

        data = np.random.randn(10000)
        ecdf.update(data)

        # Test multiple points
        points = np.array([-2, -1, 0, 1, 2])
        cdf_vals = ecdf.cdf(points)

        # CDF should be monotonically increasing
        assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1))


class TestStreamingStatistics:
    """Tests for StreamingStatistics."""

    def test_streaming_mean_variance(self):
        """Test streaming mean and variance computation."""
        stats = StreamingStatistics()

        data = np.random.randn(10000) * 5 + 10
        stats.update_batch(data)

        assert abs(stats.mean - data.mean()) < 0.1
        assert abs(stats.variance - data.var(ddof=1)) < 0.5

    def test_quantile_from_digest(self):
        """Test quantile computation via T-Digest."""
        stats = StreamingStatistics()

        data = np.random.randn(10000)
        stats.update_batch(data)

        median = stats.quantile(0.5)
        assert abs(median - np.median(data)) < 0.1


# ============================================================================
# Integration Tests
# ============================================================================

class TestMemoryEfficientValidators:
    """Integration tests for memory-efficient validators."""

    @requires_sklearn
    def test_memory_efficient_lof_validator(self, sample_lazyframe):
        """Test MemoryEfficientLOFValidator."""
        from truthound.validators.anomaly.ml_based import MemoryEfficientLOFValidator

        validator = MemoryEfficientLOFValidator(
            columns=["col0", "col1", "col2"],
            n_neighbors=20,
            contamination=0.1,
        )

        issues = validator.validate(sample_lazyframe)

        # Should complete without error
        assert isinstance(issues, list)

    @requires_sklearn
    def test_online_svm_validator(self, sample_lazyframe):
        """Test OnlineSVMValidator."""
        from truthound.validators.anomaly.ml_based import OnlineSVMValidator

        validator = OnlineSVMValidator(
            columns=["col0", "col1", "col2"],
            nu=0.1,
            n_components=50,
        )

        issues = validator.validate(sample_lazyframe)

        assert isinstance(issues, list)

    def test_streaming_ks_validator(self, large_lazyframe):
        """Test StreamingKSTestValidator."""
        from truthound.validators.drift.statistical import StreamingKSTestValidator

        # Create reference data
        ref_df = pl.DataFrame({
            "value": np.random.randn(5000) * 100 + 500,
        })

        validator = StreamingKSTestValidator(
            column="value",
            reference_data=ref_df.lazy(),
            compression=100,
        )

        issues = validator.validate(large_lazyframe)

        assert isinstance(issues, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
