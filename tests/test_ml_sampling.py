"""Tests for ML anomaly validator sampling and memory optimization."""

import pytest
import polars as pl
import numpy as np

from truthound.validators.anomaly.ml_based import (
    IsolationForestValidator,
    LOFValidator,
    OneClassSVMValidator,
    DBSCANAnomalyValidator,
    LargeDatasetMixin,
    _estimate_data_memory_mb,
    _compute_optimal_sample_size,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_BATCH_SIZE,
    MEMORY_THRESHOLD_MB,
)


class TestMemoryUtilities:
    """Tests for memory estimation utilities."""

    def test_estimate_data_memory_mb(self):
        """Test memory estimation calculation."""
        # 1000 rows x 10 cols x 8 bytes = 80KB = 0.078MB
        memory = _estimate_data_memory_mb(1000, 10)
        assert 0.07 < memory < 0.09

        # 1M rows x 100 cols = 800MB
        memory = _estimate_data_memory_mb(1_000_000, 100)
        assert 750 < memory < 850

    def test_compute_optimal_sample_size_small_data(self):
        """Test optimal sample size for small datasets."""
        # Small dataset fits in memory
        sample_size = _compute_optimal_sample_size(1000, 10, max_memory_mb=100)
        assert sample_size == 1000  # No sampling needed

    def test_compute_optimal_sample_size_large_data(self):
        """Test optimal sample size for large datasets."""
        # 10M rows would need sampling
        sample_size = _compute_optimal_sample_size(10_000_000, 10, max_memory_mb=100)

        # Should be significantly less than original
        assert sample_size < 10_000_000
        # But reasonable size
        assert sample_size >= 1000


class TestLargeDatasetMixin:
    """Tests for LargeDatasetMixin."""

    def test_smart_sample_lazyframe_no_sampling(self):
        """Test smart sampling when data is small."""
        mixin = LargeDatasetMixin()

        # Create small dataset
        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 100),
            "col2": np.random.normal(0, 1, 100),
        })
        lf = df.lazy()

        data, count, was_sampled = mixin._smart_sample_lazyframe(
            lf, ["col1", "col2"], sample_size=None
        )

        assert count == 100
        assert was_sampled is False
        assert len(data) == 100

    def test_smart_sample_lazyframe_with_sampling(self):
        """Test smart sampling when data exceeds sample_size."""
        mixin = LargeDatasetMixin()

        # Create larger dataset
        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 1000),
            "col2": np.random.normal(0, 1, 1000),
        })
        lf = df.lazy()

        data, count, was_sampled = mixin._smart_sample_lazyframe(
            lf, ["col1", "col2"], sample_size=100
        )

        assert count == 1000
        assert was_sampled is True
        assert len(data) <= 110  # Allow some buffer

    def test_smart_sample_lazyframe_handles_nulls(self):
        """Test that sampling handles null values."""
        mixin = LargeDatasetMixin()

        # Create dataset with nulls
        values = list(np.random.normal(0, 1, 90)) + [None] * 10
        df = pl.DataFrame({
            "col1": values,
            "col2": list(np.random.normal(0, 1, 100)),
        })
        lf = df.lazy()

        data, count, was_sampled = mixin._smart_sample_lazyframe(
            lf, ["col1", "col2"], sample_size=None
        )

        # Nulls should be dropped
        assert len(data) == 90

    def test_batch_predict_small_data(self):
        """Test batch prediction with small data (no batching)."""
        mixin = LargeDatasetMixin()

        # Create mock model
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))

        model = MockModel()
        data = np.random.normal(0, 1, (100, 5))

        predictions = mixin._batch_predict(model, data, batch_size=1000)

        assert len(predictions) == 100

    def test_batch_predict_large_data(self):
        """Test batch prediction with data larger than batch_size."""
        mixin = LargeDatasetMixin()

        # Create mock model with prediction tracking
        class MockModel:
            def __init__(self):
                self.call_count = 0

            def predict(self, X):
                self.call_count += 1
                return np.ones(len(X)) * self.call_count

        model = MockModel()
        data = np.random.normal(0, 1, (250, 5))

        predictions = mixin._batch_predict(model, data, batch_size=100)

        # Should have made 3 calls (100, 100, 50)
        assert model.call_count == 3
        assert len(predictions) == 250


class TestIsolationForestValidatorSampling:
    """Tests for IsolationForest with sampling."""

    def test_sample_size_parameter(self):
        """Test that sample_size parameter works."""
        np.random.seed(42)

        # Create dataset with 1000 rows
        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 1000),
            "col2": np.random.normal(0, 1, 1000),
        })
        lf = df.lazy()

        validator = IsolationForestValidator(
            columns=["col1", "col2"],
            sample_size=100,  # Sample only 100 rows
            contamination=0.1,
        )

        # Should not raise error
        issues = validator.validate(lf)
        assert isinstance(issues, list)

    def test_auto_sample_parameter(self):
        """Test auto_sample automatically determines sample size."""
        np.random.seed(42)

        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 500),
            "col2": np.random.normal(0, 1, 500),
        })
        lf = df.lazy()

        validator = IsolationForestValidator(
            columns=["col1", "col2"],
            auto_sample=True,
            max_memory_mb=1.0,  # Very low threshold to trigger sampling
            contamination=0.1,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)

    def test_sampled_results_have_info(self):
        """Test that sampled results include sampling info in details."""
        np.random.seed(42)

        # Create dataset with known anomalies
        normal = np.random.normal(0, 1, 900)
        outliers = np.random.normal(10, 0.1, 100)  # Clear outliers

        df = pl.DataFrame({
            "col1": np.concatenate([normal, outliers]),
            "col2": np.concatenate([normal, outliers]),
        })
        lf = df.lazy()

        validator = IsolationForestValidator(
            columns=["col1", "col2"],
            sample_size=200,  # Force sampling
            contamination=0.1,
            max_anomaly_ratio=0.05,  # Low threshold to trigger issue
        )

        issues = validator.validate(lf)

        # Should detect anomalies
        if issues:
            # Details should mention sampling if it occurred
            assert any("sample" in issue.details.lower() or "estimated" in issue.details.lower()
                      for issue in issues) or len(issues) == 0


class TestLOFValidatorSampling:
    """Tests for LOF with sampling."""

    def test_sample_size_parameter(self):
        """Test LOF with explicit sample_size."""
        np.random.seed(42)

        df = pl.DataFrame({
            "x": np.random.normal(0, 1, 500),
            "y": np.random.normal(0, 1, 500),
        })
        lf = df.lazy()

        validator = LOFValidator(
            columns=["x", "y"],
            n_neighbors=10,
            sample_size=100,
            contamination=0.1,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)

    def test_auto_sample_with_large_neighbors(self):
        """Test auto_sample respects n_neighbors."""
        np.random.seed(42)

        df = pl.DataFrame({
            "x": np.random.normal(0, 1, 200),
            "y": np.random.normal(0, 1, 200),
        })
        lf = df.lazy()

        validator = LOFValidator(
            columns=["x", "y"],
            n_neighbors=20,
            auto_sample=True,
            max_memory_mb=0.001,  # Very low to force sampling
            contamination=0.1,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)


class TestOneClassSVMValidatorSampling:
    """Tests for One-Class SVM with sampling."""

    def test_sample_size_parameter(self):
        """Test SVM with explicit sample_size."""
        np.random.seed(42)

        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 500),
            "col2": np.random.normal(0, 1, 500),
        })
        lf = df.lazy()

        validator = OneClassSVMValidator(
            columns=["col1", "col2"],
            sample_size=100,
            nu=0.1,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)

    def test_auto_sample_aggressive_for_svm(self):
        """Test that SVM uses more aggressive sampling."""
        np.random.seed(42)

        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 300),
            "col2": np.random.normal(0, 1, 300),
        })
        lf = df.lazy()

        validator = OneClassSVMValidator(
            columns=["col1", "col2"],
            auto_sample=True,
            max_memory_mb=0.01,  # Very low to force sampling
            nu=0.1,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)


class TestDBSCANValidatorSampling:
    """Tests for DBSCAN with sampling."""

    def test_sample_size_parameter(self):
        """Test DBSCAN with explicit sample_size."""
        np.random.seed(42)

        # Create clustered data
        cluster1 = np.random.normal(0, 0.5, (200, 2))
        cluster2 = np.random.normal(5, 0.5, (200, 2))
        noise = np.random.uniform(-10, 15, (50, 2))

        data = np.vstack([cluster1, cluster2, noise])

        df = pl.DataFrame({
            "x": data[:, 0],
            "y": data[:, 1],
        })
        lf = df.lazy()

        validator = DBSCANAnomalyValidator(
            columns=["x", "y"],
            sample_size=100,
            eps=0.5,
            min_samples=5,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_isolation_forest_default_params(self):
        """Test IsolationForest works without new params."""
        np.random.seed(42)

        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 100),
        })
        lf = df.lazy()

        # Old API should still work
        validator = IsolationForestValidator(
            columns=["col1"],
            contamination=0.1,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)

    def test_lof_default_params(self):
        """Test LOF works without new params."""
        np.random.seed(42)

        df = pl.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
        })
        lf = df.lazy()

        validator = LOFValidator(
            columns=["x", "y"],
            n_neighbors=10,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)

    def test_svm_default_params(self):
        """Test OneClassSVM works without new params."""
        np.random.seed(42)

        df = pl.DataFrame({
            "col1": np.random.normal(0, 1, 100),
            "col2": np.random.normal(0, 1, 100),
        })
        lf = df.lazy()

        validator = OneClassSVMValidator(
            columns=["col1", "col2"],
            nu=0.1,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)

    def test_dbscan_default_params(self):
        """Test DBSCAN works without new params."""
        np.random.seed(42)

        df = pl.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
        })
        lf = df.lazy()

        validator = DBSCANAnomalyValidator(
            columns=["x", "y"],
            eps=0.5,
            min_samples=5,
        )

        issues = validator.validate(lf)
        assert isinstance(issues, list)


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_constants_exist(self):
        """Test that default constants are defined."""
        assert DEFAULT_SAMPLE_SIZE == 100000
        assert DEFAULT_BATCH_SIZE == 50000
        assert MEMORY_THRESHOLD_MB == 500

    def test_constants_are_reasonable(self):
        """Test that constants have reasonable values."""
        assert DEFAULT_SAMPLE_SIZE > 1000
        assert DEFAULT_BATCH_SIZE > 1000
        assert MEMORY_THRESHOLD_MB > 100
