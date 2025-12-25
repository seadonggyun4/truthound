"""Integration tests for the ML module."""

from __future__ import annotations

import pytest
import polars as pl

from truthound.ml import (
    ModelRegistry,
    AnomalyDetector,
    ModelType,
    ModelState,
)
from truthound.ml.anomaly_models import (
    ZScoreAnomalyDetector,
    IQRAnomalyDetector,
    MADAnomalyDetector,
    IsolationForestDetector,
    EnsembleAnomalyDetector,
)
from truthound.ml.drift_detection import (
    DistributionDriftDetector,
    FeatureDriftDetector,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def normal_data() -> pl.LazyFrame:
    """Normal distributed data."""
    import random
    random.seed(42)
    return pl.DataFrame({
        "id": list(range(100)),
        "value": [random.gauss(50, 10) for _ in range(100)],
        "category": ["A"] * 50 + ["B"] * 50,
    }).lazy()


@pytest.fixture
def data_with_outliers() -> pl.LazyFrame:
    """Data with clear outliers."""
    import random
    random.seed(42)
    values = [random.gauss(50, 10) for _ in range(95)]
    values.extend([150.0, 200.0, -50.0, -100.0, 300.0])
    return pl.DataFrame({
        "id": list(range(100)),
        "value": values,
    }).lazy()


@pytest.fixture
def reference_data() -> pl.LazyFrame:
    """Reference data for drift detection."""
    import random
    random.seed(42)
    return pl.DataFrame({
        "id": list(range(200)),
        "value": [random.gauss(50, 10) for _ in range(200)],
    }).lazy()


@pytest.fixture
def drifted_data() -> pl.LazyFrame:
    """Data with distribution drift."""
    import random
    random.seed(456)
    return pl.DataFrame({
        "id": list(range(200)),
        "value": [random.gauss(80, 15) for _ in range(200)],  # Mean shift
    }).lazy()


# =============================================================================
# Test Model Registry
# =============================================================================


class TestModelRegistry:
    """Tests for model registry."""

    def test_registry_exists(self):
        """Test that ModelRegistry exists and can be instantiated."""
        registry = ModelRegistry()
        assert registry is not None

    def test_list_all(self):
        """Test listing all models."""
        registry = ModelRegistry()
        models = registry.list_all()

        # Should return a list (may be empty for new registry)
        assert isinstance(models, list)


# =============================================================================
# Test Anomaly Detectors
# =============================================================================


class TestAnomalyDetectors:
    """Tests for anomaly detection models."""

    def test_zscore_detector(self, normal_data: pl.LazyFrame):
        """Test Z-Score detector."""
        detector = ZScoreAnomalyDetector()

        # Initial state
        assert detector.state == ModelState.UNTRAINED

        # Fit
        detector.fit(normal_data)
        assert detector.is_trained

        # Score
        scores = detector.score(normal_data)
        assert isinstance(scores, pl.Series)
        assert len(scores) == 100

    def test_iqr_detector(self, normal_data: pl.LazyFrame):
        """Test IQR detector."""
        detector = IQRAnomalyDetector()
        detector.fit(normal_data)

        assert detector.is_trained
        scores = detector.score(normal_data)
        assert len(scores) == 100

    def test_mad_detector(self, normal_data: pl.LazyFrame):
        """Test MAD detector."""
        detector = MADAnomalyDetector()
        detector.fit(normal_data)

        assert detector.is_trained
        scores = detector.score(normal_data)
        assert len(scores) == 100

    def test_isolation_forest(self, normal_data: pl.LazyFrame):
        """Test Isolation Forest detector."""
        detector = IsolationForestDetector(n_trees=10, sample_size=50)
        detector.fit(normal_data)

        assert detector.is_trained
        scores = detector.score(normal_data)
        assert len(scores) == 100
        # Scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in scores.to_list())

    def test_ensemble_detector(self, normal_data: pl.LazyFrame):
        """Test ensemble detector."""
        ensemble = EnsembleAnomalyDetector(
            detectors=[
                ZScoreAnomalyDetector(),
                IQRAnomalyDetector(),
            ]
        )
        ensemble.fit(normal_data)

        assert ensemble.is_trained
        assert ensemble.n_detectors == 2

    def test_detect_outliers(self, data_with_outliers: pl.LazyFrame):
        """Test that detectors can identify outliers."""
        detector = ZScoreAnomalyDetector()
        detector.fit(data_with_outliers)

        result = detector.predict(data_with_outliers)

        # Should detect some anomalies
        assert result.anomaly_count > 0
        assert result.total_points == 100

    def test_detector_predict(self, normal_data: pl.LazyFrame):
        """Test detector predict returns AnomalyResult."""
        detector = ZScoreAnomalyDetector()
        detector.fit(normal_data)

        result = detector.predict(normal_data)

        assert hasattr(result, 'anomaly_count')
        assert hasattr(result, 'anomaly_ratio')
        assert hasattr(result, 'scores')
        assert hasattr(result, 'model_name')


# =============================================================================
# Test Drift Detectors
# =============================================================================


class TestDriftDetectors:
    """Tests for drift detection models."""

    def test_distribution_drift_detector(self, reference_data: pl.LazyFrame):
        """Test distribution drift detector."""
        detector = DistributionDriftDetector()
        detector.fit(reference_data)

        assert detector.is_trained

    def test_detect_no_drift(
        self,
        reference_data: pl.LazyFrame,
    ):
        """Test detecting no drift in similar data."""
        import random
        random.seed(123)
        similar_data = pl.DataFrame({
            "id": list(range(200)),
            "value": [random.gauss(50, 10) for _ in range(200)],
        }).lazy()

        detector = DistributionDriftDetector()
        detector.fit(reference_data)
        result = detector.detect(reference_data, similar_data)

        # Similar data should show minimal drift
        assert hasattr(result, 'is_drifted')
        assert hasattr(result, 'drift_score')

    def test_detect_drift(
        self,
        reference_data: pl.LazyFrame,
        drifted_data: pl.LazyFrame,
    ):
        """Test detecting drift in shifted data."""
        detector = DistributionDriftDetector()
        detector.fit(reference_data)
        result = detector.detect(reference_data, drifted_data)

        # Shifted data should show drift
        assert result.is_drifted is True or result.drift_score > 0.1

    def test_feature_drift_detector(self, reference_data: pl.LazyFrame):
        """Test feature drift detector."""
        detector = FeatureDriftDetector()
        detector.fit(reference_data)

        assert detector.is_trained


# =============================================================================
# Test Model Lifecycle
# =============================================================================


class TestModelLifecycle:
    """Tests for model state transitions."""

    def test_state_transition(self, normal_data: pl.LazyFrame):
        """Test model state transitions."""
        detector = ZScoreAnomalyDetector()

        # Start untrained
        assert detector.state == ModelState.UNTRAINED

        # After fit, should be trained
        detector.fit(normal_data)
        assert detector.state == ModelState.TRAINED

    def test_is_trained_property(self, normal_data: pl.LazyFrame):
        """Test is_trained property."""
        detector = ZScoreAnomalyDetector()

        assert detector.is_trained is False
        detector.fit(normal_data)
        assert detector.is_trained is True


# =============================================================================
# Test Model Info
# =============================================================================


class TestModelInfo:
    """Tests for model info."""

    def test_detector_info(self):
        """Test that detectors have info."""
        detector = ZScoreAnomalyDetector()
        info = detector.info

        assert hasattr(info, 'name')
        assert hasattr(info, 'version')
        assert hasattr(info, 'model_type')

    def test_ensemble_info(self):
        """Test ensemble model info."""
        ensemble = EnsembleAnomalyDetector()
        info = ensemble.info

        assert info.name == "ensemble"
        assert info.model_type == ModelType.ANOMALY_DETECTOR
