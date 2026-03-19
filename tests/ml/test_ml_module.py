from __future__ import annotations

import random

import polars as pl
import pytest

from truthound.ml.anomaly_models import ZScoreAnomalyDetector
from truthound.ml.base import ModelNotTrainedError, ModelState, ModelTrainingError
from truthound.ml.drift_detection import DistributionDriftDetector


pytestmark = pytest.mark.contract


@pytest.fixture
def normal_data() -> pl.LazyFrame:
    random.seed(42)
    return pl.DataFrame(
        {
            "id": list(range(100)),
            "value": [random.gauss(50, 10) for _ in range(100)],
        }
    ).lazy()


@pytest.fixture
def outlier_data() -> pl.LazyFrame:
    random.seed(42)
    values = [random.gauss(50, 10) for _ in range(95)]
    values.extend([150.0, 200.0, -50.0, -100.0, 300.0])
    return pl.DataFrame({"id": list(range(100)), "value": values}).lazy()


@pytest.fixture
def tiny_reference_data() -> pl.LazyFrame:
    return pl.DataFrame({"value": [1.0, 2.0, 3.0]}).lazy()


@pytest.mark.contract
def test_zscore_detector_flags_clear_outliers(outlier_data: pl.LazyFrame):
    detector = ZScoreAnomalyDetector()
    detector.fit(outlier_data)

    result = detector.predict(outlier_data)

    assert result.anomaly_count > 0
    assert result.total_points == 100


@pytest.mark.fault
def test_anomaly_detector_rejects_prediction_before_fit(normal_data: pl.LazyFrame):
    detector = ZScoreAnomalyDetector()

    with pytest.raises(ModelNotTrainedError):
        detector.predict(normal_data)


@pytest.mark.fault
def test_distribution_drift_detector_fails_fast_on_insufficient_reference_data(
    tiny_reference_data: pl.LazyFrame,
):
    detector = DistributionDriftDetector()

    with pytest.raises(ModelTrainingError):
        detector.fit(tiny_reference_data)

    assert detector.state == ModelState.ERROR


@pytest.mark.fault
def test_drift_predict_rejects_untrained_detector(normal_data: pl.LazyFrame):
    detector = DistributionDriftDetector()

    with pytest.raises(ModelNotTrainedError):
        detector.predict(normal_data)
