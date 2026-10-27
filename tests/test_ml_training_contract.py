"""Native model ownership and failure visibility contracts."""

import polars as pl
import pytest

from truthound.ml.anomaly_models import ZScoreAnomalyDetector
from truthound.ml.anomaly_models.statistical import StatisticalConfig
from truthound.ml.base import InsufficientDataError, ModelTrainingError
from truthound.ml.drift_detection.concept import ConceptDriftDetector


def test_model_captures_config_and_rejects_unknown_overrides():
    config = StatisticalConfig(columns=["x"])
    model = ZScoreAnomalyDetector(config, min_samples=20)
    assert config.min_samples != 20
    config.columns.append("other")
    assert model.config.columns == ["x"]
    with pytest.raises(ValueError):
        ZScoreAnomalyDetector(unknown_option=True)


def test_concept_training_does_not_convert_bad_features_to_zero_correlation():
    frame = pl.DataFrame({"x": ["not-numeric"] * 30, "target": list(range(30))}).lazy()
    model = ConceptDriftDetector(target_column="target", feature_columns=["x"])
    with pytest.raises(ModelTrainingError):
        model.fit(frame)


def test_concept_missing_target_or_feature_is_not_no_drift():
    frame = pl.DataFrame({"x": list(range(30)), "target": list(range(30))}).lazy()
    model = ConceptDriftDetector(target_column="target")
    model.fit(frame)
    with pytest.raises(ValueError):
        model.detect(frame, frame.drop("target"))
    with pytest.raises(ValueError):
        model.detect(frame, frame, columns=["absent"])


def test_failed_restore_keeps_the_live_model_and_configuration(tmp_path):
    import json

    from truthound.ml.base import ModelLoadError
    from truthound.ml.serialization import decode, encode

    data = pl.DataFrame({"x": list(range(100))}).lazy()
    model = ZScoreAnomalyDetector()
    model.fit(data)
    before = model.score(data).to_list()
    original_threshold = model.config.z_threshold
    path = tmp_path / "model.json"
    model.save(path)
    document = json.loads(path.read_text())
    payload = decode(document["payload"])
    payload["config"]["z_threshold"] = 99.0
    payload["trained_at"] = "not-a-date"
    document["payload"] = encode(payload)
    path.write_text(json.dumps(document))
    with pytest.raises(ModelLoadError):
        model.load(path)
    assert model.is_trained
    assert model.config.z_threshold == original_threshold
    assert model.score(data).to_list() == before


def test_native_training_sample_uses_seeded_rows_beyond_the_prefix():
    frame = pl.DataFrame({"x": list(range(1000))}).lazy()
    first = ZScoreAnomalyDetector(sample_size=100, random_seed=7)
    same = ZScoreAnomalyDetector(sample_size=100, random_seed=7)
    different = ZScoreAnomalyDetector(sample_size=100, random_seed=11)
    for model in (first, same, different):
        model.fit(frame)
    first_stats = first.get_statistics()
    assert first_stats == same.get_statistics()
    assert first_stats != different.get_statistics()
    assert first_stats["x"]["mean"] > 200


@pytest.mark.parametrize(
    "config",
    [{"sample_size": 0}, {"sample_size": True}, {"random_seed": -1}, {"random_seed": 2**64}],
)
def test_invalid_sampling_configuration_rejected_before_training(config):
    with pytest.raises(ValueError):
        ZScoreAnomalyDetector(**config)


@pytest.mark.parametrize(
    "algorithm",
    [
        "ZScoreAnomalyDetector",
        "IQRAnomalyDetector",
        "MADAnomalyDetector",
        "IsolationForestDetector",
        "DistributionDriftDetector",
        "FeatureDriftDetector",
        "MultivariateDriftDetector",
    ],
)
def test_failed_native_refit_preserves_trained_results(algorithm):
    import truthound.ml as ml

    model = getattr(ml, algorithm)(
        **({"n_estimators": 5} if algorithm == "IsolationForestDetector" else {})
    )
    data = pl.DataFrame({"x": list(range(100)), "y": [i % 7 for i in range(100)]}).lazy()
    model.fit(data)
    before = model.predict(data).to_dict()
    with pytest.raises((InsufficientDataError, ModelTrainingError)):
        model.fit(data.head(0))
    assert model.is_trained
    after = model.predict(data).to_dict()
    for result in (before, after):
        result.pop("detected_at", None)
        result.pop("detection_time_ms", None)
    assert after == before
