"""Ensemble failures preserve usable models and reject invalid score weights."""

import polars as pl
import pytest

from truthound.ml import EnsembleAnomalyDetector, ZScoreAnomalyDetector
from truthound.ml.base import ModelTrainingError


@pytest.mark.parametrize("weights", [[1.0], [0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0]])
def test_invalid_weights_fail_instead_of_silently_truncating_scores(weights):
    with pytest.raises((ValueError, ModelTrainingError)):
        model = EnsembleAnomalyDetector(
            detectors=[ZScoreAnomalyDetector(), ZScoreAnomalyDetector()], weights=weights
        )
        model.fit(pl.DataFrame({"x": list(range(100))}).lazy())


def test_failed_ensemble_refit_preserves_all_children():
    model = EnsembleAnomalyDetector(detectors=[ZScoreAnomalyDetector()])
    data = pl.DataFrame({"x": list(range(100))}).lazy()
    model.fit(data)
    before = model.score(data).to_list()
    with pytest.raises(ModelTrainingError):
        model.fit(pl.DataFrame({"x": ["bad"] * 100}).lazy())
    assert model.is_trained
    assert model.score(data).to_list() == before


def test_remove_before_fit_and_add_after_fit_require_consistent_retraining():
    model = EnsembleAnomalyDetector(detectors=[ZScoreAnomalyDetector(), ZScoreAnomalyDetector()])
    model.remove_detector(0)
    data = pl.DataFrame({"x": list(range(100))}).lazy()
    model.fit(data)
    model.add_detector(ZScoreAnomalyDetector(), weight=2.0)
    assert not model.is_trained
    model.fit(data)
    assert len(model.score(data)) == 100


def test_later_child_failure_does_not_publish_earlier_child_training():
    class FailingChild(ZScoreAnomalyDetector):
        def fit(self, data):
            if data.select(pl.col("x").min()).collect().item() > 0:
                raise ValueError("synthetic second-child failure")
            super().fit(data)

    supplied = ZScoreAnomalyDetector()
    model = EnsembleAnomalyDetector(detectors=[supplied, FailingChild()])
    data = pl.DataFrame({"x": list(range(100))}).lazy()
    model.fit(data)
    assert not supplied.is_trained
    before = model.score(data).to_list()
    with pytest.raises(ModelTrainingError):
        model.fit(pl.DataFrame({"x": list(range(100, 200))}).lazy())
    assert model.is_trained
    assert model.score(data).to_list() == before


def test_corrupt_saved_weights_preserve_live_ensemble(tmp_path):
    import json

    from truthound.ml.base import ModelLoadError
    from truthound.ml.serialization import decode, encode

    model = EnsembleAnomalyDetector()
    data = pl.DataFrame({"x": list(range(100))}).lazy()
    model.fit(data)
    before = model.score(data).to_list()
    path = tmp_path / "ensemble.json"
    model.save(path)
    document = json.loads(path.read_text())
    payload = decode(document["payload"])
    payload["weights"] = [0.0, 0.0, 0.0]
    document["payload"] = encode(payload)
    path.write_text(json.dumps(document))
    with pytest.raises(ModelLoadError):
        model.load(path)
    assert model.score(data).to_list() == before
