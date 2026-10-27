"""Concept method selections run real streaming detectors with native state restore."""

import json

import polars as pl
import pytest

from truthound.ml.drift_detection.concept import ConceptDriftDetector


@pytest.mark.parametrize("method", ["ddm", "adwin", "page_hinkley"])
def test_stream_method_matches_native_alarms_and_restores_without_pickle(tmp_path, method):
    from river import drift

    reference = pl.DataFrame({"error": [0.0] * 1000}).lazy()
    current = pl.DataFrame({"error": [1.0] * 1000}).lazy()
    factories = {
        "ddm": lambda: drift.binary.DDM(warm_start=30),
        "adwin": lambda: drift.ADWIN(grace_period=30),
        "page_hinkley": lambda: drift.PageHinkley(min_instances=30),
    }
    native = factories[method]()
    for value in [0.0] * 1000:
        native.update(value)
    alarms = []
    for index, value in enumerate([1.0] * 1000):
        native.update(value)
        if native.drift_detected:
            alarms.append(index)
    assert alarms
    model = ConceptDriftDetector(method=method, stream_column="error", detection_window=1000)
    model.fit(reference)
    result = model.predict(current)
    details = json.loads(result.details)
    assert result.is_drifted and details["drift_count"] == len(alarms)
    assert details["first_drift_index"] == alarms[0]
    assert not model.predict(reference).is_drifted
    path = tmp_path / "concept.json"
    model.save(path)
    restored = ConceptDriftDetector()
    restored.load(path)
    restored_result = restored.predict(current).to_dict()
    original_result = result.to_dict()
    assert restored_result.pop("detected_at") >= original_result.pop("detected_at")
    assert restored_result == original_result


@pytest.mark.parametrize("bad", [[0.5] * 1000, [float("nan")] * 1000, [None] * 1000])
def test_invalid_refit_preserves_trained_stream(bad):
    reference = pl.DataFrame({"error": [0.0] * 1000}).lazy()
    model = ConceptDriftDetector(method="ddm", stream_column="error")
    model.fit(reference)
    with pytest.raises(ValueError):
        model.fit(pl.DataFrame({"error": bad}).lazy())
    assert model.is_trained
    assert not model.predict(reference).is_drifted
    assert model.predict(pl.DataFrame({"error": [1.0] * 1000}).lazy()).is_drifted


@pytest.mark.parametrize("method", ["ddm", "adwin", "page_hinkley"])
def test_predictions_use_actual_classification_errors_and_tail_window(method):
    model = ConceptDriftDetector(
        method=method, prediction_column="prediction", target_column="label", detection_window=1000
    )
    reference = pl.DataFrame({"prediction": ["yes"] * 1000, "label": ["yes"] * 1000}).lazy()
    model.fit(reference)
    changed = pl.DataFrame({"prediction": ["yes"] * 1000, "label": ["no"] * 1000}).lazy()
    result = model.predict(pl.concat([reference, changed]))
    assert result.is_drifted
    assert json.loads(result.details)["current_samples"] == 1000
    with pytest.raises(ValueError):
        model.predict(changed.drop("label"))
    with pytest.raises(ValueError):
        model.detect(reference, changed, columns=["label"])


def test_legacy_method_label_restores_actual_correlation_behavior(tmp_path):
    from truthound.ml.serialization import decode, encode

    data = pl.DataFrame({"x": list(range(100)), "label": list(range(100))}).lazy()
    model = ConceptDriftDetector(target_column="label")
    model.fit(data)
    path = tmp_path / "legacy.json"
    model.save(path)
    document = json.loads(path.read_text())
    payload = decode(document["payload"])
    payload["info"]["version"] = "1.0.0"
    payload["config"].pop("stream_column")
    payload["config"]["method"] = "ddm"
    document["payload"] = encode(payload)
    path.write_text(json.dumps(document))
    restored = ConceptDriftDetector()
    restored.load(path)
    assert restored.config.method == "correlation"
    assert not restored.predict(data).is_drifted
