"""Native classifier configuration, failed retraining and persistence contracts."""

import pickle
import random

import pytest

from truthound.profiler.base import DataType
from truthound.profiler.ml.base import ModelConfig, TrainingData
from truthound.profiler.ml.classifier import (
    RandomForestTypeClassifier,
    RuleBasedTypeClassifier,
    train_type_classifier,
)
from truthound.profiler.ml.pretrained import PretrainedModelManager, SyntheticDataGenerator

pytestmark = pytest.mark.contract


def training():
    return TrainingData(
        features=[[float(i % 2), float(i)] for i in range(40)],
        labels=[DataType.EMAIL if i % 2 else DataType.INTEGER for i in range(40)],
        feature_names=["pat_email", "other"],
        weights=[1.0] * 40,
    )


def test_rule_model_uses_feature_names_and_measured_metrics(tmp_path):
    model = RuleBasedTypeClassifier()
    data = training()
    metrics = model.train(data)
    predictions = [p.predicted_type for p in model.predict_batch(data.features)]
    assert metrics.accuracy == sum(
        a == b for a, b in zip(predictions, data.labels, strict=True)
    ) / len(data)
    assert model.predict([1.0, 0.0]).predicted_type == DataType.EMAIL
    path = tmp_path / "rules.json"
    model.save(str(path))
    restored = RuleBasedTypeClassifier()
    restored.load(str(path))
    assert restored.predict([1.0, 0.0]).predicted_type == DataType.EMAIL


def test_unknown_classifier_does_not_train_another_backend():
    with pytest.raises(ValueError, match="model"):
        train_type_classifier(training(), "misspelled", ModelConfig(n_estimators=2, n_jobs=1))


def test_failed_retraining_preserves_previous_estimator_and_labels(monkeypatch):
    model = RandomForestTypeClassifier(
        ModelConfig(n_estimators=3, n_jobs=1, cross_validation_folds=2)
    )
    model.train(training())
    original = model._model
    before = model.predict([1, 0]).probabilities
    data = training()
    data.labels = [DataType.URL if i % 2 else DataType.FLOAT for i in range(40)]
    from sklearn.ensemble import RandomForestClassifier

    monkeypatch.setattr(
        RandomForestClassifier,
        "fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fit failure")),
    )
    with pytest.raises(RuntimeError, match="fit failure"):
        model.train(data)
    assert model._model is original and model.predict([1, 0]).probabilities == before


def test_synthetic_generator_is_seeded_and_does_not_change_global_rng():
    random.seed(178)
    state = random.getstate()
    first = SyntheticDataGenerator(7)
    expected = first.generate(DataType.UUID, 3)
    assert random.getstate() == state
    assert SyntheticDataGenerator(7).generate(DataType.UUID, 3) == expected


@pytest.mark.parametrize(
    "kind",
    [
        "random_forest",
        "gradient_boosting",
        "naive_bayes",
        "logistic_regression",
        "svm",
        "neural_network",
        "ensemble",
    ],
)
def test_native_classifiers_honor_selection_weights_cv_and_reload(kind, tmp_path):
    data = training()
    config = ModelConfig(
        n_estimators=3, n_jobs=1, max_features=1, cross_validation_folds=2, early_stopping=False
    )
    model = train_type_classifier(data, kind, config)
    assert model._model.named_steps["features"].get_support().sum() == 1
    assert len(model.get_metrics().cross_val_scores) == 2
    assert model.config.model_type.value == kind
    assert model._feature_names == data.feature_names
    before = model.predict([1.0, 7.0]).probabilities
    assert set(before) == {DataType.EMAIL, DataType.INTEGER}
    assert sum(before.values()) == pytest.approx(1)
    path = tmp_path / "native.pkl"
    model.save(str(path))
    restored = type(model)()
    restored.load(str(path))
    assert restored.predict([1.0, 7.0]).probabilities == before
    assert restored.config.to_dict() == model.config.to_dict()
    assert restored.get_metrics().to_dict() == model.get_metrics().to_dict()
    path.write_bytes(b"corrupted")
    with pytest.raises(pickle.UnpicklingError):
        restored.load(str(path))
    assert restored.predict([1.0, 7.0]).probabilities == before


def test_pretrained_rules_bind_names_without_changing_shared_model():
    import polars as pl

    from truthound.profiler.ml.feature_extraction import FeatureExtractor

    pretrained = PretrainedModelManager.load_default()
    features = FeatureExtractor().extract(pl.Series("email", ["person@example.test"] * 10))
    before = list(pretrained.model._feature_names)
    assert pretrained.predict(features).predicted_type == DataType.EMAIL
    assert pretrained.model._feature_names == before


def test_disabled_cache_never_reads_existing_pickle(tmp_path, monkeypatch):
    monkeypatch.setattr(PretrainedModelManager, "DEFAULT_CACHE_DIR", tmp_path)
    (tmp_path / "owned.pkl").write_bytes(b"not a model")

    # Stop at native training rather than paying for a second full synthetic fit.
    def stop(self, data):
        raise RuntimeError("reached native training without reading cache")

    monkeypatch.setattr(RandomForestTypeClassifier, "train", stop)
    with pytest.raises(RuntimeError, match="reached native training"):
        PretrainedModelManager.train_and_cache("owned", n_samples_per_type=250, cache=False)
    assert (tmp_path / "owned.pkl").read_bytes() == b"not a model"


def test_synthetic_training_cache_reopen_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(PretrainedModelManager, "DEFAULT_CACHE_DIR", tmp_path)
    model = PretrainedModelManager.train_and_cache("native", n_samples_per_type=250)
    assert model.model.is_trained and len(model.model.get_metrics().cross_val_scores) == 5
    assert model.model.get_metrics().accuracy == model.accuracy
    restored = PretrainedModelManager.train_and_cache("native", n_samples_per_type=250)
    assert restored.model.get_metrics().to_dict() == model.model.get_metrics().to_dict()
    path = tmp_path / "native.pkl"
    path.write_bytes(b"broken cache")
    with pytest.raises(pickle.UnpicklingError):
        PretrainedModelManager.train_and_cache("native", n_samples_per_type=250)
    assert path.read_bytes() == b"broken cache"
