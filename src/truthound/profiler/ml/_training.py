"""Transactional native sklearn training; candidate state never mutates its owner."""

import time
from copy import deepcopy

from truthound.profiler.ml.base import ModelMetrics


def metrics_for(actual, predicted, weights=None):
    import math

    actual = [getattr(value, "value", value) for value in actual]
    predicted = [getattr(value, "value", value) for value in predicted]
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Evaluation rows must align and be nonempty")
    weights = list(weights) if weights is not None else [1.0] * len(actual)
    if (
        len(weights) != len(actual)
        or any(not math.isfinite(weight) or weight < 0 for weight in weights)
        or sum(weights) <= 0
    ):
        raise ValueError("Invalid evaluation weights")
    labels = sorted(set(actual) | set(predicted))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0.0] * len(labels) for _ in labels]
    for truth, prediction, weight in zip(actual, predicted, weights, strict=True):
        matrix[index[truth]][index[prediction]] += weight
    report = {}
    for label, i in index.items():
        support = sum(matrix[i])
        count = sum(row[i] for row in matrix)
        precision = matrix[i][i] / count if count else 0.0
        recall = matrix[i][i] / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[str(label)] = dict(
            precision=precision, recall=recall, **{"f1-score": f1}, support=support
        )
    total = sum(weights)
    averages = {
        metric: sum(row[metric] * row["support"] for row in report.values()) / total
        for metric in ("precision", "recall", "f1-score")
    }
    return ModelMetrics(
        accuracy=sum(matrix[i][i] for i in range(len(labels))) / total,
        precision=averages["precision"],
        recall=averages["recall"],
        f1_score=averages["f1-score"],
        confusion_matrix=matrix,
        class_report=report,
    )


def estimator(name, config):
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    common = dict(random_state=config.random_state)
    trees = dict(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
    )
    if name == "random_forest":
        return RandomForestClassifier(
            **trees, **common, n_jobs=config.n_jobs, class_weight=config.class_weight
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            **trees,
            **common,
            n_iter_no_change=5 if config.early_stopping else None,
            validation_fraction=config.validation_size,
        )
    if name == "naive_bayes":
        return GaussianNB()
    if name == "logistic_regression":
        return LogisticRegression(**common, class_weight=config.class_weight, max_iter=1000)
    if name == "svm":
        return SVC(**common, class_weight=config.class_weight, probability=True)
    if name == "neural_network":
        return MLPClassifier(
            **common,
            early_stopping=config.early_stopping,
            validation_fraction=config.validation_size,
            max_iter=1000,
        )
    if name == "ensemble":
        return VotingClassifier(
            estimators=[
                (kind, estimator(kind, config))
                for kind in ("random_forest", "logistic_regression", "naive_bayes")
            ],
            voting="soft",
            n_jobs=config.n_jobs,
        )
    raise ValueError("Unknown model type")


def train(name, config, data):
    import numpy as np
    from sklearn.base import clone
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight

    started = time.monotonic()
    config = deepcopy(config)
    X, labels = data.to_numpy()
    encoder = LabelEncoder().fit(labels)
    y = encoder.transform(labels)
    if len(encoder.classes_) < 2:
        raise ValueError("At least two classes are required")
    if type(config.cross_validation_folds) is not int or config.cross_validation_folds < 2:
        raise ValueError("Cross validation requires at least two folds")
    if np.bincount(y).min() < config.cross_validation_folds:
        raise ValueError("Every class requires at least one example per cross-validation fold")
    if not 0 < config.validation_size < 1:
        raise ValueError("validation_size must be between zero and one")
    weights = np.asarray(data.weights, dtype=float) if data.weights is not None else np.ones(len(y))
    # Models without a class_weight parameter still honor the common setting.
    if (
        name in {"gradient_boosting", "naive_bayes", "neural_network"}
        and config.class_weight is not None
    ):
        weights = weights * compute_sample_weight(config.class_weight, y)
    stages = []
    if config.feature_selection:
        if type(config.max_features) is not int or config.max_features < 1:
            raise ValueError("max_features must be a positive integer")
        stages.append(("features", SelectKBest(f_classif, k=min(config.max_features, X.shape[1]))))
    stages.append(("model", estimator(name, config)))
    candidate = Pipeline(stages)
    train_ids, test_ids = train_test_split(
        np.arange(len(y)),
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y,
    )
    candidate.fit(X[train_ids], y[train_ids], model__sample_weight=weights[train_ids])
    metrics = metrics_for(y[test_ids], candidate.predict(X[test_ids]), weights[test_ids])
    folds = StratifiedKFold(
        config.cross_validation_folds, shuffle=True, random_state=config.random_state
    )
    scores = []
    for train_ids, test_ids in folds.split(X, y):
        fold = clone(candidate)
        fold.fit(X[train_ids], y[train_ids], model__sample_weight=weights[train_ids])
        scores.append(float(fold.score(X[test_ids], y[test_ids], sample_weight=weights[test_ids])))
    # Evaluation used held-out data; the published candidate learns all supplied rows.
    candidate.fit(X, y, model__sample_weight=weights)
    model = candidate.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        selected = (
            candidate.named_steps["features"].get_support()
            if "features" in candidate.named_steps
            else np.ones(X.shape[1], dtype=bool)
        )
        values = iter(model.feature_importances_.tolist())
        metrics.feature_importances = {
            key: next(values) if keep else 0.0
            for key, keep in zip(data.feature_names, selected, strict=True)
        }
    metrics.cross_val_scores = scores
    metrics.training_time_seconds = time.monotonic() - started
    return candidate, encoder, metrics
