"""Bounded JSON persistence for native inference models; never imports code."""

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path


def digest(path):
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Invalid inference model snapshot")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def writer_lock(path):
    from truthound.stores.concurrency.locks import (
        LockMode,
        NoOpLockStrategy,
        get_default_lock_strategy,
    )

    if path is None:
        yield
        return
    path = Path(path).absolute()
    strategy = get_default_lock_strategy()
    if isinstance(strategy, NoOpLockStrategy):
        raise RuntimeError("Inference model persistence requires a native lock")
    handle = strategy.acquire(path.with_name(path.name + ".lock"), LockMode.EXCLUSIVE, timeout=5)
    try:
        yield
    finally:
        strategy.release(handle)


def encode_model(model, depth=0):
    from truthound.profiler.ml_inference import EnsembleModel, NaiveBayesModel, RuleBasedModel

    if depth > 8:
        raise ValueError("Inference ensemble nesting exceeds its bound")
    if type(model) is RuleBasedModel:
        return {"kind": "rule_based", "version": 1, "rules": [
            [name, operator, threshold, dtype.value, weight]
            for name, operator, threshold, dtype, weight in model.rules
        ]}
    if type(model) is NaiveBayesModel:
        return {"kind": "naive_bayes", "version": 1,
                "class_priors": {key.value: value for key, value in model.class_priors.items()},
                "feature_likelihoods": {
                    name: {dtype.value: stats for dtype, stats in values.items()}
                    for name, values in model.feature_likelihoods.items()
                }, "trained": model._trained}
    if type(model) is EnsembleModel:
        return {"kind": "ensemble", "version": 1, "models": [
            {"weight": weight, "state": encode_model(child, depth + 1)}
            for child, weight in model.models
        ]}
    raise NotImplementedError("Custom inference models must implement persistence explicitly")


def finite(value):
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("Invalid inference model number")
    return value


def decode_model(data, depth=0):
    from truthound.profiler.base import DataType
    from truthound.profiler.ml_inference import EnsembleModel, NaiveBayesModel, RuleBasedModel

    if depth > 8 or not isinstance(data, dict) or data.get("version", 1) != 1:
        raise ValueError("Invalid inference model snapshot")
    kind = data.get("kind", "naive_bayes" if "class_priors" in data else None)
    if kind == "rule_based":
        rules = data["rules"]
        if not isinstance(rules, list) or len(rules) > 4096:
            raise ValueError("Invalid inference rules")
        parsed = []
        for name, operator, threshold, dtype, weight in rules:
            if not isinstance(name, str) or operator not in {">=", "<=", "=="} or finite(weight) < 0:
                raise ValueError("Invalid inference rule")
            parsed.append((name, operator, finite(threshold), DataType(dtype), weight))
        model = RuleBasedModel()
        model.rules = parsed
        return model
    if kind == "naive_bayes":
        priors = {DataType(key): finite(value) for key, value in data["class_priors"].items()}
        trained = data["trained"]
        if type(trained) is not bool or any(value < 0 or value > 1 for value in priors.values()):
            raise ValueError("Invalid inference priors")
        if trained and (not priors or not math.isclose(sum(priors.values()), 1, abs_tol=1e-6)):
            raise ValueError("Invalid trained inference priors")
        likelihoods = {}
        if len(data["feature_likelihoods"]) > 10000:
            raise ValueError("Inference feature capacity exceeded")
        for name, values in data["feature_likelihoods"].items():
            parsed = {}
            for dtype, pair in values.items():
                mean, std = pair
                if finite(std) <= 0:
                    raise ValueError("Invalid inference deviation")
                parsed[DataType(dtype)] = (finite(mean), std)
            likelihoods[name] = parsed
        model = NaiveBayesModel()
        model.class_priors, model.feature_likelihoods, model._trained = priors, likelihoods, trained
        return model
    if kind == "ensemble":
        children = data["models"]
        if not isinstance(children, list) or not 1 <= len(children) <= 32:
            raise ValueError("Invalid inference ensemble")
        models = [(decode_model(child["state"], depth + 1), finite(child["weight"])) for child in children]
        if any(weight < 0 for _, weight in models) or sum(weight for _, weight in models) <= 0:
            raise ValueError("Invalid inference ensemble weights")
        return EnsembleModel(models)
    raise ValueError("Unknown inference snapshot kind")


def save(model, path):
    from truthound.stores._filesystem_io import publish

    state = encode_model(model)
    decode_model(state)
    content = json.dumps(state, allow_nan=False).encode()
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("Inference snapshot capacity exceeded")
    publish(Path(path), content)


def load_into(model, path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Invalid inference model snapshot")
    try:
        restored = decode_model(json.loads(path.read_text()))
        if type(restored) is not type(model):
            raise ValueError("Inference model kind does not match")
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError("Invalid inference model snapshot") from None
    model.__dict__.clear()
    model.__dict__.update(restored.__dict__)
