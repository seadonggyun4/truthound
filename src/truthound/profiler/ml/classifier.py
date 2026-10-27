"""ML classifiers for type inference.

Provides scikit-learn based classifiers for data type inference.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

from truthound.profiler.base import DataType
from truthound.profiler.ml.base import (
    BaseModel,
    ModelConfig,
    ModelMetrics,
    ModelType,
    PredictionResult,
    TrainingData,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Rule-Based Classifier (No ML dependencies)
# =============================================================================


class RuleBasedTypeClassifier(BaseModel):
    """Rule-based classifier that works without ML libraries.

    Uses weighted rules to infer types based on feature values.
    Serves as fallback when scikit-learn is not available.
    """

    name = "rule_based"
    version = "1.0.0"

    # Rules: (feature_pattern, operator, threshold, type, weight)
    RULES: list[tuple[str, str, float, DataType, float]] = [
        # Email
        ("pat_email", ">=", 0.8, DataType.EMAIL, 1.0),
        ("str_has_at_sign", ">=", 0.9, DataType.EMAIL, 0.7),
        ("name_email", ">=", 0.5, DataType.EMAIL, 0.5),
        # UUID
        ("pat_uuid", ">=", 0.8, DataType.UUID, 1.0),
        ("str_uniform_length", ">=", 0.9, DataType.UUID, 0.3),
        ("name_uuid", ">=", 0.5, DataType.UUID, 0.5),
        # URL
        ("pat_url", ">=", 0.7, DataType.URL, 1.0),
        ("str_has_slash", ">=", 0.8, DataType.URL, 0.5),
        ("name_url", ">=", 0.5, DataType.URL, 0.5),
        # Date/DateTime
        ("pat_date_iso", ">=", 0.7, DataType.DATE, 0.9),
        ("pat_datetime_iso", ">=", 0.7, DataType.DATETIME, 0.9),
        ("name_date", ">=", 0.5, DataType.DATE, 0.4),
        ("name_datetime", ">=", 0.5, DataType.DATETIME, 0.4),
        ("stat_is_date_type", ">=", 0.5, DataType.DATE, 0.8),
        # Phone
        ("pat_phone_intl", ">=", 0.6, DataType.PHONE, 0.8),
        ("pat_korean_phone", ">=", 0.7, DataType.KOREAN_PHONE, 0.9),
        ("name_phone", ">=", 0.5, DataType.PHONE, 0.5),
        # Numeric types
        ("stat_is_integer_type", ">=", 0.5, DataType.INTEGER, 0.7),
        ("stat_is_float_type", ">=", 0.5, DataType.FLOAT, 0.6),
        ("stat_is_0_100_range", ">=", 0.95, DataType.PERCENTAGE, 0.5),
        ("stat_is_0_1_range", ">=", 0.95, DataType.PERCENTAGE, 0.6),
        ("stat_has_2_decimals", ">=", 0.9, DataType.CURRENCY, 0.5),
        ("name_currency", ">=", 0.5, DataType.CURRENCY, 0.6),
        ("name_percentage", ">=", 0.5, DataType.PERCENTAGE, 0.6),
        # Boolean
        ("stat_is_boolean_type", ">=", 0.5, DataType.BOOLEAN, 0.9),
        ("name_boolean", ">=", 0.5, DataType.BOOLEAN, 0.5),
        # Categorical
        ("dist_is_categorical", ">=", 0.8, DataType.CATEGORICAL, 0.7),
        ("name_categorical", ">=", 0.5, DataType.CATEGORICAL, 0.4),
        # Identifier
        ("stat_is_unique", ">=", 0.95, DataType.IDENTIFIER, 0.6),
        ("stat_is_sequential", ">=", 0.8, DataType.IDENTIFIER, 0.7),
        ("name_identifier", ">=", 0.5, DataType.IDENTIFIER, 0.5),
        ("ctx_is_first", ">=", 0.5, DataType.IDENTIFIER, 0.3),
        # IP
        ("pat_ip_v4", ">=", 0.7, DataType.IP_ADDRESS, 0.9),
        # Korean specific
        ("pat_korean_rrn", ">=", 0.7, DataType.KOREAN_RRN, 0.95),
    ]

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self._trained = True  # Always ready

    def predict(self, features: list[float]) -> PredictionResult:
        """Predict using rules."""
        start = time.time()

        import math

        if any(not math.isfinite(value) for value in features):
            raise ValueError("Features must be finite")
        # Build feature dict
        feature_dict = self._build_feature_dict(features)

        # Score each type
        type_scores: dict[DataType, float] = {}

        for feature_pattern, operator, threshold, dtype, weight in self.RULES:
            # Find matching features
            matched_value = 0.0
            for fname, fval in feature_dict.items():
                if feature_pattern in fname or fname.endswith(feature_pattern):
                    matched_value = max(matched_value, fval)

            # Check condition
            if operator == ">=" and matched_value >= threshold:
                type_scores[dtype] = type_scores.get(dtype, 0) + weight * matched_value
            elif operator == "<=" and matched_value <= threshold:
                type_scores[dtype] = type_scores.get(dtype, 0) + weight * (1 - matched_value)

        # Normalize to probabilities
        total = sum(type_scores.values()) or 1
        probabilities = {dtype: score / total for dtype, score in type_scores.items()}

        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        if sorted_probs:
            best_type, best_conf = sorted_probs[0]
            alternatives = sorted_probs[1:5]
        else:
            best_type = DataType.STRING
            best_conf = 0.5
            alternatives = []

        elapsed_ms = (time.time() - start) * 1000

        return PredictionResult(
            predicted_type=best_type,
            confidence=best_conf,
            probabilities=probabilities,
            top_alternatives=alternatives,
            model_version=self.version,
            inference_time_ms=elapsed_ms,
        )

    def predict_batch(self, features: list[list[float]]) -> list[PredictionResult]:
        """Predict for multiple samples."""
        return [self.predict(f) for f in features]

    def train(self, data: TrainingData) -> ModelMetrics:
        """Evaluate configured rules on supplied data; never invent training scores."""
        from truthound.profiler.ml._training import metrics_for

        data.to_numpy()
        names = self._feature_names
        self._feature_names = list(data.feature_names)
        try:
            metrics = metrics_for(
                data.labels,
                [p.predicted_type for p in self.predict_batch(data.features)],
                data.weights,
            )
        except BaseException:
            self._feature_names = names
            raise
        self._metrics = metrics
        return metrics

    def save(self, path: str) -> None:
        from truthound.stores._filesystem_io import publish

        payload = {
            "format": "truthound-type-rules-1",
            "version": self.version,
            "feature_names": self._feature_names,
            "metrics": self._metrics.to_dict() if self._metrics else None,
            "rules": [
                [name, op, threshold, dtype.value, weight]
                for name, op, threshold, dtype, weight in self.RULES
            ],
        }
        publish(Path(path), json.dumps(payload, allow_nan=False).encode())

    def load(self, path: str) -> None:
        import math

        data = json.loads(Path(path).read_text())
        if data.get("format") != "truthound-type-rules-1":
            raise ValueError("Unsupported rule model format")
        rules = [
            (name, op, float(threshold), DataType(dtype), float(weight))
            for name, op, threshold, dtype, weight in data["rules"]
        ]
        if any(
            not isinstance(name, str)
            or op not in {">=", "<="}
            or not math.isfinite(threshold)
            or not math.isfinite(weight)
            or weight < 0
            for name, op, threshold, dtype, weight in rules
        ):
            raise ValueError("Invalid model rules")
        names = data["feature_names"]
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Invalid feature names")
        from datetime import datetime
        metrics_data = data.get("metrics")
        if metrics_data:
            metrics_data["trained_at"] = datetime.fromisoformat(metrics_data["trained_at"])
        metrics = ModelMetrics(**metrics_data) if metrics_data else None
        if not isinstance(data["version"], str):
            raise ValueError("Invalid model version")
        self.RULES, self._feature_names = rules, names
        self._metrics, self.version = metrics, data["version"]

    def _build_feature_dict(self, features: list[float]) -> dict[str, float]:
        """Build feature dictionary from list."""
        # Use standard feature names if feature_names not set
        if self._feature_names:
            return dict(zip(self._feature_names, features, strict=True))

        # Fallback: use index-based names
        return {f"f_{i}": v for i, v in enumerate(features)}


# =============================================================================
# Scikit-Learn Based Classifiers
# =============================================================================


class TypeInferenceModel(BaseModel):
    """Base class for scikit-learn based classifiers."""

    version = "1.0.0"

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            import importlib

            importlib.import_module("sklearn")
            return True
        except ImportError:
            return False

    def _encode_labels(self, labels: list[DataType]) -> tuple[Any, Any]:
        """Encode DataType labels to integers."""
        from sklearn.preprocessing import LabelEncoder

        if self._label_encoder is None:
            self._label_encoder = LabelEncoder()
            encoded = self._label_encoder.fit_transform([label.value for label in labels])
        else:
            encoded = self._label_encoder.transform([label.value for label in labels])

        return encoded, self._label_encoder

    def _decode_labels(self, encoded: Any) -> list[DataType]:
        """Decode integer labels back to DataType."""
        if self._label_encoder is None:
            raise ValueError("Label encoder not fitted")

        decoded = self._label_encoder.inverse_transform(encoded)
        return [DataType(v) for v in decoded]

    def predict(self, features: list[float]) -> PredictionResult:
        """Predict type for a single sample."""
        if not self._trained or self._model is None:
            raise RuntimeError("Model has not been trained")

        import numpy as np

        start = time.time()

        X = np.array([features])
        proba = self._model.predict_proba(X)[0]

        # Get class probabilities
        classes = self._label_encoder.classes_
        probabilities = {DataType(c): float(p) for c, p in zip(classes, proba, strict=True)}

        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        best_type = sorted_probs[0][0]
        best_conf = sorted_probs[0][1]
        alternatives = sorted_probs[1:5]

        elapsed_ms = (time.time() - start) * 1000

        return PredictionResult(
            predicted_type=best_type,
            confidence=best_conf,
            probabilities=probabilities,
            top_alternatives=alternatives,
            features_used=self._feature_names[:10],
            model_version=self.version,
            inference_time_ms=elapsed_ms,
        )

    def predict_batch(self, features: list[list[float]]) -> list[PredictionResult]:
        """Predict for multiple samples."""
        return [self.predict(f) for f in features]

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "format": "truthound-sklearn-type-1",
            "kind": self.name,
            "model": self._model,
            "label_encoder": self._label_encoder,
            "feature_names": self._feature_names,
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict() if self._metrics else None,
            "version": self.version,
        }
        if not self._trained:
            raise RuntimeError("Model has not been trained")
        from truthound.stores._filesystem_io import publish

        publish(Path(path), pickle.dumps(data))
        logger.info(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        from datetime import datetime

        if data.get("format") != "truthound-sklearn-type-1" or data.get("kind") != self.name:
            raise ValueError("Incompatible classifier model format")
        model, encoder, names = data["model"], data["label_encoder"], data["feature_names"]
        if model is None or encoder is None or model.n_features_in_ != len(names):
            raise ValueError("Invalid trained model")
        for label in encoder.classes_:
            DataType(label)
        config_data = dict(data["config"])
        config_data["model_type"] = ModelType(config_data["model_type"])
        config = ModelConfig(**config_data)
        metrics_data = data.get("metrics")
        if metrics_data:
            metrics_data["trained_at"] = datetime.fromisoformat(metrics_data["trained_at"])
        metrics = ModelMetrics(**metrics_data) if metrics_data else None
        self._model, self._label_encoder, self._feature_names = model, encoder, list(names)
        self.config, self._metrics, self._trained = config, metrics, True
        self.version = data["version"]

    def train(self, data: TrainingData) -> ModelMetrics:
        if not self._sklearn_available:
            raise ImportError("scikit-learn is required for this classifier")
        from truthound.profiler.ml._training import train

        model, encoder, metrics = train(self.name, self.config, data)
        self._model, self._label_encoder = model, encoder
        self.config.model_type = ModelType(self.name)
        self._feature_names, self._metrics, self._trained = list(data.feature_names), metrics, True
        return metrics


class RandomForestTypeClassifier(TypeInferenceModel):
    """Native sklearn random forest type classifier."""

    name = "random_forest"
    version = "1.0.0"


class GradientBoostingTypeClassifier(TypeInferenceModel):
    """Native sklearn gradient boosting type classifier."""

    name = "gradient_boosting"
    version = "1.0.0"


class NaiveBayesTypeClassifier(TypeInferenceModel):
    name = "naive_bayes"


class LogisticRegressionTypeClassifier(TypeInferenceModel):
    name = "logistic_regression"


class SVMTypeClassifier(TypeInferenceModel):
    name = "svm"


class NeuralNetworkTypeClassifier(TypeInferenceModel):
    name = "neural_network"


class EnsembleTypeClassifier(TypeInferenceModel):
    name = "ensemble"


def train_type_classifier(
    training_data: TrainingData,
    model_type: str = "random_forest",
    config: ModelConfig | None = None,
) -> BaseModel:
    """Train the explicitly selected native classifier, failing on unknown names."""
    models = {
        cls.name: cls
        for cls in (
            RandomForestTypeClassifier,
            GradientBoostingTypeClassifier,
            RuleBasedTypeClassifier,
            NaiveBayesTypeClassifier,
            LogisticRegressionTypeClassifier,
            SVMTypeClassifier,
            NeuralNetworkTypeClassifier,
            EnsembleTypeClassifier,
        )
    }
    if model_type not in models:
        raise ValueError("Unknown model type: " + str(model_type))
    model = models[model_type](config)
    model.train(training_data)
    return model
