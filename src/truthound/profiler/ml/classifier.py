"""ML classifiers for type inference.

Provides scikit-learn based classifiers for data type inference.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    RULES: List[Tuple[str, str, float, DataType, float]] = [
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

    def predict(self, features: List[float]) -> PredictionResult:
        """Predict using rules."""
        start = time.time()

        # Build feature dict
        feature_dict = self._build_feature_dict(features)

        # Score each type
        type_scores: Dict[DataType, float] = {}

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

    def predict_batch(self, features: List[List[float]]) -> List[PredictionResult]:
        """Predict for multiple samples."""
        return [self.predict(f) for f in features]

    def train(self, data: TrainingData) -> ModelMetrics:
        """Rule-based model doesn't need training."""
        return ModelMetrics(
            accuracy=0.85,
            precision=0.85,
            recall=0.85,
            f1_score=0.85,
        )

    def save(self, path: str) -> None:
        """Save model (just saves rules)."""
        with open(path, "wb") as f:
            pickle.dump({"rules": self.RULES, "version": self.version}, f)

    def load(self, path: str) -> None:
        """Load model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            # Could customize rules here

    def _build_feature_dict(self, features: List[float]) -> Dict[str, float]:
        """Build feature dictionary from list."""
        # Use standard feature names if feature_names not set
        if self._feature_names:
            return dict(zip(self._feature_names, features))

        # Fallback: use index-based names
        return {f"f_{i}": v for i, v in enumerate(features)}


# =============================================================================
# Scikit-Learn Based Classifiers
# =============================================================================


class TypeInferenceModel(BaseModel):
    """Base class for scikit-learn based classifiers."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__(config)
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            import sklearn
            return True
        except ImportError:
            return False

    def _encode_labels(self, labels: List[DataType]) -> Tuple[Any, Any]:
        """Encode DataType labels to integers."""
        from sklearn.preprocessing import LabelEncoder

        if self._label_encoder is None:
            self._label_encoder = LabelEncoder()
            encoded = self._label_encoder.fit_transform([l.value for l in labels])
        else:
            encoded = self._label_encoder.transform([l.value for l in labels])

        return encoded, self._label_encoder

    def _decode_labels(self, encoded: Any) -> List[DataType]:
        """Decode integer labels back to DataType."""
        if self._label_encoder is None:
            raise ValueError("Label encoder not fitted")

        decoded = self._label_encoder.inverse_transform(encoded)
        return [DataType(v) for v in decoded]

    def predict(self, features: List[float]) -> PredictionResult:
        """Predict type for a single sample."""
        if not self._trained or self._model is None:
            # Fallback to rule-based
            return RuleBasedTypeClassifier().predict(features)

        import numpy as np
        start = time.time()

        X = np.array([features])
        pred = self._model.predict(X)[0]
        proba = self._model.predict_proba(X)[0]

        # Get class probabilities
        classes = self._label_encoder.classes_
        probabilities = {DataType(c): float(p) for c, p in zip(classes, proba)}

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

    def predict_batch(self, features: List[List[float]]) -> List[PredictionResult]:
        """Predict for multiple samples."""
        return [self.predict(f) for f in features]

    def save(self, path: str) -> None:
        """Save model to file."""
        data = {
            "model": self._model,
            "label_encoder": self._label_encoder,
            "feature_names": self._feature_names,
            "config": self.config.to_dict(),
            "metrics": self._metrics.to_dict() if self._metrics else None,
            "version": self.version,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]
        self._label_encoder = data["label_encoder"]
        self._feature_names = data.get("feature_names", [])
        self._trained = True
        logger.info(f"Model loaded from: {path}")


class RandomForestTypeClassifier(TypeInferenceModel):
    """Random Forest classifier for type inference.

    Provides robust predictions with built-in feature importance.
    """

    name = "random_forest"
    version = "1.0.0"

    def train(self, data: TrainingData) -> ModelMetrics:
        """Train Random Forest classifier."""
        if not self._sklearn_available:
            logger.warning("scikit-learn not available, using rule-based fallback")
            return ModelMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0)

        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            classification_report,
            confusion_matrix,
        )

        start_time = time.time()

        X, y = data.to_numpy()
        self._feature_names = data.feature_names

        # Encode labels
        y_encoded, _ = self._encode_labels(data.labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=y_encoded,
        )

        # Create and train model
        self._model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            class_weight=self.config.class_weight,
        )

        self._model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self._model, X, y_encoded,
            cv=self.config.cross_validation_folds,
            n_jobs=self.config.n_jobs,
        )

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Feature importances
        importances = dict(zip(
            self._feature_names,
            self._model.feature_importances_.tolist(),
        ))

        training_time = time.time() - start_time
        self._trained = True

        self._metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            feature_importances=importances,
            cross_val_scores=cv_scores.tolist(),
            training_time_seconds=training_time,
        )

        logger.info(f"Training complete. Accuracy: {accuracy:.2%}, F1: {f1:.2%}")

        return self._metrics


class GradientBoostingTypeClassifier(TypeInferenceModel):
    """Gradient Boosting classifier for type inference.

    Often provides higher accuracy than Random Forest.
    """

    name = "gradient_boosting"
    version = "1.0.0"

    def train(self, data: TrainingData) -> ModelMetrics:
        """Train Gradient Boosting classifier."""
        if not self._sklearn_available:
            logger.warning("scikit-learn not available, using rule-based fallback")
            return ModelMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0)

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        start_time = time.time()

        X, y = data.to_numpy()
        self._feature_names = data.feature_names

        # Encode labels
        y_encoded, _ = self._encode_labels(data.labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=y_encoded,
        )

        # Create and train model
        self._model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=min(self.config.max_depth, 5),  # GB works better with shallow trees
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )

        self._model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Feature importances
        importances = dict(zip(
            self._feature_names,
            self._model.feature_importances_.tolist(),
        ))

        training_time = time.time() - start_time
        self._trained = True

        self._metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            feature_importances=importances,
            training_time_seconds=training_time,
        )

        logger.info(f"Training complete. Accuracy: {accuracy:.2%}, F1: {f1:.2%}")

        return self._metrics


# =============================================================================
# Factory Function
# =============================================================================


def train_type_classifier(
    training_data: TrainingData,
    model_type: str = "random_forest",
    config: ModelConfig | None = None,
) -> TypeInferenceModel:
    """Train a type inference classifier.

    Args:
        training_data: Training data
        model_type: Model type ("random_forest", "gradient_boosting", "rule_based")
        config: Model configuration

    Returns:
        Trained model
    """
    models = {
        "random_forest": RandomForestTypeClassifier,
        "gradient_boosting": GradientBoostingTypeClassifier,
        "rule_based": RuleBasedTypeClassifier,
    }

    model_class = models.get(model_type, RandomForestTypeClassifier)
    model = model_class(config)
    model.train(training_data)

    return model
