"""Base types and protocols for ML module.

Defines core abstractions for machine learning based type inference.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from truthound.profiler.base import DataType

# =============================================================================
# Configuration
# =============================================================================


class ModelType(str, Enum):
    """Supported model types."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NAIVE_BAYES = "naive_bayes"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    model_type: ModelType = ModelType.RANDOM_FOREST
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1
    class_weight: str = "balanced"
    # Feature selection
    feature_selection: bool = True
    max_features: int = 50
    # Training options
    cross_validation_folds: int = 5
    early_stopping: bool = True
    validation_size: float = 0.2
    # Model persistence
    model_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "class_weight": self.class_weight,
            "feature_selection": self.feature_selection,
            "max_features": self.max_features,
            "cross_validation_folds": self.cross_validation_folds,
            "validation_size": self.validation_size,
            "early_stopping": self.early_stopping,
            "model_version": self.model_version,
            "metadata": self.metadata,
        }


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TrainingData:
    """Container for training data."""

    features: list[list[float]]
    labels: list[DataType]
    feature_names: list[str]
    sample_ids: list[str] | None = None
    weights: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.labels)

    def to_numpy(self) -> tuple[Any, Any]:
        """Convert to numpy arrays."""
        import numpy as np

        X = np.asarray(self.features, dtype=float)
        if X.ndim != 2 or not len(X) or X.shape != (len(self.labels), len(self.feature_names)):
            raise ValueError("Training rows, labels and feature names must align")
        if not np.isfinite(X).all() or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Features must be finite and names unique")
        if self.sample_ids is not None and len(self.sample_ids) != len(X):
            raise ValueError("Sample IDs must align with rows")
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if (
                weights.shape != (len(X),)
                or not np.isfinite(weights).all()
                or (weights < 0).any()
                or not weights.sum() > 0
            ):
                raise ValueError("Weights must be finite, nonnegative and aligned")
        y = np.array([DataType(d).value for d in self.labels])
        return X, y


@dataclass
class PredictionResult:
    """Result of type prediction."""

    predicted_type: DataType
    confidence: float
    probabilities: dict[DataType, float]
    top_alternatives: list[tuple[DataType, float]]
    features_used: list[str] = field(default_factory=list)
    model_version: str = ""
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_type": self.predicted_type.value,
            "confidence": self.confidence,
            "probabilities": {k.value: v for k, v in self.probabilities.items()},
            "top_alternatives": [
                {"type": t.value, "confidence": c} for t, c in self.top_alternatives
            ],
            "features_used": self.features_used,
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
        }


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]] | None = None
    class_report: dict[str, dict[str, float]] | None = None
    feature_importances: dict[str, float] | None = None
    cross_val_scores: list[float] | None = None
    training_time_seconds: float = 0.0
    trained_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
            "class_report": self.class_report,
            "feature_importances": self.feature_importances,
            "cross_val_scores": self.cross_val_scores,
            "training_time_seconds": self.training_time_seconds,
            "trained_at": self.trained_at.isoformat(),
        }


# =============================================================================
# Model Protocol
# =============================================================================


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for ML models."""

    def predict(self, features: list[float]) -> PredictionResult:
        """Predict type for a single sample."""
        ...

    def predict_batch(self, features: list[list[float]]) -> list[PredictionResult]:
        """Predict types for multiple samples."""
        ...

    def train(self, data: TrainingData) -> ModelMetrics:
        """Train the model."""
        ...

    def save(self, path: str) -> None:
        """Save model to file."""
        ...

    def load(self, path: str) -> None:
        """Load model from file."""
        ...


class BaseModel(ABC):
    """Abstract base class for ML models."""

    name: str = "base"
    version: str = "1.0.0"

    def __init__(self, config: ModelConfig | None = None):
        self.config = deepcopy(config) if config is not None else ModelConfig()
        self.version = self.config.model_version
        self._model: Any = None
        self._label_encoder: Any = None
        self._feature_names: list[str] = []
        self._trained = False
        self._metrics: ModelMetrics | None = None

    @property
    def is_trained(self) -> bool:
        return self._trained

    @abstractmethod
    def predict(self, features: list[float]) -> PredictionResult:
        """Predict type for a single sample."""
        pass

    @abstractmethod
    def predict_batch(self, features: list[list[float]]) -> list[PredictionResult]:
        """Predict types for multiple samples."""
        pass

    @abstractmethod
    def train(self, data: TrainingData) -> ModelMetrics:
        """Train the model."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from file."""
        pass

    def get_metrics(self) -> ModelMetrics | None:
        """Get training metrics."""
        return self._metrics

    def get_feature_importances(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self._metrics and self._metrics.feature_importances is not None:
            return dict(self._metrics.feature_importances)
        if hasattr(self._model, "feature_importances_") and self._feature_names:
            importances = self._model.feature_importances_
            return dict(zip(self._feature_names, importances, strict=True))
        return {}
