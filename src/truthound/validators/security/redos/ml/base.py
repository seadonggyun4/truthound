"""Base abstractions for ReDoS ML framework.

This module defines the core protocols, data classes, and base types used
throughout the ML framework. It follows the principle of dependency inversion
by defining abstractions that concrete implementations depend on.

Design Principles:
    - Protocol-based design for loose coupling
    - Immutable data classes for thread safety
    - Clear separation of concerns
    - Extensibility through composition
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from truthound.validators.security.redos.core import ReDoSRisk


# =============================================================================
# Enums
# =============================================================================


class ModelType(str, Enum):
    """Supported model types for ReDoS prediction."""

    RULE_BASED = "rule_based"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

    def __str__(self) -> str:
        return self.value


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class PatternFeatures:
    """Immutable container for extracted pattern features.

    This class holds all features extracted from a regex pattern for ML
    prediction. Features are categorized into structural, quantifier,
    dangerous pattern indicators, and complexity metrics.

    Attributes:
        length: Total length of the pattern string
        group_count: Total number of groups (capture + non-capture)
        capture_group_count: Number of capturing groups
        non_capture_group_count: Number of non-capturing groups
        max_nesting_depth: Maximum depth of nested parentheses
        alternation_count: Number of alternation operators (|)
        plus_count: Number of + quantifiers
        star_count: Number of * quantifiers
        question_count: Number of ? quantifiers (non-lookahead)
        bounded_quantifier_count: Number of {n,m} quantifiers
        unbounded_quantifier_count: Number of unbounded quantifiers (+, *, {n,})
        lazy_quantifier_count: Number of lazy quantifiers (+?, *?, etc.)
        possessive_quantifier_count: Number of possessive quantifiers
        quantifier_density: Ratio of quantifiers to pattern length
        nested_quantifier_count: Number of nested quantifier patterns
        adjacent_quantifier_count: Number of adjacent quantifiers
        quantified_alternation_count: Number of alternations with quantifiers
        quantified_backreference_count: Number of backreferences with quantifiers
        char_class_count: Number of character classes []
        negated_char_class_count: Number of negated character classes [^]
        dot_count: Number of dot metacharacters
        word_boundary_count: Number of word boundary assertions
        lookahead_count: Number of lookahead assertions
        lookbehind_count: Number of lookbehind assertions
        negative_lookaround_count: Number of negative lookaround assertions
        backreference_count: Number of backreferences
        max_backreference_index: Highest backreference index used
        start_anchor: Whether pattern starts with ^ or \\A
        end_anchor: Whether pattern ends with $ or \\Z
        anchored: Whether pattern is fully anchored (both ends)
        backtracking_potential: Estimated backtracking risk score (0-100)
        estimated_states: Estimated number of NFA states
    """

    # Structural features
    length: int = 0
    group_count: int = 0
    capture_group_count: int = 0
    non_capture_group_count: int = 0
    max_nesting_depth: int = 0
    alternation_count: int = 0

    # Quantifier features
    plus_count: int = 0
    star_count: int = 0
    question_count: int = 0
    bounded_quantifier_count: int = 0
    unbounded_quantifier_count: int = 0
    lazy_quantifier_count: int = 0
    possessive_quantifier_count: int = 0
    quantifier_density: float = 0.0

    # Dangerous pattern indicators
    nested_quantifier_count: int = 0
    adjacent_quantifier_count: int = 0
    quantified_alternation_count: int = 0
    quantified_backreference_count: int = 0

    # Character class features
    char_class_count: int = 0
    negated_char_class_count: int = 0
    dot_count: int = 0
    word_boundary_count: int = 0

    # Lookaround features
    lookahead_count: int = 0
    lookbehind_count: int = 0
    negative_lookaround_count: int = 0

    # Backreference features
    backreference_count: int = 0
    max_backreference_index: int = 0

    # Anchor features
    start_anchor: bool = False
    end_anchor: bool = False
    anchored: bool = False

    # Complexity metrics
    backtracking_potential: float = 0.0
    estimated_states: int = 0

    def to_vector(self) -> List[float]:
        """Convert features to a numeric vector for ML models.

        Returns:
            List of float values in consistent order.
        """
        return [
            float(self.length),
            float(self.group_count),
            float(self.capture_group_count),
            float(self.non_capture_group_count),
            float(self.max_nesting_depth),
            float(self.alternation_count),
            float(self.plus_count),
            float(self.star_count),
            float(self.question_count),
            float(self.bounded_quantifier_count),
            float(self.unbounded_quantifier_count),
            float(self.lazy_quantifier_count),
            float(self.possessive_quantifier_count),
            float(self.quantifier_density),
            float(self.nested_quantifier_count),
            float(self.adjacent_quantifier_count),
            float(self.quantified_alternation_count),
            float(self.quantified_backreference_count),
            float(self.char_class_count),
            float(self.negated_char_class_count),
            float(self.dot_count),
            float(self.word_boundary_count),
            float(self.lookahead_count),
            float(self.lookbehind_count),
            float(self.negative_lookaround_count),
            float(self.backreference_count),
            float(self.max_backreference_index),
            float(self.start_anchor),
            float(self.end_anchor),
            float(self.anchored),
            float(self.backtracking_potential),
            float(self.estimated_states),
        ]

    @classmethod
    def feature_names(cls) -> List[str]:
        """Get names of all features in vector order.

        Returns:
            List of feature names matching to_vector() order.
        """
        return [
            "length",
            "group_count",
            "capture_group_count",
            "non_capture_group_count",
            "max_nesting_depth",
            "alternation_count",
            "plus_count",
            "star_count",
            "question_count",
            "bounded_quantifier_count",
            "unbounded_quantifier_count",
            "lazy_quantifier_count",
            "possessive_quantifier_count",
            "quantifier_density",
            "nested_quantifier_count",
            "adjacent_quantifier_count",
            "quantified_alternation_count",
            "quantified_backreference_count",
            "char_class_count",
            "negated_char_class_count",
            "dot_count",
            "word_boundary_count",
            "lookahead_count",
            "lookbehind_count",
            "negative_lookaround_count",
            "backreference_count",
            "max_backreference_index",
            "start_anchor",
            "end_anchor",
            "anchored",
            "backtracking_potential",
            "estimated_states",
        ]

    @classmethod
    def num_features(cls) -> int:
        """Get the number of features."""
        return len(cls.feature_names())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {name: value for name, value in zip(self.feature_names(), self.to_vector())}


@dataclass
class ReDoSPrediction:
    """Result of ReDoS risk prediction.

    Attributes:
        pattern: The analyzed regex pattern
        features: Extracted feature values
        risk_probability: Probability of ReDoS vulnerability (0.0 to 1.0)
        risk_level: Categorical risk level based on probability
        confidence: Model confidence in prediction (0.0 to 1.0)
        contributing_factors: Top features influencing the prediction
        model_type: Type of model used for prediction
        model_version: Version of the model used
        inference_time_ms: Time taken for inference in milliseconds
    """

    pattern: str
    features: PatternFeatures
    risk_probability: float
    risk_level: ReDoSRisk
    confidence: float
    contributing_factors: List[Tuple[str, float]] = field(default_factory=list)
    model_type: str = ""
    model_version: str = ""
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern": self.pattern,
            "features": self.features.to_dict(),
            "risk_probability": round(self.risk_probability, 4),
            "risk_level": self.risk_level.name,
            "confidence": round(self.confidence, 4),
            "contributing_factors": [
                {"feature": name, "contribution": round(contrib, 4)}
                for name, contrib in self.contributing_factors
            ],
            "model_type": self.model_type,
            "model_version": self.model_version,
            "inference_time_ms": round(self.inference_time_ms, 3),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ReDoSTrainingData:
    """Container for training data.

    Attributes:
        patterns: List of regex patterns
        labels: Corresponding labels (0=safe, 1=vulnerable)
        features: Pre-extracted features (optional, can be computed)
        feature_names: Names of features in feature vectors
        sample_weights: Optional weights for samples
        metadata: Additional metadata about the dataset
    """

    patterns: List[str]
    labels: List[int]
    features: Optional[List[List[float]]] = None
    feature_names: List[str] = field(default_factory=list)
    sample_weights: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.patterns) != len(self.labels):
            raise ValueError(
                f"Number of patterns ({len(self.patterns)}) must match "
                f"number of labels ({len(self.labels)})"
            )
        if self.features is not None and len(self.features) != len(self.patterns):
            raise ValueError(
                f"Number of feature vectors ({len(self.features)}) must match "
                f"number of patterns ({len(self.patterns)})"
            )
        if not self.feature_names:
            self.feature_names = PatternFeatures.feature_names()

    def __len__(self) -> int:
        return len(self.patterns)

    @property
    def num_vulnerable(self) -> int:
        """Count of vulnerable patterns."""
        return sum(self.labels)

    @property
    def num_safe(self) -> int:
        """Count of safe patterns."""
        return len(self.labels) - sum(self.labels)

    @property
    def class_balance(self) -> float:
        """Ratio of vulnerable to total samples."""
        return self.num_vulnerable / len(self) if len(self) > 0 else 0.0


@dataclass
class ReDoSModelMetrics:
    """Model evaluation metrics.

    Attributes:
        accuracy: Overall classification accuracy
        precision: Precision for vulnerable class
        recall: Recall for vulnerable class (sensitivity)
        f1_score: F1 score (harmonic mean of precision and recall)
        specificity: True negative rate
        auc_roc: Area under ROC curve (if available)
        confusion_matrix: [[TN, FP], [FN, TP]]
        feature_importances: Feature importance scores (if available)
        cross_val_scores: Cross-validation scores (if available)
        training_samples: Number of training samples
        training_time_seconds: Time taken for training
        trained_at: Timestamp of training completion
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float = 0.0
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importances: Optional[Dict[str, float]] = None
    cross_val_scores: Optional[List[float]] = None
    training_samples: int = 0
    training_time_seconds: float = 0.0
    trained_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "specificity": round(self.specificity, 4),
            "training_samples": self.training_samples,
            "training_time_seconds": round(self.training_time_seconds, 3),
            "trained_at": self.trained_at.isoformat(),
        }
        if self.auc_roc is not None:
            result["auc_roc"] = round(self.auc_roc, 4)
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix
        if self.feature_importances is not None:
            result["feature_importances"] = {
                k: round(v, 4) for k, v in self.feature_importances.items()
            }
        if self.cross_val_scores is not None:
            result["cross_val_scores"] = [round(s, 4) for s in self.cross_val_scores]
        return result

    def summary(self) -> str:
        """Get a human-readable summary of metrics."""
        lines = [
            f"Accuracy:    {self.accuracy:.2%}",
            f"Precision:   {self.precision:.2%}",
            f"Recall:      {self.recall:.2%}",
            f"F1 Score:    {self.f1_score:.2%}",
            f"Specificity: {self.specificity:.2%}",
        ]
        if self.auc_roc is not None:
            lines.append(f"AUC-ROC:     {self.auc_roc:.4f}")
        if self.cross_val_scores is not None:
            mean_cv = sum(self.cross_val_scores) / len(self.cross_val_scores)
            lines.append(f"CV Mean:     {mean_cv:.2%}")
        return "\n".join(lines)


@dataclass
class ModelConfig:
    """Configuration for ReDoS ML models.

    This configuration class controls all aspects of model training and
    inference, from algorithm-specific hyperparameters to general training
    settings.

    Attributes:
        model_type: Type of model to use
        n_estimators: Number of estimators for ensemble methods
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples required to split an internal node
        min_samples_leaf: Minimum samples required at a leaf node
        learning_rate: Learning rate for gradient-based methods
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        class_weight: How to handle class imbalance
        feature_selection: Whether to perform feature selection
        max_features: Maximum number of features to use
        cross_validation_folds: Number of CV folds
        validation_split: Fraction of data for validation
        early_stopping: Whether to use early stopping
        model_version: Version string for the model
    """

    model_type: ModelType = ModelType.RANDOM_FOREST
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    learning_rate: float = 0.1
    random_state: int = 42
    n_jobs: int = -1
    class_weight: str = "balanced"
    feature_selection: bool = True
    max_features: int = 50
    cross_validation_folds: int = 5
    validation_split: float = 0.2
    early_stopping: bool = True
    model_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type.value,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "class_weight": self.class_weight,
            "feature_selection": self.feature_selection,
            "max_features": self.max_features,
            "cross_validation_folds": self.cross_validation_folds,
            "validation_split": self.validation_split,
            "early_stopping": self.early_stopping,
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        if "model_type" in data and isinstance(data["model_type"], str):
            data = dict(data)
            data["model_type"] = ModelType(data["model_type"])
        return cls(**data)

    @classmethod
    def default(cls) -> "ModelConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def fast_training(cls) -> "ModelConfig":
        """Configuration optimized for fast training."""
        return cls(
            n_estimators=50,
            max_depth=5,
            cross_validation_folds=3,
        )

    @classmethod
    def high_accuracy(cls) -> "ModelConfig":
        """Configuration optimized for high accuracy."""
        return cls(
            model_type=ModelType.GRADIENT_BOOSTING,
            n_estimators=200,
            max_depth=15,
            cross_validation_folds=10,
        )


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class FeatureExtractorProtocol(Protocol):
    """Protocol for feature extractors.

    Feature extractors are responsible for converting raw regex patterns
    into numeric feature vectors suitable for ML models.
    """

    def extract(self, pattern: str) -> PatternFeatures:
        """Extract features from a regex pattern.

        Args:
            pattern: Regex pattern string

        Returns:
            PatternFeatures instance containing all extracted features
        """
        ...

    def extract_batch(self, patterns: Sequence[str]) -> List[PatternFeatures]:
        """Extract features from multiple patterns.

        Args:
            patterns: Sequence of regex pattern strings

        Returns:
            List of PatternFeatures instances
        """
        ...


@runtime_checkable
class ReDoSModelProtocol(Protocol):
    """Protocol for ReDoS prediction models.

    This protocol defines the interface that all ReDoS ML models must
    implement, enabling polymorphic usage and easy swapping of models.
    """

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        ...

    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        ...

    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Predict risk probability and confidence.

        Args:
            features: Feature vector

        Returns:
            Tuple of (risk_probability, confidence)
        """
        ...

    def predict_batch(
        self, features: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple feature vectors.

        Args:
            features: List of feature vectors

        Returns:
            List of (risk_probability, confidence) tuples
        """
        ...

    def train(self, data: ReDoSTrainingData) -> ReDoSModelMetrics:
        """Train the model on labeled data.

        Args:
            data: Training data container

        Returns:
            Training metrics
        """
        ...

    def get_feature_importance(self) -> List[float]:
        """Get feature importance scores.

        Returns:
            List of importance scores for each feature
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        ...

    def load(self, path: str | Path) -> None:
        """Load model from disk.

        Args:
            path: Path to the saved model
        """
        ...


# =============================================================================
# Base Classes
# =============================================================================


class BaseReDoSModel(ABC):
    """Abstract base class for ReDoS ML models.

    This class provides common functionality shared by all model implementations,
    including configuration management, feature name tracking, and serialization
    utilities.

    Subclasses must implement:
        - predict(): Single sample prediction
        - predict_batch(): Batch prediction
        - train(): Model training
        - _save_model_data(): Model-specific save logic
        - _load_model_data(): Model-specific load logic
    """

    name: str = "base"
    version: str = "1.0.0"

    def __init__(self, config: ModelConfig | None = None):
        """Initialize the model.

        Args:
            config: Model configuration (uses default if None)
        """
        self._config = config or ModelConfig.default()
        self._trained = False
        self._metrics: Optional[ReDoSModelMetrics] = None
        self._feature_names: List[str] = PatternFeatures.feature_names()

    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self._trained

    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._config

    @property
    def metrics(self) -> Optional[ReDoSModelMetrics]:
        """Get training metrics if available."""
        return self._metrics

    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return self._feature_names

    @abstractmethod
    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Predict risk probability and confidence.

        Args:
            features: Feature vector

        Returns:
            Tuple of (risk_probability, confidence)
        """
        pass

    @abstractmethod
    def predict_batch(
        self, features: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple feature vectors."""
        pass

    @abstractmethod
    def train(self, data: ReDoSTrainingData) -> ReDoSModelMetrics:
        """Train the model on labeled data."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> List[float]:
        """Get feature importance scores."""
        pass

    @abstractmethod
    def _save_model_data(self) -> Dict[str, Any]:
        """Get model-specific data for saving.

        Returns:
            Dictionary of data to serialize
        """
        pass

    @abstractmethod
    def _load_model_data(self, data: Dict[str, Any]) -> None:
        """Load model-specific data.

        Args:
            data: Dictionary of serialized data
        """
        pass

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        import pickle

        path = Path(path)
        data = {
            "name": self.name,
            "version": self.version,
            "config": self._config.to_dict(),
            "trained": self._trained,
            "metrics": self._metrics.to_dict() if self._metrics else None,
            "feature_names": self._feature_names,
            "model_data": self._save_model_data(),
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load model from disk.

        Args:
            path: Path to the saved model
        """
        import pickle

        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._config = ModelConfig.from_dict(data["config"])
        self._trained = data["trained"]
        self._feature_names = data.get("feature_names", PatternFeatures.feature_names())

        if data.get("metrics"):
            # Reconstruct metrics from dict
            metrics_dict = data["metrics"]
            self._metrics = ReDoSModelMetrics(
                accuracy=metrics_dict["accuracy"],
                precision=metrics_dict["precision"],
                recall=metrics_dict["recall"],
                f1_score=metrics_dict["f1_score"],
                specificity=metrics_dict.get("specificity", 0.0),
                auc_roc=metrics_dict.get("auc_roc"),
                confusion_matrix=metrics_dict.get("confusion_matrix"),
                feature_importances=metrics_dict.get("feature_importances"),
                cross_val_scores=metrics_dict.get("cross_val_scores"),
                training_samples=metrics_dict.get("training_samples", 0),
                training_time_seconds=metrics_dict.get("training_time_seconds", 0.0),
            )

        self._load_model_data(data.get("model_data", {}))

    def get_feature_importance_dict(self) -> Dict[str, float]:
        """Get feature importance as a dictionary.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = self.get_feature_importance()
        return dict(zip(self._feature_names, importance))
