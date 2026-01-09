"""Base classes and core abstractions for ML-based validation.

This module provides the foundational abstractions for the ML system:
- MLModel: Abstract base class for all ML models
- ModelRegistry: Dynamic registration and management of ML models
- AnomalyDetector: Base class for anomaly detection models
- DriftDetector: Base class for drift detection models
- RuleLearner: Base class for rule learning models
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)
import threading
import json

import polars as pl

if TYPE_CHECKING:
    from truthound.validators.base import ValidationIssue


# =============================================================================
# Enums
# =============================================================================


class ModelType(str, Enum):
    """Types of ML models supported."""

    ANOMALY_DETECTOR = "anomaly_detector"
    DRIFT_DETECTOR = "drift_detector"
    RULE_LEARNER = "rule_learner"
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    CUSTOM = "custom"


class ModelState(str, Enum):
    """Lifecycle states for ML models."""

    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"


class AnomalyType(str, Enum):
    """Types of anomalies detected."""

    POINT = "point"  # Single data point anomaly
    CONTEXTUAL = "contextual"  # Anomaly in context
    COLLECTIVE = "collective"  # Group of data points
    PATTERN = "pattern"  # Pattern-based anomaly
    TREND = "trend"  # Trend deviation
    SEASONAL = "seasonal"  # Seasonal pattern violation


class SeverityLevel(str, Enum):
    """Severity levels for ML-detected issues."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Exceptions
# =============================================================================


class MLError(Exception):
    """Base exception for ML-related errors."""

    def __init__(self, message: str, model_name: str | None = None):
        self.model_name = model_name
        super().__init__(message)


class ModelNotTrainedError(MLError):
    """Raised when trying to use an untrained model."""

    pass


class ModelTrainingError(MLError):
    """Raised when model training fails."""

    pass


class ModelLoadError(MLError):
    """Raised when model loading fails."""

    pass


class InsufficientDataError(MLError):
    """Raised when there's not enough data for ML operations."""

    pass


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class MLConfig:
    """Base configuration for ML operations.

    Attributes:
        sample_size: Maximum samples to use for training
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        cache_predictions: Whether to cache prediction results
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
    """

    sample_size: int | None = None
    random_seed: int = 42
    n_jobs: int = 1
    cache_predictions: bool = True
    verbose: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyConfig(MLConfig):
    """Configuration for anomaly detection.

    Attributes:
        contamination: Expected proportion of outliers (0.0 to 0.5)
        sensitivity: Detection sensitivity (0.0 to 1.0)
        min_samples: Minimum samples required for detection
        window_size: Window size for temporal anomaly detection
        columns: Specific columns to analyze (None for all numeric columns)
    """

    contamination: float = 0.1
    sensitivity: float = 0.5
    min_samples: int = 100
    window_size: int | None = None
    score_threshold: float | None = None
    columns: list[str] | None = None


@dataclass
class DriftConfig(MLConfig):
    """Configuration for drift detection.

    Attributes:
        reference_window: Size of reference window
        detection_window: Size of detection window
        threshold: Drift detection threshold
        min_samples_per_window: Minimum samples per window
        n_bins: Number of bins for histogram-based methods (e.g., PSI)
        detect_gradual: Whether to detect gradual drift
        detect_sudden: Whether to detect sudden drift
    """

    reference_window: int = 1000
    detection_window: int = 100
    threshold: float = 0.05
    min_samples_per_window: int = 30
    n_bins: int = 10
    detect_gradual: bool = True
    detect_sudden: bool = True


@dataclass
class RuleLearningConfig(MLConfig):
    """Configuration for rule learning.

    Attributes:
        min_support: Minimum support for rules
        min_confidence: Minimum confidence for rules
        max_rules: Maximum number of rules to generate
        max_antecedent_length: Maximum length of rule antecedent
    """

    min_support: float = 0.1
    min_confidence: float = 0.8
    max_rules: int = 100
    max_antecedent_length: int = 3
    include_negations: bool = False


ConfigT = TypeVar("ConfigT", bound=MLConfig)


# =============================================================================
# Result Classes
# =============================================================================


@dataclass(frozen=True)
class AnomalyScore:
    """Score for a single data point or window.

    Attributes:
        index: Index or identifier of the data point
        score: Anomaly score (higher = more anomalous)
        is_anomaly: Whether classified as anomaly
        anomaly_type: Type of anomaly detected
        confidence: Confidence in the classification
        contributing_features: Features contributing to the anomaly
    """

    index: int | str
    score: float
    is_anomaly: bool
    anomaly_type: AnomalyType = AnomalyType.POINT
    confidence: float = 1.0
    contributing_features: tuple[str, ...] = field(default_factory=tuple)
    details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "score": round(self.score, 6),
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type.value,
            "confidence": round(self.confidence, 4),
            "contributing_features": list(self.contributing_features),
            "details": self.details,
        }


@dataclass(frozen=True)
class AnomalyResult:
    """Complete result of anomaly detection.

    Attributes:
        scores: Individual anomaly scores
        anomaly_count: Total number of anomalies detected
        anomaly_ratio: Ratio of anomalies to total points
        model_name: Name of the model used
        detection_time_ms: Time taken for detection in milliseconds
    """

    scores: tuple[AnomalyScore, ...] = field(default_factory=tuple)
    anomaly_count: int = 0
    anomaly_ratio: float = 0.0
    total_points: int = 0
    model_name: str = ""
    detection_time_ms: float = 0.0
    threshold_used: float | None = None
    detected_at: datetime = field(default_factory=datetime.now)

    def __iter__(self) -> Iterator[AnomalyScore]:
        return iter(self.scores)

    def get_anomalies(self) -> tuple[AnomalyScore, ...]:
        """Get only the anomalous scores."""
        return tuple(s for s in self.scores if s.is_anomaly)

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_count": self.anomaly_count,
            "anomaly_ratio": round(self.anomaly_ratio, 4),
            "total_points": self.total_points,
            "model_name": self.model_name,
            "detection_time_ms": round(self.detection_time_ms, 2),
            "threshold_used": self.threshold_used,
            "detected_at": self.detected_at.isoformat(),
            "anomalies": [s.to_dict() for s in self.get_anomalies()],
        }


@dataclass(frozen=True)
class DriftResult:
    """Result of drift detection analysis.

    Attributes:
        is_drifted: Whether drift was detected
        drift_score: Overall drift score
        column_scores: Per-column drift scores
        drift_type: Type of drift (gradual, sudden, etc.)
    """

    is_drifted: bool = False
    drift_score: float = 0.0
    column_scores: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    drift_type: str = "none"
    p_value: float | None = None
    confidence: float = 1.0
    details: str | None = None
    detected_at: datetime = field(default_factory=datetime.now)

    def get_drifted_columns(self, threshold: float = 0.5) -> list[str]:
        """Get columns with drift score above threshold."""
        return [col for col, score in self.column_scores if score >= threshold]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_drifted": self.is_drifted,
            "drift_score": round(self.drift_score, 6),
            "drift_type": self.drift_type,
            "p_value": round(self.p_value, 6) if self.p_value else None,
            "confidence": round(self.confidence, 4),
            "column_scores": {col: round(score, 6) for col, score in self.column_scores},
            "details": self.details,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass(frozen=True)
class LearnedRule:
    """A validation rule learned from data.

    Attributes:
        name: Rule name/identifier
        rule_type: Type of rule (e.g., 'range', 'pattern', 'constraint')
        column: Target column(s)
        condition: Rule condition expression
        support: Proportion of data supporting the rule
        confidence: Rule confidence
    """

    name: str
    rule_type: str
    column: str | tuple[str, ...]
    condition: str
    support: float
    confidence: float
    validator_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rule_type": self.rule_type,
            "column": self.column if isinstance(self.column, str) else list(self.column),
            "condition": self.condition,
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "validator_config": self.validator_config,
            "description": self.description,
        }

    def to_validator_spec(self) -> dict[str, Any]:
        """Convert to validator specification for use with Truthound."""
        return {
            "type": self.rule_type,
            "columns": [self.column] if isinstance(self.column, str) else list(self.column),
            **self.validator_config,
        }


@dataclass(frozen=True)
class RuleLearningResult:
    """Result of rule learning process.

    Attributes:
        rules: Learned validation rules
        data_profile: Profile of the data used for learning
        learning_time_ms: Time taken for learning
    """

    rules: tuple[LearnedRule, ...] = field(default_factory=tuple)
    total_rules: int = 0
    filtered_rules: int = 0  # Rules filtered by min_confidence/support
    learning_time_ms: float = 0.0
    data_profile: dict[str, Any] = field(default_factory=dict)
    learned_at: datetime = field(default_factory=datetime.now)

    def __iter__(self) -> Iterator[LearnedRule]:
        return iter(self.rules)

    def get_rules_by_type(self, rule_type: str) -> list[LearnedRule]:
        """Get rules of a specific type."""
        return [r for r in self.rules if r.rule_type == rule_type]

    def get_rules_for_column(self, column: str) -> list[LearnedRule]:
        """Get rules for a specific column."""
        return [r for r in self.rules if column in (
            [r.column] if isinstance(r.column, str) else list(r.column)
        )]

    def to_validation_suite(self) -> dict[str, Any]:
        """Convert to validation suite format."""
        return {
            "validators": [r.to_validator_spec() for r in self.rules],
            "generated_at": self.learned_at.isoformat(),
            "total_rules": self.total_rules,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rules": self.total_rules,
            "filtered_rules": self.filtered_rules,
            "learning_time_ms": round(self.learning_time_ms, 2),
            "rules": [r.to_dict() for r in self.rules],
            "data_profile": self.data_profile,
            "learned_at": self.learned_at.isoformat(),
        }


# =============================================================================
# Model Metadata
# =============================================================================


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about an ML model.

    Attributes:
        name: Unique model identifier
        version: Model version
        model_type: Type of model
        description: Human-readable description
        author: Model author
        created_at: Creation timestamp
        input_schema: Expected input schema
        output_schema: Output schema
    """

    name: str
    version: str
    model_type: ModelType
    description: str = ""
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    input_features: tuple[str, ...] = field(default_factory=tuple)
    supports_incremental: bool = False
    supports_online_learning: bool = False
    min_samples_required: int = 10
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type.value,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "input_features": list(self.input_features),
            "supports_incremental": self.supports_incremental,
            "supports_online_learning": self.supports_online_learning,
            "min_samples_required": self.min_samples_required,
            "tags": list(self.tags),
        }


# =============================================================================
# Base ML Model Class
# =============================================================================


class MLModel(ABC, Generic[ConfigT]):
    """Abstract base class for all ML models.

    This provides the foundational interface for ML models in Truthound.
    Subclasses must implement fit() and predict() methods.

    Example:
        class MyAnomalyModel(MLModel[AnomalyConfig]):
            @property
            def info(self) -> ModelInfo:
                return ModelInfo(
                    name="my-anomaly",
                    version="1.0.0",
                    model_type=ModelType.ANOMALY_DETECTOR,
                )

            def fit(self, data: pl.LazyFrame) -> None:
                # Train the model
                ...

            def predict(self, data: pl.LazyFrame) -> Any:
                # Make predictions
                ...
    """

    def __init__(self, config: ConfigT | None = None, **kwargs: Any):
        """Initialize the model.

        Args:
            config: Model configuration
            **kwargs: Additional parameters that override config
        """
        self._config: ConfigT = config or self._default_config()  # type: ignore
        self._state: ModelState = ModelState.UNTRAINED
        self._error: Exception | None = None
        self._trained_at: datetime | None = None
        self._training_samples: int = 0
        self._lock = threading.RLock()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                object.__setattr__(self._config, key, value)

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return model metadata.

        Returns:
            ModelInfo with model name, version, type, etc.
        """
        ...

    @abstractmethod
    def fit(self, data: pl.LazyFrame) -> None:
        """Train the model on data.

        Args:
            data: Training data as LazyFrame

        Raises:
            ModelTrainingError: If training fails
            InsufficientDataError: If not enough data
        """
        ...

    @abstractmethod
    def predict(self, data: pl.LazyFrame) -> Any:
        """Make predictions on new data.

        Args:
            data: Data to predict on

        Returns:
            Predictions (type depends on model)

        Raises:
            ModelNotTrainedError: If model not trained
        """
        ...

    def fit_predict(self, data: pl.LazyFrame) -> Any:
        """Train and predict in one step.

        Args:
            data: Data to train on and predict

        Returns:
            Predictions on the training data
        """
        self.fit(data)
        return self.predict(data)

    def partial_fit(self, data: pl.LazyFrame) -> None:
        """Incrementally update the model with new data.

        Override this for models that support online learning.

        Args:
            data: New data to learn from

        Raises:
            NotImplementedError: If not supported
        """
        if not self.info.supports_online_learning:
            raise NotImplementedError(
                f"{self.info.name} does not support online learning"
            )
        # Default implementation: just refit
        self.fit(data)

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model

        Raises:
            ModelNotTrainedError: If model not trained
        """
        if self._state not in (ModelState.TRAINED, ModelState.READY):
            raise ModelNotTrainedError(
                "Cannot save untrained model",
                model_name=self.info.name,
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = self._serialize()
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2, default=str)

    def load(self, path: str | Path) -> None:
        """Load the model from disk.

        Args:
            path: Path to load the model from

        Raises:
            ModelLoadError: If loading fails
        """
        path = Path(path)
        if not path.exists():
            raise ModelLoadError(
                f"Model file not found: {path}",
                model_name=self.info.name,
            )

        try:
            with open(path) as f:
                model_data = json.load(f)
            self._deserialize(model_data)
            self._state = ModelState.READY
        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelLoadError(
                f"Failed to load model: {e}",
                model_name=self.info.name,
            ) from e

    def _serialize(self) -> dict[str, Any]:
        """Serialize model state for saving.

        Override in subclasses to save model-specific state.
        """
        return {
            "info": self.info.to_dict(),
            "state": self._state.value,
            "trained_at": self._trained_at.isoformat() if self._trained_at else None,
            "training_samples": self._training_samples,
        }

    def _deserialize(self, data: dict[str, Any]) -> None:
        """Deserialize model state from saved data.

        Override in subclasses to restore model-specific state.
        """
        self._trained_at = (
            datetime.fromisoformat(data["trained_at"])
            if data.get("trained_at")
            else None
        )
        self._training_samples = data.get("training_samples", 0)

    def _default_config(self) -> MLConfig:
        """Return default configuration.

        Override in subclasses with specific config types.
        """
        return MLConfig()

    def _validate_data(self, data: pl.LazyFrame, min_samples: int | None = None) -> int:
        """Validate input data and return row count.

        Args:
            data: Data to validate
            min_samples: Minimum required samples

        Returns:
            Number of rows in data

        Raises:
            InsufficientDataError: If not enough data
        """
        row_count = data.select(pl.len()).collect().item()
        min_required = min_samples or self.info.min_samples_required

        if row_count < min_required:
            raise InsufficientDataError(
                f"Need at least {min_required} samples, got {row_count}",
                model_name=self.info.name,
            )

        return row_count

    def _maybe_sample(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Apply sampling if configured."""
        if self._config.sample_size is not None:
            return data.head(self._config.sample_size)
        return data

    @property
    def config(self) -> ConfigT:
        """Get model configuration."""
        return self._config

    @property
    def state(self) -> ModelState:
        """Get current model state."""
        return self._state

    @property
    def is_trained(self) -> bool:
        """Check if model is trained and ready."""
        return self._state in (ModelState.TRAINED, ModelState.READY)

    @property
    def error(self) -> Exception | None:
        """Get error if model is in error state."""
        return self._error

    @property
    def training_info(self) -> dict[str, Any]:
        """Get training information."""
        return {
            "trained_at": self._trained_at.isoformat() if self._trained_at else None,
            "training_samples": self._training_samples,
            "state": self._state.value,
        }

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.info.name!r} "
            f"state={self.state.value!r}>"
        )


# =============================================================================
# Specialized Base Classes
# =============================================================================


class AnomalyDetector(MLModel[AnomalyConfig]):
    """Abstract base class for anomaly detection models.

    Provides specialized interface for anomaly detection including
    score computation and threshold-based classification.
    """

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._get_model_name(),
            version=self._get_model_version(),
            model_type=ModelType.ANOMALY_DETECTOR,
            description=self._get_description(),
        )

    def _get_model_name(self) -> str:
        """Override to provide model name."""
        return self.__class__.__name__.lower().replace("detector", "")

    def _get_model_version(self) -> str:
        """Override to provide model version."""
        return "1.0.0"

    def _get_description(self) -> str:
        """Override to provide description."""
        return self.__class__.__doc__ or ""

    @abstractmethod
    def score(self, data: pl.LazyFrame) -> pl.Series:
        """Compute anomaly scores for data.

        Args:
            data: Data to score

        Returns:
            Series of anomaly scores (higher = more anomalous)
        """
        ...

    def predict(self, data: pl.LazyFrame) -> AnomalyResult:
        """Detect anomalies in data.

        Args:
            data: Data to analyze

        Returns:
            AnomalyResult with detected anomalies
        """
        import time
        start = time.perf_counter()

        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained before prediction",
                model_name=self.info.name,
            )

        scores = self.score(data)
        threshold = self._get_threshold()

        anomaly_scores = []
        for idx, score in enumerate(scores.to_list()):
            is_anomaly = score >= threshold
            anomaly_scores.append(
                AnomalyScore(
                    index=idx,
                    score=score,
                    is_anomaly=is_anomaly,
                    anomaly_type=AnomalyType.POINT,
                    confidence=min(1.0, score / threshold) if threshold > 0 else 1.0,
                )
            )

        anomaly_count = sum(1 for s in anomaly_scores if s.is_anomaly)
        total_points = len(anomaly_scores)

        elapsed = (time.perf_counter() - start) * 1000

        return AnomalyResult(
            scores=tuple(anomaly_scores),
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_count / total_points if total_points > 0 else 0.0,
            total_points=total_points,
            model_name=self.info.name,
            detection_time_ms=elapsed,
            threshold_used=threshold,
        )

    def _get_threshold(self) -> float:
        """Get the threshold for anomaly classification."""
        if self.config.score_threshold is not None:
            return self.config.score_threshold
        # Default: use contamination to determine threshold
        return 1.0 - self.config.contamination

    def _default_config(self) -> AnomalyConfig:
        return AnomalyConfig()


class MLDriftDetector(MLModel[DriftConfig]):
    """Abstract base class for ML-based drift detection.

    Extends the statistical drift detection in truthound.drift
    with ML-based approaches.
    """

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._get_model_name(),
            version=self._get_model_version(),
            model_type=ModelType.DRIFT_DETECTOR,
            description=self._get_description(),
            supports_incremental=True,
        )

    def _get_model_name(self) -> str:
        return self.__class__.__name__.lower().replace("detector", "")

    def _get_model_version(self) -> str:
        return "1.0.0"

    def _get_description(self) -> str:
        return self.__class__.__doc__ or ""

    @abstractmethod
    def detect(
        self,
        reference: pl.LazyFrame,
        current: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> DriftResult:
        """Detect drift between reference and current data.

        Args:
            reference: Reference (baseline) data
            current: Current data to compare
            columns: Specific columns to check (None = all)

        Returns:
            DriftResult with drift analysis
        """
        ...

    def predict(self, data: pl.LazyFrame) -> DriftResult:
        """Predict drift using stored reference data.

        Requires that fit() was called to store reference data.
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained with reference data first",
                model_name=self.info.name,
            )
        return self.detect(self._reference_data, data)

    def fit(self, data: pl.LazyFrame) -> None:
        """Store reference data for drift detection.

        Args:
            data: Reference data to store
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            self._reference_data = self._maybe_sample(data)
            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED
        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to store reference data: {e}",
                model_name=self.info.name,
            ) from e

    def _default_config(self) -> DriftConfig:
        return DriftConfig()


class RuleLearner(MLModel[RuleLearningConfig]):
    """Abstract base class for rule learning models.

    Learns validation rules from data characteristics.
    """

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._get_model_name(),
            version=self._get_model_version(),
            model_type=ModelType.RULE_LEARNER,
            description=self._get_description(),
        )

    def _get_model_name(self) -> str:
        return self.__class__.__name__.lower().replace("learner", "")

    def _get_model_version(self) -> str:
        return "1.0.0"

    def _get_description(self) -> str:
        return self.__class__.__doc__ or ""

    @abstractmethod
    def learn_rules(self, data: pl.LazyFrame) -> RuleLearningResult:
        """Learn validation rules from data.

        Args:
            data: Data to analyze

        Returns:
            RuleLearningResult with learned rules
        """
        ...

    def fit(self, data: pl.LazyFrame) -> None:
        """Learn rules from data (alias for learn_rules).

        Args:
            data: Training data
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        try:
            row_count = self._validate_data(data)
            self._learned_rules = self.learn_rules(data)
            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED
        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to learn rules: {e}",
                model_name=self.info.name,
            ) from e

    def predict(self, data: pl.LazyFrame) -> RuleLearningResult:
        """Return learned rules (rules don't make predictions per se)."""
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Model must be trained first",
                model_name=self.info.name,
            )
        return self._learned_rules

    def get_rules(self) -> tuple[LearnedRule, ...]:
        """Get learned rules."""
        if not self.is_trained:
            return tuple()
        return self._learned_rules.rules

    def _default_config(self) -> RuleLearningConfig:
        return RuleLearningConfig()


# =============================================================================
# Model Registry
# =============================================================================


class ModelRegistry:
    """Registry for ML model registration and discovery.

    Provides a centralized way to register and retrieve ML models.
    Thread-safe for concurrent access.

    Example:
        registry = ModelRegistry()
        registry.register(IsolationForestDetector)

        # Later
        model_cls = registry.get("isolation_forest")
        model = model_cls()
    """

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: dict[str, type[MLModel]] = {}
            cls._instance._by_type: dict[ModelType, dict[str, type[MLModel]]] = {}
            cls._instance._lock = threading.RLock()
            cls._instance._initialized = False
        return cls._instance

    def register(
        self,
        model_class: type[MLModel],
        name: str | None = None,
    ) -> None:
        """Register a model class.

        Args:
            model_class: Model class to register
            name: Optional name override
        """
        with self._lock:
            # Get name from class if not provided
            instance = model_class.__new__(model_class)
            instance._config = instance._default_config()
            model_name = name or instance.info.name
            model_type = instance.info.model_type

            self._models[model_name] = model_class

            if model_type not in self._by_type:
                self._by_type[model_type] = {}
            self._by_type[model_type][model_name] = model_class

    def unregister(self, name: str) -> None:
        """Unregister a model.

        Args:
            name: Model name to unregister
        """
        with self._lock:
            if name in self._models:
                model_class = self._models.pop(name)
                # Remove from type index
                for type_dict in self._by_type.values():
                    if name in type_dict:
                        del type_dict[name]

    def get(self, name: str) -> type[MLModel]:
        """Get a registered model class by name.

        Args:
            name: Model name

        Returns:
            Model class

        Raises:
            KeyError: If model not found
        """
        with self._lock:
            if name not in self._models:
                raise KeyError(
                    f"Model '{name}' not found. "
                    f"Available: {list(self._models.keys())}"
                )
            return self._models[name]

    def get_by_type(self, model_type: ModelType) -> dict[str, type[MLModel]]:
        """Get all models of a specific type.

        Args:
            model_type: Type of models to retrieve

        Returns:
            Dict of model name to model class
        """
        with self._lock:
            return dict(self._by_type.get(model_type, {}))

    def list_all(self) -> list[str]:
        """List all registered model names."""
        with self._lock:
            return list(self._models.keys())

    def list_by_type(self, model_type: ModelType) -> list[str]:
        """List model names of a specific type."""
        with self._lock:
            return list(self._by_type.get(model_type, {}).keys())

    def clear(self) -> None:
        """Clear all registered models."""
        with self._lock:
            self._models.clear()
            self._by_type.clear()
            self._initialized = False


# Global registry instance
model_registry = ModelRegistry()


def register_model(
    name: str | None = None,
) -> Callable[[type[MLModel]], type[MLModel]]:
    """Decorator to register a model class.

    Example:
        @register_model("my_detector")
        class MyAnomalyDetector(AnomalyDetector):
            ...
    """
    def decorator(cls: type[MLModel]) -> type[MLModel]:
        model_registry.register(cls, name)
        return cls
    return decorator


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class MLModelProtocol(Protocol):
    """Protocol for ML models (duck typing support)."""

    @property
    def info(self) -> ModelInfo: ...

    def fit(self, data: pl.LazyFrame) -> None: ...

    def predict(self, data: pl.LazyFrame) -> Any: ...

    @property
    def is_trained(self) -> bool: ...


@runtime_checkable
class AnomalyDetectorProtocol(Protocol):
    """Protocol for anomaly detectors."""

    def score(self, data: pl.LazyFrame) -> pl.Series: ...

    def predict(self, data: pl.LazyFrame) -> AnomalyResult: ...


@runtime_checkable
class DriftDetectorProtocol(Protocol):
    """Protocol for drift detectors."""

    def detect(
        self,
        reference: pl.LazyFrame,
        current: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> DriftResult: ...
