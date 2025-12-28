"""High-level API for ReDoS ML prediction.

This module provides a simple, user-friendly interface for ReDoS
vulnerability prediction using trained ML models.

The ReDoSMLPredictor class wraps all the complexity of feature extraction,
model management, and prediction into a clean API.

Example:
    >>> from truthound.validators.security.redos.ml import (
    ...     ReDoSMLPredictor,
    ...     train_redos_model,
    ... )
    >>>
    >>> # Quick start with pre-trained model
    >>> predictor = ReDoSMLPredictor()
    >>> result = predictor.predict("(a+)+b")
    >>> print(result.risk_level)  # ReDoSRisk.CRITICAL
    >>>
    >>> # Train custom model
    >>> predictor = train_redos_model(patterns, labels)
    >>> predictor.save("model.pkl")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from truthound.validators.security.redos.core import ReDoSRisk
from truthound.validators.security.redos.ml.base import (
    BaseReDoSModel,
    ModelConfig,
    ModelType,
    PatternFeatures,
    ReDoSModelMetrics,
    ReDoSPrediction,
    ReDoSTrainingData,
)
from truthound.validators.security.redos.ml.datasets import (
    ReDoSDatasetGenerator,
    generate_training_dataset,
)
from truthound.validators.security.redos.ml.features import PatternFeatureExtractor
from truthound.validators.security.redos.ml.models import (
    EnsembleReDoSModel,
    create_model,
)
from truthound.validators.security.redos.ml.storage import (
    ModelStorage,
    load_model,
    save_model,
)
from truthound.validators.security.redos.ml.training import (
    TrainingConfig,
    TrainingPipeline,
    TrainingResult,
)


logger = logging.getLogger(__name__)


# Risk level thresholds
RISK_THRESHOLDS: Dict[ReDoSRisk, float] = {
    ReDoSRisk.NONE: 0.1,
    ReDoSRisk.LOW: 0.3,
    ReDoSRisk.MEDIUM: 0.5,
    ReDoSRisk.HIGH: 0.7,
    ReDoSRisk.CRITICAL: 0.85,
}


class ReDoSMLPredictor:
    """High-level predictor for ReDoS vulnerability detection.

    This class provides a simple, unified interface for predicting ReDoS
    vulnerabilities in regex patterns using ML models. It handles:
    - Feature extraction
    - Model prediction
    - Result formatting
    - Batch processing
    - Model persistence

    The predictor can be used with trained models or in "rule-based" mode
    without training for quick prototyping.

    Example:
        >>> # Using default rule-based model
        >>> predictor = ReDoSMLPredictor()
        >>> result = predictor.predict("(a+)+b")
        >>> print(f"Risk: {result.risk_level.name}")
        >>> print(f"Probability: {result.risk_probability:.2%}")
        >>>
        >>> # Using trained model
        >>> predictor = ReDoSMLPredictor.from_trained("model.pkl")
        >>> results = predictor.predict_batch(["(a+)+", "^[a-z]+$"])
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        model: BaseReDoSModel | None = None,
        feature_extractor: PatternFeatureExtractor | None = None,
        risk_thresholds: Dict[ReDoSRisk, float] | None = None,
    ):
        """Initialize the predictor.

        Args:
            model: Trained model to use (defaults to EnsembleModel)
            feature_extractor: Feature extractor (defaults to PatternFeatureExtractor)
            risk_thresholds: Custom risk level thresholds
        """
        self.extractor = feature_extractor or PatternFeatureExtractor()
        self._model: BaseReDoSModel = model or EnsembleReDoSModel()
        self._thresholds = risk_thresholds or RISK_THRESHOLDS

    @classmethod
    def from_trained(cls, path: str | Path) -> "ReDoSMLPredictor":
        """Create a predictor from a saved model.

        Args:
            path: Path to the saved model file

        Returns:
            ReDoSMLPredictor with loaded model
        """
        model = load_model(path)
        return cls(model=model)

    @classmethod
    def from_storage(
        cls,
        storage: ModelStorage,
        name: str,
        version: Optional[str] = None,
    ) -> "ReDoSMLPredictor":
        """Create a predictor from a model in storage.

        Args:
            storage: ModelStorage instance
            name: Model name
            version: Model version (None for latest)

        Returns:
            ReDoSMLPredictor with loaded model
        """
        model, metadata = storage.load(name, version)
        return cls(model=model)

    @property
    def model(self) -> BaseReDoSModel:
        """Get the underlying model."""
        return self._model

    @property
    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self._model.is_trained

    def predict(self, pattern: str) -> ReDoSPrediction:
        """Predict ReDoS risk for a pattern.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            ReDoSPrediction with detailed results
        """
        start_time = time.time()

        # Extract features
        features = self.extractor.extract(pattern)
        feature_vector = features.to_vector()

        # Get prediction
        if isinstance(self._model, EnsembleReDoSModel):
            probability, confidence = self._model.predict(feature_vector, pattern)
        else:
            probability, confidence = self._model.predict(feature_vector)

        # Determine risk level
        risk_level = self._probability_to_risk(probability)

        # Get contributing factors
        contributing_factors = self._get_contributing_factors(features)

        inference_time = (time.time() - start_time) * 1000

        return ReDoSPrediction(
            pattern=pattern,
            features=features,
            risk_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            model_type=self._model.name,
            model_version=self._model.version,
            inference_time_ms=inference_time,
        )

    def predict_batch(self, patterns: Sequence[str]) -> List[ReDoSPrediction]:
        """Predict risk for multiple patterns.

        Args:
            patterns: Sequence of patterns to analyze

        Returns:
            List of ReDoSPrediction objects
        """
        return [self.predict(pattern) for pattern in patterns]

    def predict_risk_only(self, pattern: str) -> Tuple[float, ReDoSRisk]:
        """Quick prediction returning only risk probability and level.

        This is a lightweight alternative to predict() when you don't
        need the full prediction details.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            Tuple of (probability, risk_level)
        """
        features = self.extractor.extract(pattern)
        feature_vector = features.to_vector()

        if isinstance(self._model, EnsembleReDoSModel):
            probability, _ = self._model.predict(feature_vector, pattern)
        else:
            probability, _ = self._model.predict(feature_vector)

        return probability, self._probability_to_risk(probability)

    def is_safe(self, pattern: str, threshold: float = 0.5) -> bool:
        """Quick check if a pattern is safe.

        Args:
            pattern: Regex pattern to check
            threshold: Maximum acceptable risk probability

        Returns:
            True if pattern risk is below threshold
        """
        probability, _ = self.predict_risk_only(pattern)
        return probability < threshold

    def is_vulnerable(self, pattern: str, threshold: float = 0.5) -> bool:
        """Quick check if a pattern is vulnerable.

        Args:
            pattern: Regex pattern to check
            threshold: Minimum probability to consider vulnerable

        Returns:
            True if pattern risk is at or above threshold
        """
        return not self.is_safe(pattern, threshold)

    def train(
        self,
        patterns: Sequence[str],
        labels: Sequence[int],
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Train the predictor on labeled data.

        This method trains the underlying model and updates the predictor
        to use the newly trained model.

        Args:
            patterns: Training patterns
            labels: Labels (0=safe, 1=vulnerable)
            config: Training configuration

        Returns:
            TrainingResult with metrics
        """
        data = ReDoSTrainingData(
            patterns=list(patterns),
            labels=list(labels),
        )

        pipeline = TrainingPipeline(config=config)
        result = pipeline.train(data)

        self._model = result.model
        return result

    def train_from_dataset(
        self,
        dataset: ReDoSTrainingData,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Train from a ReDoSTrainingData dataset.

        Args:
            dataset: Training dataset
            config: Training configuration

        Returns:
            TrainingResult with metrics
        """
        pipeline = TrainingPipeline(config=config)
        result = pipeline.train(dataset)

        self._model = result.model
        return result

    def auto_train(
        self,
        n_samples: int = 500,
        config: TrainingConfig | None = None,
    ) -> TrainingResult:
        """Automatically train using generated dataset.

        This method generates a training dataset using built-in patterns
        and trains the model automatically.

        Args:
            n_samples: Number of training samples to generate
            config: Training configuration

        Returns:
            TrainingResult with metrics
        """
        dataset = generate_training_dataset(n_samples=n_samples)
        return self.train_from_dataset(dataset, config)

    def save(self, path: str | Path) -> None:
        """Save the trained model.

        Args:
            path: Path to save the model
        """
        save_model(self._model, path)

    def load(self, path: str | Path) -> None:
        """Load a trained model.

        Args:
            path: Path to the saved model
        """
        self._model = load_model(path)

    def get_metrics(self) -> Optional[ReDoSModelMetrics]:
        """Get training metrics if available."""
        return self._model.metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self._model.get_feature_importance_dict()

    def _probability_to_risk(self, probability: float) -> ReDoSRisk:
        """Convert probability to risk level."""
        if probability >= self._thresholds[ReDoSRisk.CRITICAL]:
            return ReDoSRisk.CRITICAL
        elif probability >= self._thresholds[ReDoSRisk.HIGH]:
            return ReDoSRisk.HIGH
        elif probability >= self._thresholds[ReDoSRisk.MEDIUM]:
            return ReDoSRisk.MEDIUM
        elif probability >= self._thresholds[ReDoSRisk.LOW]:
            return ReDoSRisk.LOW
        else:
            return ReDoSRisk.NONE

    def _get_contributing_factors(
        self, features: PatternFeatures
    ) -> List[Tuple[str, float]]:
        """Get features that contribute most to the prediction.

        Args:
            features: Extracted pattern features

        Returns:
            List of (feature_name, contribution) tuples
        """
        importance = self._model.get_feature_importance()
        feature_values = features.to_vector()
        feature_names = PatternFeatures.feature_names()

        contributions: List[Tuple[str, float]] = []
        for name, imp, value in zip(feature_names, importance, feature_values):
            contribution = imp * value
            if contribution > 0:
                contributions.append((name, contribution))

        # Sort by contribution (descending)
        contributions.sort(key=lambda x: x[1], reverse=True)

        # Return top 5
        return contributions[:5]


# =============================================================================
# Convenience Functions
# =============================================================================


def train_redos_model(
    patterns: Sequence[str],
    labels: Sequence[int],
    model_type: str | ModelType = ModelType.RANDOM_FOREST,
    config: TrainingConfig | None = None,
) -> ReDoSMLPredictor:
    """Train a ReDoS prediction model.

    This is a convenience function for quick model training.

    Args:
        patterns: Training patterns
        labels: Labels (0=safe, 1=vulnerable)
        model_type: Type of model to train
        config: Training configuration

    Returns:
        Trained ReDoSMLPredictor
    """
    if config is None:
        config = TrainingConfig(
            model_type=ModelType(model_type) if isinstance(model_type, str) else model_type
        )

    predictor = ReDoSMLPredictor()
    predictor.train(patterns, labels, config)
    return predictor


def load_trained_model(path: str | Path) -> ReDoSMLPredictor:
    """Load a trained model from disk.

    Args:
        path: Path to the saved model

    Returns:
        ReDoSMLPredictor with loaded model
    """
    return ReDoSMLPredictor.from_trained(path)


def quick_predict(pattern: str) -> ReDoSPrediction:
    """Quickly predict ReDoS risk for a pattern.

    Uses a default predictor without training. For better accuracy,
    use a trained model.

    Args:
        pattern: Regex pattern to analyze

    Returns:
        ReDoSPrediction
    """
    predictor = ReDoSMLPredictor()
    return predictor.predict(pattern)


def batch_predict(patterns: Sequence[str]) -> List[ReDoSPrediction]:
    """Predict risk for multiple patterns.

    Uses a default predictor without training.

    Args:
        patterns: Patterns to analyze

    Returns:
        List of predictions
    """
    predictor = ReDoSMLPredictor()
    return predictor.predict_batch(patterns)


# Module-level singleton for convenience
_default_predictor: Optional[ReDoSMLPredictor] = None


def get_default_predictor() -> ReDoSMLPredictor:
    """Get the default predictor singleton.

    Returns:
        Shared ReDoSMLPredictor instance
    """
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = ReDoSMLPredictor()
    return _default_predictor


def predict_redos_risk_ml(pattern: str) -> ReDoSPrediction:
    """Predict ReDoS risk using ML.

    Uses the default predictor singleton for efficiency across
    multiple calls.

    Args:
        pattern: Regex pattern to analyze

    Returns:
        ReDoSPrediction with detailed results
    """
    return get_default_predictor().predict(pattern)
