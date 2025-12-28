"""Training pipeline for ReDoS ML models.

This module provides a comprehensive training pipeline that handles:
- Data preparation and validation
- Feature extraction
- Train/test splitting with stratification
- Cross-validation
- Hyperparameter tuning (optional)
- Model evaluation and comparison
- Result persistence

The pipeline is designed for reproducibility and extensibility,
supporting both quick experiments and production training runs.

Example:
    >>> from truthound.validators.security.redos.ml.training import (
    ...     TrainingPipeline,
    ...     TrainingConfig,
    ... )
    >>> pipeline = TrainingPipeline(config=TrainingConfig())
    >>> metrics = pipeline.train(training_data)
    >>> print(metrics.summary())
    >>> pipeline.save("model.joblib")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from truthound.validators.security.redos.ml.base import (
    BaseReDoSModel,
    ModelConfig,
    ModelType,
    PatternFeatures,
    ReDoSModelMetrics,
    ReDoSTrainingData,
)
from truthound.validators.security.redos.ml.features import PatternFeatureExtractor
from truthound.validators.security.redos.ml.models import (
    create_model,
    list_available_models,
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline.

    This configuration controls all aspects of the training process,
    from data preparation to model selection and evaluation.

    Attributes:
        model_type: Type of model to train
        model_config: Configuration for the model
        test_split: Fraction of data for final test evaluation
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
        auto_balance: Whether to automatically balance class weights
        feature_selection: Whether to perform feature selection
        max_features: Maximum features to select (if feature_selection=True)
        hyperparameter_tuning: Whether to perform hyperparameter search
        tuning_iterations: Number of hyperparameter combinations to try
        early_stopping: Whether to use early stopping
        min_samples: Minimum samples required for training
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    """

    model_type: ModelType = ModelType.RANDOM_FOREST
    model_config: ModelConfig = field(default_factory=ModelConfig.default)
    test_split: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    auto_balance: bool = True
    feature_selection: bool = False
    max_features: int = 20
    hyperparameter_tuning: bool = False
    tuning_iterations: int = 20
    early_stopping: bool = True
    min_samples: int = 10
    verbose: int = 1

    def __post_init__(self):
        # Sync model config with training config
        self.model_config = ModelConfig(
            model_type=self.model_type,
            random_state=self.random_state,
            cross_validation_folds=self.cv_folds,
            validation_split=self.test_split,
            feature_selection=self.feature_selection,
            max_features=self.max_features,
            class_weight="balanced" if self.auto_balance else None,
        )

    @classmethod
    def quick(cls) -> "TrainingConfig":
        """Configuration for quick training experiments."""
        return cls(
            model_type=ModelType.RANDOM_FOREST,
            model_config=ModelConfig.fast_training(),
            cv_folds=3,
            hyperparameter_tuning=False,
        )

    @classmethod
    def thorough(cls) -> "TrainingConfig":
        """Configuration for thorough training with tuning."""
        return cls(
            model_type=ModelType.GRADIENT_BOOSTING,
            model_config=ModelConfig.high_accuracy(),
            cv_folds=10,
            hyperparameter_tuning=True,
            tuning_iterations=50,
        )

    @classmethod
    def production(cls) -> "TrainingConfig":
        """Configuration for production model training."""
        return cls(
            model_type=ModelType.ENSEMBLE,
            cv_folds=10,
            hyperparameter_tuning=True,
            tuning_iterations=100,
            feature_selection=True,
        )


@dataclass
class CrossValidationResult:
    """Result of cross-validation evaluation.

    Attributes:
        fold_metrics: Metrics for each fold
        mean_accuracy: Mean accuracy across folds
        std_accuracy: Standard deviation of accuracy
        mean_f1: Mean F1 score across folds
        std_f1: Standard deviation of F1 score
        best_fold: Index of best performing fold
        worst_fold: Index of worst performing fold
    """

    fold_metrics: List[ReDoSModelMetrics]
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    best_fold: int
    worst_fold: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fold_metrics": [m.to_dict() for m in self.fold_metrics],
            "mean_accuracy": round(self.mean_accuracy, 4),
            "std_accuracy": round(self.std_accuracy, 4),
            "mean_f1": round(self.mean_f1, 4),
            "std_f1": round(self.std_f1, 4),
            "best_fold": self.best_fold,
            "worst_fold": self.worst_fold,
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        return (
            f"Cross-Validation Results ({len(self.fold_metrics)} folds):\n"
            f"  Accuracy: {self.mean_accuracy:.2%} +/- {self.std_accuracy:.2%}\n"
            f"  F1 Score: {self.mean_f1:.2%} +/- {self.std_f1:.2%}\n"
            f"  Best fold: {self.best_fold + 1}, Worst fold: {self.worst_fold + 1}"
        )


@dataclass
class TrainingResult:
    """Complete result of training pipeline execution.

    Attributes:
        model: Trained model
        metrics: Final evaluation metrics
        cv_result: Cross-validation results (if performed)
        feature_importances: Feature importance scores
        training_time_seconds: Total training time
        config: Configuration used for training
        trained_at: Timestamp of training completion
    """

    model: BaseReDoSModel
    metrics: ReDoSModelMetrics
    cv_result: Optional[CrossValidationResult] = None
    feature_importances: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    config: Optional[TrainingConfig] = None
    trained_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model.name,
            "model_version": self.model.version,
            "metrics": self.metrics.to_dict(),
            "cv_result": self.cv_result.to_dict() if self.cv_result else None,
            "feature_importances": {
                k: round(v, 4) for k, v in self.feature_importances.items()
            },
            "training_time_seconds": round(self.training_time_seconds, 3),
            "trained_at": self.trained_at.isoformat(),
        }

    def summary(self) -> str:
        """Get a human-readable summary."""
        lines = [
            f"Training Result ({self.model.name} v{self.model.version})",
            f"Trained at: {self.trained_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Training time: {self.training_time_seconds:.2f}s",
            "",
            "Metrics:",
            self.metrics.summary(),
        ]
        if self.cv_result:
            lines.extend(["", self.cv_result.summary()])
        if self.feature_importances:
            top_features = sorted(
                self.feature_importances.items(), key=lambda x: x[1], reverse=True
            )[:5]
            lines.extend(["", "Top 5 Features:"])
            for name, importance in top_features:
                lines.append(f"  {name}: {importance:.4f}")
        return "\n".join(lines)


class TrainingPipeline:
    """Complete training pipeline for ReDoS ML models.

    This pipeline handles the full training workflow:
    1. Data validation and preparation
    2. Feature extraction
    3. Model training with cross-validation
    4. Evaluation and metrics computation
    5. Model persistence

    The pipeline is designed to be reproducible and configurable,
    supporting both quick experiments and production training.

    Example:
        >>> pipeline = TrainingPipeline(config=TrainingConfig.quick())
        >>> result = pipeline.train(training_data)
        >>> print(result.summary())
        >>> pipeline.save("model.joblib")
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        feature_extractor: PatternFeatureExtractor | None = None,
    ):
        """Initialize the training pipeline.

        Args:
            config: Training configuration
            feature_extractor: Feature extractor instance
        """
        self.config = config or TrainingConfig()
        self.extractor = feature_extractor or PatternFeatureExtractor()
        self._model: Optional[BaseReDoSModel] = None
        self._result: Optional[TrainingResult] = None

    @property
    def model(self) -> Optional[BaseReDoSModel]:
        """Get the trained model."""
        return self._model

    @property
    def result(self) -> Optional[TrainingResult]:
        """Get the training result."""
        return self._result

    def train(self, data: ReDoSTrainingData) -> TrainingResult:
        """Run the complete training pipeline.

        Args:
            data: Training data containing patterns and labels

        Returns:
            TrainingResult with trained model and metrics

        Raises:
            ValueError: If data is invalid or insufficient
        """
        start_time = time.time()

        # Validate data
        self._validate_data(data)

        if self.config.verbose >= 1:
            logger.info(
                f"Starting training with {len(data)} samples "
                f"({data.num_vulnerable} vulnerable, {data.num_safe} safe)"
            )

        # Extract features if not provided
        if data.features is None:
            if self.config.verbose >= 1:
                logger.info("Extracting features from patterns...")
            features = self.extractor.extract_vectors(data.patterns)
            data = ReDoSTrainingData(
                patterns=data.patterns,
                labels=data.labels,
                features=features,
                feature_names=PatternFeatures.feature_names(),
                sample_weights=data.sample_weights,
                metadata=data.metadata,
            )

        # Create model
        self._model = create_model(self.config.model_type, self.config.model_config)

        # Train model
        if self.config.verbose >= 1:
            logger.info(f"Training {self._model.name} model...")

        metrics = self._model.train(data)

        # Cross-validation (if data is sufficient)
        cv_result = None
        if len(data) >= self.config.cv_folds * 2:
            if self.config.verbose >= 1:
                logger.info(f"Running {self.config.cv_folds}-fold cross-validation...")
            cv_result = self._cross_validate(data)

        # Get feature importances
        feature_importances = self._model.get_feature_importance_dict()

        training_time = time.time() - start_time

        self._result = TrainingResult(
            model=self._model,
            metrics=metrics,
            cv_result=cv_result,
            feature_importances=feature_importances,
            training_time_seconds=training_time,
            config=self.config,
        )

        if self.config.verbose >= 1:
            logger.info(f"Training complete in {training_time:.2f}s")
            logger.info(f"Accuracy: {metrics.accuracy:.2%}, F1: {metrics.f1_score:.2%}")

        return self._result

    def _validate_data(self, data: ReDoSTrainingData) -> None:
        """Validate training data.

        Args:
            data: Training data to validate

        Raises:
            ValueError: If data is invalid
        """
        if len(data) < self.config.min_samples:
            raise ValueError(
                f"Insufficient training data: {len(data)} samples "
                f"(minimum: {self.config.min_samples})"
            )

        if len(set(data.labels)) < 2:
            raise ValueError(
                "Training data must contain at least two classes "
                "(vulnerable and safe patterns)"
            )

        # Check class balance
        balance = data.class_balance
        if balance < 0.1 or balance > 0.9:
            logger.warning(
                f"Highly imbalanced data: {balance:.1%} vulnerable. "
                "Consider adding more samples from minority class."
            )

    def _cross_validate(self, data: ReDoSTrainingData) -> CrossValidationResult:
        """Perform cross-validation.

        Args:
            data: Training data

        Returns:
            CrossValidationResult with fold metrics
        """
        try:
            import numpy as np
            from sklearn.model_selection import StratifiedKFold
        except ImportError:
            logger.warning("sklearn not available, skipping cross-validation")
            return None

        X = np.array(data.features)
        y = np.array(data.labels)

        kfold = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        fold_metrics = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            # Create fold data
            fold_data = ReDoSTrainingData(
                patterns=[data.patterns[i] for i in train_idx],
                labels=[data.labels[i] for i in train_idx],
                features=[data.features[i] for i in train_idx],
                feature_names=data.feature_names,
            )

            # Train model for this fold
            fold_model = create_model(
                self.config.model_type, self.config.model_config
            )
            fold_model.train(fold_data)

            # Evaluate on validation set
            val_features = [data.features[i] for i in val_idx]
            val_labels = [data.labels[i] for i in val_idx]

            predictions = [fold_model.predict(f)[0] for f in val_features]
            predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

            # Calculate metrics
            tp = sum(1 for p, l in zip(predicted_labels, val_labels) if p == 1 and l == 1)
            tn = sum(1 for p, l in zip(predicted_labels, val_labels) if p == 0 and l == 0)
            fp = sum(1 for p, l in zip(predicted_labels, val_labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(predicted_labels, val_labels) if p == 0 and l == 1)

            accuracy = (tp + tn) / max(len(val_labels), 1)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            specificity = tn / max(tn + fp, 1)

            fold_metrics.append(
                ReDoSModelMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    specificity=specificity,
                    confusion_matrix=[[tn, fp], [fn, tp]],
                )
            )

            if self.config.verbose >= 2:
                logger.info(
                    f"  Fold {fold_idx + 1}: accuracy={accuracy:.2%}, f1={f1:.2%}"
                )

        # Calculate aggregate metrics
        accuracies = [m.accuracy for m in fold_metrics]
        f1_scores = [m.f1_score for m in fold_metrics]

        return CrossValidationResult(
            fold_metrics=fold_metrics,
            mean_accuracy=np.mean(accuracies),
            std_accuracy=np.std(accuracies),
            mean_f1=np.mean(f1_scores),
            std_f1=np.std(f1_scores),
            best_fold=int(np.argmax(f1_scores)),
            worst_fold=int(np.argmin(f1_scores)),
        )

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model

        Raises:
            ValueError: If no model has been trained
        """
        if self._model is None:
            raise ValueError("No model to save. Run train() first.")

        self._model.save(path)
        if self.config.verbose >= 1:
            logger.info(f"Model saved to: {path}")

    def load(self, path: str | Path) -> None:
        """Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        self._model = create_model(self.config.model_type, self.config.model_config)
        self._model.load(path)
        if self.config.verbose >= 1:
            logger.info(f"Model loaded from: {path}")

    def predict(self, pattern: str) -> Tuple[float, float]:
        """Predict risk for a single pattern.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            Tuple of (risk_probability, confidence)

        Raises:
            ValueError: If no model has been trained
        """
        if self._model is None:
            raise ValueError("No model available. Run train() or load() first.")

        features = self.extractor.extract(pattern).to_vector()
        return self._model.predict(features)


def compare_models(
    data: ReDoSTrainingData,
    model_types: Sequence[str] | None = None,
    config: TrainingConfig | None = None,
) -> Dict[str, TrainingResult]:
    """Compare multiple model types on the same data.

    This function trains and evaluates multiple model types,
    making it easy to select the best model for a given dataset.

    Args:
        data: Training data
        model_types: List of model types to compare (default: all available)
        config: Base training configuration

    Returns:
        Dictionary mapping model types to their training results
    """
    if model_types is None:
        model_types = list_available_models()

    config = config or TrainingConfig()
    results = {}

    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        try:
            model_config = TrainingConfig(
                model_type=ModelType(model_type),
                cv_folds=config.cv_folds,
                random_state=config.random_state,
                verbose=config.verbose,
            )
            pipeline = TrainingPipeline(config=model_config)
            result = pipeline.train(data)
            results[model_type] = result
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")

    # Log comparison
    logger.info("\nModel Comparison:")
    for name, result in sorted(
        results.items(), key=lambda x: x[1].metrics.f1_score, reverse=True
    ):
        logger.info(
            f"  {name}: accuracy={result.metrics.accuracy:.2%}, "
            f"f1={result.metrics.f1_score:.2%}, "
            f"time={result.training_time_seconds:.2f}s"
        )

    return results
