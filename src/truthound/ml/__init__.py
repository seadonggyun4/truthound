"""ML-based validation module for Truthound.

This module provides machine learning capabilities for data validation:
- Anomaly detection models
- Drift detection models
- Rule learning from data

Example:
    >>> from truthound import ml
    >>> detector = ml.IsolationForestDetector()
    >>> detector.fit(train_data)
    >>> result = detector.predict(test_data)
    >>> print(f"Found {result.anomaly_count} anomalies")
"""

from truthound.ml.base import (
    # Enums
    ModelType,
    ModelState,
    AnomalyType,
    SeverityLevel,
    # Exceptions
    MLError,
    ModelNotTrainedError,
    ModelTrainingError,
    ModelLoadError,
    InsufficientDataError,
    # Configurations
    MLConfig,
    AnomalyConfig,
    DriftConfig,
    RuleLearningConfig,
    # Results
    AnomalyScore,
    AnomalyResult,
    DriftResult,
    LearnedRule,
    RuleLearningResult,
    # Model info
    ModelInfo,
    # Base classes
    MLModel,
    AnomalyDetector,
    MLDriftDetector,
    RuleLearner,
    # Registry
    ModelRegistry,
    model_registry,
    register_model,
)

from truthound.ml.anomaly_models import (
    IsolationForestDetector,
    StatisticalAnomalyDetector,
    ZScoreAnomalyDetector,
    IQRAnomalyDetector,
    MADAnomalyDetector,
    EnsembleAnomalyDetector,
)

from truthound.ml.drift_detection import (
    DistributionDriftDetector,
    FeatureDriftDetector,
    ConceptDriftDetector,
    MultivariateDriftDetector,
)

from truthound.ml.rule_learning import (
    DataProfileRuleLearner,
    ConstraintMiner,
    PatternRuleLearner,
)

__all__ = [
    # Enums
    "ModelType",
    "ModelState",
    "AnomalyType",
    "SeverityLevel",
    # Exceptions
    "MLError",
    "ModelNotTrainedError",
    "ModelTrainingError",
    "ModelLoadError",
    "InsufficientDataError",
    # Configurations
    "MLConfig",
    "AnomalyConfig",
    "DriftConfig",
    "RuleLearningConfig",
    # Results
    "AnomalyScore",
    "AnomalyResult",
    "DriftResult",
    "LearnedRule",
    "RuleLearningResult",
    # Model info
    "ModelInfo",
    # Base classes
    "MLModel",
    "AnomalyDetector",
    "MLDriftDetector",
    "RuleLearner",
    # Registry
    "ModelRegistry",
    "model_registry",
    "register_model",
    # Anomaly detectors
    "IsolationForestDetector",
    "StatisticalAnomalyDetector",
    "ZScoreAnomalyDetector",
    "IQRAnomalyDetector",
    "MADAnomalyDetector",
    "EnsembleAnomalyDetector",
    # Drift detectors
    "DistributionDriftDetector",
    "FeatureDriftDetector",
    "ConceptDriftDetector",
    "MultivariateDriftDetector",
    # Rule learners
    "DataProfileRuleLearner",
    "ConstraintMiner",
    "PatternRuleLearner",
]
