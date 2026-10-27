"""Machine Learning module for type inference and pattern detection.

This package provides ML-powered type inference with:
- Pre-trained models for common data types
- Scikit-learn based classifiers
- Transfer learning support
- Model persistence and versioning

Example:
    from truthound.profiler.ml import (
        TypeInferenceModel,
    NaiveBayesTypeClassifier,
    LogisticRegressionTypeClassifier,
    SVMTypeClassifier,
    NeuralNetworkTypeClassifier,
    EnsembleTypeClassifier,
        PretrainedModelManager,
        train_type_classifier,
    )

    # Use pre-trained model
    model = PretrainedModelManager.load_default()
    result = model.predict(column)

    # Train custom model
    model = train_type_classifier(training_data)
    model.save("custom_model.pkl")
"""

from truthound.profiler.ml.base import (
    ModelConfig,
    ModelMetrics,
    ModelProtocol,
    PredictionResult,
    TrainingData,
)
from truthound.profiler.ml.classifier import (
    EnsembleTypeClassifier,
    GradientBoostingTypeClassifier,
    LogisticRegressionTypeClassifier,
    NaiveBayesTypeClassifier,
    NeuralNetworkTypeClassifier,
    RandomForestTypeClassifier,
    RuleBasedTypeClassifier,
    SVMTypeClassifier,
    TypeInferenceModel,
    train_type_classifier,
)
from truthound.profiler.ml.feature_extraction import (
    FeatureExtractor,
    FeatureSet,
    extract_features,
)
from truthound.profiler.ml.pretrained import (
    PretrainedModel,
    PretrainedModelManager,
    get_default_model,
    list_available_models,
)

__all__ = [
    # Base types
    "ModelConfig",
    "TrainingData",
    "PredictionResult",
    "ModelMetrics",
    "ModelProtocol",
    # Feature extraction
    "FeatureExtractor",
    "FeatureSet",
    "extract_features",
    # Pre-trained
    "PretrainedModelManager",
    "PretrainedModel",
    "get_default_model",
    "list_available_models",
    # Classifiers
    "TypeInferenceModel",
    "NaiveBayesTypeClassifier",
    "LogisticRegressionTypeClassifier",
    "SVMTypeClassifier",
    "NeuralNetworkTypeClassifier",
    "EnsembleTypeClassifier",
    "RandomForestTypeClassifier",
    "GradientBoostingTypeClassifier",
    "RuleBasedTypeClassifier",
    "train_type_classifier",
]
