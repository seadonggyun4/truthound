"""ML-based ReDoS Pattern Analysis Framework.

This package provides a modular, extensible ML framework for ReDoS vulnerability
detection. It supports multiple model backends, proper training pipelines, and
model persistence.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         ML ReDoS Framework                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
    ┌────────────────┬──────────────────┼──────────────────┬────────────────────┐
    │                │                  │                  │                    │
    ▼                ▼                  ▼                  ▼                    ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────┐
│   Feature   │ │   Model     │ │   Training   │ │    Model     │ │   Dataset   │
│  Extractor  │ │  Registry   │ │   Pipeline   │ │   Storage    │ │  Generator  │
└─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘ └─────────────┘

Components:
    - base: Core abstractions (protocols, data classes, base model)
    - features: Feature extraction from regex patterns
    - models: Concrete model implementations (sklearn, rule-based, ensemble)
    - training: Training pipeline with cross-validation and metrics
    - storage: Model serialization and versioning
    - datasets: Training data generation and management

Usage:
    from truthound.validators.security.redos.ml import (
        # High-level API
        ReDoSMLPredictor,
        train_redos_model,

        # Feature extraction
        PatternFeatureExtractor,
        PatternFeatures,

        # Model types
        RandomForestReDoSModel,
        GradientBoostingReDoSModel,
        EnsembleReDoSModel,

        # Training
        TrainingPipeline,
        TrainingConfig,

        # Datasets
        generate_training_dataset,
    )

    # Quick training
    model = train_redos_model(patterns, labels)
    result = model.predict("(a+)+b")

    # Full pipeline
    pipeline = TrainingPipeline(config=TrainingConfig())
    metrics = pipeline.train(dataset)
    pipeline.save("model.joblib")
"""

from __future__ import annotations

# Base abstractions
from truthound.validators.security.redos.ml.base import (
    # Protocols
    ReDoSModelProtocol,
    FeatureExtractorProtocol,
    # Data classes
    ReDoSTrainingData,
    ReDoSPrediction,
    ReDoSModelMetrics,
    ModelConfig,
    # Enums
    ModelType,
    # Base class
    BaseReDoSModel,
)

# Feature extraction
from truthound.validators.security.redos.ml.features import (
    PatternFeatureExtractor,
    PatternFeatures,
)

# Model implementations
from truthound.validators.security.redos.ml.models import (
    RuleBasedReDoSModel,
    RandomForestReDoSModel,
    GradientBoostingReDoSModel,
    EnsembleReDoSModel,
    create_model,
)

# Training pipeline
from truthound.validators.security.redos.ml.training import (
    TrainingPipeline,
    TrainingConfig,
    CrossValidationResult,
)

# Model storage
from truthound.validators.security.redos.ml.storage import (
    ModelStorage,
    ModelMetadata,
)

# Dataset generation
from truthound.validators.security.redos.ml.datasets import (
    generate_training_dataset,
    ReDoSDatasetGenerator,
    PatternLabel,
)

# High-level API
from truthound.validators.security.redos.ml.predictor import (
    ReDoSMLPredictor,
    train_redos_model,
    load_trained_model,
)

__all__ = [
    # Base
    "ReDoSModelProtocol",
    "FeatureExtractorProtocol",
    "ReDoSTrainingData",
    "ReDoSPrediction",
    "ReDoSModelMetrics",
    "ModelConfig",
    "ModelType",
    "BaseReDoSModel",
    # Features
    "PatternFeatureExtractor",
    "PatternFeatures",
    # Models
    "RuleBasedReDoSModel",
    "RandomForestReDoSModel",
    "GradientBoostingReDoSModel",
    "EnsembleReDoSModel",
    "create_model",
    # Training
    "TrainingPipeline",
    "TrainingConfig",
    "CrossValidationResult",
    # Storage
    "ModelStorage",
    "ModelMetadata",
    # Datasets
    "generate_training_dataset",
    "ReDoSDatasetGenerator",
    "PatternLabel",
    # High-level API
    "ReDoSMLPredictor",
    "train_redos_model",
    "load_trained_model",
]
