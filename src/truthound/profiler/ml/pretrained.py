"""Pre-trained model management for type inference.

Provides pre-trained models and training data generation for
common data types without external model files.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import re
import string
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import polars as pl

from truthound.profiler.base import DataType
from truthound.profiler.ml.base import (
    BaseModel,
    ModelConfig,
    ModelMetrics,
    ModelType,
    PredictionResult,
    TrainingData,
)
from truthound.profiler.ml.feature_extraction import FeatureExtractor, FeatureSet


logger = logging.getLogger(__name__)


# =============================================================================
# Synthetic Data Generator for Training
# =============================================================================


class SyntheticDataGenerator:
    """Generate synthetic training data for each data type.

    Creates realistic sample data for training type classifiers
    without requiring external datasets.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org", "test.co.kr"]
        self._first_names = ["john", "jane", "kim", "lee", "park", "choi", "alex", "sam"]
        self._last_names = ["smith", "jones", "kim", "lee", "wang", "chen", "garcia"]

    def generate(self, data_type: DataType, n_samples: int = 100) -> List[str]:
        """Generate samples for a data type.

        Args:
            data_type: Type to generate
            n_samples: Number of samples

        Returns:
            List of string samples
        """
        generators = {
            DataType.EMAIL: self._gen_email,
            DataType.PHONE: self._gen_phone,
            DataType.URL: self._gen_url,
            DataType.UUID: self._gen_uuid,
            DataType.DATE: self._gen_date,
            DataType.DATETIME: self._gen_datetime,
            DataType.INTEGER: self._gen_integer,
            DataType.FLOAT: self._gen_float,
            DataType.BOOLEAN: self._gen_boolean,
            DataType.CURRENCY: self._gen_currency,
            DataType.PERCENTAGE: self._gen_percentage,
            DataType.CATEGORICAL: self._gen_categorical,
            DataType.IDENTIFIER: self._gen_identifier,
            DataType.STRING: self._gen_string,
            DataType.IP_ADDRESS: self._gen_ip,
            DataType.KOREAN_PHONE: self._gen_korean_phone,
            DataType.KOREAN_RRN: self._gen_korean_rrn,
            DataType.KOREAN_BUSINESS_NUMBER: self._gen_korean_business,
        }

        generator = generators.get(data_type, self._gen_string)
        return [generator() for _ in range(n_samples)]

    def _gen_email(self) -> str:
        name = random.choice(self._first_names) + random.choice(self._last_names)
        suffix = random.randint(1, 999) if random.random() > 0.5 else ""
        domain = random.choice(self._domains)
        return f"{name}{suffix}@{domain}"

    def _gen_phone(self) -> str:
        formats = [
            "+1-{}-{}-{}",
            "+44 {} {} {}",
            "+82-{}-{}-{}",
            "({}) {}-{}",
            "{}-{}-{}",
        ]
        fmt = random.choice(formats)
        return fmt.format(
            random.randint(100, 999),
            random.randint(100, 999),
            random.randint(1000, 9999),
        )

    def _gen_url(self) -> str:
        protocols = ["http://", "https://"]
        domains = ["example.com", "test.org", "sample.net", "demo.io"]
        paths = ["", "/page", "/api/v1", "/users/123", "/products"]
        return f"{random.choice(protocols)}www.{random.choice(domains)}{random.choice(paths)}"

    def _gen_uuid(self) -> str:
        return str(uuid4())

    def _gen_date(self) -> str:
        base = datetime(2020, 1, 1)
        delta = timedelta(days=random.randint(0, 1500))
        return (base + delta).strftime("%Y-%m-%d")

    def _gen_datetime(self) -> str:
        base = datetime(2020, 1, 1, 0, 0, 0)
        delta = timedelta(seconds=random.randint(0, 86400 * 1500))
        formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M"]
        return (base + delta).strftime(random.choice(formats))

    def _gen_integer(self) -> str:
        ranges = [(0, 100), (1, 1000), (1000, 999999), (-100, 100)]
        r = random.choice(ranges)
        return str(random.randint(r[0], r[1]))

    def _gen_float(self) -> str:
        value = random.uniform(-1000, 1000)
        decimals = random.choice([1, 2, 3, 4])
        return f"{value:.{decimals}f}"

    def _gen_boolean(self) -> str:
        options = [("true", "false"), ("True", "False"), ("1", "0"), ("yes", "no"), ("Y", "N")]
        pair = random.choice(options)
        return random.choice(pair)

    def _gen_currency(self) -> str:
        symbols = ["$", "", "", ""]
        amount = random.uniform(0.01, 10000)
        return f"{random.choice(symbols)}{amount:,.2f}"

    def _gen_percentage(self) -> str:
        formats = ["{:.1f}%", "{:.2f}%", "{}%"]
        value = random.uniform(0, 100)
        if "{}" in (fmt := random.choice(formats)):
            return fmt.format(int(value))
        return fmt.format(value)

    def _gen_categorical(self) -> str:
        categories = [
            ["active", "inactive", "pending"],
            ["low", "medium", "high"],
            ["A", "B", "C", "D"],
            ["open", "closed", "in_progress"],
            ["approved", "rejected", "review"],
        ]
        return random.choice(random.choice(categories))

    def _gen_identifier(self) -> str:
        formats = [
            "USR-{}",
            "ORD-{}",
            "TXN{}",
            "ID_{}",
            "{}",
        ]
        fmt = random.choice(formats)
        return fmt.format(random.randint(10000, 99999999))

    def _gen_string(self) -> str:
        word_count = random.randint(1, 5)
        words = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(word_count)
        ]
        return ' '.join(words).capitalize()

    def _gen_ip(self) -> str:
        return f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    def _gen_korean_phone(self) -> str:
        prefixes = ["010", "011", "016", "017", "018", "019"]
        prefix = random.choice(prefixes)
        return f"{prefix}-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

    def _gen_korean_rrn(self) -> str:
        year = random.randint(50, 99)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        gender = random.choice([1, 2, 3, 4])
        return f"{year:02d}{month:02d}{day:02d}-{gender}{random.randint(100000, 999999)}"

    def _gen_korean_business(self) -> str:
        return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(10000, 99999)}"


# =============================================================================
# Pre-trained Model
# =============================================================================


@dataclass
class PretrainedModel:
    """Container for a pre-trained model."""

    name: str
    version: str
    model: BaseModel
    supported_types: List[DataType]
    feature_names: List[str]
    accuracy: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def predict(self, features: FeatureSet) -> PredictionResult:
        """Predict type from features."""
        return self.model.predict(features.values)


class PretrainedModelManager:
    """Manage pre-trained models.

    Provides methods to load, save, and create pre-trained models.
    Includes a built-in rule-based model that works without training.
    """

    _instance: Optional["PretrainedModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    # Default model cache directory
    DEFAULT_CACHE_DIR = Path.home() / ".truthound" / "models"

    def __new__(cls) -> "PretrainedModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models: Dict[str, PretrainedModel] = {}
                    cls._instance._default_model: Optional[str] = None
        return cls._instance

    def register(self, model: PretrainedModel, default: bool = False) -> None:
        """Register a pre-trained model."""
        self._models[model.name] = model
        if default or self._default_model is None:
            self._default_model = model.name

    def get(self, name: Optional[str] = None) -> Optional[PretrainedModel]:
        """Get a registered model."""
        if name is None:
            name = self._default_model
        return self._models.get(name) if name else None

    def list_models(self) -> List[str]:
        """List available models."""
        return list(self._models.keys())

    @classmethod
    def load_default(cls) -> PretrainedModel:
        """Load or create the default model.

        Returns a rule-based model that works without training.
        """
        manager = cls()

        if "default" in manager._models:
            return manager._models["default"]

        # Create rule-based model
        from truthound.profiler.ml.classifier import RuleBasedTypeClassifier

        model = RuleBasedTypeClassifier()
        pretrained = PretrainedModel(
            name="default",
            version="1.0.0",
            model=model,
            supported_types=list(DataType),
            feature_names=[],
            accuracy=0.85,  # Estimated accuracy
            metadata={"type": "rule_based"},
        )

        manager.register(pretrained, default=True)
        return pretrained

    @classmethod
    def train_and_cache(
        cls,
        model_name: str = "rf_v1",
        n_samples_per_type: int = 500,
        cache: bool = True,
    ) -> PretrainedModel:
        """Train a new model using synthetic data and optionally cache it.

        Args:
            model_name: Name for the model
            n_samples_per_type: Training samples per type
            cache: Whether to cache the model to disk

        Returns:
            Trained PretrainedModel
        """
        manager = cls()

        # Check if already cached
        cache_path = cls.DEFAULT_CACHE_DIR / f"{model_name}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    pretrained = pickle.load(f)
                    manager.register(pretrained)
                    logger.info(f"Loaded cached model: {model_name}")
                    return pretrained
            except Exception as e:
                logger.warning(f"Failed to load cached model: {e}")

        # Generate training data
        logger.info(f"Generating training data ({n_samples_per_type} samples per type)...")
        generator = SyntheticDataGenerator()
        extractor = FeatureExtractor()

        features_list: List[List[float]] = []
        labels_list: List[DataType] = []
        feature_names: Optional[List[str]] = None

        types_to_train = [
            DataType.EMAIL,
            DataType.PHONE,
            DataType.URL,
            DataType.UUID,
            DataType.DATE,
            DataType.DATETIME,
            DataType.INTEGER,
            DataType.FLOAT,
            DataType.BOOLEAN,
            DataType.CURRENCY,
            DataType.PERCENTAGE,
            DataType.CATEGORICAL,
            DataType.IDENTIFIER,
            DataType.STRING,
            DataType.IP_ADDRESS,
        ]

        for dtype in types_to_train:
            samples = generator.generate(dtype, n_samples_per_type)

            # Create Polars series
            series = pl.Series("sample", samples)

            # Extract features
            feature_set = extractor.extract(series, {"column_name": f"{dtype.value}_column"})

            if feature_names is None:
                feature_names = feature_set.names

            features_list.append(feature_set.values)
            labels_list.extend([dtype] * 1)  # One aggregated sample per batch

        # Need to expand - extract features per sample
        all_features = []
        all_labels = []

        for dtype in types_to_train:
            samples = generator.generate(dtype, n_samples_per_type)

            for i in range(0, len(samples), 50):  # Batch process
                batch = samples[i:i + 50]
                if len(batch) < 10:
                    continue

                series = pl.Series("sample", batch)
                feature_set = extractor.extract(series, {"column_name": f"{dtype.value}_col"})

                if feature_names is None:
                    feature_names = feature_set.names

                all_features.append(feature_set.values)
                all_labels.append(dtype)

        training_data = TrainingData(
            features=all_features,
            labels=all_labels,
            feature_names=feature_names or [],
        )

        # Train model
        logger.info(f"Training RandomForest classifier with {len(all_features)} samples...")
        from truthound.profiler.ml.classifier import RandomForestTypeClassifier

        config = ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            n_estimators=100,
            max_depth=15,
        )
        model = RandomForestTypeClassifier(config)
        metrics = model.train(training_data)

        logger.info(f"Training complete. Accuracy: {metrics.accuracy:.2%}")

        pretrained = PretrainedModel(
            name=model_name,
            version="1.0.0",
            model=model,
            supported_types=types_to_train,
            feature_names=feature_names or [],
            accuracy=metrics.accuracy,
            metadata={
                "n_samples": len(all_features),
                "training_metrics": metrics.to_dict(),
            },
        )

        # Cache if requested
        if cache:
            cls.DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(pretrained, f)
            logger.info(f"Model cached to: {cache_path}")

        manager.register(pretrained)
        return pretrained


# =============================================================================
# Convenience Functions
# =============================================================================


def get_default_model() -> PretrainedModel:
    """Get the default pre-trained model."""
    return PretrainedModelManager.load_default()


def list_available_models() -> List[str]:
    """List available pre-trained models."""
    manager = PretrainedModelManager()
    return manager.list_models()
