"""Pre-trained model management for type inference.

Provides pre-trained models and training data generation for
common data types without external model files.
"""

from __future__ import annotations

import logging
import pickle
import random
import re
import string
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import polars as pl

from truthound.profiler.base import DataType
from truthound.profiler.ml.base import (
    BaseModel,
    ModelConfig,
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
        self._rng = random.Random(seed)
        self._domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org", "test.co.kr"]
        self._first_names = ["john", "jane", "kim", "lee", "park", "choi", "alex", "sam"]
        self._last_names = ["smith", "jones", "kim", "lee", "wang", "chen", "garcia"]

    def generate(self, data_type: DataType, n_samples: int = 100) -> list[str]:
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
        name = self._rng.choice(self._first_names) + self._rng.choice(self._last_names)
        suffix = self._rng.randint(1, 999) if self._rng.random() > 0.5 else ""
        domain = self._rng.choice(self._domains)
        return f"{name}{suffix}@{domain}"

    def _gen_phone(self) -> str:
        formats = [
            "+1-{}-{}-{}",
            "+44 {} {} {}",
            "+82-{}-{}-{}",
            "({}) {}-{}",
            "{}-{}-{}",
        ]
        fmt = self._rng.choice(formats)
        return fmt.format(
            self._rng.randint(100, 999),
            self._rng.randint(100, 999),
            self._rng.randint(1000, 9999),
        )

    def _gen_url(self) -> str:
        protocols = ["http://", "https://"]
        domains = ["example.com", "test.org", "sample.net", "demo.io"]
        paths = ["", "/page", "/api/v1", "/users/123", "/products"]
        return (
            f"{self._rng.choice(protocols)}www.{self._rng.choice(domains)}{self._rng.choice(paths)}"
        )

    def _gen_uuid(self) -> str:
        return str(UUID(int=self._rng.getrandbits(128), version=4))

    def _gen_date(self) -> str:
        base = datetime(2020, 1, 1)
        delta = timedelta(days=self._rng.randint(0, 1500))
        return (base + delta).strftime("%Y-%m-%d")

    def _gen_datetime(self) -> str:
        base = datetime(2020, 1, 1, 0, 0, 0)
        delta = timedelta(seconds=self._rng.randint(0, 86400 * 1500))
        formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M"]
        return (base + delta).strftime(self._rng.choice(formats))

    def _gen_integer(self) -> str:
        ranges = [(0, 100), (1, 1000), (1000, 999999), (-100, 100)]
        r = self._rng.choice(ranges)
        return str(self._rng.randint(r[0], r[1]))

    def _gen_float(self) -> str:
        value = self._rng.uniform(-1000, 1000)
        decimals = self._rng.choice([1, 2, 3, 4])
        return f"{value:.{decimals}f}"

    def _gen_boolean(self) -> str:
        options = [("true", "false"), ("True", "False"), ("1", "0"), ("yes", "no"), ("Y", "N")]
        pair = self._rng.choice(options)
        return self._rng.choice(pair)

    def _gen_currency(self) -> str:
        symbols = ["$", "", "", ""]
        amount = self._rng.uniform(0.01, 10000)
        return f"{self._rng.choice(symbols)}{amount:,.2f}"

    def _gen_percentage(self) -> str:
        formats = ["{:.1f}%", "{:.2f}%", "{}%"]
        value = self._rng.uniform(0, 100)
        if "{}" in (fmt := self._rng.choice(formats)):
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
        return self._rng.choice(self._rng.choice(categories))

    def _gen_identifier(self) -> str:
        formats = [
            "USR-{}",
            "ORD-{}",
            "TXN{}",
            "ID_{}",
            "{}",
        ]
        fmt = self._rng.choice(formats)
        return fmt.format(self._rng.randint(10000, 99999999))

    def _gen_string(self) -> str:
        word_count = self._rng.randint(1, 5)
        words = [
            "".join(self._rng.choices(string.ascii_lowercase, k=self._rng.randint(3, 10)))
            for _ in range(word_count)
        ]
        return " ".join(words).capitalize()

    def _gen_ip(self) -> str:
        return f"{self._rng.randint(1, 255)}.{self._rng.randint(0, 255)}.{self._rng.randint(0, 255)}.{self._rng.randint(1, 254)}"

    def _gen_korean_phone(self) -> str:
        prefixes = ["010", "011", "016", "017", "018", "019"]
        prefix = self._rng.choice(prefixes)
        return f"{prefix}-{self._rng.randint(1000, 9999)}-{self._rng.randint(1000, 9999)}"

    def _gen_korean_rrn(self) -> str:
        year = self._rng.randint(50, 99)
        month = self._rng.randint(1, 12)
        day = self._rng.randint(1, 28)
        gender = self._rng.choice([1, 2, 3, 4])
        return f"{year:02d}{month:02d}{day:02d}-{gender}{self._rng.randint(100000, 999999)}"

    def _gen_korean_business(self) -> str:
        return f"{self._rng.randint(100, 999)}-{self._rng.randint(10, 99)}-{self._rng.randint(10000, 99999)}"


# =============================================================================
# Pre-trained Model
# =============================================================================


@dataclass
class PretrainedModel:
    """Container for a pre-trained model."""

    name: str
    version: str
    model: BaseModel
    supported_types: list[DataType]
    feature_names: list[str]
    accuracy: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, features: FeatureSet) -> PredictionResult:
        """Predict type from features."""
        if len(features.names) != len(features.values) or len(set(features.names)) != len(
            features.names
        ):
            raise ValueError("Feature names and values must align")
        from truthound.profiler.ml.classifier import RuleBasedTypeClassifier

        if isinstance(self.model, RuleBasedTypeClassifier):
            from copy import copy

            model = copy(self.model)
            model._feature_names = list(features.names)
            return model.predict(features.values)
        mapping = dict(zip(features.names, features.values, strict=True))
        if set(mapping) != set(self.feature_names):
            raise ValueError("Features differ from the trained model schema")
        return self.model.predict([mapping[name] for name in self.feature_names])


class PretrainedModelManager:
    """Manage pre-trained models.

    Provides methods to load, save, and create pre-trained models.
    Includes a built-in rule-based model that works without training.
    """

    _instance: PretrainedModelManager | None = None
    _lock: threading.Lock = threading.Lock()

    # Default model cache directory
    DEFAULT_CACHE_DIR = Path.home() / ".truthound" / "models"

    def __new__(cls) -> PretrainedModelManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._models: dict[str, PretrainedModel] = {}
                    cls._instance._default_model: str | None = None
        return cls._instance

    def register(self, model: PretrainedModel, default: bool = False) -> None:
        """Register a pre-trained model."""
        self._models[model.name] = model
        if default or self._default_model is None:
            self._default_model = model.name

    def get(self, name: str | None = None) -> PretrainedModel | None:
        """Get a registered model."""
        if name is None:
            name = self._default_model
        return self._models.get(name) if name else None

    def list_models(self) -> list[str]:
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
            accuracy=0.0,  # No evaluation data has been supplied.
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
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", model_name):
            raise ValueError("Invalid model cache name")
        if type(n_samples_per_type) is not int or n_samples_per_type < 250:
            raise ValueError("At least 250 samples per type are required for five-fold training")
        manager = cls()

        # Check if already cached
        cache_path = cls.DEFAULT_CACHE_DIR / f"{model_name}.pkl"
        if cache and cache_path.exists():
            with open(cache_path, "rb") as stream:
                pretrained = pickle.load(stream)  # Operator-owned trusted native cache only.
            if (
                not isinstance(pretrained, PretrainedModel)
                or pretrained.name != model_name
                or not pretrained.model.is_trained
            ):
                raise ValueError("Invalid pretrained model cache")
            if pretrained.metadata.get("n_samples_per_type") != n_samples_per_type:
                raise ValueError("Cached training configuration differs; select a new model name")
            manager.register(pretrained)
            return pretrained

        # Generate training data
        logger.info(f"Generating training data ({n_samples_per_type} samples per type)...")
        generator = SyntheticDataGenerator()
        extractor = FeatureExtractor()

        feature_names: list[str] | None = None

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

        # Need to expand - extract features per sample
        all_features = []
        all_labels = []

        for dtype in types_to_train:
            samples = generator.generate(dtype, n_samples_per_type)

            for i in range(0, len(samples), 50):  # Batch process
                batch = samples[i : i + 50]
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
                "n_samples_per_type": n_samples_per_type,
                "training_metrics": metrics.to_dict(),
            },
        )

        # Cache if requested
        if cache:
            cls.DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            from truthound.stores._filesystem_io import publish

            publish(cache_path, pickle.dumps(pretrained))
            logger.info(f"Model cached to: {cache_path}")

        manager.register(pretrained)
        return pretrained


# =============================================================================
# Convenience Functions
# =============================================================================


def get_default_model() -> PretrainedModel:
    """Get the default pre-trained model."""
    return PretrainedModelManager.load_default()


def list_available_models() -> list[str]:
    """List available pre-trained models."""
    manager = PretrainedModelManager()
    return manager.list_models()
