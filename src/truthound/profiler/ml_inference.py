"""ML-based type inference beyond pattern matching.

This module provides machine learning based type inference that considers:
- Column context (name, position, neighboring columns)
- Value distribution patterns
- Semantic relationships
- Historical learning from user feedback

Key features:
- Pluggable model architecture
- Feature extraction pipeline
- Online learning support
- Confidence calibration

Example:
    from truthound.profiler.ml_inference import (
        MLTypeInferrer,
        ContextFeatureExtractor,
        create_inference_model,
    )

    # Create inferrer with default model
    inferrer = MLTypeInferrer()

    # Infer type with context
    result = inferrer.infer(column, context={
        "column_name": "email_address",
        "table_name": "users",
        "sample_values": ["a@b.com", "c@d.org"],
    })

    print(f"Type: {result.inferred_type}, Confidence: {result.confidence:.2%}")
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import polars as pl

from truthound.profiler.base import DataType, ColumnProfile


logger = logging.getLogger(__name__)


# =============================================================================
# Feature Types
# =============================================================================


class FeatureType(str, Enum):
    """Types of features for ML inference."""

    NAME_BASED = "name_based"
    VALUE_BASED = "value_based"
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    PATTERN_BASED = "pattern_based"


@dataclass
class Feature:
    """Single feature for ML model."""

    name: str
    value: float
    feature_type: FeatureType
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Vector of features for a column."""

    column_name: str
    features: list[Feature]
    raw_values: dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> list[float]:
        """Convert to numeric array for ML model."""
        return [f.value for f in self.features]

    def to_dict(self) -> dict[str, float]:
        """Convert to named dictionary."""
        return {f.name: f.value for f in self.features}

    def get_feature(self, name: str) -> Feature | None:
        """Get feature by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None


# =============================================================================
# Inference Result
# =============================================================================


@dataclass
class InferenceResult:
    """Result of ML type inference."""

    column_name: str
    inferred_type: DataType
    confidence: float
    alternatives: list[tuple[DataType, float]] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    features_used: list[str] = field(default_factory=list)
    model_version: str = "1.0"
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "inferred_type": self.inferred_type.value,
            "confidence": self.confidence,
            "alternatives": [
                {"type": t.value, "confidence": c}
                for t, c in self.alternatives
            ],
            "reasoning": self.reasoning,
            "features_used": self.features_used,
            "model_version": self.model_version,
            "inference_time_ms": self.inference_time_ms,
        }


# =============================================================================
# Feature Extractor Protocol
# =============================================================================


class FeatureExtractor(ABC):
    """Abstract base for feature extractors."""

    name: str = "base"
    feature_type: FeatureType = FeatureType.VALUE_BASED

    @abstractmethod
    def extract(
        self,
        column: pl.Series,
        context: dict[str, Any],
    ) -> list[Feature]:
        """Extract features from column.

        Args:
            column: Column data
            context: Additional context (column name, table info, etc.)

        Returns:
            List of extracted features
        """
        pass


class NameFeatureExtractor(FeatureExtractor):
    """Extract features from column names.

    Uses keyword matching and embedding similarity.
    """

    name = "name_features"
    feature_type = FeatureType.NAME_BASED

    # Keywords associated with each type
    TYPE_KEYWORDS: dict[DataType, list[str]] = {
        DataType.EMAIL: ["email", "mail", "e_mail", "correo"],
        DataType.PHONE: ["phone", "tel", "mobile", "cell", "fax", "telephone"],
        DataType.URL: ["url", "link", "href", "website", "uri", "endpoint"],
        DataType.UUID: ["uuid", "guid", "id", "identifier", "uid"],
        DataType.DATE: ["date", "day", "birth", "created", "updated", "modified"],
        DataType.DATETIME: ["datetime", "timestamp", "time", "at", "when"],
        DataType.INTEGER: ["count", "num", "qty", "quantity", "amount", "total", "id"],
        DataType.FLOAT: ["price", "rate", "ratio", "percent", "score", "value"],
        DataType.BOOLEAN: ["is_", "has_", "flag", "active", "enabled", "valid"],
        DataType.CURRENCY: ["price", "cost", "amount", "fee", "payment", "salary"],
        DataType.PERCENTAGE: ["percent", "pct", "ratio", "rate"],
        DataType.KOREAN_PHONE: ["phone", "hp", "tel", "mobile", "연락처", "전화"],
        DataType.KOREAN_RRN: ["rrn", "resident", "주민", "jumin"],
        DataType.KOREAN_BUSINESS_NUMBER: ["business", "사업자", "brn"],
        DataType.CATEGORICAL: ["type", "status", "category", "class", "kind", "level"],
        DataType.IDENTIFIER: ["id", "key", "code", "no", "number"],
    }

    def extract(
        self,
        column: pl.Series,
        context: dict[str, Any],
    ) -> list[Feature]:
        features = []
        col_name = context.get("column_name", column.name or "").lower()

        # Clean column name
        clean_name = re.sub(r"[^a-z0-9_]", "_", col_name)
        tokens = [t for t in clean_name.split("_") if t]

        # Check each type's keywords
        for dtype, keywords in self.TYPE_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in col_name:
                    score += 1.0
                elif any(keyword in token for token in tokens):
                    score += 0.5

            if score > 0:
                features.append(Feature(
                    name=f"name_match_{dtype.value}",
                    value=min(1.0, score / len(keywords)),
                    feature_type=self.feature_type,
                ))

        # Add general name features
        features.append(Feature(
            name="name_length",
            value=min(1.0, len(col_name) / 50),
            feature_type=self.feature_type,
        ))

        features.append(Feature(
            name="name_has_underscore",
            value=1.0 if "_" in col_name else 0.0,
            feature_type=self.feature_type,
        ))

        features.append(Feature(
            name="name_has_number",
            value=1.0 if any(c.isdigit() for c in col_name) else 0.0,
            feature_type=self.feature_type,
        ))

        return features


class ValueFeatureExtractor(FeatureExtractor):
    """Extract features from actual values."""

    name = "value_features"
    feature_type = FeatureType.VALUE_BASED

    def extract(
        self,
        column: pl.Series,
        context: dict[str, Any],
    ) -> list[Feature]:
        features = []

        # Sample values for analysis
        sample_size = min(1000, len(column))
        sample = column.drop_nulls().head(sample_size)

        if len(sample) == 0:
            return [Feature(
                name="all_null",
                value=1.0,
                feature_type=self.feature_type,
            )]

        # String analysis
        if column.dtype == pl.Utf8:
            str_features = self._extract_string_features(sample)
            features.extend(str_features)

        # Numeric analysis
        elif column.dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            num_features = self._extract_numeric_features(sample)
            features.extend(num_features)

        # Boolean
        elif column.dtype == pl.Boolean:
            features.append(Feature(
                name="is_boolean",
                value=1.0,
                feature_type=self.feature_type,
            ))

        # General features
        features.append(Feature(
            name="null_ratio",
            value=column.null_count() / len(column) if len(column) > 0 else 0,
            feature_type=self.feature_type,
        ))

        features.append(Feature(
            name="unique_ratio",
            value=column.n_unique() / len(column) if len(column) > 0 else 0,
            feature_type=self.feature_type,
        ))

        return features

    def _extract_string_features(self, sample: pl.Series) -> list[Feature]:
        """Extract features from string values."""
        features = []

        # Length statistics
        lengths = sample.str.len_chars()
        avg_len = lengths.mean() or 0
        std_len = lengths.std() or 0

        features.append(Feature(
            name="avg_string_length",
            value=min(1.0, avg_len / 100),
            feature_type=self.feature_type,
        ))

        features.append(Feature(
            name="length_variance",
            value=min(1.0, std_len / avg_len) if avg_len > 0 else 0,
            feature_type=self.feature_type,
        ))

        # Character type ratios
        sample_str = sample.to_list()[:100]  # Limit for performance

        has_at = sum(1 for s in sample_str if "@" in str(s)) / len(sample_str)
        has_dot = sum(1 for s in sample_str if "." in str(s)) / len(sample_str)
        has_slash = sum(1 for s in sample_str if "/" in str(s)) / len(sample_str)
        has_dash = sum(1 for s in sample_str if "-" in str(s)) / len(sample_str)
        has_colon = sum(1 for s in sample_str if ":" in str(s)) / len(sample_str)

        features.extend([
            Feature(name="has_at_sign", value=has_at, feature_type=self.feature_type),
            Feature(name="has_dot", value=has_dot, feature_type=self.feature_type),
            Feature(name="has_slash", value=has_slash, feature_type=self.feature_type),
            Feature(name="has_dash", value=has_dash, feature_type=self.feature_type),
            Feature(name="has_colon", value=has_colon, feature_type=self.feature_type),
        ])

        # Digit ratio
        digit_ratios = []
        for s in sample_str:
            s = str(s)
            if len(s) > 0:
                digit_ratios.append(sum(c.isdigit() for c in s) / len(s))

        avg_digit_ratio = sum(digit_ratios) / len(digit_ratios) if digit_ratios else 0
        features.append(Feature(
            name="digit_ratio",
            value=avg_digit_ratio,
            feature_type=self.feature_type,
        ))

        # Check for common patterns
        email_pattern = sum(1 for s in sample_str if re.match(r"^[^@]+@[^@]+\.[^@]+$", str(s))) / len(sample_str)
        uuid_pattern = sum(1 for s in sample_str if re.match(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
            str(s)
        )) / len(sample_str)

        features.extend([
            Feature(name="email_pattern_ratio", value=email_pattern, feature_type=self.feature_type),
            Feature(name="uuid_pattern_ratio", value=uuid_pattern, feature_type=self.feature_type),
        ])

        return features

    def _extract_numeric_features(self, sample: pl.Series) -> list[Feature]:
        """Extract features from numeric values."""
        features = []

        # Basic stats
        min_val = sample.min() or 0
        max_val = sample.max() or 0
        mean_val = sample.mean() or 0
        std_val = sample.std() or 0

        # Range features
        range_val = max_val - min_val
        features.append(Feature(
            name="numeric_range_log",
            value=math.log10(range_val + 1) / 10,  # Normalize
            feature_type=self.feature_type,
        ))

        # Check if values look like IDs (sequential integers)
        if sample.dtype in [pl.Int32, pl.Int64]:
            sorted_sample = sample.sort()
            diffs = sorted_sample.diff().drop_nulls()
            is_sequential = (diffs == 1).mean() if len(diffs) > 0 else 0
            features.append(Feature(
                name="is_sequential",
                value=is_sequential or 0,
                feature_type=self.feature_type,
            ))

        # Check for percentage-like values (0-100 or 0-1)
        in_0_1 = ((sample >= 0) & (sample <= 1)).mean()
        in_0_100 = ((sample >= 0) & (sample <= 100)).mean()

        features.extend([
            Feature(name="in_0_1_range", value=in_0_1 or 0, feature_type=self.feature_type),
            Feature(name="in_0_100_range", value=in_0_100 or 0, feature_type=self.feature_type),
        ])

        # Check for currency-like (2 decimal places)
        if sample.dtype in [pl.Float32, pl.Float64]:
            decimal_places = []
            for v in sample.head(100).to_list():
                if v is not None:
                    s = f"{v:.10f}".rstrip("0")
                    if "." in s:
                        decimal_places.append(len(s.split(".")[1]))

            if decimal_places:
                avg_decimals = sum(decimal_places) / len(decimal_places)
                is_currency_like = 1.0 if 1.5 <= avg_decimals <= 2.5 else 0.0
                features.append(Feature(
                    name="is_currency_like",
                    value=is_currency_like,
                    feature_type=self.feature_type,
                ))

        return features


class StatisticalFeatureExtractor(FeatureExtractor):
    """Extract statistical distribution features."""

    name = "statistical_features"
    feature_type = FeatureType.STATISTICAL

    def extract(
        self,
        column: pl.Series,
        context: dict[str, Any],
    ) -> list[Feature]:
        features = []

        non_null = column.drop_nulls()
        if len(non_null) == 0:
            return features

        # Cardinality
        n_unique = non_null.n_unique()
        n_total = len(non_null)
        cardinality = n_unique / n_total if n_total > 0 else 0

        features.append(Feature(
            name="cardinality",
            value=cardinality,
            feature_type=self.feature_type,
        ))

        # Is it low cardinality (categorical)?
        is_categorical = 1.0 if n_unique < 20 and cardinality < 0.05 else 0.0
        features.append(Feature(
            name="is_categorical",
            value=is_categorical,
            feature_type=self.feature_type,
        ))

        # Is it high cardinality (identifier)?
        is_identifier = 1.0 if cardinality > 0.95 else 0.0
        features.append(Feature(
            name="is_identifier",
            value=is_identifier,
            feature_type=self.feature_type,
        ))

        # Value frequency distribution
        value_counts = non_null.value_counts()
        if len(value_counts) > 0:
            counts = value_counts.get_column("count").to_list()
            max_freq = max(counts) / n_total
            features.append(Feature(
                name="max_frequency",
                value=max_freq,
                feature_type=self.feature_type,
            ))

            # Entropy
            probs = [c / n_total for c in counts]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(n_unique) if n_unique > 1 else 1
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

            features.append(Feature(
                name="normalized_entropy",
                value=normalized_entropy,
                feature_type=self.feature_type,
            ))

        return features


class ContextFeatureExtractor(FeatureExtractor):
    """Extract features from column context."""

    name = "context_features"
    feature_type = FeatureType.CONTEXTUAL

    def extract(
        self,
        column: pl.Series,
        context: dict[str, Any],
    ) -> list[Feature]:
        features = []

        # Table-level context
        table_name = context.get("table_name", "").lower()
        if table_name:
            # Check if table name gives hints
            if any(kw in table_name for kw in ["user", "customer", "member"]):
                features.append(Feature(
                    name="table_is_user_related",
                    value=1.0,
                    feature_type=self.feature_type,
                ))

            if any(kw in table_name for kw in ["order", "transaction", "payment"]):
                features.append(Feature(
                    name="table_is_transaction_related",
                    value=1.0,
                    feature_type=self.feature_type,
                ))

        # Column position
        col_index = context.get("column_index", 0)
        total_cols = context.get("total_columns", 1)
        position_ratio = col_index / total_cols if total_cols > 0 else 0

        features.append(Feature(
            name="column_position",
            value=position_ratio,
            feature_type=self.feature_type,
        ))

        # First column is often ID
        if col_index == 0:
            features.append(Feature(
                name="is_first_column",
                value=1.0,
                feature_type=self.feature_type,
            ))

        # Neighboring columns
        neighbor_names = context.get("neighbor_columns", [])
        for name in neighbor_names:
            name = name.lower()
            if "email" in name:
                features.append(Feature(
                    name="neighbor_has_email",
                    value=1.0,
                    feature_type=self.feature_type,
                ))
            if "name" in name:
                features.append(Feature(
                    name="neighbor_has_name",
                    value=1.0,
                    feature_type=self.feature_type,
                ))

        return features


# =============================================================================
# Feature Extractor Registry
# =============================================================================


class FeatureExtractorRegistry:
    """Registry for feature extractors."""

    def __init__(self) -> None:
        self._extractors: dict[str, FeatureExtractor] = {}

    def register(self, extractor: FeatureExtractor) -> None:
        """Register an extractor."""
        self._extractors[extractor.name] = extractor

    def get(self, name: str) -> FeatureExtractor:
        """Get extractor by name."""
        if name not in self._extractors:
            raise KeyError(f"Unknown extractor: {name}")
        return self._extractors[name]

    def list_extractors(self) -> list[str]:
        """List registered extractors."""
        return list(self._extractors.keys())

    def extract_all(
        self,
        column: pl.Series,
        context: dict[str, Any],
    ) -> FeatureVector:
        """Extract features using all registered extractors."""
        all_features = []
        for extractor in self._extractors.values():
            try:
                features = extractor.extract(column, context)
                all_features.extend(features)
            except Exception as e:
                logger.warning(f"Extractor {extractor.name} failed: {e}")

        return FeatureVector(
            column_name=context.get("column_name", column.name or ""),
            features=all_features,
        )


# Global registry with default extractors
feature_extractor_registry = FeatureExtractorRegistry()
feature_extractor_registry.register(NameFeatureExtractor())
feature_extractor_registry.register(ValueFeatureExtractor())
feature_extractor_registry.register(StatisticalFeatureExtractor())
feature_extractor_registry.register(ContextFeatureExtractor())


# =============================================================================
# ML Model Protocol
# =============================================================================


class InferenceModel(ABC):
    """Abstract base for inference models."""

    name: str = "base"
    version: str = "1.0"

    @abstractmethod
    def predict(
        self,
        features: FeatureVector,
    ) -> list[tuple[DataType, float]]:
        """Predict type probabilities.

        Args:
            features: Extracted features

        Returns:
            List of (DataType, probability) sorted by probability
        """
        pass

    @abstractmethod
    def train(
        self,
        training_data: list[tuple[FeatureVector, DataType]],
    ) -> None:
        """Train/update the model.

        Args:
            training_data: List of (features, true_type) pairs
        """
        pass

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        pass

    def load(self, path: str | Path) -> None:
        """Load model from file."""
        pass


class RuleBasedModel(InferenceModel):
    """Rule-based inference model.

    Uses weighted rules derived from feature values to infer types.
    Good baseline that doesn't require training data.
    """

    name = "rule_based"
    version = "1.0"

    def __init__(self) -> None:
        # Define rules as (feature_name, operator, threshold, type, weight)
        self.rules: list[tuple[str, str, float, DataType, float]] = [
            # Email rules
            ("email_pattern_ratio", ">=", 0.8, DataType.EMAIL, 0.9),
            ("has_at_sign", ">=", 0.9, DataType.EMAIL, 0.7),
            ("name_match_email", ">=", 0.5, DataType.EMAIL, 0.5),

            # UUID rules
            ("uuid_pattern_ratio", ">=", 0.8, DataType.UUID, 0.95),
            ("name_match_uuid", ">=", 0.5, DataType.UUID, 0.6),

            # Identifier rules
            ("is_identifier", ">=", 0.9, DataType.IDENTIFIER, 0.7),
            ("is_first_column", ">=", 0.5, DataType.IDENTIFIER, 0.3),
            ("name_match_identifier", ">=", 0.5, DataType.IDENTIFIER, 0.4),

            # Categorical rules
            ("is_categorical", ">=", 0.8, DataType.CATEGORICAL, 0.8),
            ("name_match_categorical", ">=", 0.5, DataType.CATEGORICAL, 0.4),

            # Date/DateTime rules
            ("name_match_date", ">=", 0.5, DataType.DATE, 0.5),
            ("name_match_datetime", ">=", 0.5, DataType.DATETIME, 0.5),

            # Numeric rules
            ("is_currency_like", ">=", 0.8, DataType.CURRENCY, 0.7),
            ("in_0_100_range", ">=", 0.9, DataType.PERCENTAGE, 0.5),
            ("in_0_1_range", ">=", 0.95, DataType.PERCENTAGE, 0.6),

            # Phone rules
            ("name_match_phone", ">=", 0.5, DataType.PHONE, 0.5),
            ("name_match_korean_phone", ">=", 0.5, DataType.KOREAN_PHONE, 0.6),

            # Boolean
            ("is_boolean", ">=", 0.9, DataType.BOOLEAN, 0.95),
        ]

    def predict(
        self,
        features: FeatureVector,
    ) -> list[tuple[DataType, float]]:
        """Apply rules to predict type."""
        type_scores: dict[DataType, float] = defaultdict(float)
        feature_dict = features.to_dict()

        for feature_name, operator, threshold, dtype, weight in self.rules:
            value = feature_dict.get(feature_name, 0.0)

            match = False
            if operator == ">=":
                match = value >= threshold
            elif operator == "<=":
                match = value <= threshold
            elif operator == "==":
                match = abs(value - threshold) < 0.01

            if match:
                type_scores[dtype] += weight * value

        # Normalize scores to probabilities
        total = sum(type_scores.values())
        if total > 0:
            probabilities = [
                (dtype, score / total)
                for dtype, score in type_scores.items()
            ]
        else:
            # Default to string if no rules match
            probabilities = [(DataType.STRING, 0.5)]

        # Sort by probability
        probabilities.sort(key=lambda x: x[1], reverse=True)

        return probabilities

    def train(
        self,
        training_data: list[tuple[FeatureVector, DataType]],
    ) -> None:
        """Rule-based model doesn't need training, but could be tuned."""
        pass


class NaiveBayesModel(InferenceModel):
    """Naive Bayes classifier for type inference.

    Simple probabilistic model that works well with limited training data.
    """

    name = "naive_bayes"
    version = "1.0"

    def __init__(self) -> None:
        self.class_priors: dict[DataType, float] = {}
        self.feature_likelihoods: dict[str, dict[DataType, tuple[float, float]]] = {}
        self._trained = False

    def predict(
        self,
        features: FeatureVector,
    ) -> list[tuple[DataType, float]]:
        """Predict using Naive Bayes."""
        if not self._trained:
            # Fall back to rule-based if not trained
            return RuleBasedModel().predict(features)

        log_posteriors: dict[DataType, float] = {}
        feature_dict = features.to_dict()

        for dtype, prior in self.class_priors.items():
            log_posterior = math.log(prior + 1e-10)

            for feature_name, value in feature_dict.items():
                if feature_name in self.feature_likelihoods:
                    mean, std = self.feature_likelihoods[feature_name].get(
                        dtype, (0.5, 0.3)
                    )
                    # Gaussian likelihood
                    if std > 0:
                        z = (value - mean) / std
                        log_likelihood = -0.5 * z * z - math.log(std) - 0.5 * math.log(2 * math.pi)
                        log_posterior += log_likelihood

            log_posteriors[dtype] = log_posterior

        # Convert to probabilities
        max_log = max(log_posteriors.values())
        exp_posteriors = {
            dtype: math.exp(lp - max_log)
            for dtype, lp in log_posteriors.items()
        }
        total = sum(exp_posteriors.values())

        probabilities = [
            (dtype, prob / total)
            for dtype, prob in exp_posteriors.items()
        ]
        probabilities.sort(key=lambda x: x[1], reverse=True)

        return probabilities

    def train(
        self,
        training_data: list[tuple[FeatureVector, DataType]],
    ) -> None:
        """Train Naive Bayes classifier."""
        if not training_data:
            return

        # Count classes
        class_counts: dict[DataType, int] = Counter()
        feature_values: dict[str, dict[DataType, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for features, dtype in training_data:
            class_counts[dtype] += 1
            for f in features.features:
                feature_values[f.name][dtype].append(f.value)

        # Calculate priors
        total = sum(class_counts.values())
        self.class_priors = {
            dtype: count / total
            for dtype, count in class_counts.items()
        }

        # Calculate feature likelihoods (mean, std for each feature per class)
        for feature_name, class_values in feature_values.items():
            self.feature_likelihoods[feature_name] = {}
            for dtype, values in class_values.items():
                if values:
                    mean = sum(values) / len(values)
                    variance = sum((v - mean) ** 2 for v in values) / len(values)
                    std = math.sqrt(variance) if variance > 0 else 0.1
                    self.feature_likelihoods[feature_name][dtype] = (mean, std)

        self._trained = True

    def save(self, path: str | Path) -> None:
        """Save model to file."""
        data = {
            "class_priors": {k.value: v for k, v in self.class_priors.items()},
            "feature_likelihoods": {
                fname: {dtype.value: stats for dtype, stats in class_stats.items()}
                for fname, class_stats in self.feature_likelihoods.items()
            },
            "trained": self._trained,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load model from file."""
        with open(path) as f:
            data = json.load(f)

        self.class_priors = {
            DataType(k): v for k, v in data["class_priors"].items()
        }
        self.feature_likelihoods = {
            fname: {DataType(dtype): tuple(stats) for dtype, stats in class_stats.items()}
            for fname, class_stats in data["feature_likelihoods"].items()
        }
        self._trained = data["trained"]


class EnsembleModel(InferenceModel):
    """Ensemble of multiple models.

    Combines predictions from multiple models using weighted voting.
    """

    name = "ensemble"
    version = "1.0"

    def __init__(
        self,
        models: list[tuple[InferenceModel, float]] | None = None,
    ):
        """Initialize ensemble.

        Args:
            models: List of (model, weight) tuples
        """
        self.models = models or [
            (RuleBasedModel(), 0.6),
            (NaiveBayesModel(), 0.4),
        ]

    def predict(
        self,
        features: FeatureVector,
    ) -> list[tuple[DataType, float]]:
        """Combine predictions from all models."""
        combined_scores: dict[DataType, float] = defaultdict(float)

        for model, weight in self.models:
            predictions = model.predict(features)
            for dtype, prob in predictions:
                combined_scores[dtype] += weight * prob

        # Normalize
        total = sum(combined_scores.values())
        if total > 0:
            probabilities = [
                (dtype, score / total)
                for dtype, score in combined_scores.items()
            ]
        else:
            probabilities = [(DataType.STRING, 1.0)]

        probabilities.sort(key=lambda x: x[1], reverse=True)
        return probabilities

    def train(
        self,
        training_data: list[tuple[FeatureVector, DataType]],
    ) -> None:
        """Train all models in ensemble."""
        for model, _ in self.models:
            model.train(training_data)


# =============================================================================
# Model Registry
# =============================================================================


class ModelRegistry:
    """Registry for inference models."""

    def __init__(self) -> None:
        self._models: dict[str, type[InferenceModel]] = {}

    def register(
        self,
        name: str,
        model_class: type[InferenceModel],
    ) -> None:
        """Register a model class."""
        self._models[name] = model_class

    def create(self, name: str, **kwargs: Any) -> InferenceModel:
        """Create a model instance."""
        if name not in self._models:
            raise KeyError(f"Unknown model: {name}")
        return self._models[name](**kwargs)

    def list_models(self) -> list[str]:
        """List available models."""
        return list(self._models.keys())


model_registry = ModelRegistry()
model_registry.register("rule_based", RuleBasedModel)
model_registry.register("naive_bayes", NaiveBayesModel)
model_registry.register("ensemble", EnsembleModel)


# =============================================================================
# ML Type Inferrer
# =============================================================================


@dataclass
class InferrerConfig:
    """Configuration for ML type inferrer."""

    model: str = "ensemble"
    confidence_threshold: float = 0.5
    use_caching: bool = True
    cache_size: int = 1000
    enable_learning: bool = True
    model_path: str | None = None


class MLTypeInferrer:
    """ML-based type inferrer.

    Main interface for ML-powered type inference.

    Example:
        inferrer = MLTypeInferrer()

        result = inferrer.infer(column, context={
            "column_name": "email",
            "table_name": "users",
        })

        print(f"Inferred: {result.inferred_type} ({result.confidence:.0%})")
    """

    def __init__(
        self,
        model: str | InferenceModel = "ensemble",
        config: InferrerConfig | None = None,
    ):
        self.config = config or InferrerConfig()

        if isinstance(model, InferenceModel):
            self._model = model
        else:
            self._model = model_registry.create(model)

        self._feature_registry = feature_extractor_registry
        self._cache: dict[str, InferenceResult] = {}
        self._feedback_buffer: list[tuple[FeatureVector, DataType]] = []
        self._lock = threading.Lock()

        # Load saved model if path provided
        if self.config.model_path and Path(self.config.model_path).exists():
            self._model.load(self.config.model_path)

    def infer(
        self,
        column: pl.Series,
        context: dict[str, Any] | None = None,
    ) -> InferenceResult:
        """Infer column type using ML.

        Args:
            column: Column data
            context: Additional context information

        Returns:
            Inference result with type and confidence
        """
        import time
        start = time.time()

        context = context or {}
        context["column_name"] = context.get("column_name", column.name or "")

        # Check cache
        cache_key = self._make_cache_key(column, context)
        if self.config.use_caching and cache_key in self._cache:
            return self._cache[cache_key]

        # Extract features
        features = self._feature_registry.extract_all(column, context)

        # Get predictions
        predictions = self._model.predict(features)

        if not predictions:
            predictions = [(DataType.STRING, 0.5)]

        # Build result
        top_type, top_confidence = predictions[0]
        alternatives = predictions[1:5]  # Top 5 alternatives

        # Generate reasoning
        reasoning = self._generate_reasoning(features, predictions)

        elapsed_ms = (time.time() - start) * 1000

        result = InferenceResult(
            column_name=context["column_name"],
            inferred_type=top_type,
            confidence=top_confidence,
            alternatives=alternatives,
            reasoning=reasoning,
            features_used=[f.name for f in features.features[:10]],
            model_version=self._model.version,
            inference_time_ms=elapsed_ms,
        )

        # Cache result
        if self.config.use_caching:
            with self._lock:
                self._cache[cache_key] = result
                # LRU eviction
                if len(self._cache) > self.config.cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

        return result

    def infer_table(
        self,
        df: pl.DataFrame,
        table_name: str = "",
    ) -> dict[str, InferenceResult]:
        """Infer types for all columns in a table.

        Args:
            df: DataFrame to analyze
            table_name: Table name for context

        Returns:
            Dictionary mapping column names to results
        """
        results = {}
        columns = df.columns

        for i, col_name in enumerate(columns):
            # Build context with neighboring columns
            neighbors = []
            if i > 0:
                neighbors.append(columns[i - 1])
            if i < len(columns) - 1:
                neighbors.append(columns[i + 1])

            context = {
                "column_name": col_name,
                "table_name": table_name,
                "column_index": i,
                "total_columns": len(columns),
                "neighbor_columns": neighbors,
            }

            result = self.infer(df.get_column(col_name), context)
            results[col_name] = result

        return results

    def provide_feedback(
        self,
        column: pl.Series,
        true_type: DataType,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Provide feedback for online learning.

        Args:
            column: Column that was classified
            true_type: The correct type
            context: Context used during inference
        """
        if not self.config.enable_learning:
            return

        context = context or {}
        context["column_name"] = context.get("column_name", column.name or "")

        features = self._feature_registry.extract_all(column, context)

        with self._lock:
            self._feedback_buffer.append((features, true_type))

            # Retrain when buffer is large enough
            if len(self._feedback_buffer) >= 100:
                self._model.train(self._feedback_buffer)
                self._feedback_buffer.clear()

                # Save model if path configured
                if self.config.model_path:
                    self._model.save(self.config.model_path)

    def _make_cache_key(
        self,
        column: pl.Series,
        context: dict[str, Any],
    ) -> str:
        """Create cache key for column + context."""
        # Use column sample and context for key
        sample = column.head(10).to_list()
        key_data = f"{context.get('column_name', '')}:{sample}:{column.dtype}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _generate_reasoning(
        self,
        features: FeatureVector,
        predictions: list[tuple[DataType, float]],
    ) -> list[str]:
        """Generate human-readable reasoning."""
        reasoning = []

        # Get top features
        sorted_features = sorted(
            features.features,
            key=lambda f: abs(f.value - 0.5),  # Deviation from neutral
            reverse=True,
        )

        for f in sorted_features[:5]:
            if f.value > 0.7:
                reasoning.append(f"High {f.name}: {f.value:.2f}")
            elif f.value < 0.3:
                reasoning.append(f"Low {f.name}: {f.value:.2f}")

        if predictions:
            top_type, top_conf = predictions[0]
            reasoning.append(f"Best match: {top_type.value} ({top_conf:.0%})")

        return reasoning

    def clear_cache(self) -> None:
        """Clear inference cache."""
        with self._lock:
            self._cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_inference_model(
    model_type: str = "ensemble",
    **kwargs: Any,
) -> InferenceModel:
    """Create an inference model.

    Args:
        model_type: Model type name
        **kwargs: Model configuration

    Returns:
        Configured model
    """
    return model_registry.create(model_type, **kwargs)


def infer_column_type_ml(
    column: pl.Series,
    context: dict[str, Any] | None = None,
    model: str = "ensemble",
) -> InferenceResult:
    """Infer column type using ML.

    Args:
        column: Column to analyze
        context: Additional context
        model: Model to use

    Returns:
        Inference result
    """
    inferrer = MLTypeInferrer(model=model)
    return inferrer.infer(column, context)


def infer_table_types_ml(
    df: pl.DataFrame,
    table_name: str = "",
    model: str = "ensemble",
) -> dict[str, InferenceResult]:
    """Infer types for all columns in a table.

    Args:
        df: DataFrame to analyze
        table_name: Table name for context
        model: Model to use

    Returns:
        Dictionary of column results
    """
    inferrer = MLTypeInferrer(model=model)
    return inferrer.infer_table(df, table_name)
