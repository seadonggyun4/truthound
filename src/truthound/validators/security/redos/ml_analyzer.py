"""ML-based ReDoS Pattern Analysis.

This module provides machine learning-based analysis for predicting
ReDoS vulnerability risk in regex patterns. It uses feature extraction
and trained models to assess pattern safety.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ML Pattern Analyzer                           │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────┬───────────────┼───────────────┬─────────────────┐
    │               │               │               │                 │
    ▼               ▼               ▼               ▼                 ▼
┌─────────┐   ┌─────────┐    ┌──────────┐   ┌──────────┐    ┌─────────┐
│ Feature │   │ Model   │    │Prediction│   │ Training │    │  Model  │
│Extractor│   │ Manager │    │ Pipeline │   │ Pipeline │    │ Storage │
└─────────┘   └─────────┘    └──────────┘   └──────────┘    └─────────┘

Features extracted:
- Structural features (length, depth, groups, etc.)
- Quantifier features (count, types, positions)
- Alternation features (count, complexity)
- Character class features (ranges, negation)
- Backtracking potential features

Usage:
    from truthound.validators.security.redos.ml_analyzer import (
        MLPatternAnalyzer,
        predict_redos_risk,
    )

    # Quick prediction
    result = predict_redos_risk(r"(a+)+b")
    print(result.risk_probability)  # 0.95
    print(result.risk_level)  # ReDoSRisk.CRITICAL

    # Full analyzer with custom model
    analyzer = MLPatternAnalyzer()
    analyzer.train(training_patterns, labels)
    result = analyzer.predict(pattern)
"""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from truthound.validators.security.redos.core import ReDoSRisk


@dataclass
class PatternFeatures:
    """Extracted features from a regex pattern.

    These features are used for ML-based risk prediction.
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

    def to_vector(self) -> list[float]:
        """Convert features to a numeric vector for ML models."""
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
    def feature_names(cls) -> list[str]:
        """Get names of all features in vector order."""
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {name: value for name, value in zip(self.feature_names(), self.to_vector())}


class FeatureExtractor:
    """Extracts ML-relevant features from regex patterns.

    This extractor analyzes regex patterns and produces a feature vector
    suitable for machine learning models.

    Example:
        extractor = FeatureExtractor()
        features = extractor.extract(r"(a+)+b")
        print(features.nested_quantifier_count)  # 1
        print(features.backtracking_potential)  # high value
    """

    # Compiled patterns for feature extraction
    _PLUS_PATTERN = re.compile(r"(?<!\\)\+")
    _STAR_PATTERN = re.compile(r"(?<!\\)\*")
    _QUESTION_PATTERN = re.compile(r"(?<!\\)\?(?![=!<:])")
    _BOUNDED_QUANT_PATTERN = re.compile(r"\{(\d+)(?:,(\d*))?\}")
    _LAZY_QUANT_PATTERN = re.compile(r"[+*?]\?|\{[^}]+\}\?")
    _CHAR_CLASS_PATTERN = re.compile(r"\[[^\]]+\]")
    _NEGATED_CLASS_PATTERN = re.compile(r"\[\^[^\]]+\]")
    _LOOKAHEAD_PATTERN = re.compile(r"\(\?[=!]")
    _LOOKBEHIND_PATTERN = re.compile(r"\(\?<[=!]")
    _BACKREFERENCE_PATTERN = re.compile(r"\\([1-9]\d*)")
    _NESTED_QUANT_PATTERN = re.compile(r"\([^)]*[+*][^)]*\)[+*]")
    _ADJACENT_QUANT_PATTERN = re.compile(r"[+*][+*]")
    _QUANTIFIED_ALT_PATTERN = re.compile(r"\([^)]*\|[^)]*\)[+*?]")
    _QUANTIFIED_BACKREF_PATTERN = re.compile(r"\\[1-9][+*]|\{[^}]+\}")
    _DOT_PATTERN = re.compile(r"(?<!\\)\.")
    _WORD_BOUNDARY_PATTERN = re.compile(r"\\b")
    _NON_CAPTURE_GROUP_PATTERN = re.compile(r"\(\?(?:[imsxLu]|:)")
    _CAPTURE_GROUP_PATTERN = re.compile(r"\((?!\?)")

    def extract(self, pattern: str) -> PatternFeatures:
        """Extract features from a regex pattern.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            PatternFeatures with all extracted features
        """
        features = PatternFeatures()

        if not pattern:
            return features

        # Structural features
        features.length = len(pattern)
        features.max_nesting_depth = self._calculate_nesting_depth(pattern)
        features.alternation_count = pattern.count("|")

        # Group counts
        features.capture_group_count = len(self._CAPTURE_GROUP_PATTERN.findall(pattern))
        features.non_capture_group_count = len(self._NON_CAPTURE_GROUP_PATTERN.findall(pattern))
        features.group_count = features.capture_group_count + features.non_capture_group_count

        # Quantifier features
        features.plus_count = len(self._PLUS_PATTERN.findall(pattern))
        features.star_count = len(self._STAR_PATTERN.findall(pattern))
        features.question_count = len(self._QUESTION_PATTERN.findall(pattern))
        features.lazy_quantifier_count = len(self._LAZY_QUANT_PATTERN.findall(pattern))

        bounded_matches = self._BOUNDED_QUANT_PATTERN.findall(pattern)
        features.bounded_quantifier_count = len(bounded_matches)

        # Unbounded quantifiers
        unbounded_count = 0
        for min_val, max_val in bounded_matches:
            if max_val == "":  # {n,} form
                unbounded_count += 1
        features.unbounded_quantifier_count = (
            features.plus_count + features.star_count + unbounded_count
        )

        # Quantifier density
        total_quantifiers = (
            features.plus_count
            + features.star_count
            + features.question_count
            + features.bounded_quantifier_count
        )
        features.quantifier_density = total_quantifiers / max(features.length, 1)

        # Dangerous patterns
        features.nested_quantifier_count = len(self._NESTED_QUANT_PATTERN.findall(pattern))
        features.adjacent_quantifier_count = len(self._ADJACENT_QUANT_PATTERN.findall(pattern))
        features.quantified_alternation_count = len(self._QUANTIFIED_ALT_PATTERN.findall(pattern))
        features.quantified_backreference_count = len(
            self._QUANTIFIED_BACKREF_PATTERN.findall(pattern)
        )

        # Character class features
        features.char_class_count = len(self._CHAR_CLASS_PATTERN.findall(pattern))
        features.negated_char_class_count = len(self._NEGATED_CLASS_PATTERN.findall(pattern))
        features.dot_count = len(self._DOT_PATTERN.findall(pattern))
        features.word_boundary_count = len(self._WORD_BOUNDARY_PATTERN.findall(pattern))

        # Lookaround features
        lookahead_matches = self._LOOKAHEAD_PATTERN.findall(pattern)
        lookbehind_matches = self._LOOKBEHIND_PATTERN.findall(pattern)
        features.lookahead_count = len(lookahead_matches)
        features.lookbehind_count = len(lookbehind_matches)
        features.negative_lookaround_count = (
            pattern.count("(?!") + pattern.count("(?<!")
        )

        # Backreference features
        backref_matches = self._BACKREFERENCE_PATTERN.findall(pattern)
        features.backreference_count = len(backref_matches)
        if backref_matches:
            features.max_backreference_index = max(int(m) for m in backref_matches)

        # Anchor features
        features.start_anchor = pattern.startswith("^") or "\\A" in pattern
        features.end_anchor = pattern.endswith("$") or "\\Z" in pattern or "\\z" in pattern
        features.anchored = features.start_anchor and features.end_anchor

        # Complexity metrics
        features.backtracking_potential = self._calculate_backtracking_potential(features)
        features.estimated_states = self._estimate_nfa_states(features)

        return features

    def _calculate_nesting_depth(self, pattern: str) -> int:
        """Calculate maximum nesting depth of groups."""
        depth = 0
        max_depth = 0
        for char in pattern:
            if char == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ")":
                depth = max(0, depth - 1)
        return max_depth

    def _calculate_backtracking_potential(self, features: PatternFeatures) -> float:
        """Estimate backtracking potential based on features.

        Higher values indicate higher risk of catastrophic backtracking.
        """
        potential = 0.0

        # Nested quantifiers are the biggest risk
        potential += features.nested_quantifier_count * 50.0

        # Quantified alternation is also risky
        potential += features.quantified_alternation_count * 30.0

        # Adjacent quantifiers
        potential += features.adjacent_quantifier_count * 20.0

        # Unbounded quantifiers increase potential
        potential += features.unbounded_quantifier_count * 5.0

        # Deep nesting increases potential
        potential += features.max_nesting_depth * 3.0

        # Backreferences with quantifiers
        potential += features.quantified_backreference_count * 40.0

        # Lack of anchoring increases potential
        if not features.anchored:
            potential *= 1.2

        return min(potential, 100.0)

    def _estimate_nfa_states(self, features: PatternFeatures) -> int:
        """Estimate number of NFA states.

        This is a rough approximation based on pattern features.
        """
        # Base states from length
        states = features.length

        # Groups add states
        states += features.group_count * 2

        # Quantifiers add states
        states += features.plus_count * 2
        states += features.star_count * 2
        states += features.question_count

        # Bounded quantifiers can add many states
        states += features.bounded_quantifier_count * 5

        # Alternations add branch states
        states += features.alternation_count * 2

        return states


@dataclass
class MLPredictionResult:
    """Result of ML-based ReDoS risk prediction.

    Attributes:
        pattern: The analyzed pattern
        features: Extracted features
        risk_probability: Probability of ReDoS vulnerability (0-1)
        risk_level: Categorical risk level
        confidence: Model confidence in the prediction
        contributing_factors: Features that most influenced the prediction
        model_version: Version of the model used
    """

    pattern: str
    features: PatternFeatures
    risk_probability: float
    risk_level: ReDoSRisk
    confidence: float
    contributing_factors: list[tuple[str, float]] = field(default_factory=list)
    model_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
            "model_version": self.model_version,
        }


class MLModelProtocol(Protocol):
    """Protocol for ML models used in ReDoS prediction."""

    def predict(self, features: list[float]) -> tuple[float, float]:
        """Predict risk probability and confidence.

        Args:
            features: Feature vector

        Returns:
            Tuple of (risk_probability, confidence)
        """
        ...

    def get_feature_importance(self) -> list[float]:
        """Get feature importance scores."""
        ...


class RuleBasedModel:
    """Rule-based model for ReDoS risk prediction.

    This model uses hand-crafted rules based on known ReDoS patterns
    to estimate risk. It serves as a baseline and fallback when
    ML models are not available.

    The model assigns weights to various pattern features and combines
    them using a logistic function to produce a probability.
    """

    # Feature weights (learned from known vulnerable patterns)
    FEATURE_WEIGHTS: dict[str, float] = {
        "nested_quantifier_count": 5.0,
        "quantified_backreference_count": 4.0,
        "quantified_alternation_count": 3.5,
        "adjacent_quantifier_count": 2.5,
        "unbounded_quantifier_count": 1.5,
        "max_nesting_depth": 0.8,
        "star_count": 0.5,
        "plus_count": 0.5,
        "alternation_count": 0.3,
        "quantifier_density": 2.0,
        "backtracking_potential": 0.1,  # Already composite
    }

    # Bias term
    BIAS = -2.0

    def __init__(self):
        """Initialize the rule-based model."""
        self._feature_names = PatternFeatures.feature_names()

    def predict(self, features: list[float]) -> tuple[float, float]:
        """Predict risk probability using rules.

        Args:
            features: Feature vector

        Returns:
            Tuple of (risk_probability, confidence)
        """
        # Map features to dictionary for easier access
        feature_dict = dict(zip(self._feature_names, features))

        # Calculate weighted sum
        weighted_sum = self.BIAS
        for feature_name, weight in self.FEATURE_WEIGHTS.items():
            if feature_name in feature_dict:
                weighted_sum += feature_dict[feature_name] * weight

        # Apply logistic function
        probability = 1.0 / (1.0 + math.exp(-weighted_sum))

        # Confidence based on how extreme the score is
        confidence = abs(2 * probability - 1)

        return probability, confidence

    def get_feature_importance(self) -> list[float]:
        """Get feature importance scores."""
        importance = []
        for name in self._feature_names:
            importance.append(self.FEATURE_WEIGHTS.get(name, 0.0))
        return importance


class EnsembleModel:
    """Ensemble model combining multiple prediction strategies.

    This model combines rule-based heuristics with pattern matching
    for more robust predictions.
    """

    def __init__(self):
        """Initialize ensemble model."""
        self._rule_model = RuleBasedModel()
        self._feature_names = PatternFeatures.feature_names()

        # Known dangerous pattern signatures with risk scores
        self._dangerous_signatures: list[tuple[re.Pattern, float]] = [
            (re.compile(r"\([^)]*[+*][^)]*\)[+*]"), 0.95),  # Nested quantifiers
            (re.compile(r"\\[1-9][+*]"), 0.85),  # Quantified backreference
            (re.compile(r"\([^)]*\|[^)]*\)[+*]"), 0.75),  # Quantified alternation
            (re.compile(r"[+*][+*]"), 0.65),  # Adjacent quantifiers
        ]

    def predict(self, features: list[float], pattern: str = "") -> tuple[float, float]:
        """Predict using ensemble of methods.

        Args:
            features: Feature vector
            pattern: Original pattern (optional, for signature matching)

        Returns:
            Tuple of (risk_probability, confidence)
        """
        # Rule-based prediction
        rule_prob, rule_conf = self._rule_model.predict(features)

        # Pattern signature matching
        sig_prob = 0.0
        for sig_pattern, risk in self._dangerous_signatures:
            if pattern and sig_pattern.search(pattern):
                sig_prob = max(sig_prob, risk)

        # Combine predictions (weighted average)
        if sig_prob > 0:
            # Signature match has high confidence
            final_prob = 0.6 * sig_prob + 0.4 * rule_prob
            final_conf = max(rule_conf, 0.9)  # High confidence when signature matches
        else:
            final_prob = rule_prob
            final_conf = rule_conf

        return final_prob, final_conf

    def get_feature_importance(self) -> list[float]:
        """Get feature importance from base model."""
        return self._rule_model.get_feature_importance()


class MLPatternAnalyzer:
    """Machine learning-based regex pattern analyzer.

    This analyzer uses ML models to predict ReDoS vulnerability risk.
    It supports multiple model backends and can be trained on custom data.

    Example:
        analyzer = MLPatternAnalyzer()

        # Predict risk
        result = analyzer.predict(r"(a+)+b")
        print(result.risk_level)  # ReDoSRisk.CRITICAL

        # Get detailed features
        features = analyzer.extract_features(r"^[a-z]+$")
        print(features.quantifier_density)

        # Train custom model
        patterns = ["(a+)+", "^[a-z]+$", ...]
        labels = [1, 0, ...]  # 1 = vulnerable, 0 = safe
        analyzer.train(patterns, labels)
    """

    VERSION = "1.0.0"

    # Risk thresholds
    RISK_THRESHOLDS: dict[ReDoSRisk, float] = {
        ReDoSRisk.NONE: 0.1,
        ReDoSRisk.LOW: 0.3,
        ReDoSRisk.MEDIUM: 0.5,
        ReDoSRisk.HIGH: 0.7,
        ReDoSRisk.CRITICAL: 0.85,
    }

    def __init__(
        self,
        model: MLModelProtocol | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ):
        """Initialize the analyzer.

        Args:
            model: ML model to use (defaults to EnsembleModel)
            feature_extractor: Feature extractor (defaults to FeatureExtractor)
        """
        self.extractor = feature_extractor or FeatureExtractor()
        self._model: Any = model or EnsembleModel()
        self._trained = False

    def extract_features(self, pattern: str) -> PatternFeatures:
        """Extract features from a pattern.

        Args:
            pattern: Regex pattern

        Returns:
            PatternFeatures object
        """
        return self.extractor.extract(pattern)

    def predict(self, pattern: str) -> MLPredictionResult:
        """Predict ReDoS risk for a pattern.

        Uses the trained ML model if available, otherwise falls back
        to the rule-based/ensemble model.

        Args:
            pattern: Regex pattern to analyze

        Returns:
            MLPredictionResult with prediction details
        """
        # Use trained ML predictor if available
        if hasattr(self, "_ml_predictor") and self._ml_predictor is not None:
            prediction = self._ml_predictor.predict(pattern)
            # Convert to legacy MLPredictionResult format
            return MLPredictionResult(
                pattern=prediction.pattern,
                features=self._convert_features(prediction.features),
                risk_probability=prediction.risk_probability,
                risk_level=prediction.risk_level,
                confidence=prediction.confidence,
                contributing_factors=prediction.contributing_factors,
                model_version=prediction.model_version,
            )

        # Fallback to original implementation
        # Extract features
        features = self.extractor.extract(pattern)
        feature_vector = features.to_vector()

        # Get prediction
        if isinstance(self._model, EnsembleModel):
            probability, confidence = self._model.predict(feature_vector, pattern)
        else:
            probability, confidence = self._model.predict(feature_vector)

        # Determine risk level
        risk_level = self._probability_to_risk_level(probability)

        # Get contributing factors
        contributing_factors = self._get_contributing_factors(features)

        return MLPredictionResult(
            pattern=pattern,
            features=features,
            risk_probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            model_version=self.VERSION,
        )

    def _convert_features(self, new_features: Any) -> PatternFeatures:
        """Convert new PatternFeatures format to legacy format.

        This is for backward compatibility with existing code that
        expects the old PatternFeatures dataclass.
        """
        # The new and old PatternFeatures have the same structure
        # Just return the new features directly
        return PatternFeatures(
            length=new_features.length,
            group_count=new_features.group_count,
            capture_group_count=new_features.capture_group_count,
            non_capture_group_count=new_features.non_capture_group_count,
            max_nesting_depth=new_features.max_nesting_depth,
            alternation_count=new_features.alternation_count,
            plus_count=new_features.plus_count,
            star_count=new_features.star_count,
            question_count=new_features.question_count,
            bounded_quantifier_count=new_features.bounded_quantifier_count,
            unbounded_quantifier_count=new_features.unbounded_quantifier_count,
            lazy_quantifier_count=new_features.lazy_quantifier_count,
            possessive_quantifier_count=new_features.possessive_quantifier_count,
            quantifier_density=new_features.quantifier_density,
            nested_quantifier_count=new_features.nested_quantifier_count,
            adjacent_quantifier_count=new_features.adjacent_quantifier_count,
            quantified_alternation_count=new_features.quantified_alternation_count,
            quantified_backreference_count=new_features.quantified_backreference_count,
            char_class_count=new_features.char_class_count,
            negated_char_class_count=new_features.negated_char_class_count,
            dot_count=new_features.dot_count,
            word_boundary_count=new_features.word_boundary_count,
            lookahead_count=new_features.lookahead_count,
            lookbehind_count=new_features.lookbehind_count,
            negative_lookaround_count=new_features.negative_lookaround_count,
            backreference_count=new_features.backreference_count,
            max_backreference_index=new_features.max_backreference_index,
            start_anchor=new_features.start_anchor,
            end_anchor=new_features.end_anchor,
            anchored=new_features.anchored,
            backtracking_potential=new_features.backtracking_potential,
            estimated_states=new_features.estimated_states,
        )

    def predict_batch(self, patterns: Sequence[str]) -> list[MLPredictionResult]:
        """Predict risk for multiple patterns.

        Args:
            patterns: Sequence of patterns to analyze

        Returns:
            List of MLPredictionResult objects
        """
        return [self.predict(pattern) for pattern in patterns]

    def train(
        self,
        patterns: Sequence[str],
        labels: Sequence[int],
        validation_split: float = 0.2,
    ) -> dict[str, float]:
        """Train the model on labeled data.

        This method trains a scikit-learn based Random Forest classifier
        on the provided patterns. If scikit-learn is not available, it
        falls back to a rule-based model that is always "trained".

        Args:
            patterns: Training patterns
            labels: Labels (1 = vulnerable, 0 = safe)
            validation_split: Fraction of data for validation

        Returns:
            Training metrics dictionary containing accuracy, precision,
            recall, f1, and sample count.

        Raises:
            ValueError: If patterns and labels have different lengths

        Example:
            >>> analyzer = MLPatternAnalyzer()
            >>> patterns = ["(a+)+", "^[a-z]+$", "(.*)+", "\\d+"]
            >>> labels = [1, 0, 1, 0]  # 1=vulnerable, 0=safe
            >>> metrics = analyzer.train(patterns, labels)
            >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        """
        if len(patterns) != len(labels):
            raise ValueError("Patterns and labels must have same length")

        # Import the new ML framework
        from truthound.validators.security.redos.ml import (
            ReDoSTrainingData,
            TrainingPipeline,
            TrainingConfig,
            ModelType,
            ReDoSMLPredictor,
        )

        # Create training data
        training_data = ReDoSTrainingData(
            patterns=list(patterns),
            labels=list(labels),
        )

        # Configure training
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            test_split=validation_split,
            cv_folds=5,
            verbose=0,
        )

        # Train using the pipeline
        pipeline = TrainingPipeline(config=config)
        result = pipeline.train(training_data)

        # Store the trained model for predictions
        self._ml_predictor = ReDoSMLPredictor(model=result.model)
        self._trained = True
        self._metrics = result.metrics

        # Return metrics as dictionary for backward compatibility
        return {
            "accuracy": result.metrics.accuracy,
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "f1": result.metrics.f1_score,
            "samples": float(len(patterns)),
        }

    def save_model(self, path: str | Path) -> None:
        """Save the trained model to disk.

        Saves the trained ML model using pickle/joblib serialization.
        The model can be loaded later using load_model().

        Args:
            path: Path to save the model (recommended: .pkl extension)

        Raises:
            ValueError: If model has not been trained

        Example:
            >>> analyzer = MLPatternAnalyzer()
            >>> analyzer.train(patterns, labels)
            >>> analyzer.save_model("redos_model.pkl")
        """
        from truthound.validators.security.redos.ml.storage import save_model

        if hasattr(self, "_ml_predictor") and self._ml_predictor is not None:
            save_model(self._ml_predictor.model, path)
        else:
            # Fallback for legacy format
            path = Path(path)
            model_data = {
                "version": self.VERSION,
                "trained": self._trained,
                "model_type": type(self._model).__name__,
                "thresholds": {k.name: v for k, v in self.RISK_THRESHOLDS.items()},
            }
            path.write_text(json.dumps(model_data, indent=2))

    def load_model(self, path: str | Path) -> None:
        """Load a trained model from disk.

        Loads a previously saved ML model. The loaded model will be
        used for all subsequent predictions.

        Args:
            path: Path to the saved model

        Example:
            >>> analyzer = MLPatternAnalyzer()
            >>> analyzer.load_model("redos_model.pkl")
            >>> result = analyzer.predict("(a+)+b")
        """
        from truthound.validators.security.redos.ml import ReDoSMLPredictor
        from truthound.validators.security.redos.ml.storage import load_model

        path = Path(path)

        # Try loading as new format first
        try:
            model = load_model(path)
            self._ml_predictor = ReDoSMLPredictor(model=model)
            self._trained = True
        except Exception:
            # Fallback to legacy JSON format
            try:
                model_data = json.loads(path.read_text())
                self._trained = model_data.get("trained", False)
            except Exception:
                self._trained = False

    def _probability_to_risk_level(self, probability: float) -> ReDoSRisk:
        """Convert probability to risk level."""
        if probability >= self.RISK_THRESHOLDS[ReDoSRisk.CRITICAL]:
            return ReDoSRisk.CRITICAL
        elif probability >= self.RISK_THRESHOLDS[ReDoSRisk.HIGH]:
            return ReDoSRisk.HIGH
        elif probability >= self.RISK_THRESHOLDS[ReDoSRisk.MEDIUM]:
            return ReDoSRisk.MEDIUM
        elif probability >= self.RISK_THRESHOLDS[ReDoSRisk.LOW]:
            return ReDoSRisk.LOW
        else:
            return ReDoSRisk.NONE

    def _get_contributing_factors(
        self,
        features: PatternFeatures,
    ) -> list[tuple[str, float]]:
        """Get features that contribute most to the risk prediction.

        Args:
            features: Extracted pattern features

        Returns:
            List of (feature_name, contribution) tuples, sorted by contribution
        """
        feature_importance = self._model.get_feature_importance()
        feature_values = features.to_vector()
        feature_names = PatternFeatures.feature_names()

        # Calculate contributions
        contributions: list[tuple[str, float]] = []
        for name, importance, value in zip(feature_names, feature_importance, feature_values):
            contribution = importance * value
            if contribution > 0:
                contributions.append((name, contribution))

        # Sort by contribution (descending)
        contributions.sort(key=lambda x: x[1], reverse=True)

        # Return top 5 contributors
        return contributions[:5]


# ============================================================================
# Convenience functions
# ============================================================================


def predict_redos_risk(
    pattern: str,
    analyzer: MLPatternAnalyzer | None = None,
) -> MLPredictionResult:
    """Predict ReDoS risk for a regex pattern using ML.

    Args:
        pattern: Regex pattern to analyze
        analyzer: Optional custom analyzer

    Returns:
        MLPredictionResult with prediction details

    Example:
        result = predict_redos_risk(r"(a+)+b")
        print(result.risk_level)  # ReDoSRisk.CRITICAL
        print(result.risk_probability)  # ~0.95
    """
    if analyzer is None:
        analyzer = MLPatternAnalyzer()
    return analyzer.predict(pattern)
