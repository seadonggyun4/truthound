"""ML model implementations for ReDoS prediction.

This module provides concrete implementations of ReDoS prediction models,
including rule-based baseline, scikit-learn models, and ensemble methods.

Available Models:
    - RuleBasedReDoSModel: Deterministic rule-based classifier (no ML deps)
    - RandomForestReDoSModel: Random Forest classifier
    - GradientBoostingReDoSModel: Gradient Boosting classifier
    - LogisticRegressionReDoSModel: Logistic Regression classifier
    - EnsembleReDoSModel: Combines multiple models for robust predictions

Example:
    >>> from truthound.validators.security.redos.ml.models import (
    ...     RandomForestReDoSModel,
    ...     create_model,
    ... )
    >>> model = create_model("random_forest")
    >>> model.train(training_data)
    >>> probability, confidence = model.predict(feature_vector)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from truthound.validators.security.redos.ml.base import (
    BaseReDoSModel,
    ModelConfig,
    ModelType,
    PatternFeatures,
    ReDoSModelMetrics,
    ReDoSTrainingData,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Rule-Based Model (No ML Dependencies)
# =============================================================================


class RuleBasedReDoSModel(BaseReDoSModel):
    """Rule-based model for ReDoS risk prediction.

    This model uses hand-crafted rules based on known ReDoS patterns
    to estimate risk. It serves as a baseline and fallback when
    ML models are not available or not trained.

    The model assigns weights to various pattern features and combines
    them using a logistic function to produce a probability.

    Feature weights are derived from analysis of known vulnerable patterns
    and security research on regex backtracking behavior.

    Attributes:
        FEATURE_WEIGHTS: Dictionary mapping feature names to weights
        BIAS: Bias term for logistic function
    """

    name = "rule_based"
    version = "1.0.0"

    # Feature weights learned from known vulnerable patterns
    FEATURE_WEIGHTS: Dict[str, float] = {
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
        "backtracking_potential": 0.1,
    }

    BIAS = -2.0

    def __init__(self, config: ModelConfig | None = None):
        """Initialize the rule-based model."""
        super().__init__(config)
        self._trained = True  # Always ready

    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Predict risk probability using rules.

        Args:
            features: Feature vector

        Returns:
            Tuple of (risk_probability, confidence)
        """
        feature_dict = dict(zip(self._feature_names, features))

        # Calculate weighted sum
        weighted_sum = self.BIAS
        for feature_name, weight in self.FEATURE_WEIGHTS.items():
            if feature_name in feature_dict:
                weighted_sum += feature_dict[feature_name] * weight

        # Apply logistic function
        probability = 1.0 / (1.0 + math.exp(-weighted_sum))

        # Confidence based on how extreme the score is
        # More extreme probabilities indicate higher confidence
        confidence = abs(2 * probability - 1)

        return probability, confidence

    def predict_batch(
        self, features: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple samples."""
        return [self.predict(f) for f in features]

    def train(self, data: ReDoSTrainingData) -> ReDoSModelMetrics:
        """Rule-based model doesn't need training.

        Returns default metrics indicating the model is ready.
        """
        # Calculate accuracy on training data for reference
        if data.features is None:
            from truthound.validators.security.redos.ml.features import (
                PatternFeatureExtractor,
            )

            extractor = PatternFeatureExtractor()
            features = [extractor.extract(p).to_vector() for p in data.patterns]
        else:
            features = data.features

        predictions = [self.predict(f)[0] for f in features]
        predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

        # Calculate metrics
        tp = sum(1 for p, l in zip(predicted_labels, data.labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(predicted_labels, data.labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predicted_labels, data.labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predicted_labels, data.labels) if p == 0 and l == 1)

        accuracy = (tp + tn) / max(len(data), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        specificity = tn / max(tn + fp, 1)

        self._metrics = ReDoSModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            confusion_matrix=[[tn, fp], [fn, tp]],
            training_samples=len(data),
        )

        return self._metrics

    def get_feature_importance(self) -> List[float]:
        """Get feature importance based on rule weights."""
        return [self.FEATURE_WEIGHTS.get(name, 0.0) for name in self._feature_names]

    def _save_model_data(self) -> Dict[str, Any]:
        """Save rule weights."""
        return {"weights": self.FEATURE_WEIGHTS, "bias": self.BIAS}

    def _load_model_data(self, data: Dict[str, Any]) -> None:
        """Load rule weights."""
        if "weights" in data:
            self.FEATURE_WEIGHTS = data["weights"]
        if "bias" in data:
            self.BIAS = data["bias"]


# =============================================================================
# Scikit-Learn Based Models
# =============================================================================


def _check_sklearn_available() -> bool:
    """Check if scikit-learn is available."""
    try:
        import sklearn

        return True
    except ImportError:
        return False


class RandomForestReDoSModel(BaseReDoSModel):
    """Random Forest classifier for ReDoS prediction.

    This model uses scikit-learn's RandomForestClassifier for robust
    predictions with built-in feature importance scores.

    Random Forest provides:
    - Robust predictions resistant to outliers
    - Built-in feature importance
    - Good performance with default hyperparameters
    - Parallel training capability
    """

    name = "random_forest"
    version = "1.0.0"

    def __init__(self, config: ModelConfig | None = None):
        """Initialize the Random Forest model."""
        super().__init__(config)
        self._model: Any = None
        self._sklearn_available = _check_sklearn_available()

    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Predict risk probability.

        Falls back to rule-based model if not trained or sklearn unavailable.
        """
        # Use rule fallback if sklearn wasn't available
        if hasattr(self, "_rule_fallback") and self._rule_fallback is not None:
            return self._rule_fallback.predict(features)

        if not self._trained or self._model is None:
            return RuleBasedReDoSModel(self._config).predict(features)

        import numpy as np

        X = np.array([features])
        proba = self._model.predict_proba(X)[0]

        # proba is [P(safe), P(vulnerable)]
        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = abs(probability - 0.5) * 2  # Confidence from certainty

        return probability, confidence

    def predict_batch(
        self, features: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple samples."""
        # Use rule fallback if sklearn wasn't available
        if hasattr(self, "_rule_fallback") and self._rule_fallback is not None:
            return self._rule_fallback.predict_batch(features)

        if not self._trained or self._model is None:
            fallback = RuleBasedReDoSModel(self._config)
            return fallback.predict_batch(features)

        import numpy as np

        X = np.array(features)
        probas = self._model.predict_proba(X)

        results = []
        for proba in probas:
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            confidence = abs(probability - 0.5) * 2
            results.append((probability, confidence))

        return results

    def train(self, data: ReDoSTrainingData) -> ReDoSModelMetrics:
        """Train the Random Forest model.

        Args:
            data: Training data container

        Returns:
            Training metrics
        """
        if not self._sklearn_available:
            logger.warning("scikit-learn not available, using rule-based fallback")
            # Use rule-based model internally but mark as trained
            self._rule_fallback = RuleBasedReDoSModel(self._config)
            metrics = self._rule_fallback.train(data)
            self._trained = True
            self._metrics = metrics
            return metrics

        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import cross_val_score, train_test_split

        start_time = time.time()

        # Prepare features
        if data.features is None:
            from truthound.validators.security.redos.ml.features import (
                PatternFeatureExtractor,
            )

            extractor = PatternFeatureExtractor()
            X = np.array([extractor.extract(p).to_vector() for p in data.patterns])
        else:
            X = np.array(data.features)

        y = np.array(data.labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self._config.validation_split,
            random_state=self._config.random_state,
            stratify=y if len(set(y)) > 1 else None,
        )

        # Create and train model
        self._model = RandomForestClassifier(
            n_estimators=self._config.n_estimators,
            max_depth=self._config.max_depth,
            min_samples_split=self._config.min_samples_split,
            min_samples_leaf=self._config.min_samples_leaf,
            random_state=self._config.random_state,
            n_jobs=self._config.n_jobs,
            class_weight=self._config.class_weight,
        )

        self._model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self._model,
            X,
            y,
            cv=min(self._config.cross_validation_folds, len(set(y))),
            n_jobs=self._config.n_jobs,
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        specificity = cm[0, 0] / max(cm[0].sum(), 1) if len(cm) > 1 else 0.0

        # AUC-ROC if we have probability predictions
        try:
            auc = roc_auc_score(y_test, y_proba[:, 1]) if len(set(y_test)) > 1 else None
        except Exception:
            auc = None

        # Feature importances
        importances = dict(
            zip(self._feature_names, self._model.feature_importances_.tolist())
        )

        training_time = time.time() - start_time
        self._trained = True

        self._metrics = ReDoSModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            auc_roc=auc,
            confusion_matrix=cm.tolist(),
            feature_importances=importances,
            cross_val_scores=cv_scores.tolist(),
            training_samples=len(data),
            training_time_seconds=training_time,
        )

        logger.info(f"Random Forest training complete. Accuracy: {accuracy:.2%}, F1: {f1:.2%}")

        return self._metrics

    def get_feature_importance(self) -> List[float]:
        """Get feature importance from trained model."""
        if self._model is not None and hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_.tolist()
        return [0.0] * len(self._feature_names)

    def _save_model_data(self) -> Dict[str, Any]:
        """Save the sklearn model."""
        return {"sklearn_model": self._model}

    def _load_model_data(self, data: Dict[str, Any]) -> None:
        """Load the sklearn model."""
        self._model = data.get("sklearn_model")
        if self._model is not None:
            self._trained = True


class GradientBoostingReDoSModel(BaseReDoSModel):
    """Gradient Boosting classifier for ReDoS prediction.

    This model uses scikit-learn's GradientBoostingClassifier which
    often provides higher accuracy than Random Forest through
    sequential boosting.

    Gradient Boosting provides:
    - Often higher accuracy than Random Forest
    - Good handling of imbalanced classes
    - Built-in feature importance
    """

    name = "gradient_boosting"
    version = "1.0.0"

    def __init__(self, config: ModelConfig | None = None):
        """Initialize the Gradient Boosting model."""
        super().__init__(config)
        self._model: Any = None
        self._sklearn_available = _check_sklearn_available()

    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Predict risk probability."""
        # Use rule fallback if sklearn wasn't available
        if hasattr(self, "_rule_fallback") and self._rule_fallback is not None:
            return self._rule_fallback.predict(features)

        if not self._trained or self._model is None:
            return RuleBasedReDoSModel(self._config).predict(features)

        import numpy as np

        X = np.array([features])
        proba = self._model.predict_proba(X)[0]

        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = abs(probability - 0.5) * 2

        return probability, confidence

    def predict_batch(
        self, features: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple samples."""
        # Use rule fallback if sklearn wasn't available
        if hasattr(self, "_rule_fallback") and self._rule_fallback is not None:
            return self._rule_fallback.predict_batch(features)

        if not self._trained or self._model is None:
            fallback = RuleBasedReDoSModel(self._config)
            return fallback.predict_batch(features)

        import numpy as np

        X = np.array(features)
        probas = self._model.predict_proba(X)

        results = []
        for proba in probas:
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            confidence = abs(probability - 0.5) * 2
            results.append((probability, confidence))

        return results

    def train(self, data: ReDoSTrainingData) -> ReDoSModelMetrics:
        """Train the Gradient Boosting model."""
        if not self._sklearn_available:
            logger.warning("scikit-learn not available, using rule-based fallback")
            self._rule_fallback = RuleBasedReDoSModel(self._config)
            metrics = self._rule_fallback.train(data)
            self._trained = True
            self._metrics = metrics
            return metrics

        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import cross_val_score, train_test_split

        start_time = time.time()

        # Prepare features
        if data.features is None:
            from truthound.validators.security.redos.ml.features import (
                PatternFeatureExtractor,
            )

            extractor = PatternFeatureExtractor()
            X = np.array([extractor.extract(p).to_vector() for p in data.patterns])
        else:
            X = np.array(data.features)

        y = np.array(data.labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self._config.validation_split,
            random_state=self._config.random_state,
            stratify=y if len(set(y)) > 1 else None,
        )

        # Create and train model (GB works better with shallow trees)
        self._model = GradientBoostingClassifier(
            n_estimators=self._config.n_estimators,
            max_depth=min(self._config.max_depth, 5),
            min_samples_split=self._config.min_samples_split,
            min_samples_leaf=self._config.min_samples_leaf,
            learning_rate=self._config.learning_rate,
            random_state=self._config.random_state,
        )

        self._model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self._model,
            X,
            y,
            cv=min(self._config.cross_validation_folds, len(set(y))),
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        specificity = cm[0, 0] / max(cm[0].sum(), 1) if len(cm) > 1 else 0.0

        try:
            auc = roc_auc_score(y_test, y_proba[:, 1]) if len(set(y_test)) > 1 else None
        except Exception:
            auc = None

        importances = dict(
            zip(self._feature_names, self._model.feature_importances_.tolist())
        )

        training_time = time.time() - start_time
        self._trained = True

        self._metrics = ReDoSModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            auc_roc=auc,
            confusion_matrix=cm.tolist(),
            feature_importances=importances,
            cross_val_scores=cv_scores.tolist(),
            training_samples=len(data),
            training_time_seconds=training_time,
        )

        logger.info(f"Gradient Boosting training complete. Accuracy: {accuracy:.2%}, F1: {f1:.2%}")

        return self._metrics

    def get_feature_importance(self) -> List[float]:
        """Get feature importance from trained model."""
        if self._model is not None and hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_.tolist()
        return [0.0] * len(self._feature_names)

    def _save_model_data(self) -> Dict[str, Any]:
        """Save the sklearn model."""
        return {"sklearn_model": self._model}

    def _load_model_data(self, data: Dict[str, Any]) -> None:
        """Load the sklearn model."""
        self._model = data.get("sklearn_model")
        if self._model is not None:
            self._trained = True


class LogisticRegressionReDoSModel(BaseReDoSModel):
    """Logistic Regression classifier for ReDoS prediction.

    Simple linear model that provides interpretable coefficients
    and fast training/inference.
    """

    name = "logistic_regression"
    version = "1.0.0"

    def __init__(self, config: ModelConfig | None = None):
        """Initialize the Logistic Regression model."""
        super().__init__(config)
        self._model: Any = None
        self._sklearn_available = _check_sklearn_available()

    def predict(self, features: List[float]) -> Tuple[float, float]:
        """Predict risk probability."""
        # Use rule fallback if sklearn wasn't available
        if hasattr(self, "_rule_fallback") and self._rule_fallback is not None:
            return self._rule_fallback.predict(features)

        if not self._trained or self._model is None:
            return RuleBasedReDoSModel(self._config).predict(features)

        import numpy as np

        X = np.array([features])
        proba = self._model.predict_proba(X)[0]

        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
        confidence = abs(probability - 0.5) * 2

        return probability, confidence

    def predict_batch(
        self, features: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple samples."""
        # Use rule fallback if sklearn wasn't available
        if hasattr(self, "_rule_fallback") and self._rule_fallback is not None:
            return self._rule_fallback.predict_batch(features)

        if not self._trained or self._model is None:
            fallback = RuleBasedReDoSModel(self._config)
            return fallback.predict_batch(features)

        import numpy as np

        X = np.array(features)
        probas = self._model.predict_proba(X)

        results = []
        for proba in probas:
            probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            confidence = abs(probability - 0.5) * 2
            results.append((probability, confidence))

        return results

    def train(self, data: ReDoSTrainingData) -> ReDoSModelMetrics:
        """Train the Logistic Regression model."""
        if not self._sklearn_available:
            logger.warning("scikit-learn not available, using rule-based fallback")
            self._rule_fallback = RuleBasedReDoSModel(self._config)
            metrics = self._rule_fallback.train(data)
            self._trained = True
            self._metrics = metrics
            return metrics

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler

        start_time = time.time()

        # Prepare features
        if data.features is None:
            from truthound.validators.security.redos.ml.features import (
                PatternFeatureExtractor,
            )

            extractor = PatternFeatureExtractor()
            X = np.array([extractor.extract(p).to_vector() for p in data.patterns])
        else:
            X = np.array(data.features)

        y = np.array(data.labels)

        # Scale features for logistic regression
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=self._config.validation_split,
            random_state=self._config.random_state,
            stratify=y if len(set(y)) > 1 else None,
        )

        # Create and train model
        self._model = LogisticRegression(
            random_state=self._config.random_state,
            class_weight=self._config.class_weight,
            max_iter=1000,
            n_jobs=self._config.n_jobs,
        )

        self._model.fit(X_train, y_train)

        # Evaluate
        y_pred = self._model.predict(X_test)
        y_proba = self._model.predict_proba(X_test)

        # Cross-validation
        cv_scores = cross_val_score(
            self._model,
            X_scaled,
            y,
            cv=min(self._config.cross_validation_folds, len(set(y))),
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        specificity = cm[0, 0] / max(cm[0].sum(), 1) if len(cm) > 1 else 0.0

        try:
            auc = roc_auc_score(y_test, y_proba[:, 1]) if len(set(y_test)) > 1 else None
        except Exception:
            auc = None

        # Feature importance from coefficients
        coeffs = self._model.coef_[0] if len(self._model.coef_.shape) > 1 else self._model.coef_
        importances = dict(zip(self._feature_names, np.abs(coeffs).tolist()))

        training_time = time.time() - start_time
        self._trained = True

        self._metrics = ReDoSModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            auc_roc=auc,
            confusion_matrix=cm.tolist(),
            feature_importances=importances,
            cross_val_scores=cv_scores.tolist(),
            training_samples=len(data),
            training_time_seconds=training_time,
        )

        logger.info(f"Logistic Regression training complete. Accuracy: {accuracy:.2%}, F1: {f1:.2%}")

        return self._metrics

    def get_feature_importance(self) -> List[float]:
        """Get feature importance from coefficients."""
        if self._model is not None and hasattr(self._model, "coef_"):
            import numpy as np

            coeffs = self._model.coef_[0] if len(self._model.coef_.shape) > 1 else self._model.coef_
            return np.abs(coeffs).tolist()
        return [0.0] * len(self._feature_names)

    def _save_model_data(self) -> Dict[str, Any]:
        """Save the sklearn model and scaler."""
        return {
            "sklearn_model": self._model,
            "scaler": getattr(self, "_scaler", None),
        }

    def _load_model_data(self, data: Dict[str, Any]) -> None:
        """Load the sklearn model and scaler."""
        self._model = data.get("sklearn_model")
        self._scaler = data.get("scaler")
        if self._model is not None:
            self._trained = True


class EnsembleReDoSModel(BaseReDoSModel):
    """Ensemble model combining multiple prediction strategies.

    This model combines rule-based heuristics with pattern signature
    matching for robust predictions even without training data.

    The ensemble uses weighted voting from:
    - Rule-based baseline model
    - Trained ML model (if available)
    - Pattern signature matching

    This approach provides robust fallback behavior while leveraging
    ML improvements when trained models are available.
    """

    name = "ensemble"
    version = "1.0.0"

    # Known dangerous pattern signatures with risk scores
    DANGEROUS_SIGNATURES: List[Tuple[str, float]] = [
        (r"\([^)]*[+*][^)]*\)[+*]", 0.95),  # Nested quantifiers
        (r"\\[1-9][+*]", 0.85),  # Quantified backreference
        (r"\([^)]*\|[^)]*\)[+*]", 0.75),  # Quantified alternation
        (r"[+*][+*]", 0.65),  # Adjacent quantifiers
        (r"\([^)]*\)\{[\d,]+\}\{", 0.70),  # Nested bounded quantifiers
    ]

    def __init__(
        self,
        config: ModelConfig | None = None,
        ml_model: BaseReDoSModel | None = None,
    ):
        """Initialize the ensemble model.

        Args:
            config: Model configuration
            ml_model: Optional trained ML model to include in ensemble
        """
        import re

        super().__init__(config)
        self._rule_model = RuleBasedReDoSModel(config)
        self._ml_model = ml_model
        self._trained = True  # Rule-based is always ready

        # Compile signature patterns
        self._compiled_signatures = [
            (re.compile(pattern), risk) for pattern, risk in self.DANGEROUS_SIGNATURES
        ]

    def predict(
        self, features: List[float], pattern: str = ""
    ) -> Tuple[float, float]:
        """Predict using ensemble of methods.

        Args:
            features: Feature vector
            pattern: Original pattern for signature matching

        Returns:
            Tuple of (risk_probability, confidence)
        """
        # Rule-based prediction
        rule_prob, rule_conf = self._rule_model.predict(features)

        # ML model prediction (if available and trained)
        ml_prob, ml_conf = 0.0, 0.0
        if self._ml_model is not None and self._ml_model.is_trained:
            ml_prob, ml_conf = self._ml_model.predict(features)

        # Pattern signature matching
        sig_prob = 0.0
        if pattern:
            for sig_pattern, risk in self._compiled_signatures:
                if sig_pattern.search(pattern):
                    sig_prob = max(sig_prob, risk)

        # Combine predictions
        if self._ml_model is not None and self._ml_model.is_trained:
            # ML model available: weighted average of all three
            if sig_prob > 0:
                final_prob = 0.4 * ml_prob + 0.35 * sig_prob + 0.25 * rule_prob
                final_conf = max(ml_conf, 0.9)
            else:
                final_prob = 0.6 * ml_prob + 0.4 * rule_prob
                final_conf = (ml_conf + rule_conf) / 2
        else:
            # No ML model: combine rule-based with signatures
            if sig_prob > 0:
                final_prob = 0.6 * sig_prob + 0.4 * rule_prob
                final_conf = max(rule_conf, 0.9)
            else:
                final_prob = rule_prob
                final_conf = rule_conf

        return final_prob, final_conf

    def predict_batch(
        self, features: List[List[float]]
    ) -> List[Tuple[float, float]]:
        """Predict for multiple samples (without pattern context)."""
        return [self.predict(f) for f in features]

    def predict_with_pattern(
        self, features: List[float], pattern: str
    ) -> Tuple[float, float]:
        """Predict with pattern context for signature matching."""
        return self.predict(features, pattern)

    def train(self, data: ReDoSTrainingData) -> ReDoSModelMetrics:
        """Train the ML component of the ensemble.

        The rule-based component doesn't need training, but the ML
        component benefits from training data.
        """
        # Create and train ML model if not provided
        if self._ml_model is None:
            self._ml_model = RandomForestReDoSModel(self._config)

        metrics = self._ml_model.train(data)
        self._metrics = metrics
        return metrics

    def get_feature_importance(self) -> List[float]:
        """Get feature importance from rule model."""
        if self._ml_model is not None and self._ml_model.is_trained:
            return self._ml_model.get_feature_importance()
        return self._rule_model.get_feature_importance()

    def _save_model_data(self) -> Dict[str, Any]:
        """Save ensemble components."""
        return {
            "ml_model_data": (
                self._ml_model._save_model_data()
                if self._ml_model is not None
                else None
            ),
            "ml_model_type": (
                self._ml_model.name if self._ml_model is not None else None
            ),
        }

    def _load_model_data(self, data: Dict[str, Any]) -> None:
        """Load ensemble components."""
        if data.get("ml_model_type") and data.get("ml_model_data"):
            model_type = data["ml_model_type"]
            self._ml_model = create_model(model_type, self._config)
            self._ml_model._load_model_data(data["ml_model_data"])


# =============================================================================
# Model Registry and Factory
# =============================================================================


MODEL_REGISTRY: Dict[str, Type[BaseReDoSModel]] = {
    "rule_based": RuleBasedReDoSModel,
    "random_forest": RandomForestReDoSModel,
    "gradient_boosting": GradientBoostingReDoSModel,
    "logistic_regression": LogisticRegressionReDoSModel,
    "ensemble": EnsembleReDoSModel,
}


def create_model(
    model_type: str | ModelType,
    config: ModelConfig | None = None,
) -> BaseReDoSModel:
    """Create a ReDoS model by type.

    Args:
        model_type: Type of model to create
        config: Optional model configuration

    Returns:
        Instantiated model

    Raises:
        ValueError: If model type is not recognized
    """
    if isinstance(model_type, ModelType):
        model_type = model_type.value

    model_class = MODEL_REGISTRY.get(model_type)
    if model_class is None:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {available}"
        )

    return model_class(config)


def register_model(name: str, model_class: Type[BaseReDoSModel]) -> None:
    """Register a custom model type.

    Args:
        name: Name to register the model under
        model_class: Model class to register
    """
    MODEL_REGISTRY[name] = model_class


def list_available_models() -> List[str]:
    """List all available model types.

    Returns:
        List of model type names
    """
    return list(MODEL_REGISTRY.keys())
