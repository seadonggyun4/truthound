"""Performance prediction models for execution time estimation.

This module provides historical data-based prediction models for
estimating validator execution times. It enables:
- Proactive timeout configuration
- Resource planning
- SLA compliance prediction

Multiple prediction models are supported:
- Moving Average: Simple rolling average
- Exponential Smoothing: Recent data weighted more heavily
- Quantile Regression: Predict percentiles for safety margins

Example:
    from truthound.validators.timeout.advanced.prediction import (
        PerformancePredictor,
        predict_execution_time,
    )

    predictor = PerformancePredictor()

    # Record historical executions
    predictor.record("null_check", 150.0, {"rows": 10000})
    predictor.record("null_check", 145.0, {"rows": 10000})
    predictor.record("null_check", 160.0, {"rows": 10000})

    # Predict execution time
    prediction = predictor.predict("null_check", {"rows": 10000})
    print(f"Predicted: {prediction.estimated_ms:.0f}ms")
    print(f"P95 estimate: {prediction.p95_ms:.0f}ms")
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque


@dataclass
class ExecutionRecord:
    """Record of a single execution.

    Attributes:
        operation: Operation name
        duration_ms: Execution duration in milliseconds
        features: Features (e.g., row count, column count)
        timestamp: When the execution occurred
        success: Whether execution succeeded
        metadata: Additional metadata
    """

    operation: str
    duration_ms: float
    features: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionHistory:
    """History of execution records for an operation.

    Maintains a sliding window of recent executions for analysis.
    """

    max_size: int = 1000
    _records: Deque[ExecutionRecord] = field(default_factory=lambda: deque(maxlen=1000))

    def __post_init__(self) -> None:
        """Initialize with proper maxlen."""
        self._records = deque(maxlen=self.max_size)

    def add(self, record: ExecutionRecord) -> None:
        """Add a record to history."""
        self._records.append(record)

    def get_durations(self) -> list[float]:
        """Get list of durations."""
        return [r.duration_ms for r in self._records]

    def get_recent(self, n: int = 10) -> list[ExecutionRecord]:
        """Get N most recent records."""
        return list(self._records)[-n:]

    def filter_by_features(
        self,
        features: dict[str, Any],
        tolerance: float = 0.2,
    ) -> list[ExecutionRecord]:
        """Filter records by similar features.

        Args:
            features: Features to match
            tolerance: Tolerance for numeric comparisons (0.2 = 20%)

        Returns:
            Matching records
        """
        matching = []
        for record in self._records:
            if self._features_match(record.features, features, tolerance):
                matching.append(record)
        return matching

    def _features_match(
        self,
        record_features: dict[str, Any],
        target_features: dict[str, Any],
        tolerance: float,
    ) -> bool:
        """Check if features match within tolerance."""
        for key, target_value in target_features.items():
            if key not in record_features:
                continue
            record_value = record_features[key]

            if isinstance(target_value, (int, float)) and isinstance(record_value, (int, float)):
                if target_value == 0:
                    if record_value != 0:
                        return False
                elif abs(record_value - target_value) / abs(target_value) > tolerance:
                    return False
            elif record_value != target_value:
                return False

        return True

    def __len__(self) -> int:
        return len(self._records)


@dataclass
class PredictionResult:
    """Result of a performance prediction.

    Attributes:
        operation: Operation name
        estimated_ms: Estimated execution time in milliseconds
        confidence: Confidence level (0.0-1.0)
        p50_ms: 50th percentile estimate
        p90_ms: 90th percentile estimate
        p95_ms: 95th percentile estimate
        p99_ms: 99th percentile estimate
        sample_count: Number of samples used for prediction
        model_used: Name of prediction model used
        features: Features used for prediction
        metadata: Additional metadata
    """

    operation: str
    estimated_ms: float
    confidence: float = 0.5
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    sample_count: int = 0
    model_used: str = "unknown"
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def suggested_timeout_ms(self) -> float:
        """Get suggested timeout (P95 with 20% buffer)."""
        if self.p95_ms is not None:
            return self.p95_ms * 1.2
        return self.estimated_ms * 2.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "estimated_ms": self.estimated_ms,
            "confidence": self.confidence,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "suggested_timeout_ms": self.suggested_timeout_ms,
            "sample_count": self.sample_count,
            "model_used": self.model_used,
        }


class PredictionModel(ABC):
    """Base class for prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @abstractmethod
    def predict(
        self,
        history: ExecutionHistory,
        features: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Make a prediction based on history.

        Args:
            history: Execution history
            features: Optional features for feature-based prediction

        Returns:
            PredictionResult
        """
        pass


class MovingAverageModel(PredictionModel):
    """Simple moving average prediction model.

    Uses the mean of the last N executions.
    """

    def __init__(self, window_size: int = 20):
        """Initialize moving average model.

        Args:
            window_size: Number of recent executions to average
        """
        self.window_size = window_size

    @property
    def name(self) -> str:
        return f"moving_average_{self.window_size}"

    def predict(
        self,
        history: ExecutionHistory,
        features: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Predict using moving average."""
        durations = history.get_durations()
        if not durations:
            return PredictionResult(
                operation="unknown",
                estimated_ms=0.0,
                confidence=0.0,
                model_used=self.name,
            )

        # Use recent values
        recent = durations[-self.window_size:]
        n = len(recent)

        mean = statistics.mean(recent)
        confidence = min(1.0, n / self.window_size)

        # Calculate percentiles
        sorted_recent = sorted(recent)
        p50 = sorted_recent[n // 2] if n > 0 else mean
        p90 = sorted_recent[int(n * 0.9)] if n >= 10 else mean * 1.5
        p95 = sorted_recent[int(n * 0.95)] if n >= 20 else mean * 1.8
        p99 = sorted_recent[int(n * 0.99)] if n >= 100 else mean * 2.5

        return PredictionResult(
            operation="",
            estimated_ms=mean,
            confidence=confidence,
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            sample_count=n,
            model_used=self.name,
            features=features or {},
        )


class ExponentialSmoothingModel(PredictionModel):
    """Exponential smoothing prediction model.

    Gives more weight to recent observations.
    """

    def __init__(self, alpha: float = 0.3):
        """Initialize exponential smoothing model.

        Args:
            alpha: Smoothing factor (0.0-1.0), higher = more weight on recent
        """
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"exponential_smoothing_{self.alpha}"

    def predict(
        self,
        history: ExecutionHistory,
        features: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Predict using exponential smoothing."""
        durations = history.get_durations()
        if not durations:
            return PredictionResult(
                operation="unknown",
                estimated_ms=0.0,
                confidence=0.0,
                model_used=self.name,
            )

        # Exponential smoothing
        smoothed = durations[0]
        for value in durations[1:]:
            smoothed = self.alpha * value + (1 - self.alpha) * smoothed

        n = len(durations)
        confidence = min(1.0, n / 20)

        # Calculate variance for percentiles
        if n > 1:
            variance = statistics.variance(durations)
            std_dev = math.sqrt(variance)
        else:
            std_dev = smoothed * 0.3  # Assume 30% variability

        return PredictionResult(
            operation="",
            estimated_ms=smoothed,
            confidence=confidence,
            p50_ms=smoothed,
            p90_ms=smoothed + 1.28 * std_dev,
            p95_ms=smoothed + 1.645 * std_dev,
            p99_ms=smoothed + 2.326 * std_dev,
            sample_count=n,
            model_used=self.name,
            features=features or {},
        )


class QuantileRegressionModel(PredictionModel):
    """Quantile regression model for percentile prediction.

    This model directly estimates percentiles from the data,
    providing more accurate tail estimates for timeout planning.
    """

    def __init__(self, quantiles: list[float] | None = None):
        """Initialize quantile regression model.

        Args:
            quantiles: Quantiles to estimate (default: [0.5, 0.9, 0.95, 0.99])
        """
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]

    @property
    def name(self) -> str:
        return "quantile_regression"

    def predict(
        self,
        history: ExecutionHistory,
        features: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Predict using quantile regression."""
        # Filter by features if provided
        if features:
            records = history.filter_by_features(features)
            durations = [r.duration_ms for r in records]
        else:
            durations = history.get_durations()

        if not durations:
            return PredictionResult(
                operation="unknown",
                estimated_ms=0.0,
                confidence=0.0,
                model_used=self.name,
            )

        n = len(durations)
        sorted_durations = sorted(durations)

        def get_quantile(q: float) -> float:
            """Get quantile value."""
            idx = int(n * q)
            idx = max(0, min(idx, n - 1))
            return sorted_durations[idx]

        p50 = get_quantile(0.5)
        p90 = get_quantile(0.9)
        p95 = get_quantile(0.95)
        p99 = get_quantile(0.99)

        # Confidence based on sample size
        confidence = min(1.0, n / 50)

        return PredictionResult(
            operation="",
            estimated_ms=p50,  # Use median as estimate
            confidence=confidence,
            p50_ms=p50,
            p90_ms=p90,
            p95_ms=p95,
            p99_ms=p99,
            sample_count=n,
            model_used=self.name,
            features=features or {},
        )


class FeatureBasedModel(PredictionModel):
    """Feature-based prediction model.

    Uses linear regression on features (e.g., row count) to
    predict execution time.
    """

    def __init__(self, feature_key: str = "rows"):
        """Initialize feature-based model.

        Args:
            feature_key: Feature to use for prediction
        """
        self.feature_key = feature_key

    @property
    def name(self) -> str:
        return f"feature_based_{self.feature_key}"

    def predict(
        self,
        history: ExecutionHistory,
        features: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Predict using feature-based regression."""
        if not features or self.feature_key not in features:
            # Fall back to simple average
            durations = history.get_durations()
            if not durations:
                return PredictionResult(
                    operation="unknown",
                    estimated_ms=0.0,
                    confidence=0.0,
                    model_used=self.name,
                )
            return PredictionResult(
                operation="",
                estimated_ms=statistics.mean(durations),
                confidence=0.3,
                model_used=self.name,
            )

        target_feature = features[self.feature_key]
        if not isinstance(target_feature, (int, float)) or target_feature <= 0:
            return PredictionResult(
                operation="",
                estimated_ms=0.0,
                confidence=0.0,
                model_used=self.name,
            )

        # Collect data points
        x_values = []
        y_values = []

        for record in history.get_recent(100):
            feature_value = record.features.get(self.feature_key)
            if isinstance(feature_value, (int, float)) and feature_value > 0:
                x_values.append(feature_value)
                y_values.append(record.duration_ms)

        if len(x_values) < 2:
            return PredictionResult(
                operation="",
                estimated_ms=0.0,
                confidence=0.0,
                model_used=self.name,
            )

        # Simple linear regression
        n = len(x_values)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n

        # Calculate slope and intercept
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        denominator = sum((x - mean_x) ** 2 for x in x_values)

        if denominator == 0:
            slope = 0
            intercept = mean_y
        else:
            slope = numerator / denominator
            intercept = mean_y - slope * mean_x

        # Predict
        predicted = intercept + slope * target_feature
        predicted = max(0, predicted)  # Ensure non-negative

        # Calculate residuals for confidence
        residuals = [
            abs(y - (intercept + slope * x))
            for x, y in zip(x_values, y_values)
        ]
        mean_residual = sum(residuals) / n
        confidence = max(0.1, 1.0 - (mean_residual / max(mean_y, 1.0)))

        return PredictionResult(
            operation="",
            estimated_ms=predicted,
            confidence=min(1.0, confidence),
            p50_ms=predicted,
            p90_ms=predicted * 1.5,
            p95_ms=predicted * 2.0,
            p99_ms=predicted * 3.0,
            sample_count=n,
            model_used=self.name,
            features=features,
            metadata={
                "slope": slope,
                "intercept": intercept,
                "r_squared": self._calculate_r_squared(x_values, y_values, slope, intercept),
            },
        )

    def _calculate_r_squared(
        self,
        x_values: list[float],
        y_values: list[float],
        slope: float,
        intercept: float,
    ) -> float:
        """Calculate R-squared for the regression."""
        n = len(y_values)
        if n == 0:
            return 0.0

        mean_y = sum(y_values) / n
        ss_tot = sum((y - mean_y) ** 2 for y in y_values)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(x_values, y_values))

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - (ss_res / ss_tot)


class PerformancePredictor:
    """Main performance predictor using ensemble of models.

    This class manages execution history and uses multiple prediction
    models to provide robust performance estimates.

    Example:
        predictor = PerformancePredictor()

        # Record executions
        predictor.record("validate", 100.0, {"rows": 10000})
        predictor.record("validate", 105.0, {"rows": 10000})

        # Predict
        result = predictor.predict("validate", {"rows": 20000})
        print(f"Estimated: {result.estimated_ms}ms")
    """

    def __init__(
        self,
        models: list[PredictionModel] | None = None,
        history_size: int = 1000,
    ):
        """Initialize predictor.

        Args:
            models: Prediction models to use (default: all available)
            history_size: Maximum history size per operation
        """
        self.models = models or [
            MovingAverageModel(20),
            ExponentialSmoothingModel(0.3),
            QuantileRegressionModel(),
            FeatureBasedModel("rows"),
        ]
        self.history_size = history_size
        self._histories: dict[str, ExecutionHistory] = defaultdict(
            lambda: ExecutionHistory(max_size=history_size)
        )
        self._lock = threading.Lock()

    def record(
        self,
        operation: str,
        duration_ms: float,
        features: dict[str, Any] | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an execution.

        Args:
            operation: Operation name
            duration_ms: Duration in milliseconds
            features: Features of the execution
            success: Whether execution succeeded
            metadata: Additional metadata
        """
        record = ExecutionRecord(
            operation=operation,
            duration_ms=duration_ms,
            features=features or {},
            success=success,
            metadata=metadata or {},
        )

        with self._lock:
            self._histories[operation].add(record)

    def predict(
        self,
        operation: str,
        features: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Predict execution time.

        Args:
            operation: Operation name
            features: Features for feature-based prediction

        Returns:
            PredictionResult with estimates
        """
        with self._lock:
            history = self._histories.get(operation)

        if not history or len(history) == 0:
            return PredictionResult(
                operation=operation,
                estimated_ms=0.0,
                confidence=0.0,
                model_used="none",
                metadata={"reason": "no_history"},
            )

        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                result = model.predict(history, features)
                if result.confidence > 0:
                    predictions.append(result)
            except Exception:
                continue

        if not predictions:
            return PredictionResult(
                operation=operation,
                estimated_ms=0.0,
                confidence=0.0,
                model_used="none",
                metadata={"reason": "all_models_failed"},
            )

        # Weighted ensemble
        total_weight = sum(p.confidence for p in predictions)
        if total_weight == 0:
            total_weight = len(predictions)
            weights = [1.0 / len(predictions)] * len(predictions)
        else:
            weights = [p.confidence / total_weight for p in predictions]

        estimated_ms = sum(p.estimated_ms * w for p, w in zip(predictions, weights))
        p50_ms = sum((p.p50_ms or p.estimated_ms) * w for p, w in zip(predictions, weights))
        p90_ms = sum((p.p90_ms or p.estimated_ms * 1.5) * w for p, w in zip(predictions, weights))
        p95_ms = sum((p.p95_ms or p.estimated_ms * 2.0) * w for p, w in zip(predictions, weights))
        p99_ms = sum((p.p99_ms or p.estimated_ms * 3.0) * w for p, w in zip(predictions, weights))

        avg_confidence = sum(p.confidence * w for p, w in zip(predictions, weights))
        total_samples = sum(p.sample_count for p in predictions)

        return PredictionResult(
            operation=operation,
            estimated_ms=estimated_ms,
            confidence=avg_confidence,
            p50_ms=p50_ms,
            p90_ms=p90_ms,
            p95_ms=p95_ms,
            p99_ms=p99_ms,
            sample_count=total_samples,
            model_used="ensemble",
            features=features or {},
            metadata={
                "models_used": [p.model_used for p in predictions],
                "model_estimates": [p.estimated_ms for p in predictions],
            },
        )

    def get_history(self, operation: str) -> ExecutionHistory | None:
        """Get execution history for an operation.

        Args:
            operation: Operation name

        Returns:
            ExecutionHistory or None
        """
        with self._lock:
            return self._histories.get(operation)

    def get_operations(self) -> list[str]:
        """Get list of operations with history.

        Returns:
            List of operation names
        """
        with self._lock:
            return list(self._histories.keys())

    def clear(self, operation: str | None = None) -> None:
        """Clear history.

        Args:
            operation: Operation to clear (None = all)
        """
        with self._lock:
            if operation:
                if operation in self._histories:
                    del self._histories[operation]
            else:
                self._histories.clear()


# Module-level predictor instance
_default_predictor: PerformancePredictor | None = None


def predict_execution_time(
    operation: str,
    features: dict[str, Any] | None = None,
) -> PredictionResult:
    """Predict execution time using default predictor.

    Args:
        operation: Operation name
        features: Features for prediction

    Returns:
        PredictionResult
    """
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = PerformancePredictor()
    return _default_predictor.predict(operation, features)


def record_execution(
    operation: str,
    duration_ms: float,
    features: dict[str, Any] | None = None,
    success: bool = True,
) -> None:
    """Record execution using default predictor.

    Args:
        operation: Operation name
        duration_ms: Duration in milliseconds
        features: Features of execution
        success: Whether execution succeeded
    """
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = PerformancePredictor()
    _default_predictor.record(operation, duration_ms, features, success)
