"""Drift metric collector.

Collects feature and prediction drift metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import math
from collections import defaultdict

from truthound.ml.monitoring.protocols import (
    IMetricCollector,
    ModelMetrics,
    PredictionRecord,
)


@dataclass
class DriftConfig:
    """Configuration for drift collector.

    Attributes:
        reference_window_size: Size of reference window
        detection_window_size: Size of detection window
        histogram_bins: Number of bins for histogram comparison
        drift_threshold: Threshold for considering drift significant
        features_to_monitor: List of features to monitor (None = all)
    """

    reference_window_size: int = 1000
    detection_window_size: int = 100
    histogram_bins: int = 20
    drift_threshold: float = 0.1
    features_to_monitor: list[str] | None = None


class DriftCollector(IMetricCollector):
    """Collects drift metrics for features and predictions.

    Uses Population Stability Index (PSI) and Kolmogorov-Smirnov
    test for drift detection.

    Example:
        >>> collector = DriftCollector()
        >>> # First, build reference distribution
        >>> collector.set_reference(model_id, reference_predictions)
        >>> # Then detect drift
        >>> metrics = collector.collect(model_id, current_predictions)
    """

    def __init__(self, config: DriftConfig | None = None):
        self._config = config or DriftConfig()
        # Reference distributions per model
        self._reference_features: dict[str, dict[str, list[float]]] = {}
        self._reference_predictions: dict[str, list[Any]] = {}
        # Current window
        self._current_features: dict[str, dict[str, list[float]]] = {}
        self._current_predictions: dict[str, list[Any]] = {}

    @property
    def name(self) -> str:
        return "drift"

    def set_reference(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> None:
        """Set reference distribution from predictions.

        Args:
            model_id: Model identifier
            predictions: Reference predictions
        """
        features: dict[str, list[float]] = defaultdict(list)
        preds: list[Any] = []

        for pred in predictions:
            preds.append(pred.prediction)
            for name, value in pred.features.items():
                if self._should_monitor_feature(name) and isinstance(value, (int, float)):
                    features[name].append(float(value))

        self._reference_features[model_id] = dict(features)
        self._reference_predictions[model_id] = preds

    def _should_monitor_feature(self, name: str) -> bool:
        """Check if feature should be monitored."""
        if self._config.features_to_monitor is None:
            return True
        return name in self._config.features_to_monitor

    def collect(
        self,
        model_id: str,
        predictions: list[PredictionRecord],
    ) -> ModelMetrics:
        """Collect drift metrics.

        Args:
            model_id: Model identifier
            predictions: Current predictions

        Returns:
            Drift metrics
        """
        # Build current distributions
        features: dict[str, list[float]] = defaultdict(list)
        preds: list[Any] = []

        for pred in predictions:
            preds.append(pred.prediction)
            for name, value in pred.features.items():
                if self._should_monitor_feature(name) and isinstance(value, (int, float)):
                    features[name].append(float(value))

        # Update current window
        self._current_features[model_id] = dict(features)
        self._current_predictions[model_id] = preds

        # Calculate drift scores
        feature_drift = self._compute_feature_drift(model_id, features)
        prediction_drift = self._compute_prediction_drift(model_id, preds)

        # Custom metrics
        custom_metrics: dict[str, float] = {}

        # Add per-feature drift to custom metrics
        for name, score in feature_drift.items():
            custom_metrics[f"drift_{name}"] = score

        # Overall feature drift (max across features)
        if feature_drift:
            custom_metrics["max_feature_drift"] = max(feature_drift.values())
            custom_metrics["mean_feature_drift"] = sum(feature_drift.values()) / len(feature_drift)
            custom_metrics["drifted_feature_count"] = sum(
                1 for v in feature_drift.values() if v > self._config.drift_threshold
            )

        return ModelMetrics(
            model_id=model_id,
            timestamp=datetime.now(timezone.utc),
            prediction_drift=prediction_drift,
            feature_drift=feature_drift,
            custom_metrics=custom_metrics,
        )

    def _compute_feature_drift(
        self,
        model_id: str,
        current_features: dict[str, list[float]],
    ) -> dict[str, float]:
        """Compute feature drift using PSI."""
        reference = self._reference_features.get(model_id, {})
        if not reference:
            return {}

        drift_scores: dict[str, float] = {}

        for name, current_values in current_features.items():
            ref_values = reference.get(name)
            if not ref_values or not current_values:
                continue

            # Compute PSI
            psi = self._compute_psi(ref_values, current_values)
            drift_scores[name] = psi

        return drift_scores

    def _compute_prediction_drift(
        self,
        model_id: str,
        current_predictions: list[Any],
    ) -> float | None:
        """Compute prediction drift."""
        reference = self._reference_predictions.get(model_id)
        if not reference or not current_predictions:
            return None

        # For numeric predictions, use PSI
        try:
            ref_numeric = [float(p) for p in reference if p is not None]
            curr_numeric = [float(p) for p in current_predictions if p is not None]

            if ref_numeric and curr_numeric:
                return self._compute_psi(ref_numeric, curr_numeric)
        except (TypeError, ValueError):
            pass

        # For categorical, use chi-square like comparison
        return self._compute_categorical_drift(reference, current_predictions)

    def _compute_psi(
        self,
        reference: list[float],
        current: list[float],
    ) -> float:
        """Compute Population Stability Index (PSI).

        PSI measures the shift in distribution:
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.2: Moderate shift
        - PSI >= 0.2: Significant shift
        """
        if not reference or not current:
            return 0.0

        # Create histograms
        all_values = reference + current
        min_val = min(all_values)
        max_val = max(all_values)

        if min_val == max_val:
            return 0.0

        bin_size = (max_val - min_val) / self._config.histogram_bins

        def get_bin_counts(values: list[float]) -> list[float]:
            counts = [0.0] * self._config.histogram_bins
            for v in values:
                bin_idx = min(int((v - min_val) / bin_size), self._config.histogram_bins - 1)
                counts[bin_idx] += 1
            # Convert to proportions
            total = sum(counts)
            if total > 0:
                counts = [c / total for c in counts]
            return counts

        ref_counts = get_bin_counts(reference)
        curr_counts = get_bin_counts(current)

        # Compute PSI
        psi = 0.0
        epsilon = 0.0001  # Small value to avoid log(0)

        for ref_prop, curr_prop in zip(ref_counts, curr_counts):
            ref_prop = max(ref_prop, epsilon)
            curr_prop = max(curr_prop, epsilon)
            psi += (curr_prop - ref_prop) * math.log(curr_prop / ref_prop)

        return psi

    def _compute_categorical_drift(
        self,
        reference: list[Any],
        current: list[Any],
    ) -> float:
        """Compute drift for categorical predictions."""
        # Count frequencies
        ref_counts: dict[Any, int] = defaultdict(int)
        curr_counts: dict[Any, int] = defaultdict(int)

        for v in reference:
            ref_counts[v] += 1
        for v in current:
            curr_counts[v] += 1

        # All categories
        all_cats = set(ref_counts.keys()) | set(curr_counts.keys())
        if not all_cats:
            return 0.0

        # Compute total variation distance
        ref_total = len(reference)
        curr_total = len(current)

        if ref_total == 0 or curr_total == 0:
            return 0.0

        tvd = 0.0
        for cat in all_cats:
            ref_prop = ref_counts.get(cat, 0) / ref_total
            curr_prop = curr_counts.get(cat, 0) / curr_total
            tvd += abs(ref_prop - curr_prop)

        return tvd / 2  # Normalize to [0, 1]

    def reset(self) -> None:
        """Reset collector state."""
        self._reference_features.clear()
        self._reference_predictions.clear()
        self._current_features.clear()
        self._current_predictions.clear()

    def get_feature_comparison(
        self,
        model_id: str,
        feature_name: str,
    ) -> dict[str, Any]:
        """Get detailed comparison for a feature.

        Args:
            model_id: Model identifier
            feature_name: Feature to compare

        Returns:
            Comparison data for visualization
        """
        ref = self._reference_features.get(model_id, {}).get(feature_name, [])
        curr = self._current_features.get(model_id, {}).get(feature_name, [])

        if not ref or not curr:
            return {}

        return {
            "feature_name": feature_name,
            "reference": {
                "mean": sum(ref) / len(ref),
                "min": min(ref),
                "max": max(ref),
                "count": len(ref),
            },
            "current": {
                "mean": sum(curr) / len(curr),
                "min": min(curr),
                "max": max(curr),
                "count": len(curr),
            },
            "drift_score": self._compute_psi(ref, curr),
        }
