"""Ensemble anomaly detection.

Combines multiple anomaly detection methods for more robust detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import polars as pl

from truthound.ml.base import (
    AnomalyConfig,
    AnomalyDetector,
    AnomalyResult,
    AnomalyScore,
    AnomalyType,
    InsufficientDataError,
    ModelInfo,
    ModelNotTrainedError,
    ModelState,
    ModelTrainingError,
    ModelType,
    register_model,
)


class EnsembleStrategy(str, Enum):
    """Strategy for combining anomaly scores."""

    AVERAGE = "average"  # Simple average
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by detector performance
    MAX = "max"  # Maximum score across detectors
    MIN = "min"  # Minimum score (conservative)
    VOTE = "vote"  # Majority voting
    UNANIMOUS = "unanimous"  # All detectors must agree


@dataclass
class EnsembleConfig(AnomalyConfig):
    """Configuration for ensemble anomaly detection.

    Attributes:
        strategy: Strategy for combining scores
        weights: Weights for weighted average (None = equal weights)
        vote_threshold: Fraction of detectors that must agree for VOTE strategy
    """

    strategy: EnsembleStrategy = EnsembleStrategy.AVERAGE
    weights: list[float] | None = None
    vote_threshold: float = 0.5


@register_model("ensemble")
class EnsembleAnomalyDetector(AnomalyDetector):
    """Ensemble of multiple anomaly detectors.

    Combines predictions from multiple detectors for more robust
    anomaly detection. Supports various combination strategies.

    Example:
        >>> from truthound.ml.anomaly_models import (
        ...     EnsembleAnomalyDetector,
        ...     ZScoreAnomalyDetector,
        ...     IQRAnomalyDetector,
        ...     IsolationForestDetector,
        ... )
        >>> ensemble = EnsembleAnomalyDetector(
        ...     detectors=[
        ...         ZScoreAnomalyDetector(),
        ...         IQRAnomalyDetector(),
        ...         IsolationForestDetector(),
        ...     ],
        ...     strategy=EnsembleStrategy.AVERAGE,
        ... )
        >>> ensemble.fit(train_data)
        >>> result = ensemble.predict(test_data)
    """

    def __init__(
        self,
        detectors: list[AnomalyDetector] | None = None,
        config: EnsembleConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self._detectors: list[AnomalyDetector] = detectors or []
        self._detector_weights: list[float] = []

    @property
    def config(self) -> EnsembleConfig:
        return self._config  # type: ignore

    def _default_config(self) -> EnsembleConfig:
        return EnsembleConfig()

    def _get_model_name(self) -> str:
        return "ensemble"

    def _get_description(self) -> str:
        return "Ensemble of multiple anomaly detectors"

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self._get_model_name(),
            version="1.0.0",
            model_type=ModelType.ANOMALY_DETECTOR,
            description=self._get_description(),
            min_samples_required=10,
            tags=("ensemble", "meta-model"),
        )

    def add_detector(
        self,
        detector: AnomalyDetector,
        weight: float = 1.0,
    ) -> None:
        """Add a detector to the ensemble.

        Args:
            detector: Detector to add
            weight: Weight for this detector
        """
        self._detectors.append(detector)
        self._detector_weights.append(weight)

    def remove_detector(self, index: int) -> AnomalyDetector:
        """Remove a detector from the ensemble.

        Args:
            index: Index of detector to remove

        Returns:
            Removed detector
        """
        detector = self._detectors.pop(index)
        self._detector_weights.pop(index)
        return detector

    def fit(self, data: pl.LazyFrame) -> None:
        """Train all detectors in the ensemble.

        Args:
            data: Training data
        """
        import time

        start = time.perf_counter()
        self._state = ModelState.TRAINING

        if not self._detectors:
            # Create default ensemble
            from truthound.ml.anomaly_models.statistical import (
                ZScoreAnomalyDetector,
                IQRAnomalyDetector,
                MADAnomalyDetector,
            )

            self._detectors = [
                ZScoreAnomalyDetector(),
                IQRAnomalyDetector(),
                MADAnomalyDetector(),
            ]

        try:
            row_count = self._validate_data(data)

            # Train each detector
            for detector in self._detectors:
                detector.fit(data)

            # Initialize weights if not provided
            if self.config.weights:
                self._detector_weights = list(self.config.weights)
            elif not self._detector_weights:
                self._detector_weights = [1.0] * len(self._detectors)

            # Normalize weights
            total_weight = sum(self._detector_weights)
            if total_weight > 0:
                self._detector_weights = [
                    w / total_weight for w in self._detector_weights
                ]

            self._training_samples = row_count
            self._trained_at = datetime.now()
            self._state = ModelState.TRAINED

        except Exception as e:
            self._state = ModelState.ERROR
            self._error = e
            raise ModelTrainingError(
                f"Failed to train ensemble: {e}",
                model_name=self.info.name,
            ) from e

    def score(self, data: pl.LazyFrame) -> pl.Series:
        """Compute ensemble anomaly scores.

        Args:
            data: Data to score

        Returns:
            Combined anomaly scores
        """
        if not self.is_trained:
            raise ModelNotTrainedError(
                "Ensemble must be trained before scoring",
                model_name=self.info.name,
            )

        # Get scores from all detectors
        all_scores: list[pl.Series] = []
        for detector in self._detectors:
            scores = detector.score(data)
            all_scores.append(scores)

        if not all_scores:
            n_rows = data.select(pl.len()).collect().item()
            return pl.Series("anomaly_score", [0.0] * n_rows)

        # Combine scores based on strategy
        n_rows = len(all_scores[0])
        combined = []

        for i in range(n_rows):
            row_scores = [scores[i] for scores in all_scores]
            combined.append(self._combine_scores(row_scores))

        return pl.Series("anomaly_score", combined)

    def _combine_scores(self, scores: list[float]) -> float:
        """Combine scores using the configured strategy."""
        strategy = self.config.strategy

        if strategy == EnsembleStrategy.AVERAGE:
            return sum(scores) / len(scores)

        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            weighted = sum(
                s * w for s, w in zip(scores, self._detector_weights)
            )
            return weighted

        elif strategy == EnsembleStrategy.MAX:
            return max(scores)

        elif strategy == EnsembleStrategy.MIN:
            return min(scores)

        elif strategy == EnsembleStrategy.VOTE:
            # For VOTE strategy, we use per-detector thresholds
            # Each detector already normalized their scores, so we count
            # how many detectors consider this point anomalous
            # Note: This is calculated at predict() level using per-detector thresholds
            # Here we just return the average score for weighting purposes
            return sum(scores) / len(scores)

        elif strategy == EnsembleStrategy.UNANIMOUS:
            # For UNANIMOUS, all detectors must agree
            # The actual voting logic is in predict()
            return sum(scores) / len(scores)

        else:
            return sum(scores) / len(scores)

    def _combine_scores_with_votes(
        self,
        scores: list[float],
        detector_anomaly_flags: list[bool],
    ) -> float:
        """Combine scores with voting information.

        Args:
            scores: Per-detector scores for this data point
            detector_anomaly_flags: Whether each detector flagged this as anomaly

        Returns:
            Combined score
        """
        strategy = self.config.strategy

        if strategy == EnsembleStrategy.VOTE:
            vote_ratio = sum(detector_anomaly_flags) / len(detector_anomaly_flags)
            if vote_ratio >= self.config.vote_threshold:
                # Return weighted average of scores from agreeing detectors
                agreeing_scores = [
                    s for s, flag in zip(scores, detector_anomaly_flags) if flag
                ]
                return sum(agreeing_scores) / len(agreeing_scores) if agreeing_scores else 0.0
            return 0.0

        elif strategy == EnsembleStrategy.UNANIMOUS:
            if all(detector_anomaly_flags):
                return sum(scores) / len(scores)
            return 0.0

        else:
            return self._combine_scores(scores)

    def predict(self, data: pl.LazyFrame) -> AnomalyResult:
        """Detect anomalies using ensemble.

        Also collects per-detector results for analysis.

        Args:
            data: Data to analyze

        Returns:
            Combined AnomalyResult
        """
        import time
        start = time.perf_counter()

        if not self.is_trained:
            raise ModelNotTrainedError(
                "Ensemble must be trained before prediction",
                model_name=self.info.name,
            )

        # Collect per-detector scores and anomaly classifications
        per_detector_scores: list[list[float]] = []
        per_detector_flags: list[list[bool]] = []
        for detector in self._detectors:
            detector_scores = detector.score(data)
            detector_threshold = detector._get_threshold()
            score_list = detector_scores.to_list()
            per_detector_scores.append(score_list)
            per_detector_flags.append([
                s >= detector_threshold for s in score_list
            ])

        # Combine scores considering the strategy
        n_rows = len(per_detector_scores[0]) if per_detector_scores else 0
        combined_scores = []
        for i in range(n_rows):
            row_scores = [scores[i] for scores in per_detector_scores]
            row_flags = [flags[i] for flags in per_detector_flags]

            # For VOTE and UNANIMOUS strategies, use voting-aware combination
            if self.config.strategy in (EnsembleStrategy.VOTE, EnsembleStrategy.UNANIMOUS):
                combined = self._combine_scores_with_votes(row_scores, row_flags)
            else:
                combined = self._combine_scores(row_scores)
            combined_scores.append(combined)

        # Determine threshold for final classification
        threshold = self._get_threshold()

        anomaly_scores = []
        for idx, score in enumerate(combined_scores):
            # For VOTE/UNANIMOUS, the combined score will be 0.0 if criteria not met
            # So we check if score > 0 (meaning it passed the voting criteria)
            if self.config.strategy in (EnsembleStrategy.VOTE, EnsembleStrategy.UNANIMOUS):
                is_anomaly = score > 0
            else:
                is_anomaly = score >= threshold

            # Count how many detectors flagged this as anomaly
            detector_votes = sum(
                1 for d_flags in per_detector_flags if d_flags[idx]
            )

            # Determine anomaly type based on consensus
            if detector_votes == len(self._detectors):
                anomaly_type = AnomalyType.COLLECTIVE
            elif detector_votes >= len(self._detectors) // 2:
                anomaly_type = AnomalyType.CONTEXTUAL
            else:
                anomaly_type = AnomalyType.POINT

            anomaly_scores.append(
                AnomalyScore(
                    index=idx,
                    score=score,
                    is_anomaly=is_anomaly,
                    anomaly_type=anomaly_type,
                    confidence=detector_votes / len(self._detectors),
                    details=f"Detected by {detector_votes}/{len(self._detectors)} detectors",
                )
            )

        anomaly_count = sum(1 for s in anomaly_scores if s.is_anomaly)
        total_points = len(anomaly_scores)

        elapsed = (time.perf_counter() - start) * 1000

        return AnomalyResult(
            scores=tuple(anomaly_scores),
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_count / total_points if total_points > 0 else 0.0,
            total_points=total_points,
            model_name=self.info.name,
            detection_time_ms=elapsed,
            threshold_used=threshold,
        )

    def get_detector_names(self) -> list[str]:
        """Get names of all detectors in ensemble."""
        return [d.info.name for d in self._detectors]

    def get_detector_weights(self) -> dict[str, float]:
        """Get detector weights."""
        return {
            d.info.name: w
            for d, w in zip(self._detectors, self._detector_weights)
        }

    @property
    def n_detectors(self) -> int:
        """Number of detectors in ensemble."""
        return len(self._detectors)
