"""Anomaly detection models for Truthound.

This module provides various anomaly detection algorithms:
- Statistical methods (Z-score, IQR, MAD)
- Tree-based methods (Isolation Forest)
- Ensemble methods

Example:
    >>> from truthound.ml.anomaly_models import IsolationForestDetector
    >>> detector = IsolationForestDetector(contamination=0.1)
    >>> detector.fit(train_data)
    >>> result = detector.predict(test_data)
"""

from truthound.ml.anomaly_models.ensemble import EnsembleAnomalyDetector
from truthound.ml.anomaly_models.isolation_forest import IsolationForestDetector
from truthound.ml.anomaly_models.statistical import (
    IQRAnomalyDetector,
    MADAnomalyDetector,
    StatisticalAnomalyDetector,
    ZScoreAnomalyDetector,
)

__all__ = [
    "StatisticalAnomalyDetector",
    "ZScoreAnomalyDetector",
    "IQRAnomalyDetector",
    "MADAnomalyDetector",
    "IsolationForestDetector",
    "EnsembleAnomalyDetector",
]
