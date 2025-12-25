"""ML-based drift detection module.

Provides advanced drift detection capabilities beyond statistical tests:
- Distribution drift detection
- Feature drift detection
- Concept drift detection
- Multivariate drift detection

Example:
    >>> from truthound.ml.drift_detection import FeatureDriftDetector
    >>> detector = FeatureDriftDetector()
    >>> detector.fit(reference_data)
    >>> result = detector.detect(reference_data, current_data)
"""

from truthound.ml.drift_detection.distribution import DistributionDriftDetector
from truthound.ml.drift_detection.feature import FeatureDriftDetector
from truthound.ml.drift_detection.concept import ConceptDriftDetector
from truthound.ml.drift_detection.multivariate import MultivariateDriftDetector

__all__ = [
    "DistributionDriftDetector",
    "FeatureDriftDetector",
    "ConceptDriftDetector",
    "MultivariateDriftDetector",
]
