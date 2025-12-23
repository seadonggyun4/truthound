"""Anomaly detection validators.

This module provides comprehensive anomaly detection capabilities:

1. Statistical Methods (no external dependencies):
   - IQRAnomalyValidator: Interquartile Range based detection
   - MADAnomalyValidator: Median Absolute Deviation
   - GrubbsTestValidator: Grubbs' test for single outliers
   - TukeyFencesValidator: Inner/outer fences
   - PercentileAnomalyValidator: Percentile-based bounds

2. ML-Based Methods (requires scikit-learn):
   - IsolationForestValidator: Tree-based isolation
   - LOFValidator: Local Outlier Factor
   - OneClassSVMValidator: Support Vector Machine
   - DBSCANAnomalyValidator: Density-based clustering

3. Multivariate Methods:
   - MahalanobisValidator: Distance-based with covariance
   - EllipticEnvelopeValidator: Robust Gaussian fitting
   - PCAAnomalyValidator: Reconstruction error
   - ZScoreMultivariateValidator: Combined Z-scores

Install ML dependencies:
    pip install truthound[anomaly]

Example usage:
    from truthound.validators.anomaly import IQRAnomalyValidator

    validator = IQRAnomalyValidator(
        column="transaction_amount",
        iqr_multiplier=1.5,
        max_anomaly_ratio=0.05,
    )
    issues = validator.validate(df.lazy())
"""

# Base classes
from truthound.validators.anomaly.base import (
    AnomalyValidator,
    ColumnAnomalyValidator,
    StatisticalAnomalyMixin,
    MLAnomalyMixin,
)

# Statistical validators (no external dependencies beyond scipy)
from truthound.validators.anomaly.statistical import (
    IQRAnomalyValidator,
    MADAnomalyValidator,
    GrubbsTestValidator,
    TukeyFencesValidator,
    PercentileAnomalyValidator,
)

# Multivariate validators
from truthound.validators.anomaly.multivariate import (
    MahalanobisValidator,
    EllipticEnvelopeValidator,
    PCAAnomalyValidator,
    ZScoreMultivariateValidator,
)

# ML-based validators (require scikit-learn)
from truthound.validators.anomaly.ml_based import (
    IsolationForestValidator,
    LOFValidator,
    OneClassSVMValidator,
    DBSCANAnomalyValidator,
)

__all__ = [
    # Base classes
    "AnomalyValidator",
    "ColumnAnomalyValidator",
    "StatisticalAnomalyMixin",
    "MLAnomalyMixin",
    # Statistical
    "IQRAnomalyValidator",
    "MADAnomalyValidator",
    "GrubbsTestValidator",
    "TukeyFencesValidator",
    "PercentileAnomalyValidator",
    # Multivariate
    "MahalanobisValidator",
    "EllipticEnvelopeValidator",
    "PCAAnomalyValidator",
    "ZScoreMultivariateValidator",
    # ML-based
    "IsolationForestValidator",
    "LOFValidator",
    "OneClassSVMValidator",
    "DBSCANAnomalyValidator",
]
