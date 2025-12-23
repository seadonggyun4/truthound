"""Data drift validators.

This module provides 12 validators for detecting distribution drift
between reference (baseline) and current datasets:

Statistical Tests:
- KSTestValidator: Kolmogorov-Smirnov test for numeric columns
- ChiSquareDriftValidator: Chi-square test for categorical columns
- WassersteinDriftValidator: Earth Mover's Distance for numeric columns

PSI (Population Stability Index):
- PSIValidator: Industry-standard PSI for drift detection
- CSIValidator: Characteristic Stability Index (per-bin PSI)

Numeric Statistics:
- MeanDriftValidator: Detect mean value drift
- VarianceDriftValidator: Detect variance/std drift
- QuantileDriftValidator: Detect drift in specific quantiles
- RangeDriftValidator: Detect min/max value drift

Multi-Feature:
- FeatureDriftValidator: Comprehensive multi-column drift detection
- JSDivergenceValidator: Jensen-Shannon divergence

Use Cases:
- ML model monitoring (detect training/serving skew)
- Data pipeline quality gates
- A/B test data validation
- Seasonal/temporal drift detection
"""

from truthound.validators.drift.base import (
    DriftValidator,
    ColumnDriftValidator,
    NumericDriftMixin,
    CategoricalDriftMixin,
)
from truthound.validators.drift.statistical import (
    KSTestValidator,
    ChiSquareDriftValidator,
    WassersteinDriftValidator,
)
from truthound.validators.drift.psi import (
    PSIValidator,
    CSIValidator,
)
from truthound.validators.drift.numeric import (
    MeanDriftValidator,
    VarianceDriftValidator,
    QuantileDriftValidator,
    RangeDriftValidator,
)
from truthound.validators.drift.multi_feature import (
    FeatureDriftValidator,
    JSDivergenceValidator,
)

__all__ = [
    # Base classes
    "DriftValidator",
    "ColumnDriftValidator",
    "NumericDriftMixin",
    "CategoricalDriftMixin",
    # Statistical tests
    "KSTestValidator",
    "ChiSquareDriftValidator",
    "WassersteinDriftValidator",
    # PSI
    "PSIValidator",
    "CSIValidator",
    # Numeric statistics
    "MeanDriftValidator",
    "VarianceDriftValidator",
    "QuantileDriftValidator",
    "RangeDriftValidator",
    # Multi-feature
    "FeatureDriftValidator",
    "JSDivergenceValidator",
]
