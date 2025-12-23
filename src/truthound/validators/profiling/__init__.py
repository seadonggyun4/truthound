"""Data profiling validators.

This module provides validators for analyzing data quality through
statistical profiling:

- **Cardinality Validators**: Unique value ratio and count validation
- **Entropy Validators**: Information entropy and gain analysis
- **Frequency Validators**: Value frequency distribution validation

Validators:
    CardinalityValidator: Validates cardinality (uniqueness ratio) bounds
    UniquenessRatioValidator: Extended uniqueness analysis with duplicates
    EntropyValidator: Validates Shannon entropy bounds
    InformationGainValidator: Validates information gain between columns
    ValueFrequencyValidator: Validates value frequency patterns
    DistributionShapeValidator: Validates distribution shape (Gini, uniformity)
"""

from truthound.validators.profiling.base import (
    ProfileMetrics,
    ProfilingValidator,
)

from truthound.validators.profiling.cardinality import (
    CardinalityValidator,
    UniquenessRatioValidator,
)

from truthound.validators.profiling.entropy import (
    EntropyValidator,
    InformationGainValidator,
)

from truthound.validators.profiling.frequency import (
    ValueFrequencyValidator,
    DistributionShapeValidator,
)

__all__ = [
    # Base classes
    "ProfileMetrics",
    "ProfilingValidator",
    # Cardinality validators
    "CardinalityValidator",
    "UniquenessRatioValidator",
    # Entropy validators
    "EntropyValidator",
    "InformationGainValidator",
    # Frequency validators
    "ValueFrequencyValidator",
    "DistributionShapeValidator",
]
