"""Time series validators.

This module provides comprehensive validators for time series data:

- **Gap Detection**: Finding missing data points and irregular intervals
- **Monotonicity**: Validating increasing/decreasing sequences
- **Seasonality**: Detecting and validating seasonal patterns
- **Trend Analysis**: Trend detection and structural break identification
- **Completeness**: Coverage and value completeness validation

Validators:
    TimeSeriesGapValidator: Gap and interval consistency detection
    TimeSeriesIntervalValidator: Interval bounds validation
    TimeSeriesDuplicateValidator: Duplicate timestamp detection
    TimeSeriesMonotonicValidator: Monotonicity validation
    TimeSeriesOrderValidator: Timestamp ordering validation
    SeasonalityValidator: Seasonal pattern detection
    SeasonalDecompositionValidator: Decomposition component validation
    TrendValidator: Trend direction and strength validation
    TrendBreakValidator: Structural break detection
    TimeSeriesCompletenessValidator: Coverage ratio validation
    TimeSeriesValueCompletenessValidator: Value null/missing validation
    TimeSeriesDateRangeValidator: Date range coverage validation
"""

from truthound.validators.timeseries.base import (
    TimeFrequency,
    TimeSeriesStats,
    TimeSeriesValidator,
    ValueTimeSeriesValidator,
)

from truthound.validators.timeseries.gap import (
    TimeSeriesGapValidator,
    TimeSeriesIntervalValidator,
    TimeSeriesDuplicateValidator,
)

from truthound.validators.timeseries.monotonic import (
    MonotonicityType,
    TimeSeriesMonotonicValidator,
    TimeSeriesOrderValidator,
)

from truthound.validators.timeseries.seasonality import (
    SeasonalPeriod,
    SeasonalityValidator,
    SeasonalDecompositionValidator,
)

from truthound.validators.timeseries.trend import (
    TrendDirection,
    TrendValidator,
    TrendBreakValidator,
)

from truthound.validators.timeseries.completeness import (
    TimeSeriesCompletenessValidator,
    TimeSeriesValueCompletenessValidator,
    TimeSeriesDateRangeValidator,
)

__all__ = [
    # Base classes and types
    "TimeFrequency",
    "TimeSeriesStats",
    "TimeSeriesValidator",
    "ValueTimeSeriesValidator",
    # Gap validators
    "TimeSeriesGapValidator",
    "TimeSeriesIntervalValidator",
    "TimeSeriesDuplicateValidator",
    # Monotonicity validators
    "MonotonicityType",
    "TimeSeriesMonotonicValidator",
    "TimeSeriesOrderValidator",
    # Seasonality validators
    "SeasonalPeriod",
    "SeasonalityValidator",
    "SeasonalDecompositionValidator",
    # Trend validators
    "TrendDirection",
    "TrendValidator",
    "TrendBreakValidator",
    # Completeness validators
    "TimeSeriesCompletenessValidator",
    "TimeSeriesValueCompletenessValidator",
    "TimeSeriesDateRangeValidator",
]
