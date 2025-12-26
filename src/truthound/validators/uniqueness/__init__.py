"""Uniqueness validators for duplicate and primary key checks."""

from truthound.validators.uniqueness.unique import (
    UniqueValidator,
    UniqueRatioValidator,
    DistinctCountValidator,
)
from truthound.validators.uniqueness.duplicate import (
    DuplicateValidator,
    DuplicateWithinGroupValidator,
)
from truthound.validators.uniqueness.primary_key import (
    PrimaryKeyValidator,
    CompoundKeyValidator,
)
from truthound.validators.uniqueness.distinct_values import (
    DistinctValuesInSetValidator,
    DistinctValuesEqualSetValidator,
    DistinctValuesContainSetValidator,
    DistinctCountBetweenValidator,
)
from truthound.validators.uniqueness.within_record import (
    UniqueWithinRecordValidator,
    AllColumnsUniqueWithinRecordValidator,
    ColumnPairUniqueValidator,
)
from truthound.validators.uniqueness.approximate import (
    HyperLogLog,
    ApproximateDistinctCountValidator,
    ApproximateUniqueRatioValidator,
    StreamingDistinctCountValidator,
)

__all__ = [
    # Exact uniqueness
    "UniqueValidator",
    "UniqueRatioValidator",
    "DistinctCountValidator",
    # Duplicate detection
    "DuplicateValidator",
    "DuplicateWithinGroupValidator",
    # Primary key
    "PrimaryKeyValidator",
    "CompoundKeyValidator",
    # Distinct value sets
    "DistinctValuesInSetValidator",
    "DistinctValuesEqualSetValidator",
    "DistinctValuesContainSetValidator",
    "DistinctCountBetweenValidator",
    # Within record
    "UniqueWithinRecordValidator",
    "AllColumnsUniqueWithinRecordValidator",
    "ColumnPairUniqueValidator",
    # Approximate (HyperLogLog)
    "HyperLogLog",
    "ApproximateDistinctCountValidator",
    "ApproximateUniqueRatioValidator",
    "StreamingDistinctCountValidator",
]
