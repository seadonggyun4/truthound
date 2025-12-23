"""Built-in validators for data quality checks.

This module provides 169 validators across 14 categories:
- schema: Table structure validation (14 validators)
- completeness: Null and missing value detection (7 validators)
- uniqueness: Duplicate, primary key, and distinct value checks (13 validators)
- distribution: Range, set, and statistical checks (15 validators)
- string: Pattern matching and format validation (17 validators)
- datetime: Date format and range checks (10 validators)
- aggregate: Statistical aggregate validation (8 validators)
- cross_table: Multi-table validation (4 validators)
- query: SQL and expression-based validation (17 validators)
- multi_column: Multi-column compound checks (18 validators)
- table: Table metadata validation (13 validators)
- geospatial: Geographic coordinate validation (8 validators)
- drift: Data drift detection (11 validators)
- anomaly: Anomaly and outlier detection (13 validators)
"""

from __future__ import annotations

# Base classes
from truthound.validators.base import (
    ValidationIssue,
    ValidatorConfig,
    Validator,
    ColumnValidator,
    AggregateValidator,
    NumericValidatorMixin,
    StringValidatorMixin,
    DatetimeValidatorMixin,
    NUMERIC_TYPES,
    STRING_TYPES,
    DATETIME_TYPES,
)

# Registry
from truthound.validators.registry import registry, register_validator

# Schema validators
from truthound.validators.schema import (
    ColumnExistsValidator,
    ColumnNotExistsValidator,
    ColumnCountValidator,
    RowCountValidator,
    ColumnTypeValidator,
    ColumnOrderValidator,
    TableSchemaValidator,
    ColumnPairValidator,
    MultiColumnUniqueValidator,
    ReferentialIntegrityValidator,
    MultiColumnSumValidator,
    MultiColumnCalculationValidator,
    ColumnPairInSetValidator,
    ColumnPairNotInSetValidator,
)

# Completeness validators
from truthound.validators.completeness import (
    NullValidator,
    NotNullValidator,
    CompletenessRatioValidator,
    EmptyStringValidator,
    WhitespaceOnlyValidator,
    ConditionalNullValidator,
    DefaultValueValidator,
)

# Uniqueness validators
from truthound.validators.uniqueness import (
    UniqueValidator,
    UniqueRatioValidator,
    DistinctCountValidator,
    DuplicateValidator,
    DuplicateWithinGroupValidator,
    PrimaryKeyValidator,
    CompoundKeyValidator,
    DistinctValuesInSetValidator,
    DistinctValuesEqualSetValidator,
    DistinctValuesContainSetValidator,
    DistinctCountBetweenValidator,
    UniqueWithinRecordValidator,
    AllColumnsUniqueWithinRecordValidator,
)

# Distribution validators
from truthound.validators.distribution import (
    BetweenValidator,
    RangeValidator,
    PositiveValidator,
    NonNegativeValidator,
    InSetValidator,
    NotInSetValidator,
    IncreasingValidator,
    DecreasingValidator,
    OutlierValidator,
    ZScoreOutlierValidator,
    QuantileValidator,
    DistributionValidator,
    KLDivergenceValidator,
    ChiSquareValidator,
    MostCommonValueValidator,
)

# String validators
from truthound.validators.string import (
    RegexValidator,
    RegexListValidator,
    NotMatchRegexValidator,
    NotMatchRegexListValidator,
    LengthValidator,
    EmailValidator,
    UrlValidator,
    PhoneValidator,
    UuidValidator,
    IpAddressValidator,
    FormatValidator,
    JsonParseableValidator,
    JsonSchemaValidator,
    AlphanumericValidator,
    ConsistentCasingValidator,
    LikePatternValidator,
    NotLikePatternValidator,
)

# Datetime validators
from truthound.validators.datetime import (
    DateFormatValidator,
    DateBetweenValidator,
    FutureDateValidator,
    PastDateValidator,
    DateOrderValidator,
    TimezoneValidator,
    RecentDataValidator,
    DatePartCoverageValidator,
    GroupedRecentDataValidator,
    DateutilParseableValidator,
)

# Aggregate validators
from truthound.validators.aggregate import (
    MeanBetweenValidator,
    MedianBetweenValidator,
    StdBetweenValidator,
    VarianceBetweenValidator,
    MinBetweenValidator,
    MaxBetweenValidator,
    SumBetweenValidator,
    TypeValidator,
)

# Cross-table validators
from truthound.validators.cross_table import (
    CrossTableRowCountValidator,
    CrossTableRowCountFactorValidator,
    CrossTableAggregateValidator,
    CrossTableDistinctCountValidator,
)

# Query validators
from truthound.validators.query import (
    QueryValidator,
    ExpressionValidator,
    QueryReturnsSingleValueValidator,
    QueryReturnsNoRowsValidator,
    QueryReturnsRowsValidator,
    QueryResultMatchesValidator,
    QueryRowCountValidator,
    QueryRowCountRatioValidator,
    QueryRowCountCompareValidator,
    QueryColumnValuesValidator,
    QueryColumnUniqueValidator,
    QueryColumnNotNullValidator,
    QueryAggregateValidator,
    QueryGroupAggregateValidator,
    QueryAggregateCompareValidator,
    CustomExpressionValidator,
    ConditionalExpressionValidator,
    MultiConditionValidator,
    RowLevelValidator,
)

# Multi-column validators
from truthound.validators.multi_column import (
    MultiColumnValidator,
    ColumnArithmeticValidator,
    ColumnSumValidator,
    ColumnProductValidator,
    ColumnDifferenceValidator,
    ColumnRatioValidator,
    ColumnPercentageValidator,
    ColumnComparisonValidator,
    ColumnChainComparisonValidator,
    ColumnMaxValidator,
    ColumnMinValidator,
    ColumnMeanValidator,
    ColumnConsistencyValidator,
    ColumnMutualExclusivityValidator,
    ColumnCoexistenceValidator,
    ColumnDependencyValidator,
    ColumnImplicationValidator,
    ColumnCorrelationValidator,
    ColumnCovarianceValidator,
    MultiColumnVarianceValidator,
)

# Table metadata validators
from truthound.validators.table import (
    TableValidator,
    TableRowCountRangeValidator,
    TableRowCountExactValidator,
    TableRowCountCompareValidator,
    TableNotEmptyValidator,
    TableColumnCountValidator,
    TableRequiredColumnsValidator,
    TableForbiddenColumnsValidator,
    TableFreshnessValidator,
    TableDataRecencyValidator,
    TableUpdateFrequencyValidator,
    TableSchemaMatchValidator,
    TableSchemaCompareValidator,
    TableColumnTypesValidator,
    TableMemorySizeValidator,
    TableRowToColumnRatioValidator,
    TableDimensionsValidator,
)

# Geospatial validators
from truthound.validators.geospatial import (
    GeoValidator,
    LatitudeValidator,
    LongitudeValidator,
    CoordinateValidator,
    CoordinateNotNullIslandValidator,
    GeoDistanceValidator,
    GeoDistanceFromPointValidator,
    GeoBoundingBoxValidator,
    GeoCountryValidator,
)

# Drift validators
from truthound.validators.drift import (
    DriftValidator,
    ColumnDriftValidator,
    KSTestValidator,
    ChiSquareDriftValidator,
    WassersteinDriftValidator,
    PSIValidator,
    CSIValidator,
    MeanDriftValidator,
    VarianceDriftValidator,
    QuantileDriftValidator,
    RangeDriftValidator,
    FeatureDriftValidator,
    JSDivergenceValidator,
)

# Anomaly validators
from truthound.validators.anomaly import (
    AnomalyValidator,
    ColumnAnomalyValidator,
    IQRAnomalyValidator,
    MADAnomalyValidator,
    GrubbsTestValidator,
    TukeyFencesValidator,
    PercentileAnomalyValidator,
    MahalanobisValidator,
    EllipticEnvelopeValidator,
    PCAAnomalyValidator,
    ZScoreMultivariateValidator,
    IsolationForestValidator,
    LOFValidator,
    OneClassSVMValidator,
    DBSCANAnomalyValidator,
)

__all__ = [
    # Base
    "ValidationIssue",
    "ValidatorConfig",
    "Validator",
    "ColumnValidator",
    "AggregateValidator",
    "NumericValidatorMixin",
    "StringValidatorMixin",
    "DatetimeValidatorMixin",
    "NUMERIC_TYPES",
    "STRING_TYPES",
    "DATETIME_TYPES",
    # Registry
    "registry",
    "register_validator",
    "get_validator",
    "list_validators",
    "list_categories",
    "BUILTIN_VALIDATORS",
    # Schema
    "ColumnExistsValidator",
    "ColumnNotExistsValidator",
    "ColumnCountValidator",
    "RowCountValidator",
    "ColumnTypeValidator",
    "ColumnOrderValidator",
    "TableSchemaValidator",
    "ColumnPairValidator",
    "MultiColumnUniqueValidator",
    "ReferentialIntegrityValidator",
    "MultiColumnSumValidator",
    "MultiColumnCalculationValidator",
    "ColumnPairInSetValidator",
    "ColumnPairNotInSetValidator",
    # Completeness
    "NullValidator",
    "NotNullValidator",
    "CompletenessRatioValidator",
    "EmptyStringValidator",
    "WhitespaceOnlyValidator",
    "ConditionalNullValidator",
    "DefaultValueValidator",
    # Uniqueness
    "UniqueValidator",
    "UniqueRatioValidator",
    "DistinctCountValidator",
    "DuplicateValidator",
    "DuplicateWithinGroupValidator",
    "PrimaryKeyValidator",
    "CompoundKeyValidator",
    "DistinctValuesInSetValidator",
    "DistinctValuesEqualSetValidator",
    "DistinctValuesContainSetValidator",
    "DistinctCountBetweenValidator",
    "UniqueWithinRecordValidator",
    "AllColumnsUniqueWithinRecordValidator",
    # Distribution
    "BetweenValidator",
    "RangeValidator",
    "PositiveValidator",
    "NonNegativeValidator",
    "InSetValidator",
    "NotInSetValidator",
    "IncreasingValidator",
    "DecreasingValidator",
    "OutlierValidator",
    "ZScoreOutlierValidator",
    "QuantileValidator",
    "DistributionValidator",
    "KLDivergenceValidator",
    "ChiSquareValidator",
    "MostCommonValueValidator",
    # String
    "RegexValidator",
    "RegexListValidator",
    "NotMatchRegexValidator",
    "NotMatchRegexListValidator",
    "LengthValidator",
    "EmailValidator",
    "UrlValidator",
    "PhoneValidator",
    "UuidValidator",
    "IpAddressValidator",
    "FormatValidator",
    "JsonParseableValidator",
    "JsonSchemaValidator",
    "AlphanumericValidator",
    "ConsistentCasingValidator",
    "LikePatternValidator",
    "NotLikePatternValidator",
    # Datetime
    "DateFormatValidator",
    "DateBetweenValidator",
    "FutureDateValidator",
    "PastDateValidator",
    "DateOrderValidator",
    "TimezoneValidator",
    "RecentDataValidator",
    "DatePartCoverageValidator",
    "GroupedRecentDataValidator",
    "DateutilParseableValidator",
    # Aggregate
    "MeanBetweenValidator",
    "MedianBetweenValidator",
    "StdBetweenValidator",
    "VarianceBetweenValidator",
    "MinBetweenValidator",
    "MaxBetweenValidator",
    "SumBetweenValidator",
    "TypeValidator",
    # Cross-table
    "CrossTableRowCountValidator",
    "CrossTableRowCountFactorValidator",
    "CrossTableAggregateValidator",
    "CrossTableDistinctCountValidator",
    # Query
    "QueryValidator",
    "ExpressionValidator",
    "QueryReturnsSingleValueValidator",
    "QueryReturnsNoRowsValidator",
    "QueryReturnsRowsValidator",
    "QueryResultMatchesValidator",
    "QueryRowCountValidator",
    "QueryRowCountRatioValidator",
    "QueryRowCountCompareValidator",
    "QueryColumnValuesValidator",
    "QueryColumnUniqueValidator",
    "QueryColumnNotNullValidator",
    "QueryAggregateValidator",
    "QueryGroupAggregateValidator",
    "QueryAggregateCompareValidator",
    "CustomExpressionValidator",
    "ConditionalExpressionValidator",
    "MultiConditionValidator",
    "RowLevelValidator",
    # Multi-column
    "MultiColumnValidator",
    "ColumnArithmeticValidator",
    "ColumnSumValidator",
    "ColumnProductValidator",
    "ColumnDifferenceValidator",
    "ColumnRatioValidator",
    "ColumnPercentageValidator",
    "ColumnComparisonValidator",
    "ColumnChainComparisonValidator",
    "ColumnMaxValidator",
    "ColumnMinValidator",
    "ColumnMeanValidator",
    "ColumnConsistencyValidator",
    "ColumnMutualExclusivityValidator",
    "ColumnCoexistenceValidator",
    "ColumnDependencyValidator",
    "ColumnImplicationValidator",
    "ColumnCorrelationValidator",
    "ColumnCovarianceValidator",
    "MultiColumnVarianceValidator",
    # Table metadata
    "TableValidator",
    "TableRowCountRangeValidator",
    "TableRowCountExactValidator",
    "TableRowCountCompareValidator",
    "TableNotEmptyValidator",
    "TableColumnCountValidator",
    "TableRequiredColumnsValidator",
    "TableForbiddenColumnsValidator",
    "TableFreshnessValidator",
    "TableDataRecencyValidator",
    "TableUpdateFrequencyValidator",
    "TableSchemaMatchValidator",
    "TableSchemaCompareValidator",
    "TableColumnTypesValidator",
    "TableMemorySizeValidator",
    "TableRowToColumnRatioValidator",
    "TableDimensionsValidator",
    # Geospatial
    "GeoValidator",
    "LatitudeValidator",
    "LongitudeValidator",
    "CoordinateValidator",
    "CoordinateNotNullIslandValidator",
    "GeoDistanceValidator",
    "GeoDistanceFromPointValidator",
    "GeoBoundingBoxValidator",
    "GeoCountryValidator",
    # Drift
    "DriftValidator",
    "ColumnDriftValidator",
    "KSTestValidator",
    "ChiSquareDriftValidator",
    "WassersteinDriftValidator",
    "PSIValidator",
    "CSIValidator",
    "MeanDriftValidator",
    "VarianceDriftValidator",
    "QuantileDriftValidator",
    "RangeDriftValidator",
    "FeatureDriftValidator",
    "JSDivergenceValidator",
    # Anomaly
    "AnomalyValidator",
    "ColumnAnomalyValidator",
    "IQRAnomalyValidator",
    "MADAnomalyValidator",
    "GrubbsTestValidator",
    "TukeyFencesValidator",
    "PercentileAnomalyValidator",
    "MahalanobisValidator",
    "EllipticEnvelopeValidator",
    "PCAAnomalyValidator",
    "ZScoreMultivariateValidator",
    "IsolationForestValidator",
    "LOFValidator",
    "OneClassSVMValidator",
    "DBSCANAnomalyValidator",
]


# Backward compatibility: BUILTIN_VALIDATORS dict
# This provides the same interface as before for api.py
BUILTIN_VALIDATORS: dict[str, type[Validator]] = {
    # Original 7 validators (backward compatibility)
    "null": NullValidator,
    "duplicate": DuplicateValidator,
    "type": TypeValidator,
    "range": RangeValidator,
    "outlier": OutlierValidator,
    "format": FormatValidator,
    "unique": UniqueValidator,
}


def get_validator(name: str) -> type[Validator]:
    """Get a validator class by name.

    Args:
        name: Name of the validator.

    Returns:
        Validator class.

    Raises:
        ValueError: If the validator name is not found.
    """
    return registry.get(name)


def list_validators(category: str | None = None) -> dict[str, type[Validator]]:
    """List all validators, optionally filtered by category.

    Args:
        category: Optional category name to filter by.

    Returns:
        Dictionary of validator name to validator class.
    """
    if category:
        return registry.get_by_category(category)
    return registry.list_all()


def list_categories() -> list[str]:
    """List all validator categories.

    Returns:
        List of category names.
    """
    return registry.list_categories()
