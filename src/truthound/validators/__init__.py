"""Built-in validators for data quality checks.

This module provides 289+ validators across 28 categories with lazy loading
for improved startup performance.

Categories:
- schema: Table structure validation (14 validators)
- completeness: Null and missing value detection (12 validators)
- uniqueness: Duplicate, primary key, and distinct value checks (17 validators)
- distribution: Range, set, and statistical checks (15 validators)
- string: Pattern matching and format validation (20 validators)
- datetime: Date format and range checks (10 validators)
- aggregate: Statistical aggregate validation (8 validators)
- cross_table: Multi-table validation (4 validators)
- query: SQL and expression-based validation (18 validators)
- multi_column: Multi-column compound checks (18 validators)
- table: Table metadata validation (17 validators)
- geospatial: Geographic coordinate validation (12 validators)
- drift: Data drift detection (14 validators)
- anomaly: Anomaly and outlier detection (17 validators)
- referential: Referential integrity validation (16 validators)
- timeseries: Time series validation (18 validators)
- business_rule: Business rule validation (8 validators)
- profiling: Data profiling validation (8 validators)
- localization: Asian localization validation (9 validators)
- ml_feature: ML feature validation (9 validators)
- privacy: GDPR/CCPA/Global privacy compliance (20+ validators)

Usage:
    # Import specific validators (recommended - lazy loaded)
    from truthound.validators import NullValidator, DuplicateValidator

    # Use registry for dynamic access (lazy loaded on access)
    from truthound.validators import registry
    validator_cls = registry.get("null")

    # List available validators (loads all - avoid if possible)
    from truthound.validators import list_validators
    all_validators = list_validators()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Base classes - eagerly loaded as they're fundamental
from truthound.validators.base import (
    ValidationIssue,
    ValidatorConfig,
    Validator,
    ColumnValidator,
    AggregateValidator,
    NumericValidatorMixin,
    StringValidatorMixin,
    DatetimeValidatorMixin,
    SampledEarlyTerminationMixin,
    EarlyTerminationResult,
    NUMERIC_TYPES,
    STRING_TYPES,
    DATETIME_TYPES,
    # Expression-based validation architecture
    ValidationExpressionSpec,
    ExpressionValidatorProtocol,
    ExpressionValidatorMixin,
    ExpressionBatchExecutor,
    # Query plan optimization utilities
    QUERY_OPTIMIZATIONS,
    optimized_collect,
)

# Registry - singleton, lazy loading inside
from truthound.validators.registry import registry, register_validator

# Lazy loading infrastructure
from truthound.validators._lazy import (
    VALIDATOR_IMPORT_MAP,
    validator_getattr,
    get_validator_import_metrics,
    preload_validators,
    preload_category,
)

if TYPE_CHECKING:
    # Type hints for IDE support - not actually imported at runtime
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
    from truthound.validators.completeness import (
        NullValidator,
        NotNullValidator,
        CompletenessRatioValidator,
        EmptyStringValidator,
        WhitespaceOnlyValidator,
        ConditionalNullValidator,
        DefaultValueValidator,
    )
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


def __getattr__(name: str) -> Any:
    """Lazy load validators on demand.

    This function is called when an attribute is not found in the module.
    It uses the lazy loading infrastructure to import validators only when needed.
    """
    # Check if it's a validator in the import map
    if name in VALIDATOR_IMPORT_MAP:
        return validator_getattr(name)

    # Special case for BUILTIN_VALIDATORS - load lazily
    if name == "BUILTIN_VALIDATORS":
        return _get_builtin_validators()

    raise AttributeError(f"module 'truthound.validators' has no attribute '{name}'")


def _get_builtin_validators() -> dict[str, type[Validator]]:
    """Get the BUILTIN_VALIDATORS dict with lazy loading.

    Note: RegexValidator is not included because it requires a mandatory
    'pattern' parameter and cannot be instantiated without configuration.
    Users should explicitly specify RegexValidator with a pattern when needed.
    """
    # Load the 7 core validators that can be instantiated without config
    return {
        "null": registry.get("NullValidator"),
        "duplicate": registry.get("DuplicateValidator"),
        "type": registry.get("TypeValidator"),
        "range": registry.get("RangeValidator"),
        "outlier": registry.get("OutlierValidator"),
        "format": registry.get("FormatValidator"),
        "unique": registry.get("UniqueValidator"),
    }


def get_validator(name: str) -> type[Validator]:
    """Get a validator class by name.

    This function uses lazy loading - the validator is only imported
    when first requested.

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

    Note: Without a category filter, this loads ALL validators which
    defeats lazy loading. Use get_validator() for individual validators
    when possible.

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
    "SampledEarlyTerminationMixin",
    "EarlyTerminationResult",
    "NUMERIC_TYPES",
    "STRING_TYPES",
    "DATETIME_TYPES",
    # Expression-based validation architecture
    "ValidationExpressionSpec",
    "ExpressionValidatorProtocol",
    "ExpressionValidatorMixin",
    "ExpressionBatchExecutor",
    # Query plan optimization utilities
    "QUERY_OPTIMIZATIONS",
    "optimized_collect",
    # Registry
    "registry",
    "register_validator",
    "get_validator",
    "list_validators",
    "list_categories",
    "BUILTIN_VALIDATORS",
    # Lazy loading utilities
    "get_validator_import_metrics",
    "preload_validators",
    "preload_category",
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
    # Referential
    "ForeignKeyRelation",
    "TableNode",
    "ReferentialValidator",
    "MultiTableValidator",
    "ForeignKeyValidator",
    "CompositeForeignKeyValidator",
    "SelfReferentialFKValidator",
    "CascadeAction",
    "CascadeRule",
    "CascadeIntegrityValidator",
    "CascadeDepthValidator",
    "OrphanRecordValidator",
    "MultiTableOrphanValidator",
    "DanglingReferenceValidator",
    "CircularReferenceValidator",
    "HierarchyCircularValidator",
    "HierarchyDepthValidator",
    # Time series
    "TimeFrequency",
    "TimeSeriesStats",
    "TimeSeriesValidator",
    "ValueTimeSeriesValidator",
    "TimeSeriesGapValidator",
    "TimeSeriesIntervalValidator",
    "TimeSeriesDuplicateValidator",
    "MonotonicityType",
    "TimeSeriesMonotonicValidator",
    "TimeSeriesOrderValidator",
    "SeasonalPeriod",
    "SeasonalityValidator",
    "SeasonalDecompositionValidator",
    "TrendDirection",
    "TrendValidator",
    "TrendBreakValidator",
    "TimeSeriesCompletenessValidator",
    "TimeSeriesValueCompletenessValidator",
    "TimeSeriesDateRangeValidator",
    # Business rule
    "BusinessRuleValidator",
    "ChecksumValidator",
    "LuhnValidator",
    "ISBNValidator",
    "CreditCardValidator",
    "IBANValidator",
    "VATValidator",
    "SWIFTValidator",
    # Profiling
    "ProfileMetrics",
    "ProfilingValidator",
    "CardinalityValidator",
    "UniquenessRatioValidator",
    "EntropyValidator",
    "InformationGainValidator",
    "ValueFrequencyValidator",
    "DistributionShapeValidator",
    # Localization
    "LocalizationValidator",
    "KoreanBusinessNumberValidator",
    "KoreanRRNValidator",
    "KoreanPhoneValidator",
    "KoreanBankAccountValidator",
    "JapanesePostalCodeValidator",
    "JapaneseMyNumberValidator",
    "ChineseIDValidator",
    "ChineseUSCCValidator",
    # ML Feature
    "MLFeatureValidator",
    "FeatureStats",
    "CorrelationResult",
    "LeakageResult",
    "ScaleType",
    "FeatureNullImpactValidator",
    "FeatureScaleValidator",
    "FeatureCorrelationMatrixValidator",
    "TargetLeakageValidator",
    # Privacy Compliance
    "PrivacyRegulation",
    "PIICategory",
    "ConsentStatus",
    "LegalBasis",
    "PIIFieldDefinition",
    "PrivacyFinding",
    "PrivacyValidator",
    "DataRetentionValidator",
    "ConsentValidator",
    "GDPRComplianceValidator",
    "GDPRSpecialCategoryValidator",
    "GDPRDataMinimizationValidator",
    "GDPRRightToErasureValidator",
    "CCPAComplianceValidator",
    "CCPASensitiveInfoValidator",
    "CCPADoNotSellValidator",
    "CCPAConsumerRightsValidator",
    "GlobalPrivacyValidator",
    "LGPDComplianceValidator",
    "PIPEDAComplianceValidator",
    "APPIComplianceValidator",
]
