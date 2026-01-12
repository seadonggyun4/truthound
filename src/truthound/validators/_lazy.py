"""Lazy loading utilities for the validators module.

This module implements PEP 562 lazy loading to improve import performance
by deferring the loading of heavy validator classes until they are actually used.

Key Features:
- Module-level __getattr__ for lazy attribute access
- Validator class mapping for on-demand loading
- Category-based loading for registry discovery
- Performance metrics tracking

Example:
    # In __init__.py
    from truthound.validators._lazy import validator_getattr

    def __getattr__(name: str):
        return validator_getattr(name)
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from truthound.validators.base import Validator

logger = logging.getLogger(__name__)


@dataclass
class ValidatorImportMetrics:
    """Metrics for tracking validator import performance."""

    total_lazy_loads: int = 0
    total_eager_loads: int = 0
    load_times: dict[str, float] = field(default_factory=dict)
    access_counts: dict[str, int] = field(default_factory=dict)
    failed_loads: list[str] = field(default_factory=list)

    def record_load(self, name: str, duration: float, lazy: bool = True) -> None:
        """Record a validator load."""
        self.load_times[name] = duration
        if lazy:
            self.total_lazy_loads += 1
        else:
            self.total_eager_loads += 1

    def record_access(self, name: str) -> None:
        """Record an attribute access."""
        self.access_counts[name] = self.access_counts.get(name, 0) + 1

    def record_failure(self, name: str) -> None:
        """Record a failed load."""
        self.failed_loads.append(name)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        total_load_time = sum(self.load_times.values())
        return {
            "total_lazy_loads": self.total_lazy_loads,
            "total_eager_loads": self.total_eager_loads,
            "total_load_time_ms": total_load_time * 1000,
            "slowest_loads": sorted(
                self.load_times.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "most_accessed": sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
            "failed_loads": self.failed_loads,
        }


# =============================================================================
# Validator Import Map
# =============================================================================

# This defines all lazy-loadable validators with their source modules
# Format: "ValidatorName": "truthound.validators.category.module"
VALIDATOR_IMPORT_MAP: dict[str, str] = {
    # === Base classes ===
    "ValidationIssue": "truthound.validators.base",
    "ValidatorConfig": "truthound.validators.base",
    "Validator": "truthound.validators.base",
    "ColumnValidator": "truthound.validators.base",
    "AggregateValidator": "truthound.validators.base",
    "NumericValidatorMixin": "truthound.validators.base",
    "StringValidatorMixin": "truthound.validators.base",
    "DatetimeValidatorMixin": "truthound.validators.base",
    "NUMERIC_TYPES": "truthound.validators.base",
    "STRING_TYPES": "truthound.validators.base",
    "DATETIME_TYPES": "truthound.validators.base",

    # === Schema validators ===
    "ColumnExistsValidator": "truthound.validators.schema.column_exists",
    "ColumnNotExistsValidator": "truthound.validators.schema.column_exists",
    "ColumnCountValidator": "truthound.validators.schema.column_count",
    "RowCountValidator": "truthound.validators.schema.column_count",
    "ColumnTypeValidator": "truthound.validators.schema.column_type",
    "ColumnOrderValidator": "truthound.validators.schema.column_order",
    "TableSchemaValidator": "truthound.validators.schema.table_schema",
    "ColumnPairValidator": "truthound.validators.schema.column_pair",
    "MultiColumnUniqueValidator": "truthound.validators.schema.multi_column",
    "ReferentialIntegrityValidator": "truthound.validators.schema.referential",
    "MultiColumnSumValidator": "truthound.validators.schema.multi_column_aggregate",
    "MultiColumnCalculationValidator": "truthound.validators.schema.multi_column_aggregate",
    "ColumnPairInSetValidator": "truthound.validators.schema.column_pair_set",
    "ColumnPairNotInSetValidator": "truthound.validators.schema.column_pair_set",

    # === Completeness validators ===
    "NullValidator": "truthound.validators.completeness.null",
    "NotNullValidator": "truthound.validators.completeness.null",
    "CompletenessRatioValidator": "truthound.validators.completeness.null",
    "EmptyStringValidator": "truthound.validators.completeness.empty",
    "WhitespaceOnlyValidator": "truthound.validators.completeness.empty",
    "ConditionalNullValidator": "truthound.validators.completeness.conditional",
    "DefaultValueValidator": "truthound.validators.completeness.default",
    "NaNValidator": "truthound.validators.completeness.nan",
    "NotNaNValidator": "truthound.validators.completeness.nan",
    "NaNRatioValidator": "truthound.validators.completeness.nan",
    "InfinityValidator": "truthound.validators.completeness.nan",
    "FiniteValidator": "truthound.validators.completeness.nan",

    # === Uniqueness validators ===
    "UniqueValidator": "truthound.validators.uniqueness.unique",
    "UniqueRatioValidator": "truthound.validators.uniqueness.unique",
    "DistinctCountValidator": "truthound.validators.uniqueness.unique",
    "DuplicateValidator": "truthound.validators.uniqueness.duplicate",
    "DuplicateWithinGroupValidator": "truthound.validators.uniqueness.duplicate",
    "PrimaryKeyValidator": "truthound.validators.uniqueness.primary_key",
    "CompoundKeyValidator": "truthound.validators.uniqueness.primary_key",
    "DistinctValuesInSetValidator": "truthound.validators.uniqueness.distinct_values",
    "DistinctValuesEqualSetValidator": "truthound.validators.uniqueness.distinct_values",
    "DistinctValuesContainSetValidator": "truthound.validators.uniqueness.distinct_values",
    "DistinctCountBetweenValidator": "truthound.validators.uniqueness.distinct_values",
    "UniqueWithinRecordValidator": "truthound.validators.uniqueness.within_record",
    "AllColumnsUniqueWithinRecordValidator": "truthound.validators.uniqueness.within_record",
    "ColumnPairUniqueValidator": "truthound.validators.uniqueness.within_record",
    "HyperLogLog": "truthound.validators.uniqueness.approximate",
    "ApproximateDistinctCountValidator": "truthound.validators.uniqueness.approximate",
    "ApproximateUniqueRatioValidator": "truthound.validators.uniqueness.approximate",
    "StreamingDistinctCountValidator": "truthound.validators.uniqueness.approximate",

    # === Distribution validators ===
    "BetweenValidator": "truthound.validators.distribution.range",
    "RangeValidator": "truthound.validators.distribution.range",
    "PositiveValidator": "truthound.validators.distribution.range",
    "NonNegativeValidator": "truthound.validators.distribution.range",
    "InSetValidator": "truthound.validators.distribution.set",
    "NotInSetValidator": "truthound.validators.distribution.set",
    "IncreasingValidator": "truthound.validators.distribution.monotonic",
    "DecreasingValidator": "truthound.validators.distribution.monotonic",
    "OutlierValidator": "truthound.validators.distribution.outlier",
    "ZScoreOutlierValidator": "truthound.validators.distribution.outlier",
    "QuantileValidator": "truthound.validators.distribution.quantile",
    "DistributionValidator": "truthound.validators.distribution.distribution",
    "KLDivergenceValidator": "truthound.validators.distribution.statistical",
    "ChiSquareValidator": "truthound.validators.distribution.statistical",
    "MostCommonValueValidator": "truthound.validators.distribution.statistical",

    # === String validators ===
    "RegexValidator": "truthound.validators.string.regex",
    "RegexListValidator": "truthound.validators.string.regex_extended",
    "NotMatchRegexValidator": "truthound.validators.string.regex_extended",
    "NotMatchRegexListValidator": "truthound.validators.string.regex_extended",
    "LengthValidator": "truthound.validators.string.length",
    "VectorizedFormatValidator": "truthound.validators.string.format",
    "EmailValidator": "truthound.validators.string.format",
    "UrlValidator": "truthound.validators.string.format",
    "PhoneValidator": "truthound.validators.string.format",
    "PhonePatterns": "truthound.validators.string.format",
    "UuidValidator": "truthound.validators.string.format",
    "IpAddressValidator": "truthound.validators.string.format",
    "Ipv6AddressValidator": "truthound.validators.string.format",
    "FormatValidator": "truthound.validators.string.format",
    "JsonParseableValidator": "truthound.validators.string.json",
    "JsonSchemaValidator": "truthound.validators.string.json_schema",
    "AlphanumericValidator": "truthound.validators.string.charset",
    "ConsistentCasingValidator": "truthound.validators.string.casing",
    "LikePatternValidator": "truthound.validators.string.like_pattern",
    "NotLikePatternValidator": "truthound.validators.string.like_pattern",

    # === Datetime validators ===
    "DateFormatValidator": "truthound.validators.datetime.format",
    "DateBetweenValidator": "truthound.validators.datetime.range",
    "FutureDateValidator": "truthound.validators.datetime.range",
    "PastDateValidator": "truthound.validators.datetime.range",
    "DateOrderValidator": "truthound.validators.datetime.order",
    "TimezoneValidator": "truthound.validators.datetime.timezone",
    "RecentDataValidator": "truthound.validators.datetime.freshness",
    "DatePartCoverageValidator": "truthound.validators.datetime.freshness",
    "GroupedRecentDataValidator": "truthound.validators.datetime.freshness",
    "DateutilParseableValidator": "truthound.validators.datetime.parseable",

    # === Aggregate validators ===
    "MeanBetweenValidator": "truthound.validators.aggregate.central",
    "MedianBetweenValidator": "truthound.validators.aggregate.central",
    "StdBetweenValidator": "truthound.validators.aggregate.spread",
    "VarianceBetweenValidator": "truthound.validators.aggregate.spread",
    "MinBetweenValidator": "truthound.validators.aggregate.extremes",
    "MaxBetweenValidator": "truthound.validators.aggregate.extremes",
    "SumBetweenValidator": "truthound.validators.aggregate.sum",
    "TypeValidator": "truthound.validators.aggregate.type",

    # === Cross-table validators ===
    "CrossTableRowCountValidator": "truthound.validators.cross_table.row_count",
    "CrossTableRowCountFactorValidator": "truthound.validators.cross_table.row_count",
    "CrossTableAggregateValidator": "truthound.validators.cross_table.aggregate",
    "CrossTableDistinctCountValidator": "truthound.validators.cross_table.aggregate",

    # === Query validators ===
    "QueryValidator": "truthound.validators.query.base",
    "ExpressionValidator": "truthound.validators.query.base",
    "QueryReturnsSingleValueValidator": "truthound.validators.query.result",
    "QueryReturnsNoRowsValidator": "truthound.validators.query.result",
    "QueryReturnsRowsValidator": "truthound.validators.query.result",
    "QueryResultMatchesValidator": "truthound.validators.query.result",
    "QueryRowCountValidator": "truthound.validators.query.row_count",
    "QueryRowCountRatioValidator": "truthound.validators.query.row_count",
    "QueryRowCountCompareValidator": "truthound.validators.query.row_count",
    "QueryColumnValuesValidator": "truthound.validators.query.column",
    "QueryColumnUniqueValidator": "truthound.validators.query.column",
    "QueryColumnNotNullValidator": "truthound.validators.query.column",
    "QueryAggregateValidator": "truthound.validators.query.aggregate",
    "QueryGroupAggregateValidator": "truthound.validators.query.aggregate",
    "QueryAggregateCompareValidator": "truthound.validators.query.aggregate",
    "CustomExpressionValidator": "truthound.validators.query.expression",
    "ConditionalExpressionValidator": "truthound.validators.query.expression",
    "MultiConditionValidator": "truthound.validators.query.expression",
    "RowLevelValidator": "truthound.validators.query.expression",

    # === Multi-column validators ===
    "MultiColumnValidator": "truthound.validators.multi_column.base",
    "ColumnArithmeticValidator": "truthound.validators.multi_column.base",
    "ColumnSumValidator": "truthound.validators.multi_column.arithmetic",
    "ColumnProductValidator": "truthound.validators.multi_column.arithmetic",
    "ColumnDifferenceValidator": "truthound.validators.multi_column.arithmetic",
    "ColumnRatioValidator": "truthound.validators.multi_column.arithmetic",
    "ColumnPercentageValidator": "truthound.validators.multi_column.arithmetic",
    "ColumnComparisonValidator": "truthound.validators.multi_column.comparison",
    "ColumnChainComparisonValidator": "truthound.validators.multi_column.comparison",
    "ColumnMaxValidator": "truthound.validators.multi_column.comparison",
    "ColumnMinValidator": "truthound.validators.multi_column.comparison",
    "ColumnMeanValidator": "truthound.validators.multi_column.comparison",
    "ColumnConsistencyValidator": "truthound.validators.multi_column.consistency",
    "ColumnMutualExclusivityValidator": "truthound.validators.multi_column.consistency",
    "ColumnCoexistenceValidator": "truthound.validators.multi_column.consistency",
    "ColumnDependencyValidator": "truthound.validators.multi_column.consistency",
    "ColumnImplicationValidator": "truthound.validators.multi_column.consistency",
    "ColumnCorrelationValidator": "truthound.validators.multi_column.statistical",
    "ColumnCovarianceValidator": "truthound.validators.multi_column.statistical",
    "MultiColumnVarianceValidator": "truthound.validators.multi_column.statistical",

    # === Table validators ===
    "TableValidator": "truthound.validators.table.base",
    "TableRowCountRangeValidator": "truthound.validators.table.row_count",
    "TableRowCountExactValidator": "truthound.validators.table.row_count",
    "TableRowCountCompareValidator": "truthound.validators.table.row_count",
    "TableNotEmptyValidator": "truthound.validators.table.row_count",
    "TableColumnCountValidator": "truthound.validators.table.column_count",
    "TableRequiredColumnsValidator": "truthound.validators.table.column_count",
    "TableForbiddenColumnsValidator": "truthound.validators.table.column_count",
    "TableFreshnessValidator": "truthound.validators.table.freshness",
    "TableDataRecencyValidator": "truthound.validators.table.freshness",
    "TableUpdateFrequencyValidator": "truthound.validators.table.freshness",
    "TableSchemaMatchValidator": "truthound.validators.table.schema",
    "TableSchemaCompareValidator": "truthound.validators.table.schema",
    "TableColumnTypesValidator": "truthound.validators.table.schema",
    "TableMemorySizeValidator": "truthound.validators.table.size",
    "TableRowToColumnRatioValidator": "truthound.validators.table.size",
    "TableDimensionsValidator": "truthound.validators.table.size",

    # === Geospatial validators ===
    "GeoValidator": "truthound.validators.geospatial.base",
    "EARTH_RADIUS_KM": "truthound.validators.geospatial.base",
    "EARTH_RADIUS_MILES": "truthound.validators.geospatial.base",
    "LatitudeValidator": "truthound.validators.geospatial.coordinate",
    "LongitudeValidator": "truthound.validators.geospatial.coordinate",
    "CoordinateValidator": "truthound.validators.geospatial.coordinate",
    "CoordinateNotNullIslandValidator": "truthound.validators.geospatial.coordinate",
    "GeoDistanceValidator": "truthound.validators.geospatial.distance",
    "GeoDistanceFromPointValidator": "truthound.validators.geospatial.distance",
    "GeoBoundingBoxValidator": "truthound.validators.geospatial.boundary",
    "GeoCountryValidator": "truthound.validators.geospatial.boundary",

    # === Drift validators ===
    "DriftValidator": "truthound.validators.drift.base",
    "ColumnDriftValidator": "truthound.validators.drift.base",
    "NumericDriftMixin": "truthound.validators.drift.base",
    "CategoricalDriftMixin": "truthound.validators.drift.base",
    "KSTestValidator": "truthound.validators.drift.statistical",
    "ChiSquareDriftValidator": "truthound.validators.drift.statistical",
    "WassersteinDriftValidator": "truthound.validators.drift.statistical",
    "PSIValidator": "truthound.validators.drift.psi",
    "CSIValidator": "truthound.validators.drift.psi",
    "MeanDriftValidator": "truthound.validators.drift.numeric",
    "VarianceDriftValidator": "truthound.validators.drift.numeric",
    "QuantileDriftValidator": "truthound.validators.drift.numeric",
    "RangeDriftValidator": "truthound.validators.drift.numeric",
    "FeatureDriftValidator": "truthound.validators.drift.multi_feature",
    "JSDivergenceValidator": "truthound.validators.drift.multi_feature",

    # === Anomaly validators ===
    "AnomalyValidator": "truthound.validators.anomaly.base",
    "ColumnAnomalyValidator": "truthound.validators.anomaly.base",
    "StatisticalAnomalyMixin": "truthound.validators.anomaly.base",
    "MLAnomalyMixin": "truthound.validators.anomaly.base",
    "IQRAnomalyValidator": "truthound.validators.anomaly.statistical",
    "MADAnomalyValidator": "truthound.validators.anomaly.statistical",
    "GrubbsTestValidator": "truthound.validators.anomaly.statistical",
    "TukeyFencesValidator": "truthound.validators.anomaly.statistical",
    "PercentileAnomalyValidator": "truthound.validators.anomaly.statistical",
    "MahalanobisValidator": "truthound.validators.anomaly.multivariate",
    "EllipticEnvelopeValidator": "truthound.validators.anomaly.multivariate",
    "PCAAnomalyValidator": "truthound.validators.anomaly.multivariate",
    "ZScoreMultivariateValidator": "truthound.validators.anomaly.multivariate",
    "IsolationForestValidator": "truthound.validators.anomaly.ml_based",
    "LOFValidator": "truthound.validators.anomaly.ml_based",
    "OneClassSVMValidator": "truthound.validators.anomaly.ml_based",
    "DBSCANAnomalyValidator": "truthound.validators.anomaly.ml_based",

    # === Referential validators ===
    "ForeignKeyRelation": "truthound.validators.referential.base",
    "TableNode": "truthound.validators.referential.base",
    "ReferentialValidator": "truthound.validators.referential.base",
    "MultiTableValidator": "truthound.validators.referential.base",
    "ForeignKeyValidator": "truthound.validators.referential.foreign_key",
    "CompositeForeignKeyValidator": "truthound.validators.referential.foreign_key",
    "SelfReferentialFKValidator": "truthound.validators.referential.foreign_key",
    "CascadeAction": "truthound.validators.referential.cascade",
    "CascadeRule": "truthound.validators.referential.cascade",
    "CascadeIntegrityValidator": "truthound.validators.referential.cascade",
    "CascadeDepthValidator": "truthound.validators.referential.cascade",
    "OrphanRecordValidator": "truthound.validators.referential.orphan",
    "MultiTableOrphanValidator": "truthound.validators.referential.orphan",
    "DanglingReferenceValidator": "truthound.validators.referential.orphan",
    "CircularReferenceValidator": "truthound.validators.referential.circular",
    "HierarchyCircularValidator": "truthound.validators.referential.circular",
    "HierarchyDepthValidator": "truthound.validators.referential.circular",

    # === Time series validators ===
    "TimeFrequency": "truthound.validators.timeseries.base",
    "TimeSeriesStats": "truthound.validators.timeseries.base",
    "TimeSeriesValidator": "truthound.validators.timeseries.base",
    "ValueTimeSeriesValidator": "truthound.validators.timeseries.base",
    "TimeSeriesGapValidator": "truthound.validators.timeseries.gap",
    "TimeSeriesIntervalValidator": "truthound.validators.timeseries.gap",
    "TimeSeriesDuplicateValidator": "truthound.validators.timeseries.gap",
    "MonotonicityType": "truthound.validators.timeseries.monotonic",
    "TimeSeriesMonotonicValidator": "truthound.validators.timeseries.monotonic",
    "TimeSeriesOrderValidator": "truthound.validators.timeseries.monotonic",
    "SeasonalPeriod": "truthound.validators.timeseries.seasonality",
    "SeasonalityValidator": "truthound.validators.timeseries.seasonality",
    "SeasonalDecompositionValidator": "truthound.validators.timeseries.seasonality",
    "TrendDirection": "truthound.validators.timeseries.trend",
    "TrendValidator": "truthound.validators.timeseries.trend",
    "TrendBreakValidator": "truthound.validators.timeseries.trend",
    "TimeSeriesCompletenessValidator": "truthound.validators.timeseries.completeness",
    "TimeSeriesValueCompletenessValidator": "truthound.validators.timeseries.completeness",
    "TimeSeriesDateRangeValidator": "truthound.validators.timeseries.completeness",

    # === Business rule validators ===
    "BusinessRuleValidator": "truthound.validators.business_rule.base",
    "ChecksumValidator": "truthound.validators.business_rule.base",
    "LuhnValidator": "truthound.validators.business_rule.checksum",
    "ISBNValidator": "truthound.validators.business_rule.checksum",
    "CreditCardValidator": "truthound.validators.business_rule.checksum",
    "IBANValidator": "truthound.validators.business_rule.financial",
    "VATValidator": "truthound.validators.business_rule.financial",
    "SWIFTValidator": "truthound.validators.business_rule.financial",

    # === Profiling validators ===
    "ProfileMetrics": "truthound.validators.profiling.base",
    "ProfilingValidator": "truthound.validators.profiling.base",
    "CardinalityValidator": "truthound.validators.profiling.cardinality",
    "UniquenessRatioValidator": "truthound.validators.profiling.cardinality",
    "EntropyValidator": "truthound.validators.profiling.entropy",
    "InformationGainValidator": "truthound.validators.profiling.entropy",
    "ValueFrequencyValidator": "truthound.validators.profiling.frequency",
    "DistributionShapeValidator": "truthound.validators.profiling.frequency",

    # === Localization validators ===
    "LocalizationValidator": "truthound.validators.localization.base",
    "KoreanBusinessNumberValidator": "truthound.validators.localization.korean",
    "KoreanRRNValidator": "truthound.validators.localization.korean",
    "KoreanPhoneValidator": "truthound.validators.localization.korean",
    "KoreanBankAccountValidator": "truthound.validators.localization.korean",
    "JapanesePostalCodeValidator": "truthound.validators.localization.japanese",
    "JapaneseMyNumberValidator": "truthound.validators.localization.japanese",
    "ChineseIDValidator": "truthound.validators.localization.chinese",
    "ChineseUSCCValidator": "truthound.validators.localization.chinese",

    # === ML feature validators ===
    "MLFeatureValidator": "truthound.validators.ml_feature.base",
    "FeatureStats": "truthound.validators.ml_feature.base",
    "CorrelationResult": "truthound.validators.ml_feature.base",
    "LeakageResult": "truthound.validators.ml_feature.base",
    "ScaleType": "truthound.validators.ml_feature.scale",
    "FeatureNullImpactValidator": "truthound.validators.ml_feature.null_impact",
    "FeatureScaleValidator": "truthound.validators.ml_feature.scale",
    "FeatureCorrelationMatrixValidator": "truthound.validators.ml_feature.correlation",
    "TargetLeakageValidator": "truthound.validators.ml_feature.leakage",

    # === Privacy validators ===
    "PrivacyRegulation": "truthound.validators.privacy.base",
    "PIICategory": "truthound.validators.privacy.base",
    "ConsentStatus": "truthound.validators.privacy.base",
    "LegalBasis": "truthound.validators.privacy.base",
    "PIIFieldDefinition": "truthound.validators.privacy.base",
    "PrivacyFinding": "truthound.validators.privacy.base",
    "PrivacyValidator": "truthound.validators.privacy.base",
    "DataRetentionValidator": "truthound.validators.privacy.base",
    "ConsentValidator": "truthound.validators.privacy.base",
    "GDPRComplianceValidator": "truthound.validators.privacy.gdpr",
    "GDPRSpecialCategoryValidator": "truthound.validators.privacy.gdpr",
    "GDPRDataMinimizationValidator": "truthound.validators.privacy.gdpr",
    "GDPRRightToErasureValidator": "truthound.validators.privacy.gdpr",
    "CCPAComplianceValidator": "truthound.validators.privacy.ccpa",
    "CCPASensitiveInfoValidator": "truthound.validators.privacy.ccpa",
    "CCPADoNotSellValidator": "truthound.validators.privacy.ccpa",
    "CCPAConsumerRightsValidator": "truthound.validators.privacy.ccpa",
    "GlobalPrivacyValidator": "truthound.validators.privacy.global_patterns",
    "LGPDComplianceValidator": "truthound.validators.privacy.global_patterns",
    "PIPEDAComplianceValidator": "truthound.validators.privacy.global_patterns",
    "APPIComplianceValidator": "truthound.validators.privacy.global_patterns",
}


# =============================================================================
# Category to Module Mapping for Registry Discovery
# =============================================================================

# Maps category name to submodules that should be loaded for that category
CATEGORY_MODULES: dict[str, list[str]] = {
    "schema": [
        "truthound.validators.schema.column_exists",
        "truthound.validators.schema.column_count",
        "truthound.validators.schema.column_type",
        "truthound.validators.schema.column_order",
        "truthound.validators.schema.table_schema",
        "truthound.validators.schema.column_pair",
        "truthound.validators.schema.multi_column",
        "truthound.validators.schema.referential",
        "truthound.validators.schema.multi_column_aggregate",
        "truthound.validators.schema.column_pair_set",
    ],
    "completeness": [
        "truthound.validators.completeness.null",
        "truthound.validators.completeness.empty",
        "truthound.validators.completeness.conditional",
        "truthound.validators.completeness.default",
        "truthound.validators.completeness.nan",
    ],
    "uniqueness": [
        "truthound.validators.uniqueness.unique",
        "truthound.validators.uniqueness.duplicate",
        "truthound.validators.uniqueness.primary_key",
        "truthound.validators.uniqueness.distinct_values",
        "truthound.validators.uniqueness.within_record",
        "truthound.validators.uniqueness.approximate",
    ],
    "distribution": [
        "truthound.validators.distribution.range",
        "truthound.validators.distribution.set",
        "truthound.validators.distribution.monotonic",
        "truthound.validators.distribution.outlier",
        "truthound.validators.distribution.quantile",
        "truthound.validators.distribution.distribution",
        "truthound.validators.distribution.statistical",
    ],
    "string": [
        "truthound.validators.string.regex",
        "truthound.validators.string.regex_extended",
        "truthound.validators.string.length",
        "truthound.validators.string.format",
        "truthound.validators.string.json",
        "truthound.validators.string.json_schema",
        "truthound.validators.string.charset",
        "truthound.validators.string.casing",
        "truthound.validators.string.like_pattern",
    ],
    "datetime": [
        "truthound.validators.datetime.format",
        "truthound.validators.datetime.range",
        "truthound.validators.datetime.order",
        "truthound.validators.datetime.timezone",
        "truthound.validators.datetime.freshness",
        "truthound.validators.datetime.parseable",
    ],
    "aggregate": [
        "truthound.validators.aggregate.central",
        "truthound.validators.aggregate.spread",
        "truthound.validators.aggregate.extremes",
        "truthound.validators.aggregate.sum",
        "truthound.validators.aggregate.type",
    ],
    "cross_table": [
        "truthound.validators.cross_table.row_count",
        "truthound.validators.cross_table.aggregate",
    ],
    "query": [
        "truthound.validators.query.base",
        "truthound.validators.query.result",
        "truthound.validators.query.row_count",
        "truthound.validators.query.column",
        "truthound.validators.query.aggregate",
        "truthound.validators.query.expression",
    ],
    "multi_column": [
        "truthound.validators.multi_column.base",
        "truthound.validators.multi_column.arithmetic",
        "truthound.validators.multi_column.comparison",
        "truthound.validators.multi_column.consistency",
        "truthound.validators.multi_column.statistical",
    ],
    "table": [
        "truthound.validators.table.base",
        "truthound.validators.table.row_count",
        "truthound.validators.table.column_count",
        "truthound.validators.table.freshness",
        "truthound.validators.table.schema",
        "truthound.validators.table.size",
    ],
    "geospatial": [
        "truthound.validators.geospatial.base",
        "truthound.validators.geospatial.coordinate",
        "truthound.validators.geospatial.distance",
        "truthound.validators.geospatial.boundary",
    ],
    "drift": [
        "truthound.validators.drift.base",
        "truthound.validators.drift.statistical",
        "truthound.validators.drift.psi",
        "truthound.validators.drift.numeric",
        "truthound.validators.drift.multi_feature",
    ],
    "anomaly": [
        "truthound.validators.anomaly.base",
        "truthound.validators.anomaly.statistical",
        "truthound.validators.anomaly.multivariate",
        "truthound.validators.anomaly.ml_based",
    ],
    "referential": [
        "truthound.validators.referential.base",
        "truthound.validators.referential.foreign_key",
        "truthound.validators.referential.cascade",
        "truthound.validators.referential.orphan",
        "truthound.validators.referential.circular",
    ],
    "timeseries": [
        "truthound.validators.timeseries.base",
        "truthound.validators.timeseries.gap",
        "truthound.validators.timeseries.monotonic",
        "truthound.validators.timeseries.seasonality",
        "truthound.validators.timeseries.trend",
        "truthound.validators.timeseries.completeness",
    ],
    "business_rule": [
        "truthound.validators.business_rule.base",
        "truthound.validators.business_rule.checksum",
        "truthound.validators.business_rule.financial",
    ],
    "profiling": [
        "truthound.validators.profiling.base",
        "truthound.validators.profiling.cardinality",
        "truthound.validators.profiling.entropy",
        "truthound.validators.profiling.frequency",
    ],
    "localization": [
        "truthound.validators.localization.base",
        "truthound.validators.localization.korean",
        "truthound.validators.localization.japanese",
        "truthound.validators.localization.chinese",
    ],
    "ml_feature": [
        "truthound.validators.ml_feature.base",
        "truthound.validators.ml_feature.null_impact",
        "truthound.validators.ml_feature.scale",
        "truthound.validators.ml_feature.correlation",
        "truthound.validators.ml_feature.leakage",
    ],
    "privacy": [
        "truthound.validators.privacy.base",
        "truthound.validators.privacy.gdpr",
        "truthound.validators.privacy.ccpa",
        "truthound.validators.privacy.global_patterns",
    ],
}


class LazyValidatorLoader:
    """Loader for lazy validator imports using name-to-module mapping.

    This loader maintains a mapping of validator names to their source modules,
    enabling lazy loading of validator classes only when they are accessed.
    """

    def __init__(
        self,
        import_map: dict[str, str],
        metrics: ValidatorImportMetrics | None = None,
    ):
        """Initialize loader.

        Args:
            import_map: Mapping of validator names to module paths.
            metrics: Optional metrics tracker.
        """
        self._import_map = import_map
        self._cache: dict[str, Any] = {}
        self._metrics = metrics or ValidatorImportMetrics()
        self._loaded_modules: dict[str, Any] = {}

    def load(self, name: str) -> Any:
        """Load and return a validator class by name.

        Args:
            name: Validator class name to load.

        Returns:
            The requested validator class.

        Raises:
            AttributeError: If the validator is not in the import map.
        """
        self._metrics.record_access(name)

        # Check cache first
        if name in self._cache:
            return self._cache[name]

        # Check if we have a mapping
        if name not in self._import_map:
            raise AttributeError(f"module has no attribute '{name}'")

        module_path = self._import_map[name]

        try:
            start_time = time.perf_counter()

            # Load the module if not cached
            if module_path not in self._loaded_modules:
                self._loaded_modules[module_path] = importlib.import_module(module_path)

            module = self._loaded_modules[module_path]
            attr = getattr(module, name)

            # Cache the result
            self._cache[name] = attr

            duration = time.perf_counter() - start_time
            self._metrics.record_load(name, duration)

            logger.debug(
                f"Lazy loaded validator '{name}' from '{module_path}' "
                f"in {duration*1000:.2f}ms"
            )

            return attr

        except (ImportError, AttributeError) as e:
            self._metrics.record_failure(name)
            logger.warning(f"Failed to lazy load '{name}' from '{module_path}': {e}")
            raise AttributeError(
                f"cannot load validator '{name}' from '{module_path}': {e}"
            ) from e

    def is_available(self, name: str) -> bool:
        """Check if a validator is available for loading."""
        return name in self._import_map

    def get_available_names(self) -> list[str]:
        """Get list of all available validator names."""
        return list(self._import_map.keys())

    def get_loaded_names(self) -> list[str]:
        """Get list of already loaded validator names."""
        return list(self._cache.keys())

    def get_metrics(self) -> ValidatorImportMetrics:
        """Get import metrics."""
        return self._metrics

    def preload(self, *names: str) -> None:
        """Preload specific validators.

        Args:
            names: Validator names to preload.
        """
        for name in names:
            if name in self._import_map and name not in self._cache:
                try:
                    self.load(name)
                except AttributeError:
                    pass

    def preload_category(self, category: str) -> None:
        """Preload all validators in a category.

        Args:
            category: Category name to preload.
        """
        if category in CATEGORY_MODULES:
            for module_path in CATEGORY_MODULES[category]:
                for name, path in self._import_map.items():
                    if path == module_path and name not in self._cache:
                        try:
                            self.load(name)
                        except AttributeError:
                            pass


# Global loader instance
_validator_loader: LazyValidatorLoader | None = None


def get_validator_loader() -> LazyValidatorLoader:
    """Get or create the global validator loader."""
    global _validator_loader
    if _validator_loader is None:
        _validator_loader = LazyValidatorLoader(VALIDATOR_IMPORT_MAP)
    return _validator_loader


def validator_getattr(name: str) -> Any:
    """Module-level __getattr__ implementation for validators.

    This function should be assigned to __getattr__ in the validators __init__.py
    to enable lazy loading.

    Example:
        # In validators/__init__.py
        from truthound.validators._lazy import validator_getattr

        def __getattr__(name: str):
            return validator_getattr(name)
    """
    return get_validator_loader().load(name)


def get_validator_import_metrics() -> dict[str, Any]:
    """Get validator import metrics summary."""
    return get_validator_loader().get_metrics().get_summary()


def preload_validators(*names: str) -> None:
    """Preload specific validators."""
    get_validator_loader().preload(*names)


def preload_category(category: str) -> None:
    """Preload all validators in a category."""
    get_validator_loader().preload_category(category)
