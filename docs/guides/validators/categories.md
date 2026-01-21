# Validator Categories

Truthound organizes validators into 21 semantic categories plus 7 infrastructure modules. Each category addresses a specific aspect of data quality.

---

## Category Overview

### Validator Categories (21 categories, 264 validators)

| Category | Validators | Module Path | Description |
|----------|------------|-------------|-------------|
| [schema](#schema) | 14 | `truthound.validators.schema` | Table structure, column types |
| [completeness](#completeness) | 12 | `truthound.validators.completeness` | Null, NaN, empty values |
| [uniqueness](#uniqueness) | 17 | `truthound.validators.uniqueness` | Duplicates, primary keys |
| [distribution](#distribution) | 15 | `truthound.validators.distribution` | Range, set, statistical |
| [string](#string) | 20 | `truthound.validators.string` | Pattern, format validation |
| [datetime](#datetime) | 10 | `truthound.validators.datetime` | Date/time validation |
| [aggregate](#aggregate) | 8 | `truthound.validators.aggregate` | Statistical aggregates |
| [cross_table](#cross_table) | 4 | `truthound.validators.cross_table` | Multi-table checks |
| [query](#query) | 18 | `truthound.validators.query` | Expression-based |
| [multi_column](#multi_column) | 18 | `truthound.validators.multi_column` | Column relationships |
| [table](#table) | 17 | `truthound.validators.table` | Table metadata |
| [geospatial](#geospatial) | 12 | `truthound.validators.geospatial` | Geographic coordinates |
| [drift](#drift) | 14 | `truthound.validators.drift` | Distribution drift |
| [anomaly](#anomaly) | 17 | `truthound.validators.anomaly` | Outlier detection |
| [referential](#referential) | 16 | `truthound.validators.referential` | Foreign keys, orphans |
| [timeseries](#timeseries) | 18 | `truthound.validators.timeseries` | Time series validation |
| [business_rule](#business_rule) | 8 | `truthound.validators.business_rule` | Checksums, IBAN, VAT |
| [profiling](#profiling) | 8 | `truthound.validators.profiling` | Cardinality, entropy |
| [localization](#localization) | 9 | `truthound.validators.localization` | Regional identifiers |
| [ml_feature](#ml_feature) | 9 | `truthound.validators.ml_feature` | Feature validation |
| [privacy](#privacy) | 20+ | `truthound.validators.privacy` | GDPR, CCPA compliance |

### Infrastructure Modules (7 modules)

| Module | Location | Description |
|--------|----------|-------------|
| [sdk](custom-validators.md) | `truthound.validators.sdk` | Custom validator development |
| [security](security.md) | `truthound.validators.security` | SQL injection, ReDoS protection |
| [i18n](i18n.md) | `truthound.validators.i18n` | Internationalization (15 languages) |
| [timeout](#timeout) | `truthound.validators.timeout` | Distributed timeout management |
| [streaming](#streaming) | `truthound.validators.streaming` | Real-time validation |
| [memory](#memory) | `truthound.validators.memory` | Memory-efficient processing |
| [optimization](optimization.md) | `truthound.validators.optimization` | DAG execution, profiling |

---

## Schema

Validates table structure, column definitions, and data types.

**Submodules:**
- `column_exists` - Column presence validation
- `column_count` - Row/column count validation
- `column_type` - Data type validation
- `column_order` - Column ordering
- `table_schema` - Complete schema matching
- `column_pair` - Column pair relationships
- `multi_column` - Composite key validation
- `referential` - Referential integrity
- `multi_column_aggregate` - Multi-column calculations
- `column_pair_set` - Column pair set membership

**Validators:**

| Validator | Description |
|-----------|-------------|
| `ColumnExistsValidator` | Ensures required columns exist |
| `ColumnNotExistsValidator` | Ensures forbidden columns are absent |
| `ColumnCountValidator` | Validates column count (exact/range) |
| `RowCountValidator` | Validates row count (exact/range) |
| `ColumnTypeValidator` | Validates column data types |
| `ColumnOrderValidator` | Validates column ordering |
| `TableSchemaValidator` | Validates complete schema |
| `ColumnPairValidator` | Validates column pair relationships |
| `MultiColumnUniqueValidator` | Validates composite key uniqueness |
| `ReferentialIntegrityValidator` | Validates foreign key references |
| `MultiColumnSumValidator` | Validates column sum equals target |
| `MultiColumnCalculationValidator` | Validates arithmetic relationships |
| `ColumnPairInSetValidator` | Validates column pairs in allowed set |
| `ColumnPairNotInSetValidator` | Validates column pairs not in forbidden set |

---

## Completeness

Detects missing values: null, NaN, empty strings, whitespace.

**Submodules:**
- `null` - Null value detection
- `empty` - Empty string detection
- `conditional` - Conditional null validation
- `default` - Default value detection
- `nan` - NaN and infinity detection

**Validators:**

| Validator | Description |
|-----------|-------------|
| `NullValidator` | Detects null values |
| `NotNullValidator` | Ensures no null values |
| `CompletenessRatioValidator` | Validates minimum completeness ratio |
| `EmptyStringValidator` | Detects empty strings |
| `WhitespaceOnlyValidator` | Detects whitespace-only values |
| `ConditionalNullValidator` | Validates nulls based on conditions |
| `DefaultValueValidator` | Detects placeholder/default values |
| `NaNValidator` | Detects NaN values |
| `NotNaNValidator` | Ensures no NaN values |
| `NaNRatioValidator` | Validates maximum NaN ratio |
| `InfinityValidator` | Detects infinite values |
| `FiniteValidator` | Ensures all values are finite |

---

## Uniqueness

Validates value distinctness, duplicates, and key constraints.

**Submodules:**
- `unique` - Basic uniqueness checks
- `duplicate` - Duplicate detection
- `primary_key` - Primary/compound key validation
- `distinct_values` - Distinct value constraints
- `within_record` - Record-level uniqueness
- `approximate` - HyperLogLog approximate counts

**Validators:**

| Validator | Description |
|-----------|-------------|
| `UniqueValidator` | Ensures column values are unique |
| `UniqueRatioValidator` | Validates unique value ratio |
| `DistinctCountValidator` | Validates distinct count range |
| `DuplicateValidator` | Detects duplicate values |
| `DuplicateWithinGroupValidator` | Detects duplicates within groups |
| `PrimaryKeyValidator` | Validates primary key (unique + non-null) |
| `CompoundKeyValidator` | Validates composite primary key |
| `DistinctValuesInSetValidator` | All distinct values must be in set |
| `DistinctValuesEqualSetValidator` | Distinct values must equal set exactly |
| `DistinctValuesContainSetValidator` | Distinct values must contain set |
| `DistinctCountBetweenValidator` | Distinct count within range |
| `UniqueWithinRecordValidator` | Values unique within each row |
| `AllColumnsUniqueWithinRecordValidator` | All values unique per row |
| `ColumnPairUniqueValidator` | Column pair uniqueness |
| `HyperLogLog` | Approximate distinct count utility |
| `ApproximateDistinctCountValidator` | HyperLogLog-based distinct count |
| `ApproximateUniqueRatioValidator` | Approximate unique ratio |

---

## Distribution

Validates value ranges, sets, and statistical distributions.

**Submodules:**
- `range` - Numeric range validation
- `set` - Set membership validation
- `monotonic` - Monotonicity checks
- `outlier` - IQR/Z-score outlier detection
- `quantile` - Quantile validation
- `distribution` - Distribution shape validation
- `statistical` - KL divergence, chi-square

**Validators:**

| Validator | Description |
|-----------|-------------|
| `BetweenValidator` | Values within inclusive range |
| `RangeValidator` | Range with configurable inclusivity |
| `PositiveValidator` | All values > 0 |
| `NonNegativeValidator` | All values >= 0 |
| `InSetValidator` | Values must be in allowed set |
| `NotInSetValidator` | Values must not be in forbidden set |
| `IncreasingValidator` | Values monotonically increasing |
| `DecreasingValidator` | Values monotonically decreasing |
| `OutlierValidator` | IQR-based outlier detection |
| `ZScoreOutlierValidator` | Z-score outlier detection |
| `QuantileValidator` | Quantile bounds validation |
| `DistributionValidator` | Distribution shape validation |
| `KLDivergenceValidator` | KL divergence threshold |
| `ChiSquareValidator` | Chi-square goodness of fit |
| `MostCommonValueValidator` | Most common value validation |

---

## String

Validates string patterns, formats, and content.

**Submodules:**
- `regex` - Basic regex matching
- `regex_extended` - Multiple patterns, negation
- `length` - String length constraints
- `format` - Common format validators
- `json` - JSON parsing validation
- `json_schema` - JSON Schema validation
- `charset` - Character set validation
- `casing` - Case consistency
- `like_pattern` - SQL LIKE patterns

**Validators:**

| Validator | Description |
|-----------|-------------|
| `RegexValidator` | Regex pattern matching |
| `RegexListValidator` | Multiple patterns (any match) |
| `NotMatchRegexValidator` | Must not match pattern |
| `NotMatchRegexListValidator` | Must not match any pattern |
| `LengthValidator` | String length constraints |
| `VectorizedFormatValidator` | Base vectorized format validator |
| `EmailValidator` | RFC 5322 email format |
| `UrlValidator` | URL format validation |
| `PhoneValidator` | Phone number format |
| `UuidValidator` | UUID format (v1-v5) |
| `IpAddressValidator` | IPv4 address format |
| `Ipv6AddressValidator` | IPv6 address format |
| `FormatValidator` | Auto-detect format by column name |
| `JsonParseableValidator` | Valid JSON strings |
| `JsonSchemaValidator` | JSON Schema validation |
| `AlphanumericValidator` | Alphanumeric characters only |
| `ConsistentCasingValidator` | Consistent case style |
| `LikePatternValidator` | SQL LIKE pattern match |
| `NotLikePatternValidator` | SQL LIKE pattern exclusion |

---

## Datetime

Validates temporal data formats, ranges, and ordering.

**Submodules:**
- `format` - Date/time format validation
- `range` - Date range validation
- `order` - Temporal ordering
- `timezone` - Timezone validation
- `freshness` - Data recency
- `parseable` - Date parsing validation

**Validators:**

| Validator | Description |
|-----------|-------------|
| `DateFormatValidator` | Date format validation |
| `DateBetweenValidator` | Date within range |
| `FutureDateValidator` | Date must be in future |
| `PastDateValidator` | Date must be in past |
| `DateOrderValidator` | Start date < end date |
| `TimezoneValidator` | Timezone-aware validation |
| `RecentDataValidator` | Data within max age |
| `DatePartCoverageValidator` | Coverage across date parts |
| `GroupedRecentDataValidator` | Recency within groups |
| `DateutilParseableValidator` | Parseable by dateutil |

---

## Aggregate

Validates column-level statistical aggregates.

**Submodules:**
- `central` - Mean, median validation
- `spread` - Std, variance validation
- `extremes` - Min, max validation
- `sum` - Sum validation
- `type` - Type validation at aggregate level

**Validators:**

| Validator | Description |
|-----------|-------------|
| `MeanBetweenValidator` | Column mean within range |
| `MedianBetweenValidator` | Column median within range |
| `StdBetweenValidator` | Standard deviation within range |
| `VarianceBetweenValidator` | Variance within range |
| `MinBetweenValidator` | Column minimum within range |
| `MaxBetweenValidator` | Column maximum within range |
| `SumBetweenValidator` | Column sum within range |
| `TypeValidator` | Aggregate-level type validation |

---

## Cross_table

Validates relationships between multiple tables.

**Submodules:**
- `row_count` - Row count comparisons
- `aggregate` - Aggregate comparisons

**Validators:**

| Validator | Description |
|-----------|-------------|
| `CrossTableRowCountValidator` | Compare row counts |
| `CrossTableRowCountFactorValidator` | Row count ratio validation |
| `CrossTableAggregateValidator` | Compare aggregates |
| `CrossTableDistinctCountValidator` | Compare distinct counts |

---

## Query

Expression-based validation using Polars expressions.

**Submodules:**
- `base` - Base query validators
- `result` - Query result validation
- `row_count` - Query row count validation
- `column` - Query column validation
- `aggregate` - Query aggregate validation
- `expression` - Custom expression validation

**Validators:**

| Validator | Description |
|-----------|-------------|
| `QueryValidator` | Base query validator |
| `ExpressionValidator` | Polars expression validation |
| `QueryReturnsSingleValueValidator` | Query returns one value |
| `QueryReturnsNoRowsValidator` | Query returns no rows |
| `QueryReturnsRowsValidator` | Query returns at least one row |
| `QueryResultMatchesValidator` | Query result matches expected |
| `QueryRowCountValidator` | Query result row count |
| `QueryRowCountRatioValidator` | Ratio of matching rows |
| `QueryRowCountCompareValidator` | Compare row counts |
| `QueryColumnValuesValidator` | Validate column values |
| `QueryColumnUniqueValidator` | Query column uniqueness |
| `QueryColumnNotNullValidator` | Query column non-null |
| `QueryAggregateValidator` | Query aggregate validation |
| `QueryGroupAggregateValidator` | Group aggregate validation |
| `QueryAggregateCompareValidator` | Compare aggregates |
| `CustomExpressionValidator` | Custom expression strings |
| `ConditionalExpressionValidator` | Conditional expressions |
| `MultiConditionValidator` | Multiple conditions |
| `RowLevelValidator` | Row-level validation |

---

## Multi_column

Validates relationships across multiple columns.

**Submodules:**
- `base` - Base multi-column validators
- `arithmetic` - Arithmetic relationships
- `comparison` - Column comparisons
- `consistency` - Consistency patterns
- `statistical` - Statistical relationships

**Validators:**

| Validator | Description |
|-----------|-------------|
| `MultiColumnValidator` | Base multi-column validator |
| `ColumnArithmeticValidator` | Arithmetic relationships |
| `ColumnSumValidator` | Columns sum to target |
| `ColumnProductValidator` | Columns multiply to target |
| `ColumnDifferenceValidator` | Column difference validation |
| `ColumnRatioValidator` | Column ratio validation |
| `ColumnPercentageValidator` | Percentage validation |
| `ColumnComparisonValidator` | Column comparison (>, <, ==) |
| `ColumnChainComparisonValidator` | Ordered column chain |
| `ColumnMaxValidator` | Max across columns |
| `ColumnMinValidator` | Min across columns |
| `ColumnMeanValidator` | Mean across columns |
| `ColumnConsistencyValidator` | Consistency patterns |
| `ColumnMutualExclusivityValidator` | At most one non-null |
| `ColumnCoexistenceValidator` | All null or all non-null |
| `ColumnDependencyValidator` | Functional dependencies |
| `ColumnImplicationValidator` | If A then B |
| `ColumnCorrelationValidator` | Correlation validation |
| `ColumnCovarianceValidator` | Covariance validation |
| `MultiColumnVarianceValidator` | Variance across columns |

---

## Table

Validates table-level metadata and properties.

**Submodules:**
- `base` - Base table validator
- `row_count` - Row count validation
- `column_count` - Column count validation
- `freshness` - Data freshness
- `schema` - Schema matching
- `size` - Size validation

**Validators:**

| Validator | Description |
|-----------|-------------|
| `TableValidator` | Base table validator |
| `TableRowCountRangeValidator` | Row count within range |
| `TableRowCountExactValidator` | Exact row count |
| `TableRowCountCompareValidator` | Compare with reference |
| `TableNotEmptyValidator` | Table not empty |
| `TableColumnCountValidator` | Column count validation |
| `TableRequiredColumnsValidator` | Required columns present |
| `TableForbiddenColumnsValidator` | Forbidden columns absent |
| `TableFreshnessValidator` | Data freshness validation |
| `TableDataRecencyValidator` | Recent data presence |
| `TableUpdateFrequencyValidator` | Update frequency check |
| `TableSchemaMatchValidator` | Schema matches spec |
| `TableSchemaCompareValidator` | Compare with reference |
| `TableColumnTypesValidator` | Column types match |
| `TableMemorySizeValidator` | Memory size bounds |
| `TableRowToColumnRatioValidator` | Row/column ratio |
| `TableDimensionsValidator` | Table dimensions validation |

---

## Geospatial

Validates geographic coordinates and spatial data.

**Submodules:**
- `base` - Base geo validator with Haversine
- `coordinate` - Coordinate validation
- `distance` - Distance calculations
- `boundary` - Bounding box, country

**Validators:**

| Validator | Description |
|-----------|-------------|
| `GeoValidator` | Base geospatial validator |
| `LatitudeValidator` | Latitude range (-90 to 90) |
| `LongitudeValidator` | Longitude range (-180 to 180) |
| `CoordinateValidator` | Coordinate pair validation |
| `CoordinateNotNullIslandValidator` | Detects (0, 0) coordinates |
| `GeoDistanceValidator` | Distance between coordinate pairs |
| `GeoDistanceFromPointValidator` | Distance from reference point |
| `GeoBoundingBoxValidator` | Within bounding box |
| `GeoCountryValidator` | Within country boundaries |

**Constants:**
- `EARTH_RADIUS_KM` = 6371.0
- `EARTH_RADIUS_MILES` = 3958.8

---

## Drift

Detects distribution changes between reference and current data.

**Installation:** `pip install truthound[drift]`

**Submodules:**
- `base` - Base drift validators
- `statistical` - KS test, chi-square, Wasserstein
- `psi` - Population Stability Index
- `numeric` - Mean, variance, quantile drift
- `multi_feature` - Multi-feature drift

**Validators:**

| Validator | Description |
|-----------|-------------|
| `DriftValidator` | Base drift validator |
| `ColumnDriftValidator` | Single-column drift base |
| `KSTestValidator` | Kolmogorov-Smirnov test |
| `ChiSquareDriftValidator` | Chi-square for categorical |
| `WassersteinDriftValidator` | Earth Mover's Distance |
| `PSIValidator` | Population Stability Index |
| `CSIValidator` | Characteristic Stability Index |
| `MeanDriftValidator` | Mean change detection |
| `VarianceDriftValidator` | Variance change detection |
| `QuantileDriftValidator` | Quantile change detection |
| `RangeDriftValidator` | Range change detection |
| `FeatureDriftValidator` | Multi-feature drift |
| `JSDivergenceValidator` | Jensen-Shannon divergence |

**Mixins:**
- `NumericDriftMixin` - Numeric drift utilities
- `CategoricalDriftMixin` - Categorical drift utilities

---

## Anomaly

Detects outliers using statistical and ML methods.

**Installation:** `pip install truthound[anomaly]`

**Submodules:**
- `base` - Base anomaly validators
- `statistical` - IQR, MAD, Grubbs, Tukey
- `multivariate` - Mahalanobis, Elliptic, PCA
- `ml_based` - Isolation Forest, LOF, SVM, DBSCAN

**Validators:**

| Validator | Description |
|-----------|-------------|
| `AnomalyValidator` | Base anomaly validator |
| `ColumnAnomalyValidator` | Single-column anomaly base |
| `IQRAnomalyValidator` | Interquartile range method |
| `MADAnomalyValidator` | Median absolute deviation |
| `GrubbsTestValidator` | Grubbs' test for single outlier |
| `TukeyFencesValidator` | Inner/outer fences |
| `PercentileAnomalyValidator` | Percentile-based bounds |
| `MahalanobisValidator` | Multivariate Mahalanobis |
| `EllipticEnvelopeValidator` | Robust Gaussian fitting |
| `PCAAnomalyValidator` | PCA reconstruction error |
| `ZScoreMultivariateValidator` | Multi-column Z-score |
| `IsolationForestValidator` | Isolation Forest |
| `LOFValidator` | Local Outlier Factor |
| `OneClassSVMValidator` | One-Class SVM |
| `DBSCANAnomalyValidator` | DBSCAN clustering |

**Mixins:**
- `StatisticalAnomalyMixin` - Statistical method utilities
- `MLAnomalyMixin` - ML method utilities

---

## Referential

Validates foreign key relationships and hierarchy integrity.

**Submodules:**
- `base` - Base referential validators
- `foreign_key` - Foreign key validation
- `cascade` - Cascade integrity
- `orphan` - Orphan record detection
- `circular` - Circular reference detection

**Validators:**

| Validator | Description |
|-----------|-------------|
| `ReferentialValidator` | Base referential validator |
| `MultiTableValidator` | Multi-table validation base |
| `ForeignKeyValidator` | Foreign key validation |
| `CompositeForeignKeyValidator` | Composite FK validation |
| `SelfReferentialFKValidator` | Self-referential FK |
| `CascadeIntegrityValidator` | Cascade action integrity |
| `CascadeDepthValidator` | Cascade depth limits |
| `OrphanRecordValidator` | Orphan record detection |
| `MultiTableOrphanValidator` | Multi-table orphan check |
| `DanglingReferenceValidator` | Dangling reference detection |
| `CircularReferenceValidator` | Circular reference detection |
| `HierarchyCircularValidator` | Hierarchy cycle detection |
| `HierarchyDepthValidator` | Hierarchy depth limits |

**Data Classes:**
- `ForeignKeyRelation` - FK relationship definition
- `TableNode` - Table graph node
- `CascadeAction` - Cascade action enum
- `CascadeRule` - Cascade rule definition

---

## Timeseries

Validates time series data properties.

**Submodules:**
- `base` - Base time series validators
- `gap` - Gap and duplicate detection
- `monotonic` - Monotonicity validation
- `seasonality` - Seasonal pattern detection
- `trend` - Trend analysis
- `completeness` - Time series completeness

**Validators:**

| Validator | Description |
|-----------|-------------|
| `TimeSeriesValidator` | Base time series validator |
| `ValueTimeSeriesValidator` | Value-based time series |
| `TimeSeriesGapValidator` | Gap detection |
| `TimeSeriesIntervalValidator` | Interval regularity |
| `TimeSeriesDuplicateValidator` | Duplicate timestamp detection |
| `TimeSeriesMonotonicValidator` | Monotonicity validation |
| `TimeSeriesOrderValidator` | Timestamp ordering |
| `SeasonalityValidator` | Seasonal pattern validation |
| `SeasonalDecompositionValidator` | Decomposition validation |
| `TrendValidator` | Trend direction validation |
| `TrendBreakValidator` | Trend break detection |
| `TimeSeriesCompletenessValidator` | Series completeness |
| `TimeSeriesValueCompletenessValidator` | Value completeness |
| `TimeSeriesDateRangeValidator` | Date range validation |

**Enums & Data Classes:**
- `TimeFrequency` - Frequency types (DAILY, HOURLY, etc.)
- `TimeSeriesStats` - Time series statistics
- `MonotonicityType` - Monotonicity types
- `SeasonalPeriod` - Seasonal period definitions
- `TrendDirection` - Trend direction enum

---

## Business_rule

Validates domain-specific business rules and checksums.

**Submodules:**
- `base` - Base business rule validator
- `checksum` - Luhn, ISBN, credit card
- `financial` - IBAN, VAT, SWIFT

**Validators:**

| Validator | Description |
|-----------|-------------|
| `BusinessRuleValidator` | Base business rule validator |
| `ChecksumValidator` | Generic checksum validation |
| `LuhnValidator` | Luhn algorithm (credit cards) |
| `ISBNValidator` | ISBN validation |
| `CreditCardValidator` | Credit card number validation |
| `IBANValidator` | International Bank Account Number |
| `VATValidator` | VAT number validation |
| `SWIFTValidator` | SWIFT/BIC code validation |

---

## Profiling

Validates data profiling metrics and distributions.

**Submodules:**
- `base` - Base profiling validator
- `cardinality` - Cardinality metrics
- `entropy` - Entropy calculations
- `frequency` - Value frequency analysis

**Validators:**

| Validator | Description |
|-----------|-------------|
| `ProfilingValidator` | Base profiling validator |
| `CardinalityValidator` | Cardinality bounds |
| `UniquenessRatioValidator` | Uniqueness ratio validation |
| `EntropyValidator` | Shannon entropy bounds |
| `InformationGainValidator` | Information gain validation |
| `ValueFrequencyValidator` | Value frequency distribution |
| `DistributionShapeValidator` | Distribution shape validation |

**Data Classes:**
- `ProfileMetrics` - Profiling metrics container

---

## Localization

Validates regional identifier formats.

**Submodules:**
- `base` - Base localization validator
- `korean` - Korean formats
- `japanese` - Japanese formats
- `chinese` - Chinese formats

**Validators:**

| Validator | Description |
|-----------|-------------|
| `LocalizationValidator` | Base localization validator |
| `KoreanBusinessNumberValidator` | Korean business number (사업자등록번호) |
| `KoreanRRNValidator` | Korean Resident Registration Number |
| `KoreanPhoneValidator` | Korean phone format |
| `KoreanBankAccountValidator` | Korean bank account |
| `JapanesePostalCodeValidator` | Japanese postal code |
| `JapaneseMyNumberValidator` | Japanese My Number |
| `ChineseIDValidator` | Chinese ID number |
| `ChineseUSCCValidator` | Chinese USCC (统一社会信用代码) |

---

## ML_feature

Validates machine learning feature quality.

**Submodules:**
- `base` - Base ML feature validator
- `null_impact` - Null impact analysis
- `scale` - Feature scale validation
- `correlation` - Correlation analysis
- `leakage` - Target leakage detection

**Validators:**

| Validator | Description |
|-----------|-------------|
| `MLFeatureValidator` | Base ML feature validator |
| `FeatureNullImpactValidator` | Null value impact analysis |
| `FeatureScaleValidator` | Feature scale validation |
| `FeatureCorrelationMatrixValidator` | Correlation matrix validation |
| `TargetLeakageValidator` | Target leakage detection |

**Enums & Data Classes:**
- `ScaleType` - Scale types (STANDARD, MINMAX, etc.)
- `FeatureStats` - Feature statistics
- `CorrelationResult` - Correlation results
- `LeakageResult` - Leakage detection results

---

## Privacy

Validates privacy compliance (GDPR, CCPA, LGPD, etc.).

**Submodules:**
- `base` - Base privacy validators
- `gdpr` - GDPR compliance
- `ccpa` - CCPA compliance
- `global_patterns` - Global privacy patterns

**Validators:**

| Validator | Description |
|-----------|-------------|
| `PrivacyValidator` | Base privacy validator |
| `DataRetentionValidator` | Data retention compliance |
| `ConsentValidator` | Consent validation |
| `GDPRComplianceValidator` | GDPR overall compliance |
| `GDPRSpecialCategoryValidator` | GDPR special category data |
| `GDPRDataMinimizationValidator` | Data minimization principle |
| `GDPRRightToErasureValidator` | Right to erasure validation |
| `CCPAComplianceValidator` | CCPA overall compliance |
| `CCPASensitiveInfoValidator` | CCPA sensitive info |
| `CCPADoNotSellValidator` | Do-not-sell compliance |
| `CCPAConsumerRightsValidator` | Consumer rights validation |
| `GlobalPrivacyValidator` | Global privacy patterns |
| `LGPDComplianceValidator` | Brazilian LGPD compliance |
| `PIPEDAComplianceValidator` | Canadian PIPEDA compliance |
| `APPIComplianceValidator` | Japanese APPI compliance |

**Enums & Data Classes:**
- `PrivacyRegulation` - Regulation types
- `PIICategory` - PII category types
- `ConsentStatus` - Consent status
- `LegalBasis` - Legal basis types
- `PIIFieldDefinition` - PII field definition
- `PrivacyFinding` - Privacy finding results

---

## Timeout

Distributed timeout management for validation operations.

**Key Components:**
- `DeadlineContext` - Deadline propagation context
- `TimeoutBudget` - Time budget allocation
- `CascadeTimeoutHandler` - Cascading timeout management
- `GracefulDegradation` - Fallback on timeout

See [Built-in Validators](built-in.md#20-distributed-timeout) for detailed usage.

---

## Streaming

Real-time streaming data validation.

**Key Components:**
- `StreamingValidatorMixin` - Base mixin for streaming
- Streaming-compatible completeness validators
- Streaming-compatible range validators

---

## Memory

Memory-efficient validation for large datasets.

**Key Components:**
- Memory-aware processing algorithms
- Approximate algorithms (HyperLogLog, streaming ECDF)
- SGD online learning for outlier detection

---

## Next Steps

- [Built-in Validators Reference](built-in.md) - Detailed parameter reference
- [Custom Validators](custom-validators.md) - Build your own validators
- [Security Features](security.md) - ReDoS protection, SQL injection prevention
- [i18n Support](i18n.md) - Internationalized error messages
- [Performance Optimization](optimization.md) - DAG execution, profiling
