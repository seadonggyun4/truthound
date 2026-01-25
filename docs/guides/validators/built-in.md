# Built-in Validators Reference

Complete parameter reference for all 264 built-in validators organized by category.

---

## Table of Contents

1. [Schema Validators](#1-schema-validators) (14)
2. [Completeness Validators](#2-completeness-validators) (12)
3. [Uniqueness Validators](#3-uniqueness-validators) (17)
4. [Distribution Validators](#4-distribution-validators) (15)
5. [String Validators](#5-string-validators) (20)
6. [Datetime Validators](#6-datetime-validators) (10)
7. [Aggregate Validators](#7-aggregate-validators) (8)
8. [Cross-Table Validators](#8-cross-table-validators) (4)
9. [Query Validators](#9-query-validators) (19)
10. [Multi-Column Validators](#10-multi-column-validators) (20)
11. [Table Validators](#11-table-validators) (17)
12. [Geospatial Validators](#12-geospatial-validators) (9)
13. [Drift Validators](#13-drift-validators) (14)
14. [Anomaly Validators](#14-anomaly-validators) (15)
15. [Referential Validators](#15-referential-validators) (13)
16. [Time Series Validators](#16-time-series-validators) (14)
17. [Business Rule Validators](#17-business-rule-validators) (8)
18. [Profiling Validators](#18-profiling-validators) (7)
19. [Localization Validators](#19-localization-validators) (9)
20. [ML Feature Validators](#20-ml-feature-validators) (5)
21. [Privacy Validators](#21-privacy-validators) (15)

---

## 1. Schema Validators

Import: `from truthound.validators.schema import *`

### ColumnExistsValidator

Validates the presence of specified columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Column names that must exist |

```python
validator = ColumnExistsValidator(columns=["id", "name", "email"])
```

### ColumnNotExistsValidator

Ensures specified columns are absent.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Column names that must not exist |

### ColumnCountValidator

Validates column count.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `exact_count` | `int \| None` | No | `None` | Exact expected count |
| `min_count` | `int \| None` | No | `None` | Minimum count |
| `max_count` | `int \| None` | No | `None` | Maximum count |

```python
validator = ColumnCountValidator(min_count=5, max_count=20)
```

### RowCountValidator

Validates row count.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `exact_count` | `int \| None` | No | `None` | Exact expected count |
| `min_count` | `int \| None` | No | `None` | Minimum count |
| `max_count` | `int \| None` | No | `None` | Maximum count |

### ColumnTypeValidator

Validates column data types.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expected_types` | `dict[str, str \| type[pl.DataType]]` | Yes | - | Column to type mapping |

Supported type aliases: `"int"`, `"uint"`, `"float"`, `"numeric"`, `"string"`, `"bool"`, `"date"`, `"datetime"`, `"time"`.

```python
# Using Polars types directly
validator = ColumnTypeValidator(expected_types={
    "id": pl.Int64,
    "name": pl.String,
    "price": pl.Float64,
})

# Using type aliases
validator = ColumnTypeValidator(expected_types={
    "id": "int",           # Matches Int8, Int16, Int32, Int64
    "amount": "numeric",   # Matches any int, uint, or float
    "email": "string",     # Matches String, Utf8
})
```

### ColumnOrderValidator

Validates column ordering.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expected_order` | `list[str]` | Yes | - | Expected column order |
| `strict` | `bool` | No | `False` | No extra columns allowed |

### TableSchemaValidator

Validates complete schema.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `schema` | `dict[str, type]` | Yes | - | Column to type mapping |
| `strict` | `bool` | No | `False` | Reject extra columns |

### ColumnPairValidator

Validates column pair relationships.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |
| `relationship` | `str` | Yes | - | Relationship type |

### MultiColumnUniqueValidator

Validates composite key uniqueness.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns forming composite key |

### ReferentialIntegrityValidator

Validates foreign key references.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Foreign key column |
| `reference_data` | `pl.DataFrame` | Yes | - | Reference table |
| `reference_column` | `str` | Yes | - | Primary key in reference |

```python
validator = ReferentialIntegrityValidator(
    column="department_id",
    reference_data=departments_df,
    reference_column="id"
)
```

### MultiColumnSumValidator

Validates column sum equals target.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns to sum |
| `expected_sum` | `float` | Yes | - | Expected sum |
| `tolerance` | `float` | No | `0.0` | Acceptable tolerance |

### MultiColumnCalculationValidator

Validates arithmetic relationships.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expression` | `str` | Yes | - | Math expression |
| `tolerance` | `float` | No | `0.0` | Acceptable tolerance |

### ColumnPairInSetValidator

Validates column pairs in allowed set.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |
| `valid_pairs` | `set[tuple]` | Yes | - | Valid (a, b) pairs |

### ColumnPairNotInSetValidator

Validates column pairs not in forbidden set.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |
| `invalid_pairs` | `set[tuple]` | Yes | - | Forbidden (a, b) pairs |

---

## 2. Completeness Validators

Import: `from truthound.validators.completeness import *`

### NullValidator

Detects null values.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target columns (all if None) |

### NotNullValidator

Ensures no null values.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target columns (all if None) |

```python
# Single column
validator = NotNullValidator(columns=["email"])

# Multiple columns
validator = NotNullValidator(columns=["user_id", "username", "email"])

# All columns (default)
validator = NotNullValidator()
```

### CompletenessRatioValidator

Validates minimum completeness.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_ratio` | `float` | No | `0.95` | Min completeness (0.0-1.0) |
| `columns` | `list[str] \| None` | No | `None` | Target columns (all if None) |

```python
# Single column with custom ratio
validator = CompletenessRatioValidator(min_ratio=0.95, columns=["phone"])

# Multiple columns
validator = CompletenessRatioValidator(min_ratio=0.90, columns=["email", "phone"])
```

### EmptyStringValidator

Detects empty strings.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target columns |

### WhitespaceOnlyValidator

Detects whitespace-only values.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target columns |

### ConditionalNullValidator

Validates nulls based on condition.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `condition` | `pl.Expr` | Yes | - | Polars condition expression |

```python
# If status is "active", email must not be null
validator = ConditionalNullValidator(
    column="email",
    condition=pl.col("status") == "active"
)
```

### DefaultValueValidator

Detects default/placeholder values.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `default_values` | `list` | Yes | - | Values considered defaults |

### NaNValidator

Detects NaN values in float columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target float columns |
| `mostly` | `float \| None` | No | `None` | Min non-NaN ratio |

### NotNaNValidator

Ensures no NaN values.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### NaNRatioValidator

Validates maximum NaN ratio.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `max_ratio` | `float` | Yes | - | Max NaN ratio (0.0-1.0) |

### InfinityValidator

Detects infinite values.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target numeric columns |

### FiniteValidator

Ensures all values are finite.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

---

## 3. Uniqueness Validators

Import: `from truthound.validators.uniqueness import *`

### UniqueValidator

Ensures column values are unique.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target columns (None = all) |

### UniqueRatioValidator

Validates unique value ratio.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_ratio` | `float` | No | `0.0` | Min unique ratio |
| `max_ratio` | `float` | No | `1.0` | Max unique ratio |

### DistinctCountValidator

Validates distinct value count.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_count` | `int \| None` | No | `None` | Min distinct count |
| `max_count` | `int \| None` | No | `None` | Max distinct count |

### DuplicateValidator

Detects duplicates.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Columns to check |

### DuplicateWithinGroupValidator

Detects duplicates within groups.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Column to check |
| `group_by` | `list[str]` | Yes | - | Grouping columns |

### PrimaryKeyValidator

Validates primary key (unique + non-null).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Primary key column |

### CompoundKeyValidator

Validates composite primary key.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns forming compound key |

### DistinctValuesInSetValidator

All distinct values must be in set.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `value_set` | `set` | Yes | - | Allowed values |

### DistinctValuesEqualSetValidator

Distinct values must exactly match set.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `value_set` | `set` | Yes | - | Expected values |

### DistinctValuesContainSetValidator

Distinct values must contain set.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `value_set` | `set` | Yes | - | Required values |

### DistinctCountBetweenValidator

Distinct count within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_count` | `int` | Yes | - | Min count |
| `max_count` | `int` | Yes | - | Max count |

### UniqueWithinRecordValidator

Values unique within each row.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns to compare |

### AllColumnsUniqueWithinRecordValidator

All values unique within each row.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Columns to check (all if None) |

### ColumnPairUniqueValidator

Column pair uniqueness.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |

### ApproximateDistinctCountValidator

HyperLogLog-based distinct count.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_count` | `int \| None` | No | `None` | Min distinct count |
| `max_count` | `int \| None` | No | `None` | Max distinct count |
| `precision` | `int` | No | `14` | HLL precision bits |

### ApproximateUniqueRatioValidator

Approximate unique ratio.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_ratio` | `float` | No | `0.0` | Min unique ratio |
| `max_ratio` | `float` | No | `1.0` | Max unique ratio |

---

## 4. Distribution Validators

Import: `from truthound.validators.distribution import *`

### BetweenValidator

Values within specified range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_value` | `float \| None` | No | `None` | Minimum value |
| `max_value` | `float \| None` | No | `None` | Maximum value |
| `inclusive` | `bool` | No | `True` | Include boundaries |

### RangeValidator

Auto-detects expected ranges based on column names.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| (no parameters) | - | - | - | Uses built-in KNOWN_RANGES mapping |

**Auto-detected ranges:**
- `age`: [0, 150]
- `percentage/percent/pct/rate/score`: [0, 100]
- `rating`: [0, 5]
- `year`: [1900, 2100]
- `month`: [1, 12], `day`: [1, 31]
- `hour`: [0, 23], `minute/second`: [0, 59]
- `price/quantity/count/amount`: [0, None]

### PositiveValidator

All values > 0. Automatically applies to all numeric columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| (no parameters) | - | - | - | Uses `NumericValidatorMixin` to auto-detect numeric columns |

```python
# Validates all numeric columns have positive values
validator = PositiveValidator()
issues = validator.validate(lf)
```

### NonNegativeValidator

All values >= 0. Automatically applies to all numeric columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| (no parameters) | - | - | - | Uses `NumericValidatorMixin` to auto-detect numeric columns |

```python
# Validates all numeric columns have non-negative values
validator = NonNegativeValidator()
issues = validator.validate(lf)
```

### InSetValidator

Values must be in allowed set. Applies to all columns (or specified via `columns` kwarg).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `allowed_values` | `list[Any]` | Yes | - | List of allowed values |

```python
validator = InSetValidator(allowed_values=["active", "inactive", "pending"])
issues = validator.validate(lf)
```

### NotInSetValidator

Values must not be in forbidden set. Applies to all columns (or specified via `columns` kwarg).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `forbidden_values` | `list[Any]` | Yes | - | List of forbidden values |

```python
validator = NotInSetValidator(forbidden_values=["N/A", "unknown", ""])
issues = validator.validate(lf)
```

### IncreasingValidator

Values monotonically increasing. Automatically applies to all numeric columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `strict` | `bool` | No | `False` | Strictly increasing (no equal values) |

```python
# Check for monotonic increase across all numeric columns
validator = IncreasingValidator(strict=True)
issues = validator.validate(lf)
```

### DecreasingValidator

Values monotonically decreasing. Automatically applies to all numeric columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `strict` | `bool` | No | `False` | Strictly decreasing (no equal values) |

```python
validator = DecreasingValidator()
issues = validator.validate(lf)
```

### OutlierValidator

IQR-based outlier detection. Automatically applies to all numeric columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `iqr_multiplier` | `float` | No | `1.5` | IQR multiplier for bounds |

```python
validator = OutlierValidator(iqr_multiplier=2.0)
issues = validator.validate(lf)
```

**Formula:** `Bounds = [Q1 - k×IQR, Q3 + k×IQR]`

### ZScoreOutlierValidator

Z-score outlier detection. Automatically applies to all numeric columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `threshold` | `float` | No | `3.0` | Z-score threshold for outlier detection |

```python
validator = ZScoreOutlierValidator(threshold=2.5)
issues = validator.validate(lf)
```

### QuantileValidator

Quantile bounds validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `quantile` | `float` | Yes | - | Quantile (0.0-1.0) |
| `max_value` | `float` | Yes | - | Max value at quantile |

### DistributionValidator

Distribution shape validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `distribution` | `str` | Yes | - | Expected distribution |

### KLDivergenceValidator

KL divergence threshold.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `reference_distribution` | `dict` | Yes | - | Reference distribution |
| `threshold` | `float` | Yes | - | Max KL divergence |

**Formula:** `D_KL(P||Q) = Σ P(x) × log(P(x) / Q(x))`

### ChiSquareValidator

Chi-square goodness of fit.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `expected_frequencies` | `dict` | Yes | - | Expected frequencies |
| `alpha` | `float` | No | `0.05` | Significance level |

### MostCommonValueValidator

Most common value validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `expected_value` | `Any` | Yes | - | Expected most common value |
| `min_frequency` | `float` | No | `0.0` | Min frequency ratio |

---

## 5. String Validators

Import: `from truthound.validators.string import *`

### RegexValidator

Regex pattern matching.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pattern` | `str` | **Yes** | - | Regex pattern |
| `columns` | `list[str] \| None` | No | `None` | Target columns |
| `match_full` | `bool` | No | `True` | Match entire string |
| `case_insensitive` | `bool` | No | `False` | Ignore case |
| `mostly` | `float \| None` | No | `None` | Acceptable match ratio |

> **Note:** `RegexValidator` requires a `pattern` parameter and is **not** in `BUILTIN_VALIDATORS`.

```python
validator = RegexValidator(pattern=r"^[A-Z]{3}-\d{4}$")
```

### RegexListValidator

Multiple patterns (any match passes).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `patterns` | `list[str]` | Yes | - | List of patterns |

### NotMatchRegexValidator

Must not match pattern.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `pattern` | `str` | Yes | - | Forbidden pattern |

### NotMatchRegexListValidator

Must not match any pattern.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `patterns` | `list[str]` | Yes | - | Forbidden patterns |

### LengthValidator

String length constraints.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_length` | `int \| None` | No | `None` | Min length |
| `max_length` | `int \| None` | No | `None` | Max length |

### EmailValidator

RFC 5322 email format.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### UrlValidator

URL format validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### PhoneValidator

Phone number format.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `country_code` | `str \| None` | No | `None` | Expected country code |

### UuidValidator

UUID format (v1-v5).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### IpAddressValidator

IPv4 address format.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `version` | `int \| None` | No | `None` | IP version (4 or 6) |

### Ipv6AddressValidator

IPv6 address format.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### FormatValidator

Auto-detect format by column name.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target columns |

**Auto-detected formats:**
- `email` - email, e-mail, mail
- `phone` - phone, tel, mobile, cell
- `url` - url, link, website
- `uuid` - uuid, guid
- `ip` - ip, ip_address
- `date` - date, dob, birth
- `code` - product_code, sku, barcode

### JsonParseableValidator

Valid JSON strings.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### JsonSchemaValidator

JSON Schema validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `schema` | `dict` | Yes | - | JSON Schema spec |

### AlphanumericValidator

Alphanumeric characters only.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### ConsistentCasingValidator

Consistent case style.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `casing` | `str` | Yes | - | Expected style (snake_case, camelCase, etc.) |

### LikePatternValidator

SQL LIKE pattern match.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `pattern` | `str` | Yes | - | LIKE pattern |

```python
validator = LikePatternValidator(column="code", pattern="PRD-%")
```

### NotLikePatternValidator

SQL LIKE pattern exclusion.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `pattern` | `str` | Yes | - | Forbidden LIKE pattern |

---

## 6. Datetime Validators

Import: `from truthound.validators.datetime import *`

### DateFormatValidator

Date format validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `format` | `str` | Yes | - | strptime format |

### DateBetweenValidator

Date within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_date` | `date \| datetime` | Yes | - | Min date |
| `max_date` | `date \| datetime` | Yes | - | Max date |

### FutureDateValidator

Date must be in future.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### PastDateValidator

Date must be in past.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

### DateOrderValidator

Start date < end date.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `start_column` | `str` | Yes | - | Start date column |
| `end_column` | `str` | Yes | - | End date column |

```python
validator = DateOrderValidator(start_column="start_date", end_column="end_date")
```

### TimezoneValidator

Timezone-aware validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `expected_timezone` | `str \| None` | No | `None` | Expected timezone |

### RecentDataValidator

Data within max age.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Datetime column |
| `max_age_days` | `int` | Yes | - | Max age in days |

### DatePartCoverageValidator

Coverage across date parts.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Datetime column |
| `date_part` | `str` | Yes | - | Part to check (day, month, hour) |
| `min_coverage` | `float` | Yes | - | Min coverage ratio |

### GroupedRecentDataValidator

Recency within groups.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `datetime_column` | `str` | Yes | - | Datetime column |
| `group_column` | `str` | Yes | - | Grouping column |
| `max_age_days` | `int` | Yes | - | Max age per group |

### DateutilParseableValidator

Parseable by dateutil.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |

---

## 7. Aggregate Validators

Import: `from truthound.validators.aggregate import *`

### MeanBetweenValidator

Column mean within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_value` | `float` | Yes | - | Min mean |
| `max_value` | `float` | Yes | - | Max mean |

### MedianBetweenValidator

Column median within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_value` | `float` | Yes | - | Min median |
| `max_value` | `float` | Yes | - | Max median |

### StdBetweenValidator

Standard deviation within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_value` | `float` | Yes | - | Min std |
| `max_value` | `float` | Yes | - | Max std |

### VarianceBetweenValidator

Variance within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_value` | `float` | Yes | - | Min variance |
| `max_value` | `float` | Yes | - | Max variance |

### MinBetweenValidator

Column minimum within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_value` | `float` | Yes | - | Min of min |
| `max_value` | `float` | Yes | - | Max of min |

### MaxBetweenValidator

Column maximum within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_value` | `float` | Yes | - | Min of max |
| `max_value` | `float` | Yes | - | Max of max |

### SumBetweenValidator

Column sum within range.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `min_value` | `float` | Yes | - | Min sum |
| `max_value` | `float` | Yes | - | Max sum |

### TypeValidator

Aggregate-level type validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str] \| None` | No | `None` | Target columns |

---

## 8. Cross-Table Validators

Import: `from truthound.validators.cross_table import *`

### CrossTableRowCountValidator

Compare row counts between tables.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `reference_data` | `pl.DataFrame` | Yes | - | Reference table |
| `operator` | `str` | Yes | - | Comparison operator |
| `tolerance` | `float` | No | `0.0` | Acceptable tolerance |

### CrossTableRowCountFactorValidator

Row count ratio validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `reference_data` | `pl.DataFrame` | Yes | - | Reference table |
| `expected_factor` | `float` | Yes | - | Expected ratio |
| `tolerance` | `float` | No | `0.0` | Acceptable tolerance |

### CrossTableAggregateValidator

Compare aggregates between tables.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `reference_data` | `pl.DataFrame` | Yes | - | Reference table |
| `aggregate` | `str` | Yes | - | Aggregate function |
| `operator` | `str` | Yes | - | Comparison operator |

### CrossTableDistinctCountValidator

Compare distinct counts between tables.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Target column |
| `reference_data` | `pl.DataFrame` | Yes | - | Reference table |

---

## 9. Query Validators

Import: `from truthound.validators.query import *`

### ExpressionValidator

Polars expression validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expression` | `pl.Expr` | Yes | - | Polars expression |
| `description` | `str` | No | `""` | Description |

```python
validator = ExpressionValidator(
    expression=pl.col("price") > 0,
    description="Price must be positive"
)
```

### QueryReturnsSingleValueValidator

Query returns exactly one value.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `pl.Expr` | Yes | - | Query expression |

### QueryReturnsNoRowsValidator

Query returns no rows.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `pl.Expr` | Yes | - | Filter expression |

### QueryReturnsRowsValidator

Query returns at least one row.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `pl.Expr` | Yes | - | Filter expression |

### QueryResultMatchesValidator

Query result matches expected.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `pl.Expr` | Yes | - | Query expression |
| `expected` | `Any` | Yes | - | Expected value |

### QueryRowCountValidator

Query result row count.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `filter_expr` | `pl.Expr` | Yes | - | Filter expression |
| `expected_count` | `int \| None` | No | `None` | Expected count |
| `min_count` | `int \| None` | No | `None` | Min count |
| `max_count` | `int \| None` | No | `None` | Max count |

### QueryRowCountRatioValidator

Ratio of matching rows.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `filter_expr` | `pl.Expr` | Yes | - | Filter expression |
| `min_ratio` | `float` | No | `0.0` | Min ratio |
| `max_ratio` | `float` | No | `1.0` | Max ratio |

### QueryRowCountCompareValidator

Compare row counts between queries.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query_a` | `pl.Expr` | Yes | - | First query |
| `query_b` | `pl.Expr` | Yes | - | Second query |
| `operator` | `str` | Yes | - | Comparison operator |

### QueryColumnValuesValidator

Validate column values in query results.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `pl.Expr` | Yes | - | Query expression |
| `column` | `str` | Yes | - | Column to validate |
| `expected_values` | `set` | Yes | - | Expected values |

### QueryColumnUniqueValidator

Query column uniqueness.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `pl.Expr` | Yes | - | Query expression |
| `column` | `str` | Yes | - | Column to check |

### QueryColumnNotNullValidator

Query column non-null.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `pl.Expr` | Yes | - | Query expression |
| `column` | `str` | Yes | - | Column to check |

### QueryAggregateValidator

Query aggregate validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Column |
| `aggregate` | `str` | Yes | - | Aggregate function |
| `expected_value` | `float` | Yes | - | Expected value |
| `tolerance` | `float` | No | `0.0` | Tolerance |

### QueryGroupAggregateValidator

Group aggregate validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `group_by` | `list[str]` | Yes | - | Group columns |
| `column` | `str` | Yes | - | Aggregate column |
| `aggregate` | `str` | Yes | - | Aggregate function |

### QueryAggregateCompareValidator

Compare aggregates between queries.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column` | `str` | Yes | - | Column |
| `aggregate` | `str` | Yes | - | Aggregate function |
| `reference_data` | `pl.DataFrame` | Yes | - | Reference data |

### CustomExpressionValidator

Custom expression strings.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expression_str` | `str` | Yes | - | Expression string |

### ConditionalExpressionValidator

Conditional expressions.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `condition` | `pl.Expr` | Yes | - | Condition |
| `then_expr` | `pl.Expr` | Yes | - | Expression if true |

### MultiConditionValidator

Multiple conditions.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `conditions` | `list[pl.Expr]` | Yes | - | List of conditions |
| `mode` | `str` | No | `"all"` | all/any |

### RowLevelValidator

Row-level validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `check_expr` | `pl.Expr` | Yes | - | Check expression |

---

## 10. Multi-Column Validators

Import: `from truthound.validators.multi_column import *`

### ColumnSumValidator

Columns sum to target.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns to sum |
| `target_column` | `str` | Yes | - | Expected sum column |
| `tolerance` | `float` | No | `0.0` | Tolerance |

```python
validator = ColumnSumValidator(
    columns=["subtotal", "tax", "shipping"],
    target_column="total",
    tolerance=0.01
)
```

### ColumnProductValidator

Column multiplication.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns to multiply |
| `target_column` | `str` | Yes | - | Expected product column |
| `tolerance` | `float` | No | `0.0` | Tolerance |

### ColumnDifferenceValidator

Column difference validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |
| `expected_difference` | `float` | Yes | - | Expected difference |

### ColumnRatioValidator

Column ratio validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `numerator_column` | `str` | Yes | - | Numerator |
| `denominator_column` | `str` | Yes | - | Denominator |
| `expected_ratio` | `float` | Yes | - | Expected ratio |
| `tolerance` | `float` | No | `0.0` | Tolerance |

### ColumnPercentageValidator

Percentage validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `part_column` | `str` | Yes | - | Part column |
| `whole_column` | `str` | Yes | - | Whole column |
| `min_percentage` | `float` | No | `0.0` | Min percentage |
| `max_percentage` | `float` | No | `100.0` | Max percentage |

### ColumnComparisonValidator

Column comparison.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |
| `operator` | `str` | Yes | - | Operator (>, <, >=, <=, ==, !=) |

```python
validator = ColumnComparisonValidator(
    column_a="end_date",
    column_b="start_date",
    operator=">"
)
```

### ColumnChainComparisonValidator

Ordered column chain.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns in order |
| `operator` | `str` | Yes | - | Comparison operator |

### ColumnMutualExclusivityValidator

At most one non-null per row.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Mutually exclusive columns |

### ColumnCoexistenceValidator

All null or all non-null.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Coexisting columns |

### ColumnDependencyValidator

Functional dependency.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `determinant_column` | `str` | Yes | - | Determinant |
| `dependent_column` | `str` | Yes | - | Dependent |

### ColumnImplicationValidator

If A then B.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `antecedent` | `pl.Expr` | Yes | - | Condition |
| `consequent` | `pl.Expr` | Yes | - | Result |

### ColumnCorrelationValidator

Correlation validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |
| `min_correlation` | `float` | No | `-1.0` | Min correlation |
| `max_correlation` | `float` | No | `1.0` | Max correlation |

### ColumnCovarianceValidator

Covariance validation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `column_a` | `str` | Yes | - | First column |
| `column_b` | `str` | Yes | - | Second column |
| `expected_covariance` | `float` | Yes | - | Expected covariance |
| `tolerance` | `float` | No | `0.0` | Tolerance |

### MultiColumnVarianceValidator

Variance across columns.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `columns` | `list[str]` | Yes | - | Columns to analyze |
| `max_variance` | `float` | Yes | - | Max variance |

---

## 11-21. Remaining Categories

For detailed parameter references for:

- **Table Validators** - See [Categories Reference](categories.md#table)
- **Geospatial Validators** - See [Categories Reference](categories.md#geospatial)
- **Drift Validators** - See [Categories Reference](categories.md#drift)
- **Anomaly Validators** - See [Categories Reference](categories.md#anomaly)
- **Referential Validators** - See [Categories Reference](categories.md#referential)
- **Time Series Validators** - See [Categories Reference](categories.md#timeseries)
- **Business Rule Validators** - See [Categories Reference](categories.md#business_rule)
- **Profiling Validators** - See [Categories Reference](categories.md#profiling)
- **Localization Validators** - See [Categories Reference](categories.md#localization)
- **ML Feature Validators** - See [Categories Reference](categories.md#ml_feature)
- **Privacy Validators** - See [Categories Reference](categories.md#privacy)

---

## Common Parameters

These parameters are shared across many validators:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `column` | `str` | - | Target column name (legacy, prefer `columns`) |
| `columns` | `list[str]` | `None` | Multiple columns (None = all) |
| `mostly` | `float` | `None` | Min pass ratio (0.0-1.0) |
| `max_anomaly_ratio` | `float` | `0.1` | Max anomaly ratio |
| `reference_data` | `pl.DataFrame` | - | Baseline dataset for drift |
| `tolerance` | `float` | `0.0` | Acceptable numeric tolerance |

> **Note**: Most validators support both `column` (single) and `columns` (multiple) parameters via `**kwargs`. The `columns` parameter is preferred for consistency and flexibility. When `columns=None`, validators typically apply to all applicable columns.

---

## Severity Levels

| Severity | Description | Typical Condition |
|----------|-------------|-------------------|
| `LOW` | Minor issue | < 1% affected |
| `MEDIUM` | Moderate concern | 1-5% affected |
| `HIGH` | Significant issue | 5-10% affected |
| `CRITICAL` | Severe problem | > 10% affected |

---

## Installation Options

```bash
# Core validators only
pip install truthound

# With drift detection (scipy)
pip install truthound[drift]

# With anomaly detection (scipy, scikit-learn)
pip install truthound[anomaly]

# All dependencies
pip install truthound[all]
```

---

## Next Steps

- [Categories Reference](categories.md) - Category descriptions and submodules
- [Custom Validators](custom-validators.md) - Build your own validators
- [Security Features](security.md) - ReDoS and SQL injection protection
