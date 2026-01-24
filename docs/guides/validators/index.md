# Validators Guide

This guide covers data validation with Truthound's Python API. It includes practical usage patterns, error handling, and the complete validator reference.

**Current Implementation: 264 validators across 28 categories (23 validator categories + 5 infrastructure categories).**

---

## Quick Start

```python
import truthound as th

# Basic validation with all built-in validators
report = th.check("data.csv")

# Specific validators only
report = th.check(df, validators=["null", "duplicate", "range"])

# With validator configuration
report = th.check(
    df,
    validators=["regex"],
    validator_config={"regex": {"pattern": r"^[A-Z]{3}-\d{4}$"}}
)
```

---

## Common Usage Patterns

### Pattern 1: Combining Multiple Validators

```python
import truthound as th
from truthound.validators import (
    NullValidator,
    RegexValidator,
    CompletenessRatioValidator,
    BetweenValidator,
)

# Create validator instances
validators = [
    NullValidator(columns=["email", "phone"]),
    RegexValidator(pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$", columns=["email"]),
    CompletenessRatioValidator(column="phone", min_ratio=0.95),
    BetweenValidator(min_value=0, max_value=150, columns=["age"]),
]

# Run validation
report = th.check(df, validators=validators)
```

### Pattern 2: Conditional Validation

```python
import polars as pl
from truthound.validators import ConditionalNullValidator, ExpressionValidator

# If status is "active", email must not be null
conditional_null = ConditionalNullValidator(
    column="email",
    condition=pl.col("status") == "active"
)

# Custom condition with expression
custom_check = ExpressionValidator(
    expression=(pl.col("end_date") > pl.col("start_date")),
    description="End date must be after start date"
)

report = th.check(df, validators=[conditional_null, custom_check])
```

### Pattern 3: Schema-Based Validation

```python
import truthound as th

# Learn schema from baseline data
schema = th.learn("baseline.csv")

# Validate new data against schema
report = th.check("new_data.csv", schema=schema)

# Filter schema-specific issues
schema_issues = [i for i in report.issues if i.validator == "schema"]
```

### Pattern 4: Processing Validation Results

```python
import truthound as th

report = th.check("data.csv")

# Filter by severity
critical_issues = [i for i in report.issues if i.severity == "critical"]
high_issues = report.filter_by_severity("high").issues

# Group by column
from collections import defaultdict
by_column = defaultdict(list)
for issue in report.issues:
    by_column[issue.column].append(issue)

# Group by validator
by_validator = defaultdict(list)
for issue in report.issues:
    by_validator[issue.validator].append(issue)

# Export to different formats
from truthound.reporters import JSONReporter, JUnitXMLReporter

json_output = JSONReporter().render(report)
junit_output = JUnitXMLReporter().render(report)
```

### Pattern 5: Cross-Table Validation

```python
import polars as pl
from truthound.validators import (
    ReferentialIntegrityValidator,
    CrossTableRowCountValidator,
)

# Load reference data
departments = pl.read_csv("departments.csv")

# Validate foreign key relationship
fk_validator = ReferentialIntegrityValidator(
    column="department_id",
    reference_data=departments,
    reference_column="id"
)

# Compare row counts
row_count_validator = CrossTableRowCountValidator(
    reference_data=departments,
    operator=">=",
    tolerance=0.0
)

report = th.check(employees_df, validators=[fk_validator, row_count_validator])
```

---

## Error Handling

### ValidationIssue Structure

```python
from truthound.validators.base import ValidationIssue

# Each issue contains:
issue.column         # str: Affected column name
issue.issue_type     # str: Issue type (e.g., "null_value", "pattern_mismatch")
issue.count          # int: Number of affected rows
issue.severity       # Severity: LOW, MEDIUM, HIGH, CRITICAL
issue.details        # str | None: Human-readable description
issue.expected       # Any | None: Expected value
issue.actual         # Any | None: Actual value found
issue.sample_values  # list | None: Sample of problematic values
```

### Handling Validation Errors

```python
import truthound as th
from truthound.validators.base import (
    ValidationTimeoutError,
    ColumnNotFoundError,
)
from truthound.datasources.base import DataSourceError

try:
    report = th.check("data.csv", validators=["null", "regex"])
except ColumnNotFoundError as e:
    print(f"Column not found: {e.column}")
except ValidationTimeoutError as e:
    print(f"Validation timed out after {e.timeout}s")
except DataSourceError as e:
    print(f"Could not read data source: {e}")
```

### Strict Mode for Missing Columns

```python
import truthound as th

# Default: warns and skips missing columns
report = th.check(df, validators=["null"])  # MaskingWarning if column missing

# Strict mode: raises error for missing columns
from truthound.validators import NullValidator
validator = NullValidator(columns=["nonexistent"], strict=True)
# Raises ColumnNotFoundError
```

---

## Document Structure

The documents in this directory cover all aspects of the Truthound validator system:

| Document | Description |
|----------|-------------|
| **[index.md](index.md)** (this document) | Overview, quick reference for all 264 validators |
| **[categories.md](categories.md)** | Detailed descriptions of 28 categories |
| **[built-in.md](built-in.md)** | Complete reference for 289 built-in validators |
| **[custom-validators.md](custom-validators.md)** | Developing custom validators with SDK |
| **[enterprise-sdk.md](enterprise-sdk.md)** | Enterprise SDK (sandbox, signing, licensing) |
| **[security.md](security.md)** | ReDoS protection, SQL injection prevention |
| **[i18n.md](i18n.md)** | Internationalization support for 15 languages |
| **[optimization.md](optimization.md)** | DAG execution, performance optimization, profiling |

---

## Table of Contents

1. [Schema Validators](#1-schema-validators)
2. [Completeness Validators](#2-completeness-validators)
3. [Uniqueness Validators](#3-uniqueness-validators)
4. [Distribution Validators](#4-distribution-validators)
5. [String Validators](#5-string-validators)
6. [Datetime Validators](#6-datetime-validators)
7. [Aggregate Validators](#7-aggregate-validators)
8. [Cross-Table Validators](#8-cross-table-validators)
9. [Query Validators](#9-query-validators)
10. [Multi-Column Validators](#10-multi-column-validators)
11. [Table Validators](#11-table-validators)
12. [Geospatial Validators](#12-geospatial-validators)
13. [Drift Validators](#13-drift-validators)
14. [Anomaly Validators](#14-anomaly-validators)
15. [DAG-Based Execution](#15-dag-based-execution)
16. [Validator Performance Profiling](#16-validator-performance-profiling)
17. [Custom Validator SDK](#17-custom-validator-sdk)
18. [ReDoS Protection](#18-redos-protection)
19. [Internationalization (i18n)](#19-internationalization-i18n)
20. [Distributed Timeout](#20-distributed-timeout)

---

## 1. Schema Validators

Schema validators verify the structural integrity of datasets, ensuring that column definitions, data types, and relationships conform to specified constraints.

### 1.1 ColumnExistsValidator

Validates the presence of specified columns within a dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Column names that must exist |

```python
validator = ColumnExistsValidator(columns=["id", "name", "email"])
```

### 1.2 ColumnNotExistsValidator

Ensures that specified columns are absent from the dataset, useful for preventing the inclusion of deprecated or sensitive fields.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Column names that must not exist |

### 1.3 ColumnCountValidator

Validates that the dataset contains an expected number of columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `exact_count` | `int \| None` | Exact expected column count |
| `min_count` | `int \| None` | Minimum column count |
| `max_count` | `int \| None` | Maximum column count |

```python
# Exact count
validator = ColumnCountValidator(exact_count=10)

# Range
validator = ColumnCountValidator(min_count=5, max_count=20)
```

### 1.4 RowCountValidator

Ensures the dataset contains a specified number of rows.

| Parameter | Type | Description |
|-----------|------|-------------|
| `exact_count` | `int \| None` | Exact expected row count |
| `min_count` | `int \| None` | Minimum row count |
| `max_count` | `int \| None` | Maximum row count |

```python
# Exact count
validator = RowCountValidator(exact_count=1000)

# Range
validator = RowCountValidator(min_count=100, max_count=10000)
```

### 1.5 ColumnTypeValidator

Validates that columns conform to expected data types.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column name |
| `expected_type` | `type[pl.DataType]` | Expected Polars data type |

```python
validator = ColumnTypeValidator(column="age", expected_type=pl.Int64)
```

### 1.6 ColumnOrderValidator

Ensures columns appear in a specified order within the dataset.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expected_order` | `list[str]` | Expected column order |
| `strict` | `bool` | If True, no additional columns allowed |

### 1.7 TableSchemaValidator

Validates the complete schema of a dataset against a reference specification.

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema` | `dict[str, type]` | Column name to type mapping |
| `strict` | `bool` | Reject extra columns if True |

### 1.8 ColumnPairValidator

Validates relationships between two columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_a` | `str` | First column |
| `column_b` | `str` | Second column |
| `relationship` | `str` | Expected relationship type |

### 1.9 MultiColumnUniqueValidator

Ensures uniqueness across a combination of columns (composite key validation).

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Columns forming composite key |

### 1.10 ReferentialIntegrityValidator

Validates foreign key relationships between tables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Foreign key column |
| `reference_data` | `pl.DataFrame` | Reference table |
| `reference_column` | `str` | Primary key column in reference |

```python
validator = ReferentialIntegrityValidator(
    column="department_id",
    reference_data=departments_df,
    reference_column="id"
)
```

### 1.11 MultiColumnSumValidator

Validates that the sum of specified columns equals an expected value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Columns to sum |
| `expected_sum` | `float` | Expected sum value |
| `tolerance` | `float` | Acceptable tolerance |

### 1.12 MultiColumnCalculationValidator

Validates arbitrary arithmetic relationships between columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expression` | `str` | Mathematical expression |
| `tolerance` | `float` | Acceptable tolerance |

### 1.13 ColumnPairInSetValidator

Validates that column value pairs exist within a predefined set.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_a` | `str` | First column |
| `column_b` | `str` | Second column |
| `valid_pairs` | `set[tuple]` | Set of valid (a, b) pairs |

### 1.14 ColumnPairNotInSetValidator

Ensures column value pairs do not exist within a forbidden set.

---

## 2. Completeness Validators

Completeness validators assess the presence and validity of data values, detecting null values, empty strings, and other indicators of missing information.

### 2.1 NullValidator

Detects and reports null values within specified columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Target columns (None = all) |

### 2.2 NotNullValidator

Ensures specified columns contain no null values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |

### 2.3 CompletenessRatioValidator

Validates that the completeness ratio meets a minimum threshold.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_ratio` | `float` | Minimum completeness ratio (0.0-1.0) |

```python
validator = CompletenessRatioValidator(column="phone", min_ratio=0.95)
```

### 2.4 EmptyStringValidator

Detects empty strings in string columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Target string columns |

### 2.5 WhitespaceOnlyValidator

Identifies values containing only whitespace characters.

### 2.6 ConditionalNullValidator

Validates null values based on conditional logic.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `condition` | `pl.Expr` | Polars expression defining condition |

```python
# If status is "active", email must not be null
validator = ConditionalNullValidator(
    column="email",
    condition=pl.col("status") == "active"
)
```

### 2.7 DefaultValueValidator

Detects values matching default or placeholder patterns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `default_values` | `list` | Values considered as defaults |

### 2.8 NaNValidator

Detects NaN (Not a Number) values in floating-point columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Target float columns (None = all float columns) |
| `mostly` | `float \| None` | Acceptable non-NaN ratio |

```python
validator = NaNValidator(columns=["temperature", "humidity"])
```

### 2.9 NotNaNValidator

Ensures specified columns contain no NaN values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |

### 2.10 NaNRatioValidator

Validates that the NaN ratio is within acceptable bounds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `max_ratio` | `float` | Maximum allowed NaN ratio (0.0-1.0) |

```python
validator = NaNRatioValidator(column="sensor_reading", max_ratio=0.01)
```

### 2.11 InfinityValidator

Detects infinite values (positive or negative infinity) in numeric columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Target numeric columns |

```python
validator = InfinityValidator(columns=["calculation_result"])
```

### 2.12 FiniteValidator

Ensures all values are finite (not NaN and not infinite).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |

```python
validator = FiniteValidator(column="price")
```

---

## 3. Uniqueness Validators

Uniqueness validators verify the distinctness of values, detecting duplicates and validating primary key constraints.

### 3.1 UniqueValidator

Ensures all values in a column are unique.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Target columns (None = all) |

### 3.2 UniqueRatioValidator

Validates that the unique value ratio meets a threshold.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_ratio` | `float` | Minimum unique ratio |
| `max_ratio` | `float` | Maximum unique ratio |

### 3.3 DistinctCountValidator

Validates the number of distinct values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_count` | `int \| None` | Minimum distinct count |
| `max_count` | `int \| None` | Maximum distinct count |

### 3.4 DuplicateValidator

Detects duplicate values within specified columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to check |

### 3.5 DuplicateWithinGroupValidator

Detects duplicates within specified groups.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Column to check for duplicates |
| `group_by` | `list[str]` | Grouping columns |

### 3.6 PrimaryKeyValidator

Validates primary key constraints (unique and non-null).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Primary key column |

### 3.7 CompoundKeyValidator

Validates composite primary key constraints.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Columns forming compound key |

### 3.8 DistinctValuesInSetValidator

Ensures all distinct values belong to a specified set.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `value_set` | `set` | Allowed values |

### 3.9 DistinctValuesEqualSetValidator

Validates that distinct values exactly match a specified set.

### 3.10 DistinctValuesContainSetValidator

Ensures distinct values contain all elements of a specified set.

### 3.11 DistinctCountBetweenValidator

Validates that distinct count falls within a range.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_count` | `int` | Minimum distinct count |
| `max_count` | `int` | Maximum distinct count |

### 3.12 UniqueWithinRecordValidator

Ensures specified columns have unique values within each record.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Columns to compare |

### 3.13 AllColumnsUniqueWithinRecordValidator

Validates that all values within each record are unique.

---

## 4. Distribution Validators

Distribution validators analyze the statistical properties of data, verifying that values fall within expected ranges and conform to anticipated distributions.

### 4.1 BetweenValidator

Validates that values fall within a specified range.

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_value` | `float \| None` | Minimum value |
| `max_value` | `float \| None` | Maximum value |
| `inclusive` | `bool` | Include boundaries (default: True) |

### 4.2 RangeValidator

Auto-detects expected ranges based on column names (e.g., "age", "percentage", "rating").

| Parameter | Type | Description |
|-----------|------|-------------|
| (no parameters) | - | Uses built-in KNOWN_RANGES mapping |

**Auto-detected ranges:**
- `age`: [0, 150]
- `percentage/percent/pct/rate/score`: [0, 100]
- `rating`: [0, 5]
- `year`: [1900, 2100]
- `month`: [1, 12]
- `day`: [1, 31]
- `hour`: [0, 23]
- `minute/second`: [0, 59]
- `price/quantity/count/amount`: [0, None]

### 4.3 PositiveValidator

Ensures all values are strictly positive (> 0).

### 4.4 NonNegativeValidator

Ensures all values are non-negative (>= 0).

### 4.5 InSetValidator

Validates that values belong to a predefined set.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `value_set` | `set` | Allowed values |
| `mostly` | `float \| None` | Acceptable pass ratio |

### 4.6 NotInSetValidator

Ensures values do not belong to a forbidden set.

### 4.7 IncreasingValidator

Validates that values are monotonically increasing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `strict` | `bool` | Strictly increasing if True |

### 4.8 DecreasingValidator

Validates that values are monotonically decreasing.

### 4.9 OutlierValidator

Detects statistical outliers using the IQR method.

| Parameter | Type | Description |
|-----------|------|-------------|
| `iqr_multiplier` | `float` | IQR multiplier (default: 1.5) |

**Mathematical Definition**:
```
IQR = Q3 - Q1
Lower Bound = Q1 - k × IQR
Upper Bound = Q3 + k × IQR
```

### 4.10 ZScoreOutlierValidator

Detects outliers using Z-score methodology.

| Parameter | Type | Description |
|-----------|------|-------------|
| `threshold` | `float` | Z-score threshold (default: 3.0) |

### 4.11 QuantileValidator

Validates that values fall within specified quantile bounds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `quantile` | `float` | Quantile value (0.0-1.0) |
| `max_value` | `float` | Maximum allowed value at quantile |

### 4.12 DistributionValidator

Validates that data follows an expected distribution pattern.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `distribution` | `str` | Expected distribution type |

### 4.13 KLDivergenceValidator

Validates distribution similarity using Kullback-Leibler divergence.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_distribution` | `dict` | Reference probability distribution |
| `threshold` | `float` | Maximum allowed KL divergence |

**Mathematical Definition**:
```
D_KL(P||Q) = Σ P(x) × log(P(x) / Q(x))
```

### 4.14 ChiSquareValidator

Performs chi-square goodness-of-fit testing.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `expected_frequencies` | `dict` | Expected frequency distribution |
| `alpha` | `float` | Significance level (default: 0.05) |

### 4.15 MostCommonValueValidator

Validates the most common value and its frequency.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `expected_value` | `Any` | Expected most common value |
| `min_frequency` | `float` | Minimum frequency ratio |

---

## 5. String Validators

String validators verify the format, structure, and content of text data using pattern matching and format validation.

### 5.1 RegexValidator

Validates strings against a regular expression pattern.

> **Note:** `RegexValidator` requires a mandatory `pattern` parameter and is **not included** in `BUILTIN_VALIDATORS`. It must be used explicitly with the pattern specified.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `str` | Regular expression pattern (**required**) |
| `columns` | `list[str] \| None` | Target columns (None = all string columns) |
| `match_full` | `bool` | Match entire string (default: True) |
| `case_insensitive` | `bool` | Ignore case (default: False) |
| `mostly` | `float \| None` | Acceptable match ratio |

**Usage:**

```python
import truthound as th
from truthound.validators import RegexValidator

# Option 1: Pass validator instance directly
validator = RegexValidator(pattern=r"^[A-Z]{3}-\d{4}$")
report = th.check("data.csv", validators=[validator])

# Option 2: Apply to specific columns
validator = RegexValidator(
    pattern=r"^[A-Z]{3}-\d{4}$",
    columns=["product_code", "item_id"]
)
report = th.check("data.csv", validators=[validator])

# Option 3: Use by name with validator_config
report = th.check(
    "data.csv",
    validators=["regex"],
    validator_config={"regex": {"pattern": r"^[A-Z]{3}-\d{4}$"}}
)
```

### 5.2 RegexListValidator

Validates against multiple regex patterns (any match passes).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `patterns` | `list[str]` | List of patterns |

### 5.3 NotMatchRegexValidator

Ensures values do not match a specified pattern.

### 5.4 NotMatchRegexListValidator

Ensures values do not match any pattern in a list.

### 5.5 LengthValidator

Validates string length constraints.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_length` | `int \| None` | Minimum length |
| `max_length` | `int \| None` | Maximum length |

### 5.6 EmailValidator

Validates email address format using RFC 5322 patterns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |

### 5.7 UrlValidator

Validates URL format.

### 5.8 PhoneValidator

Validates phone number format with international support.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `country_code` | `str \| None` | Expected country code |

### 5.9 UuidValidator

Validates UUID format (versions 1-5).

### 5.10 IpAddressValidator

Validates IPv4 and IPv6 address formats.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `version` | `int \| None` | IP version (4 or 6) |

### 5.11 FormatValidator

Auto-detects and validates common format types based on column names. This validator is included in the default `th.check()` validation and automatically detects patterns for columns with recognizable names.

**Auto-detected formats based on column name patterns:**

| Format | Column Name Patterns |
|--------|---------------------|
| `email` | email, e-mail, mail |
| `phone` | phone, tel, mobile, cell, fax |
| `url` | url, link, website, href |
| `uuid` | uuid, guid |
| `ip` | ip, ip_address, ipaddress |
| `date` | date, dob, birth |
| `code` | product_code, item_code, sku, part_number, model_number, serial_number, barcode, upc, ean |

**Usage:**

```python
import truthound as th

# Auto-detection: columns named "email", "product_code", etc. are validated
report = th.check("data.csv")  # Uses FormatValidator by default

# Explicit format validation
from truthound.validators import FormatValidator
validator = FormatValidator()
issues = validator.validate(lf)
```

**Note:** For custom pattern validation on columns with non-standard names, use `RegexValidator` explicitly or define patterns in a schema:

```python
# Option 1: Use RegexValidator explicitly
from truthound.validators import RegexValidator
validator = RegexValidator(pattern=r"^[A-Z]{3}-\d{4}$")
report = th.check("data.csv", validators=[validator])

# Option 2: Use schema with pattern constraints
schema = th.learn("data.csv")
schema["my_column"].pattern = r"^[A-Z]{3}-\d{4}$"
report = th.check("data.csv", schema=schema)
```

### 5.12 JsonParseableValidator

Ensures string values are valid JSON.

### 5.13 JsonSchemaValidator

Validates JSON strings against a JSON Schema.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `schema` | `dict` | JSON Schema specification |

### 5.14 AlphanumericValidator

Ensures values contain only alphanumeric characters.

### 5.15 ConsistentCasingValidator

Validates consistent casing patterns (snake_case, camelCase, etc.).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `casing` | `str` | Expected casing style |

### 5.16 LikePatternValidator

SQL LIKE pattern matching with `%` and `_` wildcards.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `pattern` | `str` | LIKE pattern |

```python
validator = LikePatternValidator(column="code", pattern="PRD-%")
```

### 5.17 NotLikePatternValidator

Ensures values do not match a LIKE pattern.

---

## 6. Datetime Validators

Datetime validators verify temporal data integrity, ensuring dates and times conform to expected formats and ranges.

### 6.1 DateFormatValidator

Validates date/datetime format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `format` | `str` | Expected strptime format |

### 6.2 DateBetweenValidator

Validates dates within a specified range.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_date` | `date \| datetime` | Minimum date |
| `max_date` | `date \| datetime` | Maximum date |

### 6.3 FutureDateValidator

Ensures dates are in the future.

### 6.4 PastDateValidator

Ensures dates are in the past.

### 6.5 DateOrderValidator

Validates chronological ordering between date columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_column` | `str` | Start date column |
| `end_column` | `str` | End date column |

```python
validator = DateOrderValidator(
    start_column="start_date",
    end_column="end_date"
)
```

### 6.6 TimezoneValidator

Validates timezone-aware datetime values.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `expected_timezone` | `str \| None` | Expected timezone |

### 6.7 RecentDataValidator

Ensures data contains recent entries.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Datetime column |
| `max_age_days` | `int` | Maximum age in days |

### 6.8 DatePartCoverageValidator

Validates coverage across date parts (days, months, hours).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Datetime column |
| `date_part` | `str` | Date part to check |
| `min_coverage` | `float` | Minimum coverage ratio |

### 6.9 GroupedRecentDataValidator

Validates recency within groups.

| Parameter | Type | Description |
|-----------|------|-------------|
| `datetime_column` | `str` | Datetime column |
| `group_column` | `str` | Grouping column |
| `max_age_days` | `int` | Maximum age per group |

### 6.10 DateutilParseableValidator

Validates that strings can be parsed as dates using dateutil.

---

## 7. Aggregate Validators

Aggregate validators verify statistical properties computed across entire columns.

### 7.1 MeanBetweenValidator

Validates column mean within a range.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_value` | `float` | Minimum mean |
| `max_value` | `float` | Maximum mean |

### 7.2 MedianBetweenValidator

Validates column median within a range.

### 7.3 StdBetweenValidator

Validates column standard deviation within a range.

### 7.4 VarianceBetweenValidator

Validates column variance within a range.

### 7.5 MinBetweenValidator

Validates column minimum within a range.

### 7.6 MaxBetweenValidator

Validates column maximum within a range.

### 7.7 SumBetweenValidator

Validates column sum within a range.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `min_value` | `float` | Minimum sum |
| `max_value` | `float` | Maximum sum |

### 7.8 TypeValidator

Validates column data types at an aggregate level.

---

## 8. Cross-Table Validators

Cross-table validators verify relationships and consistency between multiple datasets.

### 8.1 CrossTableRowCountValidator

Compares row counts between tables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_data` | `pl.DataFrame` | Reference table |
| `operator` | `str` | Comparison operator |
| `tolerance` | `float` | Acceptable tolerance |

### 8.2 CrossTableRowCountFactorValidator

Validates row count ratio between tables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_data` | `pl.DataFrame` | Reference table |
| `expected_factor` | `float` | Expected ratio |
| `tolerance` | `float` | Acceptable tolerance |

### 8.3 CrossTableAggregateValidator

Compares aggregate statistics between tables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference table |
| `aggregate` | `str` | Aggregate function |
| `operator` | `str` | Comparison operator |

### 8.4 CrossTableDistinctCountValidator

Compares distinct value counts between tables.

---

## 9. Query Validators

Query validators enable flexible validation through SQL-like expressions and custom queries.

### 9.1 QueryValidator

Base class for query-based validation.

### 9.2 ExpressionValidator

Validates using Polars expressions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expression` | `pl.Expr` | Polars expression |
| `description` | `str` | Validation description |

```python
validator = ExpressionValidator(
    expression=pl.col("price") > 0,
    description="Price must be positive"
)
```

### 9.3 QueryReturnsSingleValueValidator

Ensures a query returns exactly one value.

### 9.4 QueryReturnsNoRowsValidator

Ensures a query returns no rows (useful for finding violations).

### 9.5 QueryReturnsRowsValidator

Ensures a query returns at least one row.

### 9.6 QueryResultMatchesValidator

Validates that query results match expected values.

### 9.7 QueryRowCountValidator

Validates query result row count.

### 9.8 QueryRowCountRatioValidator

Validates ratio of rows matching a condition.

### 9.9 QueryRowCountCompareValidator

Compares row counts between queries.

### 9.10 QueryColumnValuesValidator

Validates column values in query results.

### 9.11 QueryColumnUniqueValidator

Ensures query result column values are unique.

### 9.12 QueryColumnNotNullValidator

Ensures query result column has no nulls.

### 9.13 QueryAggregateValidator

Validates aggregate values from queries.

### 9.14 QueryGroupAggregateValidator

Validates aggregates within groups.

### 9.15 QueryAggregateCompareValidator

Compares aggregates between queries.

### 9.16 CustomExpressionValidator

Validates using custom Polars expression strings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expression_str` | `str` | Expression as string |

### 9.17 ConditionalExpressionValidator

Validates expressions with conditional logic.

### 9.18 MultiConditionValidator

Validates multiple conditions simultaneously.

### 9.19 RowLevelValidator

Validates at the individual row level.

---

## 10. Multi-Column Validators

Multi-column validators verify relationships and constraints across multiple columns within the same dataset.

### 10.1 MultiColumnValidator

Base class for multi-column validation.

### 10.2 ColumnArithmeticValidator

Validates arithmetic relationships between columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Input columns |
| `operation` | `str` | Arithmetic operation |
| `result_column` | `str` | Result column |

### 10.3 ColumnSumValidator

Validates that columns sum to a target column.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Columns to sum |
| `target_column` | `str` | Expected sum column |
| `tolerance` | `float` | Acceptable tolerance |

```python
validator = ColumnSumValidator(
    columns=["subtotal", "tax", "shipping"],
    target_column="total",
    tolerance=0.01
)
```

### 10.4 ColumnProductValidator

Validates column multiplication relationships.

### 10.5 ColumnDifferenceValidator

Validates difference between columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_a` | `str` | First column |
| `column_b` | `str` | Second column |
| `expected_difference` | `float` | Expected difference |

### 10.6 ColumnRatioValidator

Validates ratio between columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `numerator_column` | `str` | Numerator column |
| `denominator_column` | `str` | Denominator column |
| `expected_ratio` | `float` | Expected ratio |
| `tolerance` | `float` | Acceptable tolerance |

### 10.7 ColumnPercentageValidator

Validates percentage relationships.

### 10.8 ColumnComparisonValidator

Validates comparison relationships (>, <, >=, <=, ==, !=).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_a` | `str` | First column |
| `column_b` | `str` | Second column |
| `operator` | `str` | Comparison operator |

```python
validator = ColumnComparisonValidator(
    column_a="end_date",
    column_b="start_date",
    operator=">"
)
```

### 10.9 ColumnChainComparisonValidator

Validates ordered relationships across multiple columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Columns in order |
| `operator` | `str` | Comparison operator |

### 10.10 ColumnMaxValidator

Identifies maximum value across columns.

### 10.11 ColumnMinValidator

Identifies minimum value across columns.

### 10.12 ColumnMeanValidator

Validates mean across columns.

### 10.13 ColumnConsistencyValidator

Validates consistency patterns between columns.

### 10.14 ColumnMutualExclusivityValidator

Ensures columns are mutually exclusive (at most one non-null per row).

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Mutually exclusive columns |

### 10.15 ColumnCoexistenceValidator

Ensures columns coexist (all null or all non-null).

### 10.16 ColumnDependencyValidator

Validates functional dependencies between columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `determinant_column` | `str` | Determinant column |
| `dependent_column` | `str` | Dependent column |

### 10.17 ColumnImplicationValidator

Validates conditional implications (if A then B).

| Parameter | Type | Description |
|-----------|------|-------------|
| `antecedent` | `pl.Expr` | Condition expression |
| `consequent` | `pl.Expr` | Result expression |

### 10.18 ColumnCorrelationValidator

Validates correlation between numeric columns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_a` | `str` | First column |
| `column_b` | `str` | Second column |
| `min_correlation` | `float` | Minimum correlation |
| `max_correlation` | `float` | Maximum correlation |

### 10.19 ColumnCovarianceValidator

Validates covariance between columns.

### 10.20 MultiColumnVarianceValidator

Validates variance across multiple columns.

---

## 11. Table Validators

Table validators verify metadata and structural properties of entire datasets.

### 11.1 TableValidator

Base class for table-level validation.

### 11.2 TableRowCountRangeValidator

Validates row count within a range.

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_rows` | `int \| None` | Minimum row count |
| `max_rows` | `int \| None` | Maximum row count |

### 11.3 TableRowCountExactValidator

Validates exact row count.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expected_rows` | `int` | Expected row count |
| `tolerance` | `int` | Acceptable tolerance |

### 11.4 TableRowCountCompareValidator

Compares row count with reference table.

### 11.5 TableNotEmptyValidator

Ensures table is not empty.

### 11.6 TableColumnCountValidator

Validates column count.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expected_count` | `int \| None` | Expected column count |
| `min_count` | `int \| None` | Minimum column count |
| `max_count` | `int \| None` | Maximum column count |

### 11.7 TableRequiredColumnsValidator

Ensures required columns are present.

| Parameter | Type | Description |
|-----------|------|-------------|
| `required_columns` | `list[str]` | Required column names |

### 11.8 TableForbiddenColumnsValidator

Ensures forbidden columns are absent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `forbidden_columns` | `list[str]` | Forbidden column names |

### 11.9 TableFreshnessValidator

Validates data freshness.

| Parameter | Type | Description |
|-----------|------|-------------|
| `datetime_column` | `str` | Timestamp column |
| `max_age_hours` | `int` | Maximum age in hours |

### 11.10 TableDataRecencyValidator

Validates recent data presence.

### 11.11 TableUpdateFrequencyValidator

Validates expected update frequency.

### 11.12 TableSchemaMatchValidator

Validates schema matches specification.

| Parameter | Type | Description |
|-----------|------|-------------|
| `expected_schema` | `dict` | Expected schema |
| `strict` | `bool` | Reject extra columns |

### 11.13 TableSchemaCompareValidator

Compares schema with reference table.

### 11.14 TableColumnTypesValidator

Validates column types match expectations.

### 11.15 TableMemorySizeValidator

Validates estimated memory usage.

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_size_mb` | `float \| None` | Minimum size in MB |
| `max_size_mb` | `float \| None` | Maximum size in MB |

### 11.16 TableRowToColumnRatioValidator

Validates row-to-column ratio.

### 11.17 TableDimensionsValidator

Validates table dimensions (rows and columns).

---

## 12. Geospatial Validators

Geospatial validators verify geographic coordinates and spatial data integrity.

### 12.1 GeoValidator

Base class for geospatial validation with Haversine distance calculation.

**Haversine Formula**:
```
a = sin²(Δlat/2) + cos(lat₁) × cos(lat₂) × sin²(Δlon/2)
c = 2 × atan2(√a, √(1-a))
d = R × c
```

### 12.2 LatitudeValidator

Validates latitude values (-90 to 90).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Latitude column |
| `min_lat` | `float` | Minimum latitude |
| `max_lat` | `float` | Maximum latitude |

### 12.3 LongitudeValidator

Validates longitude values (-180 to 180).

### 12.4 CoordinateValidator

Validates latitude/longitude coordinate pairs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat_column` | `str` | Latitude column |
| `lon_column` | `str` | Longitude column |

### 12.5 CoordinateNotNullIslandValidator

Detects "null island" coordinates (0, 0).

### 12.6 GeoDistanceValidator

Validates distances between coordinate pairs.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat_column_a` | `str` | First latitude column |
| `lon_column_a` | `str` | First longitude column |
| `lat_column_b` | `str` | Second latitude column |
| `lon_column_b` | `str` | Second longitude column |
| `max_distance_km` | `float` | Maximum distance |

### 12.7 GeoDistanceFromPointValidator

Validates distance from a reference point.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat_column` | `str` | Latitude column |
| `lon_column` | `str` | Longitude column |
| `reference_lat` | `float` | Reference latitude |
| `reference_lon` | `float` | Reference longitude |
| `max_distance_km` | `float` | Maximum distance |

### 12.8 GeoBoundingBoxValidator

Validates coordinates within a bounding box.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat_column` | `str` | Latitude column |
| `lon_column` | `str` | Longitude column |
| `min_lat` | `float` | Minimum latitude |
| `max_lat` | `float` | Maximum latitude |
| `min_lon` | `float` | Minimum longitude |
| `max_lon` | `float` | Maximum longitude |

### 12.9 GeoCountryValidator

Validates coordinates are within a country's boundaries.

| Parameter | Type | Description |
|-----------|------|-------------|
| `lat_column` | `str` | Latitude column |
| `lon_column` | `str` | Longitude column |
| `country_code` | `str` | ISO country code |

---

## 13. Drift Validators

Drift validators detect distributional changes between reference and current datasets, essential for monitoring machine learning model inputs and data pipeline integrity.

**Installation**: `pip install truthound[drift]`

### 13.1 DriftValidator

Base class for drift detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `reference_data` | `pl.DataFrame` | Baseline dataset |

### 13.2 ColumnDriftValidator

Base class for single-column drift detection.

### 13.3 KSTestValidator

Kolmogorov-Smirnov test for numeric distribution drift.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `p_value_threshold` | `float` | Significance level (default: 0.05) |
| `statistic_threshold` | `float \| None` | Optional KS statistic threshold |

**Mathematical Definition**:
```
D = max|F_ref(x) - F_curr(x)|
```

The test rejects the null hypothesis (no drift) when D exceeds the critical value.

```python
validator = KSTestValidator(
    column="feature_value",
    reference_data=training_df,
    p_value_threshold=0.05
)
```

### 13.4 ChiSquareDriftValidator

Chi-square test for categorical distribution drift.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `p_value_threshold` | `float` | Significance level |
| `min_expected_frequency` | `float` | Minimum expected frequency per category |

**Mathematical Definition**:
```
χ² = Σ (O_i - E_i)² / E_i
```

### 13.5 WassersteinDriftValidator

Wasserstein distance (Earth Mover's Distance) for numeric drift detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `threshold` | `float` | Maximum allowed distance |
| `normalize` | `bool` | Normalize by standard deviation |

The Wasserstein distance measures the minimum "work" required to transform one distribution into another, providing an interpretable metric in the same units as the data.

### 13.6 PSIValidator

Population Stability Index for distribution drift monitoring.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `threshold` | `float` | PSI threshold (default: 0.25) |
| `n_bins` | `int` | Number of bins (default: 10) |
| `is_categorical` | `bool` | Categorical column flag |

**Mathematical Definition**:
```
PSI = Σ (P_curr,i - P_ref,i) × ln(P_curr,i / P_ref,i)
```

**Industry Standard Thresholds**:
- PSI < 0.10: No significant change
- 0.10 ≤ PSI < 0.25: Moderate change (investigation recommended)
- PSI ≥ 0.25: Significant change (action required)

```python
validator = PSIValidator(
    column="credit_score",
    reference_data=baseline_df,
    threshold=0.25,
    n_bins=10
)
```

### 13.7 CSIValidator

Characteristic Stability Index for per-bin PSI analysis.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `threshold_per_bin` | `float` | Per-bin PSI threshold |
| `n_bins` | `int` | Number of bins |

### 13.8 MeanDriftValidator

Detects drift in column mean.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `threshold_pct` | `float \| None` | Percentage threshold |
| `threshold_abs` | `float \| None` | Absolute threshold |

### 13.9 VarianceDriftValidator

Detects drift in column variance or standard deviation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `threshold_pct` | `float` | Percentage threshold |
| `use_std` | `bool` | Use standard deviation instead of variance |

### 13.10 QuantileDriftValidator

Detects drift in specific quantiles.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `quantiles` | `list[float]` | Quantiles to monitor |
| `threshold_pct` | `float` | Percentage threshold |

### 13.11 RangeDriftValidator

Detects drift in value range (min/max).

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `threshold_pct` | `float` | Percentage threshold |
| `allow_expansion` | `bool` | Allow range expansion |

### 13.12 FeatureDriftValidator

Multi-feature drift detection with configurable methods.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str]` | Columns to monitor |
| `reference_data` | `pl.DataFrame` | Reference data |
| `method` | `str` | Detection method (psi, ks, wasserstein, chi_square) |
| `threshold` | `float` | Drift threshold |
| `alert_on_any` | `bool` | Alert if any column drifts |
| `min_drift_count` | `int` | Minimum columns that must drift |
| `categorical_columns` | `list[str]` | Categorical column list |

```python
validator = FeatureDriftValidator(
    columns=["age", "income", "credit_score"],
    reference_data=training_df,
    method="psi",
    threshold=0.15,
    alert_on_any=True
)
```

### 13.13 JSDivergenceValidator

Jensen-Shannon divergence for distribution similarity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `reference_data` | `pl.DataFrame` | Reference data |
| `threshold` | `float` | Maximum JS divergence |
| `is_categorical` | `bool` | Categorical column flag |

**Mathematical Definition**:
```
JS(P||Q) = ½ D_KL(P||M) + ½ D_KL(Q||M)
where M = ½(P + Q)
```

Jensen-Shannon divergence is symmetric and bounded [0, 1], making it suitable for comparing distributions of different sizes.

---

## 14. Anomaly Validators

Anomaly validators detect unusual patterns, outliers, and data points that deviate significantly from expected behavior. These validators employ both statistical and machine learning approaches.

**Installation**: `pip install truthound[anomaly]`

### 14.1 AnomalyValidator

Base class for table-wide anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `max_anomaly_ratio` | `float` | Maximum acceptable anomaly ratio |

### 14.2 ColumnAnomalyValidator

Base class for single-column anomaly detection.

### 14.3 IQRAnomalyValidator

Interquartile Range anomaly detection with configurable sensitivity.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `iqr_multiplier` | `float` | IQR multiplier (1.5=standard, 3.0=extreme) |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |
| `detect_lower` | `bool` | Detect lower bound anomalies |
| `detect_upper` | `bool` | Detect upper bound anomalies |

**Mathematical Definition**:
```
IQR = Q3 - Q1
Lower Bound = Q1 - k × IQR
Upper Bound = Q3 + k × IQR
```

- k = 1.5: Standard outliers (Tukey's fences)
- k = 3.0: Extreme outliers

```python
validator = IQRAnomalyValidator(
    column="transaction_amount",
    iqr_multiplier=1.5,
    max_anomaly_ratio=0.05
)
```

### 14.4 MADAnomalyValidator

Median Absolute Deviation anomaly detection, robust to outliers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `threshold` | `float` | Modified Z-score threshold (default: 3.5) |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

**Mathematical Definition**:
```
MAD = median(|X_i - median(X)|)
Modified Z-score = 0.6745 × (X - median(X)) / MAD
```

The constant 0.6745 makes MAD consistent with standard deviation for normal distributions. MAD is preferred over standard deviation when data contains outliers.

### 14.5 GrubbsTestValidator

Grubbs' test for iterative single outlier detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `alpha` | `float` | Significance level (default: 0.05) |
| `max_iterations` | `int` | Maximum iterations |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

**Mathematical Definition**:
```
G = max|X_i - mean(X)| / std(X)
```

Grubbs' test assumes the data follows an approximately normal distribution and is designed to detect a single outlier at a time. The validator runs iteratively, removing detected outliers and retesting until no more outliers are found or the maximum iterations are reached.

### 14.6 TukeyFencesValidator

Tukey's method with inner and outer fences for classifying outliers.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `detect_mild` | `bool` | Report mild outliers (k=1.5) |
| `detect_extreme` | `bool` | Report extreme outliers (k=3.0) |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

**Fence Definitions**:
- Inner fences: [Q1 - 1.5×IQR, Q3 + 1.5×IQR] — mild outliers outside
- Outer fences: [Q1 - 3.0×IQR, Q3 + 3.0×IQR] — extreme outliers outside

### 14.7 PercentileAnomalyValidator

Percentile-based anomaly detection with configurable bounds.

| Parameter | Type | Description |
|-----------|------|-------------|
| `column` | `str` | Target column |
| `lower_percentile` | `float` | Lower percentile (default: 1.0) |
| `upper_percentile` | `float` | Upper percentile (default: 99.0) |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

```python
validator = PercentileAnomalyValidator(
    column="response_time",
    lower_percentile=1,
    upper_percentile=99,
    max_anomaly_ratio=0.05
)
```

### 14.8 MahalanobisValidator

Multivariate anomaly detection using Mahalanobis distance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `threshold_percentile` | `float` | Chi-squared percentile threshold |
| `use_robust_covariance` | `bool` | Use robust covariance estimation |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

**Mathematical Definition**:
```
D²_M(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

The Mahalanobis distance accounts for correlations between variables and is scale-invariant. For multivariate normal data, squared Mahalanobis distances follow a chi-squared distribution with degrees of freedom equal to the number of dimensions.

```python
validator = MahalanobisValidator(
    columns=["feature1", "feature2", "feature3"],
    threshold_percentile=97.5,
    use_robust_covariance=True
)
```

### 14.9 EllipticEnvelopeValidator

Robust Gaussian fitting for multivariate anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `contamination` | `float` | Expected proportion of outliers |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

The Elliptic Envelope fits a robust covariance estimate to the data and defines an elliptical decision boundary. Points outside this boundary are classified as anomalies. Best suited for approximately Gaussian data.

### 14.10 PCAAnomalyValidator

Principal Component Analysis reconstruction error for anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `n_components` | `int \| float \| None` | Number of components |
| `error_percentile` | `float` | Error threshold percentile |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

This method projects data onto principal components and uses reconstruction error to identify anomalies. Points with high reconstruction error (deviating from the main variance directions) are classified as anomalies.

### 14.11 ZScoreMultivariateValidator

Multivariate Z-score anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `threshold` | `float` | Z-score threshold (default: 3.0) |
| `method` | `str` | Combination method (any, all, mean) |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

**Combination Methods**:
- `any`: Anomaly if any column exceeds threshold
- `all`: Anomaly if all columns exceed threshold
- `mean`: Anomaly if mean Z-score exceeds threshold

### 14.12 IsolationForestValidator

Isolation Forest for efficient high-dimensional anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `contamination` | `float \| str` | Expected anomaly proportion or "auto" |
| `n_estimators` | `int` | Number of trees (default: 100) |
| `max_samples` | `int \| float \| str` | Samples per tree |
| `random_state` | `int \| None` | Random seed |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

Isolation Forest isolates anomalies by randomly selecting features and split values. Anomalies are easier to isolate, resulting in shorter path lengths. This method is computationally efficient and works well for high-dimensional data without assuming any particular distribution.

```python
validator = IsolationForestValidator(
    columns=["feature1", "feature2", "feature3"],
    contamination=0.05,
    n_estimators=100,
    random_state=42
)
```

### 14.13 LOFValidator

Local Outlier Factor for density-based anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `n_neighbors` | `int` | Number of neighbors (default: 20) |
| `contamination` | `float \| str` | Expected anomaly proportion |
| `metric` | `str` | Distance metric |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

LOF measures the local density deviation of a point compared to its neighbors. Points with substantially lower density than their neighbors are considered outliers. This method excels at detecting local anomalies in clustered data.

### 14.14 OneClassSVMValidator

One-Class Support Vector Machine for boundary-based anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `kernel` | `str` | Kernel type (rbf, linear, poly, sigmoid) |
| `nu` | `float` | Upper bound on anomaly fraction |
| `gamma` | `str \| float` | Kernel coefficient |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

One-Class SVM learns a decision boundary around normal data. Points outside this boundary are classified as anomalies. Works well for high-dimensional data but can be slower than tree-based methods.

### 14.15 DBSCANAnomalyValidator

DBSCAN clustering for density-based anomaly detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `list[str] \| None` | Columns to analyze |
| `eps` | `float` | Maximum distance for neighbors |
| `min_samples` | `int` | Minimum cluster size |
| `metric` | `str` | Distance metric |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio |

DBSCAN identifies outliers as noise points that don't belong to any cluster. This method can discover clusters of arbitrary shape and automatically determines the number of clusters.

```python
validator = DBSCANAnomalyValidator(
    columns=["x", "y"],
    eps=0.5,
    min_samples=5,
    max_anomaly_ratio=0.1
)
```

---

## 15. DAG-Based Execution

Truthound supports DAG (Directed Acyclic Graph) based validator execution, enabling dependency-aware parallel processing for improved performance with multiple validators.

### 15.1 Overview

When running multiple validators, Truthound can automatically:
- Detect dependencies between validators
- Group independent validators for parallel execution
- Execute dependent validators in correct order
- Optimize execution based on validator phases

### 15.2 Execution Phases

Validators are automatically categorized into phases for optimal execution order:

| Phase | Order | Description |
|-------|-------|-------------|
| SCHEMA | 1 | Column structure, types |
| COMPLETENESS | 2 | Null detection, required fields |
| UNIQUENESS | 3 | Duplicates, primary keys |
| FORMAT | 4 | String patterns, regex |
| RANGE | 5 | Value bounds, distributions |
| STATISTICAL | 6 | Aggregates, drift, anomalies |
| CROSS_TABLE | 7 | Multi-table relationships |
| CUSTOM | 8 | User-defined validators |

### 15.3 Usage

```python
import truthound as th

# Enable parallel execution with th.check()
report = th.check("data.csv", parallel=True)

# With custom worker count
report = th.check("data.csv", parallel=True, max_workers=8)
```

### 15.4 Advanced Usage

```python
from truthound.validators.optimization import (
    ValidatorDAG,
    ParallelExecutionStrategy,
    AdaptiveExecutionStrategy,
)

# Create DAG manually
dag = ValidatorDAG()

# Add validators with explicit dependencies
dag.add_validator(schema_validator)
dag.add_validator(null_validator, dependencies={"schema"})
dag.add_validator(range_validator, dependencies={"null"})

# Build execution plan
plan = dag.build_execution_plan()

# Execute with parallel strategy
strategy = ParallelExecutionStrategy(max_workers=4)
result = plan.execute(lf, strategy)

print(f"Total issues: {len(result.all_issues)}")
print(f"Execution time: {result.total_duration_ms:.2f}ms")
```

### 15.5 Execution Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `SequentialExecutionStrategy` | Single-threaded execution | Debugging, small datasets |
| `ParallelExecutionStrategy` | ThreadPoolExecutor-based | Large datasets, many validators |
| `AdaptiveExecutionStrategy` | Auto-selects based on validator count | General use (default) |

### 15.6 Custom Validator Dependencies

Define dependencies in custom validators:

```python
class MyValidator(Validator):
    name = "my_validator"
    category = "custom"
    dependencies = {"null", "schema"}  # Runs after null and schema
    provides = {"my_check"}            # Other validators can depend on this
    priority = 50                       # Lower = runs earlier within phase

    def validate(self, lf):
        ...
```

---

## 16. Validator Performance Profiling

Truthound provides comprehensive profiling capabilities to measure and analyze validator performance.

### 16.1 Overview

The profiling framework tracks:
- **Timing**: Mean, median, p50, p90, p95, p99 execution times
- **Memory**: Peak usage, delta, GC collections
- **Throughput**: Rows processed per second
- **Issues**: Validation issues found per execution

### 16.2 Basic Usage

```python
from truthound.validators.optimization import (
    ValidatorProfiler,
    profile_validator,
)

# Simple profiling with context manager
with profile_validator(my_validator, rows_processed=10000) as ctx:
    issues = my_validator.validate(lf)
    ctx.set_issue_count(len(issues))

print(ctx.metrics.to_dict())
```

### 16.3 Session-Based Profiling

```python
from truthound.validators.optimization import ValidatorProfiler

profiler = ValidatorProfiler()
profiler.start_session("validation_run_1")

for validator in validators:
    with profiler.profile(validator, rows_processed=row_count) as ctx:
        issues = validator.validate(lf)
        ctx.set_issue_count(len(issues))

session = profiler.end_session()

# Get performance insights
slowest = profiler.get_slowest_validators(10)
memory_heavy = profiler.get_memory_intensive_validators(10)

# Export results
print(session.to_json())
print(profiler.to_prometheus())
```

### 16.4 Profiler Modes

| Mode | Description | Metrics |
|------|-------------|---------|
| `DISABLED` | No profiling (zero overhead) | None |
| `BASIC` | Timing only | Execution time |
| `STANDARD` | Default mode | Timing + memory |
| `DETAILED` | Full metrics + snapshots | All metrics + per-execution snapshots |
| `DIAGNOSTIC` | Maximum detail | All metrics + extended snapshots |

```python
from truthound.validators.optimization import ProfilerConfig, ValidatorProfiler

# Use detailed mode for debugging
profiler = ValidatorProfiler(ProfilerConfig.detailed())
```

### 16.5 Decorator-Based Profiling

```python
from truthound.validators.optimization import profiled

class MyValidator(Validator):
    name = "my_validator"
    category = "custom"

    @profiled()
    def validate(self, lf):
        # Automatically profiled
        return issues
```

### 16.6 Report Generation

```python
from truthound.validators.optimization import ProfilingReport

report = ProfilingReport(profiler)

# Text summary
print(report.text_summary())

# HTML report
html = report.html_report()
with open("profiling_report.html", "w") as f:
    f.write(html)

# Prometheus format (for monitoring integration)
prometheus_metrics = profiler.to_prometheus()
```

---

## 17. Custom Validator SDK

Truthound provides a comprehensive SDK for developing custom validators with decorators, fluent builder, testing utilities, and pre-built templates.

### 17.1 Decorator-Based Development

```python
from truthound.validators.sdk import custom_validator
from truthound.validators.base import Validator, StringValidatorMixin

@custom_validator(
    name="email_format",
    category="string",
    description="Validates email address format",
    tags=["format", "string", "email"],
)
class EmailFormatValidator(Validator, StringValidatorMixin):
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Implementation
        ...
```

### 17.2 Fluent Builder API

```python
from truthound.validators.sdk import ValidatorBuilder

validator = (
    ValidatorBuilder("null_check")
    .category("completeness")
    .for_numeric_columns()
    .check(lambda col, lf: lf.filter(pl.col(col).is_null()).collect().height)
    .with_issue_type("null_value")
    .with_severity("high")
    .build()
)
```

### 17.3 Pre-built Templates

| Template | Description |
|----------|-------------|
| `SimplePatternValidator` | Regex pattern matching |
| `SimpleRangeValidator` | Numeric range validation |
| `SimpleColumnValidator` | Single column validation |
| `SimpleAggregateValidator` | Aggregate function validation |

```python
from truthound.validators.sdk import SimplePatternValidator

validator = SimplePatternValidator(
    name="phone_format",
    pattern=r"\d{3}-\d{4}-\d{4}",
    description="Validates Korean phone format",
)
```

### 17.4 Testing Utilities

```python
from truthound.validators.sdk import ValidatorTestCase

class TestMyValidator(ValidatorTestCase):
    validator_class = MyValidator

    def test_valid_data(self):
        result = self.validate_with({"column": [1, 2, 3]})
        self.assert_no_issues(result)

    def test_invalid_data(self):
        result = self.validate_with({"column": [None, 2, None]})
        self.assert_has_issues(result, count=2)
```

---

## 18. ReDoS Protection

Truthound includes comprehensive protection against Regular Expression Denial of Service (ReDoS) attacks.

### 18.1 Regex Safety Checker

```python
from truthound.validators.security import RegexSafetyChecker, ReDoSRisk

checker = RegexSafetyChecker()
result = checker.analyze("(a+)+$")

print(result.risk)  # ReDoSRisk.CRITICAL
print(result.patterns)  # ["nested_quantifiers"]
```

### 18.2 Risk Levels

| Level | Description | Action |
|-------|-------------|--------|
| `NONE` | Safe pattern | Allow |
| `LOW` | Minor complexity | Allow with monitoring |
| `MEDIUM` | Moderate complexity | Review recommended |
| `HIGH` | Significant risk | Consider alternatives |
| `CRITICAL` | Dangerous pattern | Block or refactor |

### 18.3 Safe Regex Executor

```python
from truthound.validators.security import SafeRegexExecutor

executor = SafeRegexExecutor(timeout_seconds=1.0)
result = executor.execute(pattern, text)

if result.matched:
    print(result.groups)
else:
    print(result.error)  # Timeout or pattern error
```

### 18.4 Complexity Analysis

```python
from truthound.validators.security import RegexComplexityAnalyzer

analyzer = RegexComplexityAnalyzer()
analysis = analyzer.analyze("(a|b)*c+d?")

print(analysis.complexity_score)  # 0-100
print(analysis.nesting_depth)
print(analysis.quantifier_count)
```

---

## 19. Internationalization (i18n)

Truthound validator error messages support 7 languages (en, ko, ja, zh, de, fr, es).

### 19.1 Supported Languages

| Code | Language |
|------|----------|
| `en` | English (default) |
| `ko` | 한국어 (Korean) |
| `ja` | 日本語 (Japanese) |
| `zh` | 中文 (Chinese) |
| `de` | Deutsch (German) |
| `fr` | Français (French) |
| `es` | Español (Spanish) |

### 19.2 Usage

```python
from truthound.validators.i18n import (
    set_validator_locale,
    get_validator_message,
    ValidatorMessageCode,
)

# Set locale
set_validator_locale("ko")

# Get localized message
msg = get_validator_message(
    ValidatorMessageCode.NULL_VALUES_FOUND,
    column="email",
    count=10,
)
# -> "'email' 컬럼에서 10개의 null 값이 발견되었습니다"
```

### 19.3 Custom Catalogs

```python
from truthound.validators.i18n import ValidatorMessageCatalog

catalog = (
    ValidatorMessageCatalog.builder("ko_formal")
    .add_null(values_found="'{column}' 항목에서 {count}건의 누락이 확인되었습니다")
    .add_unique(duplicates_found="'{column}' 항목에서 {count}건의 중복이 확인되었습니다")
    .build()
)
```

### 19.4 Message Codes

| Category | Codes |
|----------|-------|
| Null | `NULL_VALUES_FOUND`, `NULL_COLUMN_EMPTY`, `NULL_ABOVE_THRESHOLD` |
| Unique | `UNIQUE_DUPLICATES_FOUND`, `UNIQUE_COMPOSITE_DUPLICATES`, `UNIQUE_KEY_VIOLATION` |
| Type | `TYPE_MISMATCH`, `TYPE_COERCION_FAILED`, `TYPE_INFERENCE_FAILED` |
| Format | `FORMAT_INVALID_EMAIL`, `FORMAT_INVALID_PHONE`, `FORMAT_PATTERN_MISMATCH` |
| Range | `RANGE_OUT_OF_BOUNDS`, `RANGE_BELOW_MINIMUM`, `RANGE_ABOVE_MAXIMUM` |
| Timeout | `TIMEOUT_EXCEEDED`, `TIMEOUT_PARTIAL_RESULT` |

---

## 20. Distributed Timeout

Truthound provides timeout management for distributed validation environments.

### 20.1 Deadline Context

```python
from truthound.validators.timeout import DeadlineContext

with DeadlineContext.from_seconds(60) as ctx:
    # Allocate budget for sub-operations
    budget = ctx.allocate(validators=40, reporting=15, cleanup=5)

    # Execute with deadline awareness
    result = validate_with_deadline(data, budget.validators)

    # Check remaining time
    if ctx.remaining_seconds < 10:
        # Use fast path
        ...
```

### 20.2 Timeout Budget

```python
from truthound.validators.timeout import TimeoutBudget

budget = TimeoutBudget(total_seconds=120)

# Allocate time for operations
validation_budget = budget.allocate("validation", 60)
report_budget = budget.allocate("reporting", 30)

# Use allocated time
with budget.use("validation") as ctx:
    validate(data)

print(budget.get_summary())
```

### 20.3 Graceful Degradation

```python
from truthound.validators.timeout import (
    GracefulDegradation,
    DegradationPolicy,
)

degradation = GracefulDegradation(
    policy=DegradationPolicy.SAMPLE,
    sample_fraction=0.1,
)

# Execute with fallback
result = degradation.execute(operation, timeout=10)

if result.degraded:
    print(f"Used fallback: {result.policy}")
```

### 20.4 Cascade Timeout

```python
from truthound.validators.timeout import (
    CascadeTimeoutHandler,
    CascadePolicy,
)

handler = CascadeTimeoutHandler(
    total_timeout=60,
    policy=CascadePolicy.PROPORTIONAL,
)

# Execute with cascading timeouts
with handler.cascade() as cascade:
    result1 = cascade.execute("phase1", operation1, 20)
    result2 = cascade.execute("phase2", operation2, 30)
```

---

## Appendix A: Validator Categories Summary

### Validator Classes (23 categories, 264 validators)

| Category | Count | Dependencies | Description |
|----------|-------|--------------|-------------|
| Schema | 15 | Core | Structural validation |
| Completeness | 12 | Core | Missing value detection |
| Uniqueness | 17 | Core | Duplicate and key validation |
| Distribution | 15 | Core | Range and statistical checks |
| String | 20 | Core | Pattern and format validation |
| Datetime | 10 | Core | Temporal data validation |
| Aggregate | 8 | Core | Statistical aggregate checks |
| Cross-Table | 5 | Core | Multi-table relationships |
| Query | 19 | Core | Expression-based validation |
| Multi-Column | 20 | Core | Column relationships |
| Table | 18 | Core | Table metadata validation |
| Geospatial | 13 | Core | Geographic coordinates |
| Drift | 14 | scipy | Distribution change detection |
| Anomaly | 18 | scipy, scikit-learn | Outlier detection |
| Business Rule | 8 | Core | Domain-specific validation (Luhn, IBAN, VAT) |
| Localization | 9 | Core | Regional identifiers (Korean, Japanese, Chinese) |
| ML Feature | 5 | Core | Leakage detection, correlation analysis |
| Profiling | 7 | Core | Cardinality, entropy, frequency analysis |
| Referential | 14 | Core | Foreign key and orphan record validation |
| Time Series | 14 | Core | Gap detection, seasonality, trend analysis |
| Privacy | 16 | Core | GDPR, CCPA, LGPD compliance patterns |
| Streaming | 7 | Core | Streaming data validation |
| SDK | 5 | Core | Custom validator development tools |
| **Subtotal** | **289** | | |

### Infrastructure Modules (5 categories, 26 modules)

| Category | Modules | Dependencies | Description |
|----------|---------|--------------|-------------|
| Security | 2 | Core | SQL injection, ReDoS protection |
| Memory | 4 | Core | Memory-aware processing algorithms |
| Optimization | 6 | Core | DAG orchestration, profiling |
| i18n | 10 | Core | Internationalized error messages (7 languages) |
| Timeout | 4 | Core | Distributed timeout management |
| **Subtotal** | **26** | | |

| **Total** | **28 categories** | | **264 validators + 26 modules** |

---

## Appendix B: Installation Options

```bash
# Core validators only
pip install truthound

# With drift detection (adds scipy)
pip install truthound[drift]

# With anomaly detection (adds scipy and scikit-learn)
pip install truthound[anomaly]

# All optional dependencies
pip install truthound[all]
```

---

## Appendix C: Quick Reference

### Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `column` | `str` | Target column name | Required |
| `columns` | `list[str]` | Multiple target columns | `None` (all) |
| `mostly` | `float` | Minimum pass ratio | `None` (100%) |
| `max_anomaly_ratio` | `float` | Maximum anomaly ratio | `0.1` |
| `reference_data` | `pl.DataFrame` | Baseline dataset | Required (drift) |

### Severity Levels

| Severity | Description | Typical Ratio |
|----------|-------------|---------------|
| `LOW` | Minor issue | < 1% |
| `MEDIUM` | Moderate concern | 1-5% |
| `HIGH` | Significant issue | 5-10% |
| `CRITICAL` | Severe problem | > 10% |

---

*Document Version: 1.0.0*
*Last Updated: December 2025*
