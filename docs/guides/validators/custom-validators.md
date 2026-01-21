# Custom Validator Development Guide

This document describes how to develop custom validators using the Truthound SDK.

## Overview

The Truthound SDK provides three approaches for creating custom validators:

1. **Decorator Approach** - Add metadata to classes and register them with the registry
2. **Builder Pattern** - Create validators using a fluent API without subclassing
3. **Template Inheritance** - Extend predefined template classes

---

## 1. Decorator Approach

### @custom_validator

The most common approach for creating custom validators.

```python
from truthound.validators.sdk import custom_validator
from truthound.validators.base import Validator, ValidationIssue, NumericValidatorMixin
import polars as pl

@custom_validator(
    name="percentage_range",
    category="numeric",
    description="Validates values are valid percentages (0-100)",
    version="1.0.0",
    author="your-name@example.com",
    tags=["numeric", "range", "percentage"],
    examples=[
        "PercentageValidator()",
        "PercentageValidator(columns=('rate', 'ratio'))",
    ],
    config_schema={
        "type": "object",
        "properties": {
            "allow_zero": {"type": "boolean", "default": True}
        }
    },
    auto_register=True,  # Default: True
)
class PercentageValidator(Validator, NumericValidatorMixin):
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues = []
        for col in self._get_numeric_columns(lf):
            # Range validation logic for 0-100
            count = (
                lf.filter(
                    (pl.col(col) < 0) | (pl.col(col) > 100)
                )
                .select(pl.len())
                .collect()
                .item()
            )
            if count > 0:
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="invalid_percentage",
                        count=count,
                        severity=self.config.severity_override or Severity.MEDIUM,
                        details=f"Found {count} values outside 0-100 range",
                    )
                )
        return issues
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | (required) | Unique validator name |
| `category` | `str` | `"custom"` | Validator category |
| `description` | `str` | `""` | Human-readable description |
| `version` | `str` | `"1.0.0"` | Semantic version |
| `author` | `str` | `""` | Author name/email |
| `tags` | `list[str]` | `None` | Tags for filtering/searching |
| `examples` | `list[str]` | `None` | Usage examples for documentation |
| `config_schema` | `dict` | `None` | Configuration JSON schema |
| `auto_register` | `bool` | `True` | Whether to automatically register with the registry |

### @register_validator

Registers an existing validator class with the registry.

```python
from truthound.validators.sdk import register_validator

@register_validator
class MyValidator(Validator):
    name = "my_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        ...
```

### @validator_metadata

Adds detailed metadata to an existing validator.

```python
from truthound.validators.sdk import validator_metadata, register_validator

@validator_metadata(
    description="Validates phone number format",
    version="1.0.0",
    author="team@example.com",
    tags=["string", "phone", "format"],
    examples=[
        "PhoneValidator(format='US')",
        "PhoneValidator(format='KR')",
    ],
)
@register_validator
class PhoneValidator(Validator):
    name = "phone"
    category = "string"
    ...
```

### @deprecated_validator

Marks a validator as deprecated.

```python
from truthound.validators.sdk import deprecated_validator

@deprecated_validator(
    message="Use 'email_v2' for RFC 5322 compliance",
    replacement="email_v2",
    remove_in_version="2.0.0",
)
class OldEmailValidator(Validator):
    name = "email_v1"
    ...
```

A `DeprecationWarning` is raised upon instantiation:
```
DeprecationWarning: Validator 'email_v1' is deprecated. Use 'email_v2' for RFC 5322 compliance. Use 'email_v2' instead. Will be removed in version 2.0.0.
```

---

## 2. Registry API

Functions for querying and managing registered validators.

```python
from truthound.validators.sdk import (
    get_registered_validators,
    get_validator_by_name,
    get_validator_metadata,
    get_validators_by_category,
    get_validators_by_tag,
    list_validator_categories,
    list_validator_tags,
    unregister_validator,
    clear_registry,
)
```

### Query Functions

| Function | Return Type | Description |
|----------|-------------|-------------|
| `get_registered_validators()` | `dict[str, type]` | All registered validators |
| `get_validator_by_name(name)` | `type \| None` | Retrieve validator class by name |
| `get_validator_metadata(name)` | `ValidatorMeta \| None` | Retrieve validator metadata |
| `get_validators_by_category(category)` | `list[type]` | List of validators by category |
| `get_validators_by_tag(tag)` | `list[type]` | List of validators by tag |
| `list_validator_categories()` | `list[str]` | List of all categories |
| `list_validator_tags()` | `list[str]` | List of all tags |

### Management Functions

| Function | Return Type | Description |
|----------|-------------|-------------|
| `unregister_validator(name)` | `bool` | Unregister a validator |
| `clear_registry()` | `None` | Unregister all validators (for testing) |

### ValidatorMeta Dataclass

```python
@dataclass(frozen=True)
class ValidatorMeta:
    name: str
    category: str = "general"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    deprecated: bool = False
    deprecated_message: str = ""
    replacement: str = ""
    examples: tuple[str, ...] = field(default_factory=tuple)
    config_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]: ...
```

---

## 3. Builder Pattern

Create validators using a fluent API without subclassing.

### ValidatorBuilder

```python
from truthound.validators.sdk import ValidatorBuilder
from truthound.types import Severity
import polars as pl

validator = (
    ValidatorBuilder("positive_values")
    .category("numeric")
    .description("Checks that numeric values are positive")
    .for_numeric_columns()
    .check_column(
        lambda col, lf: lf.filter(pl.col(col) < 0).select(pl.len()).collect().item()
    )
    .with_issue_type("negative_value")
    .with_severity(Severity.HIGH)
    .with_message("Column '{column}' has {count} negative values")
    .with_samples(
        lambda col, lf: lf.filter(pl.col(col) < 0)
        .select(col)
        .head(5)
        .collect()
        .to_series()
        .to_list()
    )
    .build()
)

# Usage
issues = validator.validate(lf)
```

#### Method Chain

| Method | Description |
|--------|-------------|
| `category(name)` | Set category |
| `description(text)` | Set description |
| `for_columns(dtype_filter)` | Set data type filter |
| `for_numeric_columns()` | Target only numeric columns |
| `for_string_columns()` | Target only string columns |
| `for_datetime_columns()` | Target only datetime columns |
| `for_float_columns()` | Target only float columns |
| `check_column(fn)` / `check(fn)` | Add check function (col, lf) -> count |
| `with_issue_type(type)` | Set issue type |
| `with_severity(severity)` | Set severity |
| `with_message(template)` | Set message template ({column}, {count}) |
| `with_samples(fn)` | Set sample collection function |
| `with_config(config)` | Set ValidatorConfig |
| `build()` | Create Validator instance |

### ColumnCheckBuilder

Defines individual column checks.

```python
from truthound.validators.sdk import ColumnCheckBuilder
from truthound.types import Severity
import polars as pl

check = (
    ColumnCheckBuilder()
    .violation_filter(pl.col("value") < 0)
    .issue_type("negative_value")
    .severity(Severity.HIGH)
    .message("Found {count} negative values in '{column}'")
    .build()
)
```

### AggregateCheckBuilder

Defines aggregate-level checks.

```python
from truthound.validators.sdk import AggregateCheckBuilder
from truthound.types import Severity

check = (
    AggregateCheckBuilder()
    .check(lambda col, stats: stats["mean"] > 0)
    .issue_type("non_positive_mean")
    .severity(Severity.MEDIUM)
    .message("Column '{column}' has non-positive mean")
    .build()
)
```

### Convenience Functions

#### simple_column_validator

Create a simple column validator in one line:

```python
from truthound.validators.sdk import simple_column_validator
from truthound.types import Severity
import polars as pl

validator = simple_column_validator(
    name="no_nulls",
    check_fn=lambda col, lf: lf.filter(
        pl.col(col).is_null()
    ).select(pl.len()).collect().item(),
    issue_type="null_value",
    severity=Severity.HIGH,
    category="completeness",
    dtype_filter=None,  # All types
)
```

#### simple_expression_validator

Create a validator from a Polars expression:

```python
from truthound.validators.sdk import simple_expression_validator
from truthound.types import Severity
import polars as pl

validator = simple_expression_validator(
    name="positive_values",
    violation_expr=pl.col("amount") <= 0,  # True = violation
    issue_type="non_positive",
    severity=Severity.HIGH,
    category="numeric",
    columns=["amount", "quantity"],  # Specific columns only
)
```

---

## 4. Template Classes

Abstract template classes are provided for common patterns.

### SimpleColumnValidator

Base template for per-column validation:

```python
from truthound.validators.sdk import SimpleColumnValidator
from truthound.validators.base import NUMERIC_TYPES
from truthound.types import Severity
import polars as pl

class PositiveValidator(SimpleColumnValidator):
    name = "positive"
    category = "numeric"
    issue_type = "non_positive_value"
    default_severity = Severity.HIGH
    dtype_filter = NUMERIC_TYPES  # Numeric columns only

    def check_column_values(self, lf: pl.LazyFrame, col: str) -> int:
        """Return violation count (required implementation)"""
        return lf.filter(pl.col(col) <= 0).select(pl.len()).collect().item()

    def get_violation_samples(self, lf: pl.LazyFrame, col: str) -> list | None:
        """Return violation samples (optional)"""
        return (
            lf.filter(pl.col(col) <= 0)
            .select(col)
            .head(5)
            .collect()
            .to_series()
            .to_list()
        )

    def get_issue_details(self, col: str, count: int, total: int) -> str:
        """Return issue detail message (optional)"""
        pct = (count / total * 100) if total > 0 else 0
        return f"Found {count} non-positive values ({pct:.1f}%)"
```

#### Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"simple_column"` | Validator name |
| `category` | `str` | `"custom"` | Category |
| `issue_type` | `str` | `"validation_failed"` | Issue type |
| `default_severity` | `Severity` | `MEDIUM` | Default severity |
| `dtype_filter` | `set[type] \| None` | `None` | Data type filter |

### SimplePatternValidator

Regex-based string validation:

```python
from truthound.validators.sdk import SimplePatternValidator
from truthound.types import Severity

class EmailValidator(SimplePatternValidator):
    name = "email"
    category = "string"
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    issue_type = "invalid_email"
    match_full = True  # Full string matching
    case_sensitive = True

class NoSSNValidator(SimplePatternValidator):
    name = "no_ssn"
    category = "privacy"
    pattern = r"\d{3}-\d{2}-\d{4}"
    invert_match = True  # Pattern presence is a violation
    issue_type = "contains_ssn"
```

#### Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `pattern` | `str` | `""` (required) | Regex pattern |
| `match_full` | `bool` | `True` | Whether to match the full string |
| `invert_match` | `bool` | `False` | If True, matching values are violations |
| `case_sensitive` | `bool` | `True` | Case sensitivity |

### SimpleRangeValidator

Numeric range validation:

```python
from truthound.validators.sdk import SimpleRangeValidator
from truthound.types import Severity

class PercentageValidator(SimpleRangeValidator):
    name = "percentage"
    min_value = 0
    max_value = 100
    issue_type = "invalid_percentage"

class PositiveOnlyValidator(SimpleRangeValidator):
    name = "positive_only"
    min_value = 0
    inclusive_min = False  # Exclude 0 (> 0)
    issue_type = "non_positive"

class AgeValidator(SimpleRangeValidator):
    name = "valid_age"
    min_value = 0
    max_value = 150
    issue_type = "invalid_age"
```

#### Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_value` | `float \| int \| None` | `None` | Minimum value |
| `max_value` | `float \| int \| None` | `None` | Maximum value |
| `inclusive_min` | `bool` | `True` | Whether to include minimum value |
| `inclusive_max` | `bool` | `True` | Whether to include maximum value |

### SimpleComparisonValidator

Cross-column comparison validation:

```python
from truthound.validators.sdk import SimpleComparisonValidator
from truthound.types import Severity

class StartBeforeEndValidator(SimpleComparisonValidator):
    name = "start_before_end"
    left_column = "start_date"
    right_column = "end_date"
    operator = "lt"  # start_date < end_date
    issue_type = "invalid_date_range"

class AmountMatchesValidator(SimpleComparisonValidator):
    name = "amounts_match"
    left_column = "calculated_total"
    right_column = "reported_total"
    operator = "eq"  # Must be equal
    issue_type = "amount_mismatch"
```

#### Supported Operators

| Operator | Symbol | Meaning |
|----------|--------|---------|
| `"eq"` | `==` | Equal |
| `"ne"` | `!=` | Not equal |
| `"lt"` | `<` | Less than |
| `"le"` | `<=` | Less than or equal |
| `"gt"` | `>` | Greater than |
| `"ge"` | `>=` | Greater than or equal |

### CompositeValidator

Combine multiple validators:

```python
from truthound.validators.sdk import CompositeValidator
from truthound.validators import NullValidator, UniqueValidator, RangeValidator

# Method 1: Class inheritance
class CustomerDataValidator(CompositeValidator):
    name = "customer_data"
    category = "business"

    def get_validators(self) -> list[Validator]:
        return [
            EmailValidator(columns=("email",)),
            PhoneValidator(columns=("phone",)),
            AgeValidator(columns=("age",)),
        ]

# Method 2: Inline creation
composite = CompositeValidator(
    validators=[
        NullValidator(columns=("id", "name")),
        UniqueValidator(columns=("id",)),
        RangeValidator(columns=("age",), min_value=0, max_value=150),
    ]
)

# Method 3: Builder style
composite = CompositeValidator()
composite.add_validator(NullValidator())
composite.add_validator(UniqueValidator(columns=("id",)))
```

---

## 5. Factory Functions

Factory functions for dynamic class creation.

### create_pattern_validator

```python
from truthound.validators.sdk import create_pattern_validator

# Create class
EmailValidator = create_pattern_validator(
    name="email",
    pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    issue_type="invalid_email",
    invert=False,  # Non-matching is a violation
    case_sensitive=True,
)

# Create instance and use
validator = EmailValidator()
issues = validator.validate(lf)
```

### create_range_validator

```python
from truthound.validators.sdk import create_range_validator

# Create class
PercentageValidator = create_range_validator(
    name="percentage",
    min_value=0,
    max_value=100,
    issue_type="invalid_percentage",
    inclusive=True,  # Include boundary values
)

# Create instance and use
validator = PercentageValidator()
issues = validator.validate(lf)
```

---

## 6. Testing Framework

### ValidatorTestCase

Base class for unittest-based testing:

```python
from truthound.validators.sdk import ValidatorTestCase
from truthound.types import Severity

class TestPositiveValidator(ValidatorTestCase):
    validator_class = PositiveValidator

    def test_detects_negative_values(self):
        df = self.create_df({"col1": [1, -1, 2, -2, 3]})
        self.validate(df)
        self.assert_has_issue(
            column="col1",
            issue_type="non_positive_value",
            min_count=2,
        )

    def test_no_issues_for_valid_data(self):
        df = self.create_df({"col1": [1, 2, 3, 4, 5]})
        self.validate(df)
        self.assert_no_issues()

    def test_severity_level(self):
        df = self.create_df({"col1": [-1]})
        self.validate(df)
        self.assert_has_issue(
            column="col1",
            severity=Severity.HIGH,
        )

    def test_performance(self):
        df = self.create_large_df(rows=1_000_000)
        self.validate(df)
        self.assert_performance(max_ms=1000, rows=1_000_000)
```

#### Test Methods

| Method | Description |
|--------|-------------|
| `create_validator(**kwargs)` | Create validator instance |
| `create_df(data)` | Create test LazyFrame |
| `create_large_df(rows, schema, seed)` | Create large test data |
| `validate(lf, validator)` | Execute validation |
| `validate_safe(lf, validator)` | Validation with error handling |

#### Assertion Methods

| Method | Description |
|--------|-------------|
| `assert_no_issues()` | Verify no issues |
| `assert_has_issue(column, issue_type, min_count, exact_count, severity)` | Verify specific issue exists |
| `assert_issue_count(expected)` | Verify issue count |
| `assert_total_violations(expected)` | Verify total violations |
| `assert_no_error()` | Verify no errors |
| `assert_error(error_type)` | Verify specific error occurred |
| `assert_performance(max_ms, rows)` | Verify performance criteria met |

### Test Data Generation

```python
from truthound.validators.sdk import create_test_dataframe, create_edge_case_data

# Basic test data
df = create_test_dataframe(rows=1000, include_nulls=True)

# Explicit data
df = create_test_dataframe(data={"col1": [1, 2, 3]})

# Edge case collection
edge_cases = create_edge_case_data()
# Returns: {
#   "empty": Empty DataFrame,
#   "single_row": 1-row DataFrame,
#   "all_nulls": All values null,
#   "uniform_values": All values identical,
#   "large_values": Very large numbers,
#   "small_values": Very small numbers,
#   "unicode": Unicode strings,
#   "empty_strings": Empty strings,
#   "whitespace": Whitespace characters,
#   "special_floats": inf, -inf, nan, 0.0, -0.0,
# }
```

### Standalone Assertion Functions

```python
from truthound.validators.sdk import (
    assert_no_issues,
    assert_has_issue,
    assert_issue_count,
)

issues = validator.validate(lf)

assert_no_issues(issues)
assert_has_issue(issues, column="col1", issue_type="null_value", min_count=5)
assert_issue_count(issues, expected=3)
```

### Performance Benchmarking

```python
from truthound.validators.sdk import benchmark_validator, ValidatorBenchmark

# Single validator benchmarking
result = benchmark_validator(
    validator=PositiveValidator(),
    lf=large_dataframe,
    iterations=10,
    warmup=2,
)
print(f"Mean: {result.mean_ms:.2f}ms")
print(f"Throughput: {result.throughput_rows_per_sec:,.0f} rows/sec")

# Compare multiple validators
benchmark = ValidatorBenchmark()
benchmark.add_validator(NullValidator())
benchmark.add_validator(UniqueValidator())
benchmark.add_validator(PositiveValidator())

results = benchmark.run(
    row_counts=[1000, 10000, 100000, 1000000],
    iterations=10,
)
benchmark.print_report()
```

#### BenchmarkResult

```python
@dataclass
class BenchmarkResult:
    validator_name: str
    row_count: int
    iterations: int
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    throughput_rows_per_sec: float

    def to_dict(self) -> dict[str, Any]: ...
```

---

## 7. CLI Scaffolding

Generate validator templates using the `th new validator` command.

```bash
# Basic template (7 types available)
th new validator my_validator

# Specify template
th new validator email_format --template pattern

# Immediate installation (editable mode)
th new validator my_validator --install
```

### Available Templates

| Template | Description |
|----------|-------------|
| `basic` | Basic Validator subclass |
| `column` | SimpleColumnValidator inheritance |
| `pattern` | SimplePatternValidator inheritance |
| `range` | SimpleRangeValidator inheritance |
| `comparison` | SimpleComparisonValidator inheritance |
| `composite` | CompositeValidator inheritance |
| `ml` | ML-based validator (anomaly detection) |

Generated file structure:

```
my_validator/
├── pyproject.toml         # Package metadata
├── README.md
├── src/
│   └── my_validator/
│       ├── __init__.py
│       └── validator.py   # Validator implementation
└── tests/
    └── test_validator.py  # Test code
```

---

## Next Steps

- [Enterprise SDK](enterprise-sdk.md) - Sandbox, code signing, license management
- [Security Guide](security.md) - ReDoS protection, SQL injection prevention
- [Built-in Validators](built-in.md) - 289 built-in validators reference
