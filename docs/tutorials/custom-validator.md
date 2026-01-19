# Creating Custom Validators

Learn how to create custom validators for your specific data quality needs.

## Overview

Truthound's validator SDK makes it easy to create custom validators that integrate seamlessly with the framework. The SDK provides three approaches:

1. **Decorator-Based** - Quick and simple for straightforward validators
2. **Class-Based** - Full control with inheritance for complex validators
3. **Fluent Builder** - Chainable API for one-off validators

## Prerequisites

- Basic Python knowledge
- Truthound installed (`pip install truthound`)
- Familiarity with Polars DataFrames

## Method 1: Decorator-Based Validators

The `@custom_validator` decorator is the simplest way to create a validator:

```python
from truthound.validators.sdk import (
    custom_validator,
    Validator,
    ValidationIssue,
)
from truthound.types import Severity
import polars as pl

@custom_validator(
    name="positive_values",
    category="numeric",
    description="Checks that all values are positive",
    tags=["numeric", "range", "positive"],
)
class PositiveValuesValidator(Validator):
    """Validate that column values are positive."""

    def __init__(self, allow_zero: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.allow_zero = allow_zero

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues = []

        for col in self._get_target_columns(lf):
            # Check column type
            schema = lf.collect_schema()
            if col not in schema or not schema[col].is_numeric():
                continue

            # Count violations
            if self.allow_zero:
                invalid_count = (
                    lf.filter(pl.col(col) < 0)
                    .select(pl.len())
                    .collect()
                    .item()
                )
            else:
                invalid_count = (
                    lf.filter(pl.col(col) <= 0)
                    .select(pl.len())
                    .collect()
                    .item()
                )

            if invalid_count > 0:
                total = lf.select(pl.len()).collect().item()
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="non_positive_value",
                        count=invalid_count,
                        severity=Severity.HIGH,
                        details=f"Found {invalid_count}/{total} non-positive values",
                    )
                )

        return issues
```

### Decorator Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique validator name (required) |
| `category` | `str` | Category for grouping (default: "custom") |
| `description` | `str` | Human-readable description |
| `version` | `str` | Semantic version (default: "1.0.0") |
| `author` | `str` | Author name or email |
| `tags` | `list[str]` | Tags for filtering and discovery |
| `auto_register` | `bool` | Auto-register in global registry (default: True) |

## Method 2: Class-Based Validators

For more complex validators with state or multiple checks:

```python
from truthound.validators.sdk import (
    Validator,
    ValidationIssue,
    NumericValidatorMixin,
)
from truthound.types import Severity
import polars as pl

class RangeValidator(Validator, NumericValidatorMixin):
    """Validates that values fall within a specified range."""

    name = "custom_range"
    category = "numeric"

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues = []
        columns = self._get_numeric_columns(lf)

        for col in columns:
            # Build filter for violations
            conditions = []

            if self.min_value is not None:
                if self.inclusive:
                    conditions.append(pl.col(col) < self.min_value)
                else:
                    conditions.append(pl.col(col) <= self.min_value)

            if self.max_value is not None:
                if self.inclusive:
                    conditions.append(pl.col(col) > self.max_value)
                else:
                    conditions.append(pl.col(col) >= self.max_value)

            if not conditions:
                continue

            # Combine conditions with OR
            combined = conditions[0]
            for cond in conditions[1:]:
                combined = combined | cond

            invalid_count = (
                lf.filter(combined)
                .select(pl.len())
                .collect()
                .item()
            )

            if invalid_count > 0:
                total = lf.select(pl.len()).collect().item()
                issues.append(
                    ValidationIssue(
                        column=col,
                        issue_type="out_of_range",
                        count=invalid_count,
                        severity=Severity.MEDIUM,
                        details=self._format_message(col, invalid_count, total),
                    )
                )

        return issues

    def _format_message(self, column: str, invalid: int, total: int) -> str:
        range_str = ""
        if self.min_value is not None and self.max_value is not None:
            range_str = f"[{self.min_value}, {self.max_value}]"
        elif self.min_value is not None:
            range_str = f">= {self.min_value}"
        elif self.max_value is not None:
            range_str = f"<= {self.max_value}"

        return f"{column}: {invalid}/{total} values outside range {range_str}"
```

### Available Mixins

The SDK provides mixins for common patterns:

| Mixin | Description |
|-------|-------------|
| `NumericValidatorMixin` | `_get_numeric_columns()` helper |
| `StringValidatorMixin` | `_get_string_columns()` helper |
| `DatetimeValidatorMixin` | `_get_datetime_columns()` helper |
| `FloatValidatorMixin` | `_get_float_columns()` helper |
| `RegexValidatorMixin` | Safe regex execution with ReDoS protection |
| `StreamingValidatorMixin` | Support for streaming large datasets |

## Method 3: Fluent Builder

For quick one-off validators without creating a class:

```python
from truthound.validators.sdk import ValidatorBuilder
from truthound.types import Severity
import polars as pl

# Create validator using builder
email_domain_validator = (
    ValidatorBuilder("email_domain")
    .category("string")
    .description("Validates email domain is from allowed list")
    .for_string_columns()
    .check_column(
        lambda col, lf: lf.filter(
            ~pl.col(col).str.contains(r"@(company\.com|partner\.com)$")
        ).select(pl.len()).collect().item()
    )
    .with_issue_type("invalid_email_domain")
    .with_severity(Severity.MEDIUM)
    .with_message("Column '{column}' has {count} emails with invalid domains")
    .build()
)

# Use the validator
issues = email_domain_validator.validate(df.lazy())
```

### Builder Methods

| Method | Description |
|--------|-------------|
| `.category(str)` | Set validator category |
| `.description(str)` | Set description |
| `.for_numeric_columns()` | Filter to numeric columns |
| `.for_string_columns()` | Filter to string columns |
| `.for_datetime_columns()` | Filter to datetime columns |
| `.check_column(fn)` | Add check function `(col, lf) -> count` |
| `.with_issue_type(str)` | Set issue type |
| `.with_severity(Severity)` | Set severity level |
| `.with_message(str)` | Set message template |
| `.build()` | Build and return validator |

## Registering Validators

### Automatic Registration

Using `@custom_validator` with `auto_register=True` (default) automatically registers the validator:

```python
@custom_validator(name="my_validator", category="custom")
class MyValidator(Validator):
    ...

# Now usable via name
import truthound as th
report = th.check(df, validators=["my_validator"])
```

### Manual Registration

For validators without the decorator:

```python
from truthound.validators.sdk import register_validator

@register_validator
class MyValidator(Validator):
    name = "my_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        ...
```

### Using Validator Instances

You can also pass validator instances directly:

```python
import truthound as th

# Create validator instance
validator = RangeValidator(min_value=0, max_value=100)

# Use with th.check()
report = th.check(df, validators=[validator])

# Or call directly
issues = validator.validate(df.lazy())
```

## Testing Your Validator

The SDK provides testing utilities:

```python
import pytest
import polars as pl
from truthound.validators.sdk import (
    ValidatorTestCase,
    create_test_dataframe,
    assert_no_issues,
    assert_has_issue,
    assert_issue_count,
)
from my_validators import PositiveValuesValidator

class TestPositiveValuesValidator(ValidatorTestCase):
    """Tests for PositiveValuesValidator."""

    def test_passes_for_positive_values(self):
        """Test that validator passes for positive values."""
        df = create_test_dataframe({
            "amount": [1, 2, 3, 4, 5]
        })

        validator = PositiveValuesValidator()
        issues = validator.validate(df.lazy())

        assert_no_issues(issues)

    def test_fails_for_negative_values(self):
        """Test that validator fails for negative values."""
        df = create_test_dataframe({
            "amount": [1, -2, 3, -4, 5]
        })

        validator = PositiveValuesValidator()
        issues = validator.validate(df.lazy())

        assert_issue_count(issues, 1)
        assert_has_issue(issues, column="amount", issue_type="non_positive_value")

    def test_allow_zero_option(self):
        """Test zero handling with allow_zero option."""
        df = create_test_dataframe({
            "amount": [0, 1, 2]
        })

        # Without allow_zero (default)
        validator = PositiveValuesValidator(allow_zero=False)
        issues = validator.validate(df.lazy())
        assert_issue_count(issues, 1)

        # With allow_zero
        validator = PositiveValuesValidator(allow_zero=True)
        issues = validator.validate(df.lazy())
        assert_no_issues(issues)
```

## Best Practices

### 1. Use Lazy Evaluation

Work with LazyFrames when possible for better performance:

```python
# Good - uses LazyFrame operations
def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
    count = (
        lf.filter(pl.col(col).is_null())
        .select(pl.len())
        .collect()
        .item()
    )
    ...

# Avoid - collects too early
def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
    data = lf.collect()  # Don't do this!
    ...
```

### 2. Provide Clear Messages

Make validation messages actionable:

```python
# Good
details = f"Column '{column}' has {count} values below minimum threshold {min_val}"

# Bad
details = "Validation failed"
```

### 3. Include Relevant Details

Always include information useful for debugging:

```python
ValidationIssue(
    column=col,
    issue_type="out_of_range",
    count=invalid_count,
    severity=Severity.MEDIUM,
    details=f"Found {invalid_count} values outside [{min_val}, {max_val}]",
    sample_values=samples[:5] if samples else None,
)
```

### 4. Handle Edge Cases

Account for empty data and missing columns:

```python
def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
    issues = []
    schema = lf.collect_schema()

    for col in self._get_target_columns(lf):
        # Check if column exists
        if col not in schema:
            continue

        # Check for empty data
        count = lf.select(pl.len()).collect().item()
        if count == 0:
            continue

        # Regular validation
        ...

    return issues
```

### 5. Use Type Filters

Leverage the SDK's type filtering:

```python
from truthound.validators.sdk import NUMERIC_TYPES, STRING_TYPES

class MyValidator(Validator, NumericValidatorMixin):
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Only processes numeric columns
        for col in self._get_numeric_columns(lf):
            ...
```

## Enterprise Features

For production environments, the SDK includes enterprise features:

```python
from truthound.validators.sdk import (
    EnterpriseSDKManager,
    EnterpriseConfig,
    SandboxBackend,
    ResourceLimits,
)

# Configure enterprise features
config = EnterpriseConfig(
    sandbox_backend=SandboxBackend.PROCESS,
    resource_limits=ResourceLimits(
        max_memory_mb=512,
        max_cpu_percent=50,
        max_execution_time_seconds=30,
    ),
    enable_signing=True,
)

manager = EnterpriseSDKManager(config)

# Execute validator in sandbox
result = manager.execute_validator(my_validator, df.lazy())
```

See the [API Reference](../python-api/validators.md) for complete enterprise SDK documentation.

## Next Steps

- [Validators Guide](../guides/validators.md) - All built-in validators
- [Python API Reference](../python-api/validators.md) - Complete SDK documentation
- [Data Profiling Guide](../guides/profiler.md) - Auto-generate validation rules
