# Creating Custom Validators

Learn how to create custom validators for your specific data quality needs.

## Overview

Truthound's validator SDK makes it easy to create custom validators that integrate seamlessly with the framework.

## Prerequisites

- Basic Python knowledge
- Truthound installed
- Familiarity with Polars DataFrames

## Method 1: Decorator-Based Validators

The simplest way to create a validator:

```python
from truthound.validators.sdk import validator, ValidationResult
import polars as pl

@validator(
    name="positive_values",
    category="custom",
    severity="high",
    description="Checks that all values are positive"
)
def positive_values_validator(
    df: pl.LazyFrame,
    column: str,
    allow_zero: bool = False
) -> ValidationResult:
    """Validate that column values are positive.

    Args:
        df: Data to validate
        column: Column to check
        allow_zero: Whether to allow zero values

    Returns:
        ValidationResult with pass/fail status
    """
    # Collect to DataFrame
    data = df.select(column).collect()

    if allow_zero:
        invalid = (data[column] < 0).sum()
    else:
        invalid = (data[column] <= 0).sum()

    total = len(data)
    passed = invalid == 0

    return ValidationResult(
        passed=passed,
        message=f"Found {invalid} non-positive values in {column}",
        severity="high",
        details={
            "column": column,
            "invalid_count": invalid,
            "total_count": total,
            "invalid_ratio": invalid / total if total > 0 else 0
        }
    )
```

## Method 2: Class-Based Validators

For more complex validators with state:

```python
from truthound.validators.sdk import BaseValidator, ValidationResult
import polars as pl

class RangeValidator(BaseValidator):
    """Validates that values fall within a specified range."""

    name = "custom_range"
    category = "custom"
    description = "Validates values within a custom range"

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(
        self,
        df: pl.LazyFrame,
        column: str
    ) -> ValidationResult:
        """Run the validation.

        Args:
            df: Data to validate
            column: Column to check

        Returns:
            ValidationResult
        """
        data = df.select(column).collect()
        values = data[column]

        # Build filter expression
        filters = []

        if self.min_value is not None:
            if self.inclusive:
                filters.append(values < self.min_value)
            else:
                filters.append(values <= self.min_value)

        if self.max_value is not None:
            if self.inclusive:
                filters.append(values > self.max_value)
            else:
                filters.append(values >= self.max_value)

        if filters:
            invalid_mask = filters[0]
            for f in filters[1:]:
                invalid_mask = invalid_mask | f
            invalid_count = invalid_mask.sum()
        else:
            invalid_count = 0

        passed = invalid_count == 0

        return ValidationResult(
            passed=passed,
            message=self._format_message(column, invalid_count, len(values)),
            details={
                "column": column,
                "min_value": self.min_value,
                "max_value": self.max_value,
                "invalid_count": invalid_count
            }
        )

    def _format_message(
        self,
        column: str,
        invalid: int,
        total: int
    ) -> str:
        range_str = ""
        if self.min_value is not None and self.max_value is not None:
            range_str = f"[{self.min_value}, {self.max_value}]"
        elif self.min_value is not None:
            range_str = f">= {self.min_value}"
        elif self.max_value is not None:
            range_str = f"<= {self.max_value}"

        return f"{column}: {invalid}/{total} values outside range {range_str}"
```

## Method 3: Fluent Builder

For quick one-off validators:

```python
from truthound.validators.sdk import ValidatorBuilder
import polars as pl

# Create validator using builder
email_domain_validator = (
    ValidatorBuilder("email_domain")
    .category("custom")
    .severity("medium")
    .description("Validates email domain")
    .with_parameter("allowed_domains", list, required=True)
    .validate_with(lambda df, col, allowed_domains: (
        df.select(col)
        .collect()[col]
        .str.extract(r"@(.+)$")
        .is_in(allowed_domains)
        .all()
    ))
    .build()
)
```

## Registering Validators

Register your validator to use it with Truthound:

```python
from truthound.validators import register_validator

# Register decorator-based validator
register_validator(positive_values_validator)

# Register class-based validator
register_validator(RangeValidator(min_value=0, max_value=100))

# Now use it
import truthound as th

report = th.check(
    "data.csv",
    validators=["positive_values", "custom_range"]
)
```

## Testing Your Validator

Write tests to ensure your validator works correctly:

```python
import pytest
import polars as pl
from my_validators import positive_values_validator

def test_positive_values_passes():
    """Test that validator passes for positive values."""
    df = pl.DataFrame({
        "amount": [1, 2, 3, 4, 5]
    }).lazy()

    result = positive_values_validator(df, "amount")

    assert result.passed
    assert result.details["invalid_count"] == 0

def test_positive_values_fails():
    """Test that validator fails for negative values."""
    df = pl.DataFrame({
        "amount": [1, -2, 3, -4, 5]
    }).lazy()

    result = positive_values_validator(df, "amount")

    assert not result.passed
    assert result.details["invalid_count"] == 2

def test_positive_values_with_zero():
    """Test zero handling."""
    df = pl.DataFrame({
        "amount": [0, 1, 2]
    }).lazy()

    # Without allow_zero
    result = positive_values_validator(df, "amount", allow_zero=False)
    assert not result.passed

    # With allow_zero
    result = positive_values_validator(df, "amount", allow_zero=True)
    assert result.passed
```

## Best Practices

### 1. Use Lazy Evaluation

Work with LazyFrames when possible for better performance:

```python
# Good - uses LazyFrame operations
def my_validator(df: pl.LazyFrame, column: str) -> ValidationResult:
    result = (
        df.select(column)
        .filter(pl.col(column).is_null())
        .collect()
    )
    ...

# Avoid - collects too early
def my_validator(df: pl.LazyFrame, column: str) -> ValidationResult:
    data = df.collect()  # Don't do this
    ...
```

### 2. Provide Clear Messages

Make validation messages actionable:

```python
# Good
message = f"Column '{column}' has {count} values below minimum threshold {min_val}"

# Bad
message = "Validation failed"
```

### 3. Include Details

Always include relevant details for debugging:

```python
return ValidationResult(
    passed=passed,
    message=message,
    details={
        "column": column,
        "threshold": threshold,
        "actual_value": actual,
        "expected": expected,
        "sample_invalid_rows": invalid_rows[:5]  # Sample for debugging
    }
)
```

### 4. Handle Edge Cases

Account for empty data and missing columns:

```python
def my_validator(df: pl.LazyFrame, column: str) -> ValidationResult:
    # Check if column exists
    if column not in df.columns:
        return ValidationResult(
            passed=False,
            message=f"Column '{column}' not found",
            severity="critical"
        )

    # Check for empty data
    count = df.select(column).collect().shape[0]
    if count == 0:
        return ValidationResult(
            passed=True,
            message=f"No data to validate in '{column}'"
        )

    # Regular validation
    ...
```

## Next Steps

- [Validators Guide](../user-guide/validators.md) - All built-in validators
- [API Reference](../api-reference/validators/sdk/index.md) - SDK API documentation
