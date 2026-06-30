# Creating 사용자 정의 검증기s

튜토리얼에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

튜토리얼에서 Truthound, SDK을(를) 다루는 항목입니다:

1. 튜토리얼에서 Decorator-Based, Quick을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. 튜토리얼에서 Class-Based, Full을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
3. 튜토리얼에서 API, Fluent, Builder, Chainable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Prerequisites

- 튜토리얼에서 Basic, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 Truthound, `pip install truthound`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 Polars, Familiarity, DataFrames을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Method 1: Decorator-Based 검증기

튜토리얼에서 `@custom_validator`을(를) 다루는 항목입니다:

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

| 튜토리얼에서 Parameter을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| 튜토리얼에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Unique 검증기 name (required) |
| 튜토리얼에서 `category`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `description`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Human-readable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `version`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Semantic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `author`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Author을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `tags`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `list[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Tags을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `auto_register`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Auto-register, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Method 2: Class-Based 검증기

튜토리얼에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

튜토리얼에서 SDK을(를) 다루는 항목입니다:

| 튜토리얼에서 Mixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| 튜토리얼에서 `NumericValidatorMixin`, NumericValidatorMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `_get_numeric_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `StringValidatorMixin`, StringValidatorMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `_get_string_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `DatetimeValidatorMixin`, DatetimeValidatorMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `_get_datetime_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `FloatValidatorMixin`, FloatValidatorMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `_get_float_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `RegexValidatorMixin`, RegexValidatorMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Safe, ReDoS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `StreamingValidatorMixin`, StreamingValidatorMixin을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Support을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Method 3: Fluent Builder

튜토리얼에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

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

| 튜토리얼에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 튜토리얼에서 `.category(str)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Set 검증기 category |
| 튜토리얼에서 `.description(str)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `.for_numeric_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Filter to numeric 컬럼 |
| 튜토리얼에서 `.for_string_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Filter to string 컬럼 |
| 튜토리얼에서 `.for_datetime_columns()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Filter to datetime 컬럼 |
| 튜토리얼에서 `.check_column(fn)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 `(col, lf) -> count`, Add을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `.with_issue_type(str)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `.with_severity(Severity)`, Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `.with_message(str)`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 튜토리얼에서 Set을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 튜토리얼에서 `.build()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Build and return 검증기 |

## Registering 검증기

### Automatic Registration

튜토리얼에서 `@custom_validator`, `auto_register=True`, True을(를) 다루는 항목입니다:

```python
@custom_validator(name="my_validator", category="custom")
class MyValidator(Validator):
    ...

# Now usable via name
import truthound as th
report = th.check(df, validators=["my_validator"])
```

### Manual Registration

튜토리얼에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

```python
from truthound.validators.sdk import register_validator

@register_validator
class MyValidator(Validator):
    name = "my_validator"
    category = "custom"

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        ...
```

### Using 검증기 Instances

튜토리얼에서 You을(를) 다루는 항목입니다:

```python
import truthound as th

# Create validator instance
validator = RangeValidator(min_value=0, max_value=100)

# Use with th.check()
report = th.check(df, validators=[validator])

# Or call directly
issues = validator.validate(df.lazy())
```

## Testing Your 검증기

튜토리얼에서 SDK을(를) 다루는 항목입니다:

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

## 권장 방식

### 1. Use Lazy Evaluation

튜토리얼에서 Work, LazyFrames을(를) 다루는 항목입니다:

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

Make 검증 messages actionable:

```python
# Good
details = f"Column '{column}' has {count} values below minimum threshold {min_val}"

# Bad
details = "Validation failed"
```

### 3. Include Relevant Details

튜토리얼에서 Always을(를) 다루는 항목입니다:

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

튜토리얼에서 Account을(를) 다루는 항목입니다:

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

튜토리얼에서 Leverage, SDK을(를) 다루는 항목입니다:

```python
from truthound.validators.sdk import NUMERIC_TYPES, STRING_TYPES

class MyValidator(Validator, NumericValidatorMixin):
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        # Only processes numeric columns
        for col in self._get_numeric_columns(lf):
            ...
```

## Enterprise Features

튜토리얼에서 SDK을(를) 다루는 항목입니다:

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

튜토리얼에서 API, See, Reference, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 다음 단계

- 튜토리얼에서 Validators, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 API, Python, Reference, Complete, SDK을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 튜토리얼에서 Data, Profiling, Guide, Auto-generate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
