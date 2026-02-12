# Validators

Validator interface and registration system.

## Validator Base Class

All validators inherit from the `Validator` base class.

### Definition

```python
from truthound.validators.base import Validator

class Validator(ABC):
    """Abstract base class for all validators."""

    name: str = "base"        # Unique validator identifier
    category: str = "general" # Validator category

    # DAG execution metadata
    dependencies: set[str] = set()  # Validators that must run before this
    provides: set[str] = set()      # Capabilities this validator provides
    priority: int = 100             # Lower = runs earlier within same phase

    @abstractmethod
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Execute validation and return issues."""

    def validate_safe(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
        """Run validation with graceful error handling."""

    def validate_with_timeout(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation with timeout protection."""

    # VE-3: Metric deduplication
    def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
        """Declare metrics this validator needs (for deduplication)."""

    def validate_with_metrics(self, lf: pl.LazyFrame, metric_store: SharedMetricStore) -> list[ValidationIssue]:
        """Run validation using pre-computed metrics from the store."""

    # VE-4: Conditional execution
    def should_skip(self, prior_results: dict[str, ValidatorExecutionResult]) -> tuple[bool, str | None]:
        """Decide whether to skip based on prior results."""

    def get_skip_conditions(self) -> list[SkipCondition]:
        """Declare fine-grained skip conditions."""
```

### Class Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique validator identifier |
| `category` | `str` | Category (completeness, uniqueness, etc.) |
| `dependencies` | `set[str]` | Validators that must run before this |
| `provides` | `set[str]` | Capabilities this validator provides |
| `priority` | `int` | Execution priority (lower = earlier) |

### Methods

#### `validate()`

Execute validation on a LazyFrame.

```python
@abstractmethod
def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
    """
    Execute validation.

    Args:
        lf: Polars LazyFrame to validate

    Returns:
        List of ValidationIssue objects
    """
```

#### `validate_safe()`

Execute validation with error handling.

```python
def validate_safe(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
    """Run validation with graceful error handling."""
```

#### `validate_with_timeout()`

Execute validation with timeout protection.

```python
def validate_with_timeout(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
    """Run validation with timeout protection."""
```

#### `get_required_metrics()` (VE-3)

Declare shared metric dependencies for deduplication across validators.

```python
def get_required_metrics(self, columns: list[str]) -> list[MetricKey]:
    """
    Declare base metrics this validator needs.

    Args:
        columns: Columns this validator will operate on.

    Returns:
        List of MetricKey objects to be pre-computed.
    """
    return []  # Override in subclasses
```

Validators with built-in metric declarations: `NullValidator`, `NotNullValidator`, `CompletenessRatioValidator`, `UniqueValidator`, `UniqueRatioValidator`, `DistinctCountValidator`, `BetweenValidator`.

#### `validate_with_metrics()` (VE-3)

Execute validation using pre-computed metrics from the `SharedMetricStore`.

```python
def validate_with_metrics(
    self,
    lf: pl.LazyFrame,
    metric_store: SharedMetricStore,
) -> list[ValidationIssue]:
    """
    Run validation using pre-computed metrics from the store.
    Default: delegates to validate().
    """
    return self.validate(lf)
```

#### `should_skip()` (VE-4)

Decide whether to skip this validator based on prior results.

```python
def should_skip(
    self,
    prior_results: dict[str, ValidatorExecutionResult],
) -> tuple[bool, str | None]:
    """
    Returns:
        (should_skip, reason) — True when this validator should be skipped.
    """
```

The method checks two conditions:
1. Whether any declared `dependencies` ended in FAILED/TIMEOUT/SKIPPED status.
2. Whether any `get_skip_conditions()` evaluate to True against prior results.

#### `get_skip_conditions()` (VE-4)

Declare fine-grained skip conditions beyond basic dependency failure.

```python
def get_skip_conditions(self) -> list[SkipCondition]:
    """
    Returns:
        List of SkipCondition objects.

    Example:
        return [
            SkipCondition(depends_on="schema_check", skip_when="failed"),
            SkipCondition(depends_on="null_check", skip_when="critical"),
        ]
    """
    return []
```

---

## ValidationIssue

Represents a single validation issue.

### Definition

```python
from truthound.validators.base import ValidationIssue

@dataclass
class ValidationIssue:
    """Represents a single data quality issue found during validation."""

    # Core fields (always populated)
    column: str
    issue_type: str
    count: int
    severity: Severity

    # Legacy detail fields (backward compatible)
    details: str | None = None
    expected: Any | None = None
    actual: Any | None = None
    sample_values: list[Any] | None = None

    # VE-2: Structured validation result
    result: ValidationDetail | None = None
    validator_name: str | None = None
    success: bool = False

    # VE-5: Exception context
    exception_info: ExceptionInfo | None = None
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `column` | `str` | Column name (or "*" for table-level) |
| `issue_type` | `str` | Type of issue (e.g., "null_values", "out_of_range") |
| `count` | `int` | Number of affected rows |
| `severity` | `Severity` | Issue severity |
| `details` | `str \| None` | Human-readable description |
| `expected` | `Any \| None` | Expected value or constraint |
| `actual` | `Any \| None` | Actual value found |
| `sample_values` | `list[Any] \| None` | Sample of problematic values |
| `result` | `ValidationDetail \| None` | Structured result detail (VE-2) |
| `validator_name` | `str \| None` | Name of the validator that produced this issue |
| `success` | `bool` | Whether the validation passed |
| `exception_info` | `ExceptionInfo \| None` | Exception context if validation failed (VE-5) |

### Convenience Properties (VE-2)

```python
@property
def unexpected_percent(self) -> float | None:
    """Delegates to result.unexpected_percent."""

@property
def unexpected_rows(self) -> pl.DataFrame | None:
    """Delegates to result.unexpected_rows."""

@property
def debug_query(self) -> str | None:
    """Delegates to result.debug_query."""
```

### Methods

```python
def to_dict(self) -> dict:
    """Convert to dictionary for JSON serialization.
    Includes result and exception_info when present."""
```

---

## Report

Container for validation results.

### Definition

```python
from truthound.report import Report

@dataclass
class Report:
    """Validation report containing all issues found."""

    issues: list[ValidationIssue]
    source: str = "unknown"
    row_count: int = 0
    column_count: int = 0
```

### Properties and Methods

| Property/Method | Type | Description |
|-----------------|------|-------------|
| `issues` | `list[ValidationIssue]` | All found issues |
| `source` | `str` | Data source name |
| `row_count` | `int` | Total row count |
| `column_count` | `int` | Total column count |

### Methods

```python
def add_issue(self, issue: ValidationIssue) -> None:
    """Add an issue maintaining heap property."""

def add_issues(self, issues: list[ValidationIssue]) -> None:
    """Add multiple issues efficiently."""

def get_sorted_issues(self) -> list[ValidationIssue]:
    """Get issues sorted by severity (highest first)."""

def get_top_issues(self, k: int) -> list[ValidationIssue]:
    """Get top k issues by severity efficiently."""

def get_most_severe(self) -> ValidationIssue | None:
    """Get the most severe issue in O(1) time."""

def iter_by_severity(self) -> Iterator[ValidationIssue]:
    """Iterate through issues in severity order."""

def filter_by_severity(self, min_severity: Severity) -> Report:
    """Return a new report with only issues at or above the given severity."""

def to_dict(self) -> dict:
    """Convert report to dictionary for JSON serialization."""

def to_json(self, indent: int = 2) -> str:
    """Convert report to JSON string."""

def print(self) -> None:
    """Print the report to stdout."""
```

### Properties

```python
@property
def has_issues(self) -> bool:
    """Check if the report contains any issues."""

@property
def has_critical(self) -> bool:
    """Check if the report contains critical issues."""

@property
def has_high(self) -> bool:
    """Check if the report contains high or critical severity issues."""
```

---

## Severity Enum

Severity levels indicate the importance of validation issues.

```python
from truthound.types import Severity

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### Severity Guidelines

| Severity | Use For | Default Threshold |
|----------|---------|-------------------|
| `LOW` | Minor issues, informational | < 5% affected rows |
| `MEDIUM` | Issues that should be addressed | 5-20% affected rows |
| `HIGH` | Significant issues affecting data quality | 20-50% affected rows |
| `CRITICAL` | Blocking issues, data unusable | > 50% affected rows |

!!! info "No INFO Severity Level"
    Truthound does not have an `INFO` severity level. The lowest level is `LOW`.

### Comparison Support

```python
# Severity supports comparison operators (custom __ge__, __gt__, __le__, __lt__)
Severity.HIGH > Severity.LOW      # True
Severity.MEDIUM >= Severity.LOW   # True
Severity.CRITICAL > Severity.HIGH # True
Severity.LOW < Severity.MEDIUM    # True
```

---

## ValidatorConfig

Immutable configuration for validators.

### Definition

```python
from truthound.validators.base import ValidatorConfig

@dataclass(frozen=True)
class ValidatorConfig:
    """Immutable configuration for validators."""

    columns: tuple[str, ...] | None = None
    exclude_columns: tuple[str, ...] | None = None
    severity_override: Severity | None = None
    sample_size: int = 5
    mostly: float | None = None  # Fraction of rows that must pass
    timeout_seconds: float | None = 300.0
    graceful_degradation: bool = True
    log_errors: bool = True

    # VE-1: Result format
    result_format: ResultFormat | ResultFormatConfig = ResultFormat.SUMMARY

    # VE-5: Exception isolation
    catch_exceptions: bool = True          # False = abort on first error
    max_retries: int = 0                   # Retry count for transient errors
    partial_failure_mode: str = "collect"   # "collect" | "skip" | "raise"
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `columns` | `tuple[str, ...]` | `None` | Columns to validate |
| `exclude_columns` | `tuple[str, ...]` | `None` | Columns to exclude |
| `severity_override` | `Severity` | `None` | Override calculated severity |
| `sample_size` | `int` | `5` | Number of sample values to collect |
| `mostly` | `float` | `None` | Pass if this fraction of rows pass (0.0-1.0) |
| `timeout_seconds` | `float` | `300.0` | Timeout for validation |
| `graceful_degradation` | `bool` | `True` | Skip on error instead of failing |
| `log_errors` | `bool` | `True` | Log errors when they occur |
| `result_format` | `ResultFormat \| ResultFormatConfig` | `SUMMARY` | Detail level for results (VE-1) |
| `catch_exceptions` | `bool` | `True` | Isolate exceptions instead of aborting (VE-5) |
| `max_retries` | `int` | `0` | Retry count for transient errors (VE-5) |
| `partial_failure_mode` | `str` | `"collect"` | How to handle partial failures: `"collect"` (gather partial results), `"skip"` (discard), `"raise"` (re-raise) (VE-5) |

---

## Creating Custom Validators

### Simple Example

```python
from truthound.validators.base import Validator, ValidationIssue, ValidatorConfig
from truthound.types import Severity
import polars as pl

class PositiveValidator(Validator):
    """Check that numeric values are positive."""

    name = "positive_check"
    category = "range"

    def __init__(self, config: ValidatorConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues = []
        df = lf.collect()

        # Get numeric columns to validate
        cols = self._get_target_columns(lf, dtype_filter=NUMERIC_TYPES)

        for col in cols:
            negative_count = df.filter(pl.col(col) < 0).height
            if negative_count > 0:
                total = df.height
                ratio = negative_count / total

                issues.append(ValidationIssue(
                    column=col,
                    issue_type="negative_values",
                    count=negative_count,
                    severity=self._calculate_severity(ratio),
                    details=f"{negative_count} negative values ({ratio:.1%})",
                ))

        return issues
```

### Using with th.check()

```python
import truthound as th
from my_validators import PositiveValidator

# Use custom validator instance
validator = PositiveValidator(columns=("amount", "quantity"))
report = th.check("data.csv", validators=[validator])

# Or register and use by name
from truthound.validators import BUILTIN_VALIDATORS
BUILTIN_VALIDATORS["positive"] = PositiveValidator

report = th.check("data.csv", validators=["positive"])
```

### Using the @validator Decorator

The `@validator` decorator provides a simple way to create custom validators from
value-level validation functions. The function should take a single value and return
`True` if valid, `False` if invalid.

```python
import truthound as th

@th.validator
def is_positive(value: int) -> bool:
    """Check if a value is positive."""
    return value > 0

@th.validator
def is_valid_email_domain(value: str) -> bool:
    """Check if email has valid domain."""
    return value.endswith("@company.com")

# Use with th.check()
report = th.check("data.csv", validators=[is_positive, is_valid_email_domain])
```

The decorator:

- Wraps your function in a `CustomValidator` class
- Automatically applies validation to all applicable columns
- Calculates severity based on failure rate (>30% = HIGH, >10% = MEDIUM, else LOW)
- Uses the function name as the validator name

For more complex validation logic that requires access to the full DataFrame
or LazyFrame, subclass `Validator` directly instead (see [Simple Example](#simple-example) above).

---

## Built-in Validators

### Completeness

| Validator | Description |
|-----------|-------------|
| `NullValidator` | Check for null values |
| `NotNullValidator` | Ensure no nulls |
| `CompletenessRatioValidator` | Check completeness percentage |

### Uniqueness

| Validator | Description |
|-----------|-------------|
| `DuplicateValidator` | Check for duplicate rows |
| `UniqueValidator` | Ensure column uniqueness |
| `PrimaryKeyValidator` | Validate primary key constraints |

### Range

| Validator | Description |
|-----------|-------------|
| `RangeValidator` | Check numeric ranges |
| `BetweenValidator` | Check value bounds |
| `PositiveValidator` | Ensure positive values |
| `NonNegativeValidator` | Ensure non-negative values |

### Format

| Validator | Description |
|-----------|-------------|
| `EmailValidator` | Validate email format |
| `PhoneValidator` | Validate phone format |
| `URLValidator` | Validate URL format |
| `RegexValidator` | Custom regex pattern |

### Schema

| Validator | Description |
|-----------|-------------|
| `TypeValidator` | Validate data types |
| `ColumnExistsValidator` | Check column presence |
| `SchemaValidator` | Full schema validation |

For complete validator documentation (289 validators), see [Validators Guide](../guides/validators.md).

---

## Error Handling

### ValidationTimeoutError

Raised when validation exceeds the configured timeout.

```python
from truthound.validators.base import ValidationTimeoutError

try:
    issues = validator.validate_with_timeout(lf)
except ValidationTimeoutError as e:
    print(f"Timed out after {e.timeout_seconds}s")
```

### ColumnNotFoundError

Raised when a required column is not found.

```python
from truthound.validators.base import ColumnNotFoundError

try:
    issues = validator.validate(lf)
except ColumnNotFoundError as e:
    print(f"Column '{e.column}' not found")
    print(f"Available: {e.available_columns}")
```

### ValidatorExecutionResult

Result of validation with error handling.

```python
from truthound.validators.base import ValidatorExecutionResult, ValidationResult

result = validator.validate_safe(lf)

if result.status == ValidationResult.SUCCESS:
    print(f"Found {len(result.issues)} issues")
elif result.status == ValidationResult.TIMEOUT:
    print(f"Timed out: {result.error_message}")
elif result.status == ValidationResult.SKIPPED:
    print(f"Skipped: {result.error_message}")
elif result.status == ValidationResult.FAILED:
    print(f"Failed: {result.error_message}")

# VE-5: Extended fields
if result.exception_info:
    print(f"Category: {result.exception_info.failure_category}")
    print(f"Retries: {result.retry_count}")
if result.partial_issues:
    print(f"Partial results collected: {len(result.partial_issues)}")
```

---

## SkipCondition (VE-4)

Declares when a Validator should be skipped based on prior results.

```python
from truthound.validators.base import SkipCondition

@dataclass(frozen=True)
class SkipCondition:
    depends_on: str                          # Upstream validator name
    skip_when: str = "failed"                # "failed" | "critical" | "any_issue"
    reason_template: str = "Skipped due to {depends_on} {skip_when}"
```

**skip_when Modes:**

| Mode | Skip When |
|------|-----------|
| `"failed"` | Upstream ended FAILED or TIMEOUT |
| `"critical"` | Upstream produced CRITICAL-severity issue |
| `"any_issue"` | Upstream produced any issue |

**Usage:**

```python
class RangeValidator(Validator):
    def get_skip_conditions(self) -> list[SkipCondition]:
        return [
            SkipCondition(depends_on="schema_check", skip_when="failed"),
            SkipCondition(depends_on="null_check", skip_when="critical"),
        ]
```

---

## ExceptionInfo (VE-5)

Detailed exception information with automatic classification.

```python
from truthound.validators.base import ExceptionInfo

@dataclass
class ExceptionInfo:
    raised_exception: bool = False
    exception_type: str | None = None
    exception_message: str | None = None
    exception_traceback: str | None = None
    retry_count: int = 0
    max_retries: int = 0
    is_retryable: bool = False
    validator_name: str | None = None
    column: str | None = None
    expression_alias: str | None = None
    failure_category: str = "unknown"
```

**Exception Classification:**

| Category | Exception Types |
|----------|----------------|
| `transient` | `TimeoutError`, `ConnectionError`, `OSError`, `ValidationTimeoutError` |
| `configuration` | `ValueError`, `TypeError`, `KeyError`, `ColumnNotFoundError` |
| `data` | Polars `ComputeError`, `SchemaError` |
| `permanent` | All other exceptions |

**Factory:**

```python
info = ExceptionInfo.from_exception(
    exc=some_error,
    validator_name="null_check",
    column="email",
)
print(info.failure_category)  # e.g., "transient"
print(info.is_retryable)      # True for transient errors
```

## See Also

- [Validators Guide](../guides/validators.md) - All 289 validators
- [Custom Validator Tutorial](../tutorials/custom-validator.md) - Step-by-step guide
- [th.check()](core-functions.md#thcheck) - Using validators
