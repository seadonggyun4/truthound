# 검증기

Python API 사용에서 Validator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 검증기 Base Class

Python API 사용에서 `Validator`, Validator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

| Python API 사용에서 Attribute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------|-------------|
| Python API 사용에서 `name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Unique 검증기 identifier |
| Python API 사용에서 `category`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `dependencies`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `set[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Validators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `provides`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `set[str]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Capabilities this 검증기 provides |
| Python API 사용에서 `priority`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Execution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Methods

#### `validate()`

Execute 검증 on a LazyFrame.

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

Python API 사용에서 Execute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
def validate_safe(self, lf: pl.LazyFrame) -> ValidatorExecutionResult:
    """Run validation with graceful error handling."""
```

#### `validate_with_timeout()`

Python API 사용에서 Execute을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
def validate_with_timeout(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
    """Run validation with timeout protection."""
```

#### `get_required_metrics()` (VE-3)

Python API 사용에서 Declare을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Python API 사용에서 `NullValidator`, `NotNullValidator`, `CompletenessRatioValidator`, `UniqueValidator`, `UniqueRatioValidator`, `DistinctCountValidator`, `BetweenValidator`, Validators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

#### `validate_with_metrics()` (VE-3)

Python API 사용에서 `SharedMetricStore`, Execute, SharedMetricStore을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Python API 사용에서 Decide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:
1. Python API 사용에서 `dependencies`, Whether, FAILED/TIMEOUT/SKIPPED을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
2. Python API 사용에서 `get_skip_conditions()`, Whether, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

#### `get_skip_conditions()` (VE-4)

Python API 사용에서 Declare을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 검증Issue

Represents a single 검증 issue.

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

| Python API 사용에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|------|-------------|
| Python API 사용에서 `column`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name (or "*" for 테이블-level) |
| Python API 사용에서 `issue_type`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `count`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `severity`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Severity`, Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Issue을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `details`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Human-readable을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `expected`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Expected을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `actual`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Actual을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `sample_values`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Any을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Sample을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `result`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | `검증Detail \ | Python API 사용에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Structured 결과 detail (VE-2) |
| Python API 사용에서 `validator_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Name을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `success`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Whether the 검증 passed |
| Python API 사용에서 `exception_info`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ExceptionInfo을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Exception, VE-5을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

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

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 검증RunResult 통합

Python API 사용에서 ValidationRunResult, Validators을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
런타임 output returned by `th.check()`.

### Definition

```python
from truthound.core.results import CheckResult, ValidationRunResult

@dataclass(frozen=True)
class ValidationRunResult:
    suite_name: str
    source: str
    row_count: int
    column_count: int
    checks: tuple[CheckResult, ...]
    issues: tuple[ValidationIssue, ...]
    execution_issues: tuple[ExecutionIssue, ...]
    metadata: dict[str, Any]
```

### What 검증기 Contribute

| Python API 사용에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|-------------|
| Python API 사용에서 `checks`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Per-check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `issues`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Flattened issues emitted by 검증기 |
| Python API 사용에서 `execution_issues`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `catch_exceptions=True`, Exceptions, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `metadata`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Planner을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Usage

```python
import truthound as th
from truthound.types import Severity

run = th.check("data.csv", validators=["null", "duplicate"])

print(run.source)
print(len(run.checks))
print(len(run.issues))

for issue in run.issues:
    print(issue.validator_name, issue.column, issue.issue_type)

critical_only = run.filter_by_severity(Severity.CRITICAL)
print(len(critical_only.issues))
```

### CheckResult

```python
@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str = "general"
    success: bool = True
    issue_count: int = 0
    issues: tuple[ValidationIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Severity Enum

Python API 사용에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.types import Severity

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### Severity Guidelines

| Python API 사용에서 Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default, Threshold을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|---------|-------------------|
| Python API 사용에서 `LOW`, LOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Minor을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `MEDIUM`, MEDIUM을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Issues을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `HIGH`, HIGH을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Significant issues affecting 데이터 품질 | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `CRITICAL`, CRITICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Blocking을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

!!! info "참고"
Python API 사용에서 Truthound, `INFO`, `LOW`, INFO, LOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Comparison Support

```python
# Severity supports comparison operators (custom __ge__, __gt__, __le__, __lt__)
Severity.HIGH > Severity.LOW      # True
Severity.MEDIUM >= Severity.LOW   # True
Severity.CRITICAL > Severity.HIGH # True
Severity.LOW < Severity.MEDIUM    # True
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## ValidatorConfig

Immutable 설정 for 검증기.

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

| Python API 사용에서 Field을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Default을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-------|------|---------|-------------|
| Python API 사용에서 `columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `tuple[str, ...]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 to validate |
| Python API 사용에서 `exclude_columns`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `tuple[str, ...]`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 to exclude |
| Python API 사용에서 `severity_override`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `Severity`, Severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Override을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `sample_size`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `5`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Number을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `mostly`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `None`, None을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Pass을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `timeout_seconds`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `float`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `300.0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Timeout for 검증 |
| Python API 사용에서 `graceful_degradation`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Skip을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `log_errors`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Log을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `result_format`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ResultFormat을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 ResultFormatConfig을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `SUMMARY`, SUMMARY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Detail, VE-1을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `catch_exceptions`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `bool`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `True`, True을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Isolate, VE-5을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `max_retries`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `int`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `0`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Retry, VE-5을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `partial_failure_mode`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `str`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `"collect"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `"collect"`, `"skip"`, `"raise"`, VE-5을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Creating 사용자 정의 검증기s

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

### Using the @검증기 Decorator

Python API 사용에서 `@validator`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 `True`, `False`, True, False을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Python API 사용에서 관련 설정과 실행 흐름을(를) 다루는 항목입니다:

- Python API 사용에서 `CustomValidator`, Wraps, CustomValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Automatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Calculates, HIGH, MEDIUM, LOW을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 DataFrame을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
Python API 사용에서 `Validator`, LazyFrame, Validator, Simple, Example을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Built-in 검증기

### Completeness

| 검증기 | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| Python API 사용에서 `NullValidator`, NullValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `NotNullValidator`, NotNullValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Ensure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `CompletenessRatioValidator`, CompletenessRatioValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Uniqueness

| 검증기 | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| Python API 사용에서 `DuplicateValidator`, DuplicateValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `UniqueValidator`, UniqueValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Ensure 컬럼 uniqueness |
| Python API 사용에서 `PrimaryKeyValidator`, PrimaryKeyValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Range

| 검증기 | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| Python API 사용에서 `RangeValidator`, RangeValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `BetweenValidator`, BetweenValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Check을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `PositiveValidator`, PositiveValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Ensure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `NonNegativeValidator`, NonNegativeValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Ensure을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Format

| 검증기 | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| Python API 사용에서 `EmailValidator`, EmailValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `PhoneValidator`, PhoneValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `URLValidator`, URLValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Validate, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `RegexValidator`, RegexValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 스키마

| 검증기 | Python API 사용에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| Python API 사용에서 `TypeValidator`, TypeValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Validate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `ColumnExistsValidator`, ColumnExistsValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Check 컬럼 presence |
| Python API 사용에서 `SchemaValidator`, SchemaValidator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Full 스키마 검증 |

Python API 사용에서 Validators, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Error Handling

### 검증TimeoutError

Python API 사용에서 Raised을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.validators.base import ValidationTimeoutError

try:
    issues = validator.validate_with_timeout(lf)
except ValidationTimeoutError as e:
    print(f"Timed out after {e.timeout_seconds}s")
```

### ColumnNotFoundError

Python API 사용에서 Raised을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.validators.base import ColumnNotFoundError

try:
    issues = validator.validate(lf)
except ColumnNotFoundError as e:
    print(f"Column '{e.column}' not found")
    print(f"Available: {e.available_columns}")
```

### ValidatorExecutionResult

결과 of 검증 with error handling.

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

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## SkipCondition (VE-4)

Python API 사용에서 Declares, Validator을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.validators.base import SkipCondition

@dataclass(frozen=True)
class SkipCondition:
    depends_on: str                          # Upstream validator name
    skip_when: str = "failed"                # "failed" | "critical" | "any_issue"
    reason_template: str = "Skipped due to {depends_on} {skip_when}"
```

Python API 사용에서 Modes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| Python API 사용에서 Mode을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Skip을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-----------|
| Python API 사용에서 `"failed"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Upstream, FAILED, TIMEOUT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `"critical"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Upstream, CRITICAL-severity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `"any_issue"`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Upstream을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

Python API 사용에서 Usage을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class RangeValidator(Validator):
    def get_skip_conditions(self) -> list[SkipCondition]:
        return [
            SkipCondition(depends_on="schema_check", skip_when="failed"),
            SkipCondition(depends_on="null_check", skip_when="critical"),
        ]
```

Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## ExceptionInfo (VE-5)

Python API 사용에서 Detailed을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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

Python API 사용에서 Exception, Classification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| Python API 사용에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Exception, Types을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------------|
| Python API 사용에서 `transient`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `TimeoutError`, `ConnectionError`, `OSError`, `ValidationTimeoutError`, TimeoutError, ConnectionError, OSError, ValidationTimeoutError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `configuration`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 `ValueError`, `TypeError`, `KeyError`, `ColumnNotFoundError`, ValueError, TypeError, KeyError, ColumnNotFoundError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `data`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 Polars, `ComputeError`, `SchemaError`, ComputeError, SchemaError을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Python API 사용에서 `permanent`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

Python API 사용에서 Factory을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
info = ExceptionInfo.from_exception(
    exc=some_error,
    validator_name="null_check",
    column="email",
)
print(info.failure_category)  # e.g., "transient"
print(info.is_retryable)      # True for transient errors
```

## 함께 보기

- Python API 사용에서 Validators, Guide을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 Custom, Validator, Tutorial, Step-by-step을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- Python API 사용에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
