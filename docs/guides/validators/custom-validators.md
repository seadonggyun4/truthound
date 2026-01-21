# 커스텀 검증기 개발 가이드

이 문서는 Truthound SDK를 사용하여 커스텀 검증기를 개발하는 방법을 설명합니다.

## 개요

Truthound SDK는 세 가지 방식으로 커스텀 검증기를 생성할 수 있습니다:

1. **데코레이터 방식** - 클래스에 메타데이터를 추가하고 레지스트리에 등록
2. **빌더 패턴** - 서브클래싱 없이 플루언트 API로 검증기 생성
3. **템플릿 상속** - 미리 정의된 템플릿 클래스를 상속하여 구현

---

## 1. 데코레이터 방식

### @custom_validator

가장 일반적인 커스텀 검증기 생성 방식입니다.

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
    auto_register=True,  # 기본값: True
)
class PercentageValidator(Validator, NumericValidatorMixin):
    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        issues = []
        for col in self._get_numeric_columns(lf):
            # 0-100 범위 검사 로직
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

#### 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `name` | `str` | (필수) | 고유한 검증기 이름 |
| `category` | `str` | `"custom"` | 검증기 카테고리 |
| `description` | `str` | `""` | 사람이 읽을 수 있는 설명 |
| `version` | `str` | `"1.0.0"` | 시맨틱 버전 |
| `author` | `str` | `""` | 작성자 이름/이메일 |
| `tags` | `list[str]` | `None` | 필터링/검색용 태그 |
| `examples` | `list[str]` | `None` | 문서화용 사용 예시 |
| `config_schema` | `dict` | `None` | 설정 JSON 스키마 |
| `auto_register` | `bool` | `True` | 자동 레지스트리 등록 여부 |

### @register_validator

기존에 정의된 검증기 클래스를 레지스트리에 등록합니다.

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

기존 검증기에 상세 메타데이터를 추가합니다.

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

검증기를 폐기 예정으로 표시합니다.

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

인스턴스화 시 `DeprecationWarning`이 발생합니다:
```
DeprecationWarning: Validator 'email_v1' is deprecated. Use 'email_v2' for RFC 5322 compliance. Use 'email_v2' instead. Will be removed in version 2.0.0.
```

---

## 2. 레지스트리 API

등록된 검증기를 조회하고 관리하는 함수들입니다.

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

### 조회 함수

| 함수 | 반환 타입 | 설명 |
|------|-----------|------|
| `get_registered_validators()` | `dict[str, type]` | 모든 등록된 검증기 |
| `get_validator_by_name(name)` | `type \| None` | 이름으로 검증기 클래스 조회 |
| `get_validator_metadata(name)` | `ValidatorMeta \| None` | 검증기 메타데이터 조회 |
| `get_validators_by_category(category)` | `list[type]` | 카테고리별 검증기 목록 |
| `get_validators_by_tag(tag)` | `list[type]` | 태그별 검증기 목록 |
| `list_validator_categories()` | `list[str]` | 모든 카테고리 목록 |
| `list_validator_tags()` | `list[str]` | 모든 태그 목록 |

### 관리 함수

| 함수 | 반환 타입 | 설명 |
|------|-----------|------|
| `unregister_validator(name)` | `bool` | 검증기 등록 해제 |
| `clear_registry()` | `None` | 모든 검증기 등록 해제 (테스트용) |

### ValidatorMeta 데이터클래스

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

## 3. 빌더 패턴

서브클래싱 없이 플루언트 API로 검증기를 생성합니다.

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

# 사용
issues = validator.validate(lf)
```

#### 메서드 체인

| 메서드 | 설명 |
|--------|------|
| `category(name)` | 카테고리 설정 |
| `description(text)` | 설명 설정 |
| `for_columns(dtype_filter)` | 데이터 타입 필터 설정 |
| `for_numeric_columns()` | 숫자 컬럼만 대상 |
| `for_string_columns()` | 문자열 컬럼만 대상 |
| `for_datetime_columns()` | 날짜/시간 컬럼만 대상 |
| `for_float_columns()` | 실수 컬럼만 대상 |
| `check_column(fn)` / `check(fn)` | 검사 함수 추가 (col, lf) -> count |
| `with_issue_type(type)` | 이슈 타입 설정 |
| `with_severity(severity)` | 심각도 설정 |
| `with_message(template)` | 메시지 템플릿 설정 ({column}, {count}) |
| `with_samples(fn)` | 샘플 수집 함수 설정 |
| `with_config(config)` | ValidatorConfig 설정 |
| `build()` | Validator 인스턴스 생성 |

### ColumnCheckBuilder

개별 컬럼 검사를 정의합니다.

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

집계 수준 검사를 정의합니다.

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

### 편의 함수

#### simple_column_validator

한 줄로 간단한 컬럼 검증기 생성:

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
    dtype_filter=None,  # 모든 타입
)
```

#### simple_expression_validator

Polars 표현식으로 검증기 생성:

```python
from truthound.validators.sdk import simple_expression_validator
from truthound.types import Severity
import polars as pl

validator = simple_expression_validator(
    name="positive_values",
    violation_expr=pl.col("amount") <= 0,  # True = 위반
    issue_type="non_positive",
    severity=Severity.HIGH,
    category="numeric",
    columns=["amount", "quantity"],  # 특정 컬럼만
)
```

---

## 4. 템플릿 클래스

일반적인 패턴에 대한 추상 템플릿 클래스를 제공합니다.

### SimpleColumnValidator

컬럼별 검사의 기본 템플릿:

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
    dtype_filter = NUMERIC_TYPES  # 숫자 컬럼만

    def check_column_values(self, lf: pl.LazyFrame, col: str) -> int:
        """위반 개수 반환 (필수 구현)"""
        return lf.filter(pl.col(col) <= 0).select(pl.len()).collect().item()

    def get_violation_samples(self, lf: pl.LazyFrame, col: str) -> list | None:
        """위반 샘플 반환 (선택적)"""
        return (
            lf.filter(pl.col(col) <= 0)
            .select(col)
            .head(5)
            .collect()
            .to_series()
            .to_list()
        )

    def get_issue_details(self, col: str, count: int, total: int) -> str:
        """이슈 상세 메시지 (선택적)"""
        pct = (count / total * 100) if total > 0 else 0
        return f"Found {count} non-positive values ({pct:.1f}%)"
```

#### 클래스 속성

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `name` | `str` | `"simple_column"` | 검증기 이름 |
| `category` | `str` | `"custom"` | 카테고리 |
| `issue_type` | `str` | `"validation_failed"` | 이슈 타입 |
| `default_severity` | `Severity` | `MEDIUM` | 기본 심각도 |
| `dtype_filter` | `set[type] \| None` | `None` | 데이터 타입 필터 |

### SimplePatternValidator

정규식 기반 문자열 검증:

```python
from truthound.validators.sdk import SimplePatternValidator
from truthound.types import Severity

class EmailValidator(SimplePatternValidator):
    name = "email"
    category = "string"
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    issue_type = "invalid_email"
    match_full = True  # 전체 문자열 매칭
    case_sensitive = True

class NoSSNValidator(SimplePatternValidator):
    name = "no_ssn"
    category = "privacy"
    pattern = r"\d{3}-\d{2}-\d{4}"
    invert_match = True  # 패턴이 있으면 위반
    issue_type = "contains_ssn"
```

#### 클래스 속성

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `pattern` | `str` | `""` (필수) | 정규식 패턴 |
| `match_full` | `bool` | `True` | 전체 문자열 매칭 여부 |
| `invert_match` | `bool` | `False` | True면 매칭되는 값이 위반 |
| `case_sensitive` | `bool` | `True` | 대소문자 구분 |

### SimpleRangeValidator

숫자 범위 검증:

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
    inclusive_min = False  # 0 제외 (> 0)
    issue_type = "non_positive"

class AgeValidator(SimpleRangeValidator):
    name = "valid_age"
    min_value = 0
    max_value = 150
    issue_type = "invalid_age"
```

#### 클래스 속성

| 속성 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `min_value` | `float \| int \| None` | `None` | 최솟값 |
| `max_value` | `float \| int \| None` | `None` | 최댓값 |
| `inclusive_min` | `bool` | `True` | 최솟값 포함 여부 |
| `inclusive_max` | `bool` | `True` | 최댓값 포함 여부 |

### SimpleComparisonValidator

컬럼 간 비교 검증:

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
    operator = "eq"  # 같아야 함
    issue_type = "amount_mismatch"
```

#### 지원 연산자

| 연산자 | 기호 | 의미 |
|--------|------|------|
| `"eq"` | `==` | 같음 |
| `"ne"` | `!=` | 다름 |
| `"lt"` | `<` | 미만 |
| `"le"` | `<=` | 이하 |
| `"gt"` | `>` | 초과 |
| `"ge"` | `>=` | 이상 |

### CompositeValidator

여러 검증기를 조합:

```python
from truthound.validators.sdk import CompositeValidator
from truthound.validators import NullValidator, UniqueValidator, RangeValidator

# 방법 1: 클래스 상속
class CustomerDataValidator(CompositeValidator):
    name = "customer_data"
    category = "business"

    def get_validators(self) -> list[Validator]:
        return [
            EmailValidator(columns=("email",)),
            PhoneValidator(columns=("phone",)),
            AgeValidator(columns=("age",)),
        ]

# 방법 2: 인라인 생성
composite = CompositeValidator(
    validators=[
        NullValidator(columns=("id", "name")),
        UniqueValidator(columns=("id",)),
        RangeValidator(columns=("age",), min_value=0, max_value=150),
    ]
)

# 방법 3: 빌더 스타일
composite = CompositeValidator()
composite.add_validator(NullValidator())
composite.add_validator(UniqueValidator(columns=("id",)))
```

---

## 5. 팩토리 함수

클래스를 동적으로 생성하는 팩토리 함수입니다.

### create_pattern_validator

```python
from truthound.validators.sdk import create_pattern_validator

# 클래스 생성
EmailValidator = create_pattern_validator(
    name="email",
    pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    issue_type="invalid_email",
    invert=False,  # 매칭 안 되면 위반
    case_sensitive=True,
)

# 인스턴스 생성 및 사용
validator = EmailValidator()
issues = validator.validate(lf)
```

### create_range_validator

```python
from truthound.validators.sdk import create_range_validator

# 클래스 생성
PercentageValidator = create_range_validator(
    name="percentage",
    min_value=0,
    max_value=100,
    issue_type="invalid_percentage",
    inclusive=True,  # 경계값 포함
)

# 인스턴스 생성 및 사용
validator = PercentageValidator()
issues = validator.validate(lf)
```

---

## 6. 테스트 프레임워크

### ValidatorTestCase

unittest 기반 테스트 베이스 클래스:

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

#### 테스트 메서드

| 메서드 | 설명 |
|--------|------|
| `create_validator(**kwargs)` | 검증기 인스턴스 생성 |
| `create_df(data)` | 테스트 LazyFrame 생성 |
| `create_large_df(rows, schema, seed)` | 대용량 테스트 데이터 생성 |
| `validate(lf, validator)` | 검증 실행 |
| `validate_safe(lf, validator)` | 에러 핸들링 포함 검증 |

#### 어설션 메서드

| 메서드 | 설명 |
|--------|------|
| `assert_no_issues()` | 이슈 없음 확인 |
| `assert_has_issue(column, issue_type, min_count, exact_count, severity)` | 특정 이슈 존재 확인 |
| `assert_issue_count(expected)` | 이슈 개수 확인 |
| `assert_total_violations(expected)` | 총 위반 수 확인 |
| `assert_no_error()` | 에러 없음 확인 |
| `assert_error(error_type)` | 특정 에러 발생 확인 |
| `assert_performance(max_ms, rows)` | 성능 기준 충족 확인 |

### 테스트 데이터 생성

```python
from truthound.validators.sdk import create_test_dataframe, create_edge_case_data

# 기본 테스트 데이터
df = create_test_dataframe(rows=1000, include_nulls=True)

# 명시적 데이터
df = create_test_dataframe(data={"col1": [1, 2, 3]})

# 엣지 케이스 모음
edge_cases = create_edge_case_data()
# 반환: {
#   "empty": 빈 DataFrame,
#   "single_row": 1행 DataFrame,
#   "all_nulls": 모든 값이 null,
#   "uniform_values": 모든 값 동일,
#   "large_values": 매우 큰 숫자,
#   "small_values": 매우 작은 숫자,
#   "unicode": 유니코드 문자열,
#   "empty_strings": 빈 문자열,
#   "whitespace": 공백 문자,
#   "special_floats": inf, -inf, nan, 0.0, -0.0,
# }
```

### 독립 어설션 함수

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

### 성능 벤치마킹

```python
from truthound.validators.sdk import benchmark_validator, ValidatorBenchmark

# 단일 검증기 벤치마킹
result = benchmark_validator(
    validator=PositiveValidator(),
    lf=large_dataframe,
    iterations=10,
    warmup=2,
)
print(f"Mean: {result.mean_ms:.2f}ms")
print(f"Throughput: {result.throughput_rows_per_sec:,.0f} rows/sec")

# 여러 검증기 비교
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

## 7. CLI 스캐폴딩

`th new validator` 명령어로 검증기 템플릿을 생성할 수 있습니다.

```bash
# 기본 템플릿 (7종 선택)
th new validator my_validator

# 특정 템플릿 지정
th new validator email_format --template pattern

# 즉시 설치 (editable mode)
th new validator my_validator --install
```

### 사용 가능한 템플릿

| 템플릿 | 설명 |
|--------|------|
| `basic` | 기본 Validator 서브클래스 |
| `column` | SimpleColumnValidator 상속 |
| `pattern` | SimplePatternValidator 상속 |
| `range` | SimpleRangeValidator 상속 |
| `comparison` | SimpleComparisonValidator 상속 |
| `composite` | CompositeValidator 상속 |
| `ml` | ML 기반 검증기 (이상 탐지) |

생성되는 파일 구조:

```
my_validator/
├── pyproject.toml         # 패키지 메타데이터
├── README.md
├── src/
│   └── my_validator/
│       ├── __init__.py
│       └── validator.py   # 검증기 구현
└── tests/
    └── test_validator.py  # 테스트 코드
```

---

## 다음 단계

- [엔터프라이즈 SDK](enterprise-sdk.md) - 샌드박스, 코드 서명, 라이선스 관리
- [보안 가이드](security.md) - ReDoS 보호, SQL 인젝션 방지
- [내장 검증기](built-in.md) - 289개 내장 검증기 참조
