# Pattern Matching

이 문서는 Polars 네이티브 패턴 매칭 시스템을 설명합니다.

## 개요

`src/truthound/profiler/native_patterns.py`에 구현된 패턴 매칭 시스템은 Polars의 벡터화된 `str.contains()` 연산을 사용하여 고성능으로 패턴을 감지합니다.

## PatternSpec

패턴 정의를 위한 데이터 클래스입니다.

```python
@dataclass
class PatternSpec:
    """패턴 스펙 정의"""

    name: str                      # 패턴 이름 (예: "email", "phone")
    regex: str                     # 정규식 패턴
    data_type: DataType            # 매칭 시 추론할 데이터 타입
    priority: int = 0              # 우선순위 (높을수록 먼저 매칭)
    examples: list[str] = field(default_factory=list)  # 예시 값
    description: str = ""          # 패턴 설명
    category: str = "general"      # 패턴 카테고리
```

## PatternBuilder

Fluent API를 사용한 패턴 정의입니다.

```python
from truthound.profiler.native_patterns import PatternBuilder

# Fluent 스타일로 패턴 생성
pattern = (
    PatternBuilder("korean_mobile")
    .regex(r"^01[0-9]-?[0-9]{3,4}-?[0-9]{4}$")
    .data_type(DataType.KOREAN_PHONE)
    .priority(100)
    .examples(["010-1234-5678", "01012345678"])
    .description("한국 휴대폰 번호")
    .category("korean")
    .build()
)
```

## NativePatternMatcher

Polars 네이티브 연산을 사용하는 패턴 매처입니다.

```python
from truthound.profiler.native_patterns import NativePatternMatcher

# 매처 생성
matcher = NativePatternMatcher()

# 컬럼에서 패턴 매칭
results = matcher.match(lf, "email_column")

for result in results:
    print(f"Pattern: {result.pattern_name}")
    print(f"Match ratio: {result.match_ratio:.2%}")
    print(f"Data type: {result.data_type}")
```

### 내부 구현

```python
class NativePatternMatcher:
    """Polars 네이티브 패턴 매처"""

    def match(self, lf: pl.LazyFrame, column: str) -> list[PatternMatch]:
        """
        Polars의 벡터화된 str.contains()를 사용하여
        고성능 패턴 매칭 수행
        """
        col = pl.col(column)

        for pattern in self._patterns:
            # 네이티브 Polars 연산 (map_elements 없음)
            match_expr = col.str.contains(pattern.regex)
            match_count = match_expr.sum()
            # ...
```

## 내장 패턴

### 일반 패턴

| 패턴 이름 | 데이터 타입 | 설명 |
|-----------|------------|------|
| `email` | `EMAIL` | 이메일 주소 |
| `url` | `URL` | URL/URI |
| `uuid` | `UUID` | UUID (v1-v5) |
| `ip_address` | `IP_ADDRESS` | IPv4/IPv6 |
| `phone` | `PHONE` | 국제 전화번호 |
| `date_iso` | `DATE` | ISO 8601 날짜 |
| `datetime_iso` | `DATETIME` | ISO 8601 날짜시간 |
| `json` | `JSON` | JSON 객체/배열 |
| `currency` | `CURRENCY` | 통화 금액 |
| `percentage` | `PERCENTAGE` | 백분율 |

### 한국어 특화 패턴

| 패턴 이름 | 데이터 타입 | 설명 |
|-----------|------------|------|
| `korean_rrn` | `KOREAN_RRN` | 주민등록번호 |
| `korean_phone` | `KOREAN_PHONE` | 한국 전화번호 |
| `korean_mobile` | `KOREAN_PHONE` | 한국 휴대폰 번호 |
| `korean_business_number` | `KOREAN_BUSINESS_NUMBER` | 사업자등록번호 |

## 패턴 레지스트리

```python
from truthound.profiler.native_patterns import PatternRegistry

# 기본 패턴 조회
email_pattern = PatternRegistry.get("email")

# 카테고리별 패턴 조회
korean_patterns = PatternRegistry.get_by_category("korean")

# 커스텀 패턴 등록
PatternRegistry.register(
    PatternSpec(
        name="custom_id",
        regex=r"^[A-Z]{2}\d{6}$",
        data_type=DataType.IDENTIFIER,
        priority=50,
        examples=["AB123456"],
        description="회사 고유 ID 형식",
    )
)

# 패턴 제거
PatternRegistry.unregister("custom_id")
```

## PatternMatch 결과

```python
@dataclass
class PatternMatch:
    """패턴 매칭 결과"""

    pattern_name: str       # 매칭된 패턴 이름
    regex: str              # 사용된 정규식
    data_type: DataType     # 추론된 데이터 타입
    match_count: int        # 매칭된 행 수
    total_count: int        # 전체 행 수 (null 제외)
    match_ratio: float      # 매칭 비율 (0.0-1.0)
    confidence: float       # 신뢰도
    sample_matches: list[str]  # 매칭된 값 샘플
```

## 우선순위 기반 매칭

여러 패턴이 매칭될 경우 우선순위에 따라 결과를 반환합니다.

```python
# 우선순위 예시
patterns = [
    PatternSpec("korean_mobile", ..., priority=100),  # 가장 먼저 체크
    PatternSpec("phone", ..., priority=50),           # 일반 전화번호
    PatternSpec("numeric", ..., priority=10),         # 숫자
]

# 한국 휴대폰 번호가 일반 전화번호보다 먼저 매칭됨
```

## 성능 최적화

### 벡터화 연산

```python
# 내부 구현 - Python 콜백 없음
def _count_matches(self, lf: pl.LazyFrame, column: str, pattern: str) -> int:
    return (
        lf.select(
            pl.col(column)
            .str.contains(pattern)  # Polars 네이티브
            .sum()
        )
        .collect()
        .item()
    )
```

### 샘플링 결합

```python
from truthound.profiler.native_patterns import NativePatternMatcher
from truthound.profiler.sampling import Sampler, SamplingConfig

# 대용량 데이터에서 샘플링 후 패턴 매칭
sampler = Sampler(SamplingConfig(max_rows=10_000))
sampled_result = sampler.sample(lf)

matcher = NativePatternMatcher()
patterns = matcher.match(sampled_result.data.lazy(), "email")
```

## CLI 사용법

```bash
# 패턴 감지 포함 프로파일링
th profile data.csv --include-patterns

# 패턴 감지 비활성화
th profile data.csv --no-patterns

# 특정 컬럼만 패턴 감지
th profile data.csv --pattern-columns email,phone
```

## 커스텀 패턴 파일

YAML 형식으로 커스텀 패턴을 정의할 수 있습니다.

```yaml
# custom_patterns.yaml
patterns:
  - name: employee_id
    regex: "^EMP\\d{5}$"
    data_type: identifier
    priority: 80
    examples:
      - EMP00001
      - EMP12345
    description: 직원 ID

  - name: product_sku
    regex: "^[A-Z]{3}-\\d{4}-[A-Z]$"
    data_type: identifier
    priority: 70
    examples:
      - ABC-1234-X
    description: 제품 SKU
```

```python
from truthound.profiler.native_patterns import load_patterns_from_yaml

# 커스텀 패턴 로드
patterns = load_patterns_from_yaml("custom_patterns.yaml")
PatternRegistry.register_all(patterns)
```

## 다음 단계

- [규칙 생성](rule-generation.md) - 감지된 패턴에서 검증 규칙 생성
- [ML 추론](ml-inference.md) - ML 기반 타입 추론
