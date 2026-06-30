# Pattern Matching

실무 운영 가이드에서 Polars을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 Polars, `src/truthound/profiler/native_patterns.py`, `str.contains()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## PatternSpec

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
@dataclass
class PatternSpec:
    """Pattern specification definition"""

    name: str                      # Pattern name (e.g., "email", "phone")
    regex: str                     # Regular expression pattern
    data_type: DataType            # Data type to infer upon matching
    priority: int = 0              # Priority (higher values match first)
    examples: list[str] = field(default_factory=list)  # Example values
    description: str = ""          # Pattern description
    category: str = "general"      # Pattern category
```

## PatternBuilder

실무 운영 가이드에서 API, Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.native_patterns import PatternBuilder

# Create pattern using fluent style
pattern = (
    PatternBuilder("korean_mobile")
    .regex(r"^01[0-9]-?[0-9]{3,4}-?[0-9]{4}$")
    .data_type(DataType.KOREAN_PHONE)
    .priority(100)
    .examples(["010-1234-5678", "01012345678"])
    .description("Korean mobile phone number")
    .category("korean")
    .build()
)
```

## NativePatternMatcher

실무 운영 가이드에서 Polars을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.native_patterns import NativePatternMatcher

# Create matcher
matcher = NativePatternMatcher()

# Match patterns in column
results = matcher.match(lf, "email_column")

for result in results:
    print(f"Pattern: {result.pattern_name}")
    print(f"Match ratio: {result.match_ratio:.2%}")
    print(f"Data type: {result.data_type}")
```

### Internal Implementation

```python
class NativePatternMatcher:
    """Polars native pattern matcher"""

    def match(self, lf: pl.LazyFrame, column: str) -> list[PatternMatch]:
        """
        Perform high-performance pattern matching using
        Polars' vectorized str.contains()
        """
        col = pl.col(column)

        for pattern in self._patterns:
            # Native Polars operation (no map_elements)
            match_expr = col.str.contains(pattern.regex)
            match_count = match_expr.sum()
            # ...
```

## Built-in Patterns

### General Patterns

| 실무 운영 가이드에서 Pattern, Name을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data, Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------------|-----------|-------------|
| 실무 운영 가이드에서 `email`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `EMAIL`, EMAIL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Email을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `url`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `URL`, URL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 URL/URI을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `uuid`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `UUID`, UUID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 UUID을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `ip_address`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `IP_ADDRESS`, IP_ADDRESS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 IPv4/IPv6을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `phone`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `PHONE`, PHONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 International을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `date_iso`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `DATE`, DATE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ISO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `datetime_iso`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `DATETIME`, DATETIME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 ISO을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `json`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 JSON을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `currency`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `CURRENCY`, CURRENCY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Currency을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `percentage`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `PERCENTAGE`, PERCENTAGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Percentages을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Korean-Specific Patterns

| 실무 운영 가이드에서 Pattern, Name을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Data, Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------------|-----------|-------------|
| 실무 운영 가이드에서 `korean_rrn`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `KOREAN_RRN`, KOREAN_RRN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Resident을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `korean_phone`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `KOREAN_PHONE`, KOREAN_PHONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Korean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `korean_mobile`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `KOREAN_PHONE`, KOREAN_PHONE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Korean을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `korean_business_number`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `KOREAN_BUSINESS_NUMBER`, KOREAN_BUSINESS_NUMBER을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Business을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## Pattern Registry

```python
from truthound.profiler.native_patterns import PatternRegistry

# Retrieve default patterns
email_pattern = PatternRegistry.get("email")

# Retrieve patterns by category
korean_patterns = PatternRegistry.get_by_category("korean")

# Register custom patterns
PatternRegistry.register(
    PatternSpec(
        name="custom_id",
        regex=r"^[A-Z]{2}\d{6}$",
        data_type=DataType.IDENTIFIER,
        priority=50,
        examples=["AB123456"],
        description="Company-specific ID format",
    )
)

# Remove pattern
PatternRegistry.unregister("custom_id")
```

## PatternMatch 결과

```python
@dataclass
class PatternMatch:
    """Pattern matching result"""

    pattern_name: str       # Matched pattern name
    regex: str              # Regular expression used
    data_type: DataType     # Inferred data type
    match_count: int        # Number of matched rows
    total_count: int        # Total rows (excluding null)
    match_ratio: float      # Match ratio (0.0-1.0)
    confidence: float       # Confidence level
    sample_matches: list[str]  # Sample matched values
```

## Priority-Based Matching

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
# Priority example
patterns = [
    PatternSpec("korean_mobile", ..., priority=100),  # Checked first
    PatternSpec("phone", ..., priority=50),           # General phone number
    PatternSpec("numeric", ..., priority=10),         # Numeric
]

# Korean mobile numbers match before general phone numbers
```

## 성능 Optimization

### Vectorized Operations

```python
# Internal implementation - no Python callbacks
def _count_matches(self, lf: pl.LazyFrame, column: str, pattern: str) -> int:
    return (
        lf.select(
            pl.col(column)
            .str.contains(pattern)  # Polars native
            .sum()
        )
        .collect()
        .item()
    )
```

### Combining with Sampling

```python
from truthound.profiler.native_patterns import NativePatternMatcher
from truthound.profiler.sampling import Sampler, SamplingConfig

# Sample from large data then perform pattern matching
sampler = Sampler(SamplingConfig(max_rows=10_000))
sampled_result = sampler.sample(lf)

matcher = NativePatternMatcher()
patterns = matcher.match(sampled_result.data.lazy(), "email")
```

## CLI Usage

```bash
# Profile with pattern detection
th profile data.csv --include-patterns

# Disable pattern detection
th profile data.csv --no-patterns

# Pattern detection for specific columns only
th profile data.csv --pattern-columns email,phone
```

## Custom Pattern 파일

실무 운영 가이드에서 YAML, Custom을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

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
    description: Employee ID

  - name: product_sku
    regex: "^[A-Z]{3}-\\d{4}-[A-Z]$"
    data_type: identifier
    priority: 70
    examples:
      - ABC-1234-X
    description: Product SKU
```

```python
from truthound.profiler.native_patterns import load_patterns_from_yaml

# Load custom patterns
patterns = load_patterns_from_yaml("custom_patterns.yaml")
PatternRegistry.register_all(patterns)
```

## 다음 단계

- 실무 운영 가이드에서 Rule, Generation, Generate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Inference, ML-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
