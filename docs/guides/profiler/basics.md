# Basic Profiling

이 문서는 Truthound Profiler의 기본 사용법을 설명합니다.

## DataProfiler

`DataProfiler`는 데이터셋 전체를 프로파일링하는 핵심 클래스입니다.

### 기본 사용법

```python
from truthound.profiler import DataProfiler, ProfilerConfig
import polars as pl

# 설정
config = ProfilerConfig(
    sample_size=50000,
    include_patterns=True,
    include_correlations=False,
    top_n_values=10,
)

# 프로파일러 생성
profiler = DataProfiler(config)

# LazyFrame 프로파일링
lf = pl.scan_csv("data.csv")
profile = profiler.profile(lf)

# 결과 확인
print(f"Rows: {profile.row_count}")
print(f"Columns: {profile.column_count}")
```

### TableProfile 구조

```python
@dataclass(frozen=True)
class TableProfile:
    """테이블 프로파일 결과 (불변 객체)"""

    # 테이블 정보
    name: str = ""
    row_count: int = 0
    column_count: int = 0

    # 메모리 추정
    estimated_memory_bytes: int = 0

    # 컬럼 프로파일 (불변 튜플)
    columns: tuple[ColumnProfile, ...] = field(default_factory=tuple)

    # 테이블 수준 메트릭
    duplicate_row_count: int = 0
    duplicate_row_ratio: float = 0.0

    # 상관관계 매트릭스
    correlations: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)

    # 메타데이터
    source: str = ""
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_duration_ms: float = 0.0
```

### 주요 메서드

```python
# 딕셔너리로 변환
profile_dict = profile.to_dict()

# 특정 컬럼 프로파일 조회
col_profile = profile.get("email")

# 컬럼 이름 목록
names = profile.column_names
```

## ColumnProfiler

`ColumnProfiler`는 개별 컬럼을 분석합니다.

### 기본 사용법

```python
from truthound.profiler import ColumnProfiler

column_profiler = ColumnProfiler()
col_profile = column_profiler.profile(lf, "email")

print(f"Type: {col_profile.inferred_type}")
print(f"Null ratio: {col_profile.null_ratio:.2%}")
print(f"Unique ratio: {col_profile.unique_ratio:.2%}")
```

### ColumnProfile 구조

```python
@dataclass(frozen=True)
class ColumnProfile:
    """컬럼 프로파일 결과 (불변 객체)"""

    # 기본 정보
    name: str
    physical_type: str  # Polars 데이터 타입 (문자열)
    inferred_type: DataType = DataType.UNKNOWN  # 추론된 논리적 타입

    # 완전성
    row_count: int = 0
    null_count: int = 0
    null_ratio: float = 0.0
    empty_string_count: int = 0

    # 고유성
    distinct_count: int = 0
    unique_ratio: float = 0.0
    is_unique: bool = False
    is_constant: bool = False

    # 통계 분포 (수치형 컬럼)
    distribution: DistributionStats | None = None
    # DistributionStats 포함 필드: mean, std, min, max, median, q1, q3, skewness, kurtosis

    # 상위/하위 값 (ValueFrequency 튜플)
    top_values: tuple[ValueFrequency, ...] = field(default_factory=tuple)
    bottom_values: tuple[ValueFrequency, ...] = field(default_factory=tuple)
    # ValueFrequency 포함 필드: value, count, ratio

    # 길이 통계 (문자열 컬럼)
    min_length: int | None = None
    max_length: int | None = None
    avg_length: float | None = None

    # 패턴 분석 (문자열 컬럼)
    detected_patterns: tuple[PatternMatch, ...] = field(default_factory=tuple)
    # PatternMatch 포함 필드: pattern, regex, match_ratio, sample_matches

    # 시간 분석 (datetime 컬럼)
    min_date: datetime | None = None
    max_date: datetime | None = None
    date_gaps: int = 0

    # 추천 validator 목록
    suggested_validators: tuple[str, ...] = field(default_factory=tuple)

    # 메타데이터
    profiled_at: datetime = field(default_factory=datetime.now)
    profile_duration_ms: float = 0.0
```

## DataType (추론된 타입)

```python
class DataType(str, Enum):
    """추론된 논리적 데이터 타입"""

    # 기본 타입
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    TIME = "time"
    DURATION = "duration"

    # 시맨틱 타입 (패턴 감지)
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"
    UUID = "uuid"
    IP_ADDRESS = "ip_address"
    JSON = "json"

    # 식별자
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"

    # 수치형 하위 타입
    CURRENCY = "currency"
    PERCENTAGE = "percentage"

    # 한국어 특화
    KOREAN_RRN = "korean_rrn"
    KOREAN_PHONE = "korean_phone"
    KOREAN_BUSINESS_NUMBER = "korean_business_number"

    # 미확인
    UNKNOWN = "unknown"
```

## ProfilerConfig 옵션

```python
@dataclass
class ProfilerConfig:
    """프로파일링 설정"""

    # 샘플링
    sample_size: int | None = None  # None = 전체 데이터 사용
    random_seed: int = 42

    # 분석 옵션
    include_patterns: bool = True      # 패턴 감지 활성화
    include_correlations: bool = False # 상관관계 계산
    include_distributions: bool = True # 분포 통계 포함

    # 성능 튜닝
    top_n_values: int = 10             # 상위/하위 값 개수
    pattern_sample_size: int = 1000    # 패턴 매칭 샘플 크기
    correlation_threshold: float = 0.7 # 상관관계 임계값

    # 패턴 감지
    min_pattern_match_ratio: float = 0.8  # 최소 패턴 매칭 비율

    # 병렬 처리
    n_jobs: int = 1
```

## CLI 사용법

```bash
# 기본 프로파일링
th profile data.csv

# JSON 출력
th profile data.csv -o profile.json

# 샘플링 적용
th profile data.csv --sample-size 10000

# 패턴 감지 비활성화
th profile data.csv --no-patterns

# 스트리밍 모드 (대용량 파일)
th profile large_file.csv --streaming --chunk-size 100000
```

## 파일 포맷 지원

| 포맷 | 확장자 | 지원 |
|------|--------|------|
| CSV | `.csv` | ✅ |
| Parquet | `.parquet` | ✅ |
| JSON | `.json`, `.jsonl` | ✅ |
| Excel | `.xlsx`, `.xls` | ✅ |
| Arrow | `.arrow`, `.feather` | ✅ |

```python
# 다양한 포맷 프로파일링
profile = profiler.profile_file("data.parquet")
profile = profiler.profile_file("data.json")
profile = profiler.profile_file("data.xlsx")
```

## 결과 직렬화

```python
import json

# JSON으로 저장
with open("profile.json", "w") as f:
    json.dump(profile.to_dict(), f, indent=2, default=str)

# JSON에서 로드 (수동 복원)
with open("profile.json") as f:
    data = json.load(f)
# TableProfile은 frozen=True이므로 직접 역직렬화 필요
```

## 다음 단계

- [샘플링 전략](sampling.md) - 대용량 데이터 처리
- [패턴 매칭](patterns.md) - 이메일, 전화번호 등 자동 감지
- [규칙 생성](rule-generation.md) - 검증 규칙 자동 생성
