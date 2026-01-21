# Sampling Strategies

이 문서는 대용량 데이터 처리를 위한 샘플링 전략을 설명합니다.

## 개요

`src/truthound/profiler/sampling.py`에 구현된 샘플링 시스템은 8가지 전략을 제공합니다.

## SamplingMethod Enum

```python
class SamplingMethod(str, Enum):
    """샘플링 전략"""

    NONE = "none"           # 샘플링 없음 (전체 데이터)
    RANDOM = "random"       # 랜덤 샘플링
    SYSTEMATIC = "systematic"  # 체계적 샘플링 (매 N번째 행)
    STRATIFIED = "stratified"  # 층화 샘플링
    RESERVOIR = "reservoir"    # 저수지 샘플링 (스트리밍)
    ADAPTIVE = "adaptive"      # 적응적 샘플링 (자동 선택)
    HEAD = "head"              # 첫 N개 행
    HASH = "hash"              # 해시 기반 (재현 가능)
```

## SamplingConfig

```python
@dataclass
class SamplingConfig:
    """샘플링 설정"""

    strategy: SamplingMethod = SamplingMethod.ADAPTIVE
    max_rows: int = 100_000          # 최대 샘플 크기
    confidence_level: float = 0.95   # 신뢰 수준 (0.0-1.0)
    random_seed: int | None = None   # 랜덤 시드 (재현성)

    # 층화 샘플링 옵션
    stratify_column: str | None = None

    # 해시 샘플링 옵션
    hash_column: str | None = None
```

## SamplingMetrics

```python
@dataclass
class SamplingMetrics:
    """샘플링 결과 메트릭"""

    original_row_count: int      # 원본 행 수
    sampled_row_count: int       # 샘플 행 수
    sampling_ratio: float        # 샘플링 비율
    confidence_level: float      # 신뢰 수준
    margin_of_error: float       # 오차 한계
    strategy_used: SamplingMethod
    execution_time_ms: float
```

## 전략별 사용법

### NONE - 샘플링 없음

```python
from truthound.profiler.sampling import Sampler, SamplingConfig, SamplingMethod

config = SamplingConfig(strategy=SamplingMethod.NONE)
sampler = Sampler(config)
result = sampler.sample(lf)
# 전체 데이터 반환
```

### RANDOM - 랜덤 샘플링

```python
config = SamplingConfig(
    strategy=SamplingMethod.RANDOM,
    max_rows=10_000,
    random_seed=42,
)
sampler = Sampler(config)
result = sampler.sample(lf)

print(f"Sampled: {result.metrics.sampled_row_count}")
print(f"Margin of error: {result.metrics.margin_of_error:.2%}")
```

### SYSTEMATIC - 체계적 샘플링

매 N번째 행을 선택합니다.

```python
config = SamplingConfig(
    strategy=SamplingMethod.SYSTEMATIC,
    max_rows=10_000,
)
sampler = Sampler(config)
result = sampler.sample(lf)
# 정렬된 데이터에서 균등 간격 샘플링
```

### STRATIFIED - 층화 샘플링

특정 컬럼의 분포를 유지하면서 샘플링합니다.

```python
config = SamplingConfig(
    strategy=SamplingMethod.STRATIFIED,
    max_rows=10_000,
    stratify_column="category",  # 이 컬럼의 분포 유지
)
sampler = Sampler(config)
result = sampler.sample(lf)
# category 컬럼의 비율이 원본과 동일하게 유지됨
```

### RESERVOIR - 저수지 샘플링

스트리밍 데이터에 적합한 알고리즘입니다.

```python
config = SamplingConfig(
    strategy=SamplingMethod.RESERVOIR,
    max_rows=10_000,
)
sampler = Sampler(config)
result = sampler.sample(lf)
# O(1) 메모리로 균등 확률 샘플링
```

### ADAPTIVE - 적응적 샘플링

데이터 크기에 따라 자동으로 최적 전략을 선택합니다.

```python
config = SamplingConfig(
    strategy=SamplingMethod.ADAPTIVE,
    max_rows=50_000,
    confidence_level=0.95,
)
sampler = Sampler(config)
result = sampler.sample(lf)

# 자동 선택 로직:
# - 작은 데이터셋: NONE
# - 중간 데이터셋: RANDOM
# - 대용량 데이터셋: RESERVOIR 또는 HASH
```

### HEAD - 첫 N개 행

가장 빠른 샘플링 방법입니다.

```python
config = SamplingConfig(
    strategy=SamplingMethod.HEAD,
    max_rows=1_000,
)
sampler = Sampler(config)
result = sampler.sample(lf)
# 첫 1,000개 행만 반환
```

### HASH - 해시 기반 샘플링

재현 가능한 결정적 샘플링입니다.

```python
config = SamplingConfig(
    strategy=SamplingMethod.HASH,
    max_rows=10_000,
    hash_column="id",  # 해시 기준 컬럼
)
sampler = Sampler(config)
result = sampler.sample(lf)
# 동일한 ID는 항상 동일한 샘플에 포함됨
```

## SamplingMethodRegistry

스레드 안전한 전략 레지스트리입니다.

```python
from truthound.profiler.sampling import SamplingMethodRegistry

# 전략 조회
strategy_class = SamplingMethodRegistry.get(SamplingMethod.RANDOM)

# 커스텀 전략 등록
@SamplingMethodRegistry.register("my_strategy")
class MyCustomStrategy:
    def sample(self, lf: pl.LazyFrame, config: SamplingConfig) -> SamplingResult:
        # 커스텀 샘플링 로직
        pass
```

## 통계적 샘플 크기 계산

```python
from truthound.profiler.sampling import calculate_sample_size

# 95% 신뢰수준, 5% 오차한계
sample_size = calculate_sample_size(
    population_size=1_000_000,
    confidence_level=0.95,
    margin_of_error=0.05,
)
print(f"Required sample size: {sample_size}")  # ~385
```

## 메모리 안전 샘플링

Sampler는 내부적으로 `.head(limit).collect()`를 사용하여 OOM을 방지합니다:

```python
# 안전한 구현 (내부)
def _safe_sample(self, lf: pl.LazyFrame) -> pl.DataFrame:
    # 전체 collect() 호출 없이 limit 적용
    return lf.head(self.config.max_rows).collect()
```

## CLI 사용법

```bash
# 랜덤 샘플링
th profile data.csv --sample-size 10000 --sample-strategy random

# 해시 기반 샘플링
th profile data.csv --sample-size 10000 --sample-strategy hash --hash-column id

# 적응적 샘플링 (기본값)
th profile data.csv --sample-size 50000
```

## 전략 선택 가이드

| 상황 | 권장 전략 |
|------|-----------|
| 소규모 데이터 (<100K) | `NONE` |
| 빠른 미리보기 | `HEAD` |
| 일반적인 분석 | `RANDOM` 또는 `ADAPTIVE` |
| 분포 유지 필요 | `STRATIFIED` |
| 스트리밍 데이터 | `RESERVOIR` |
| 재현 가능성 필요 | `HASH` |
| 정렬된 데이터 | `SYSTEMATIC` |

## 다음 단계

- [패턴 매칭](patterns.md) - 샘플링된 데이터에서 패턴 감지
- [분산 처리](distributed.md) - 대용량 데이터 병렬 처리
