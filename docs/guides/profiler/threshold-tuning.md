# Automatic Threshold Tuning

이 문서는 데이터 특성에 따른 자동 임계값 조정 시스템을 설명합니다.

## 개요

`src/truthound/profiler/auto_threshold.py`에 구현된 임계값 튜닝 시스템은 데이터 분포를 분석하여 최적의 검증 임계값을 자동으로 결정합니다.

## TuningStrategy

```python
class TuningStrategy(str, Enum):
    """임계값 튜닝 전략"""

    CONSERVATIVE = "conservative"    # 보수적 (엄격한 임계값, 적은 오탐)
    BALANCED = "balanced"            # 균형 (정밀도와 재현율 균형)
    PERMISSIVE = "permissive"        # 관대 (느슨한 임계값, 적은 미탐)
    ADAPTIVE = "adaptive"            # 적응적 (데이터에서 학습)
    STATISTICAL = "statistical"      # 통계적 (신뢰구간 기반)
    DOMAIN_AWARE = "domain_aware"    # 도메인 인식 (업무 지식 적용)
```

## ColumnThresholds

컬럼별 임계값 설정입니다.

```python
@dataclass
class ColumnThresholds:
    """컬럼별 임계값"""

    column_name: str
    null_threshold: float = 0.0                  # 최대 허용 null 비율
    uniqueness_threshold: float | None = None    # 최소 고유 비율
    min_value: float | None = None               # 최소값
    max_value: float | None = None               # 최대값
    min_length: int | None = None                # 최소 길이
    max_length: int | None = None                # 최대 길이
    pattern_match_threshold: float = 0.8         # 최소 패턴 매칭 비율
    allowed_values: set[Any] | None = None       # 허용 값 집합
    outlier_threshold: float = 0.01              # 이상치 비율
    confidence: float = 0.5                      # 임계값 신뢰도
    reasoning: list[str] = field(default_factory=list)  # 튜닝 근거
```

## TableThresholds

테이블 수준 임계값 컬렉션입니다.

```python
@dataclass
class TableThresholds:
    """테이블 임계값"""

    table_name: str
    columns: dict[str, ColumnThresholds] = field(default_factory=dict)
    duplicate_threshold: float = 0.0           # 중복 행 허용 비율
    row_count_min: int | None = None           # 최소 행 수
    row_count_max: int | None = None           # 최대 행 수
    global_null_threshold: float = 0.1         # 전역 null 비율
    strategy_used: TuningStrategy = TuningStrategy.BALANCED
    tuned_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_column(self, name: str) -> ColumnThresholds | None:
        """컬럼별 임계값 조회"""
        return self.columns.get(name)
```

## ThresholdTuner

자동 임계값 튜닝 클래스입니다. 문자열 또는 TuningStrategy Enum으로 전략을 지정할 수 있습니다.

```python
from truthound.profiler.auto_threshold import ThresholdTuner, TuningStrategy

# 문자열로 전략 지정 (권장)
tuner = ThresholdTuner(strategy="balanced")

# 또는 Enum 사용
tuner = ThresholdTuner(strategy=TuningStrategy.BALANCED)

# 프로파일에서 임계값 튜닝
thresholds = tuner.tune(profile)

# 컬럼별 임계값 확인
for col_name, col_thresholds in thresholds.columns.items():
    print(f"{col_name}:")
    print(f"  null_threshold: {col_thresholds.null_threshold:.2%}")
    print(f"  min_value: {col_thresholds.min_value}")
    print(f"  max_value: {col_thresholds.max_value}")
    print(f"  confidence: {col_thresholds.confidence:.2%}")
```

## 전략별 동작

### CONSERVATIVE - 보수적

```python
tuner = ThresholdTuner(strategy=TuningStrategy.CONSERVATIVE)
thresholds = tuner.tune(profile)

# 특성:
# - 엄격한 범위 (평균 ± 2σ)
# - 낮은 null 허용치
# - 높은 패턴 매칭 요구
```

### BALANCED - 균형

```python
tuner = ThresholdTuner(strategy=TuningStrategy.BALANCED)
thresholds = tuner.tune(profile)

# 특성:
# - 중간 범위 (평균 ± 3σ)
# - 합리적인 null 허용치
# - 중간 패턴 매칭 요구
```

### PERMISSIVE - 관대

```python
tuner = ThresholdTuner(strategy=TuningStrategy.PERMISSIVE)
thresholds = tuner.tune(profile)

# 특성:
# - 넓은 범위 (평균 ± 4σ)
# - 높은 null 허용치
# - 낮은 패턴 매칭 요구
```

### ADAPTIVE - 적응적

데이터 분포를 분석하여 자동으로 전략을 선택합니다.

```python
tuner = ThresholdTuner(strategy=TuningStrategy.ADAPTIVE)
thresholds = tuner.tune(profile)

# 내부 로직:
# - 데이터 변동성 분석
# - 이상치 비율 분석
# - 분포 형태 분석
# → 최적 전략 자동 선택
```

### STATISTICAL - 통계적

IQR, 백분위수, Wilson 신뢰구간 기반 임계값 설정입니다.

```python
# StatisticalStrategy는 생성자에서 파라미터 설정
from truthound.profiler.auto_threshold import StatisticalStrategy, ThresholdTuner

# 통계 전략 직접 생성
stat_strategy = StatisticalStrategy(
    percentile_low=0.01,    # 하위 1% 백분위
    percentile_high=0.99,   # 상위 99% 백분위
    iqr_multiplier=1.5,     # IQR 배수
)

tuner = ThresholdTuner(strategy=stat_strategy)
thresholds = tuner.tune(profile)

# IQR 및 Wilson CI 기반 범위/null 임계값 설정
```

### DOMAIN_AWARE - 도메인 인식

DataType별 도메인 지식을 자동으로 적용합니다. 내장된 DOMAIN_DEFAULTS를 사용합니다.

```python
from truthound.profiler.auto_threshold import ThresholdTuner

# DomainAwareStrategy는 DataType별 기본값 사용
# - EMAIL: null_threshold=0.1, pattern_threshold=0.95, min_length=5, max_length=254
# - PHONE: null_threshold=0.2, pattern_threshold=0.9, min_length=7, max_length=20
# - UUID: null_threshold=0.0, uniqueness_threshold=1.0, min_length=36, max_length=36
# - PERCENTAGE: min_value=0.0, max_value=100.0
# 기타 DataType.CURRENCY, BOOLEAN, KOREAN_PHONE, KOREAN_RRN 등 지원

tuner = ThresholdTuner(strategy="domain_aware")
thresholds = tuner.tune(profile)
```

## IQR 기반 분석

사분위 범위(IQR)를 사용한 이상치 감지입니다.

```python
def _compute_iqr_bounds(self, profile: ColumnProfile) -> tuple[float, float]:
    """IQR 기반 경계 계산"""
    q1 = profile.quantiles.get(0.25, profile.min_value)
    q3 = profile.quantiles.get(0.75, profile.max_value)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper
```

## 백분위수 기반 분석

```python
def _compute_percentile_bounds(
    self,
    profile: ColumnProfile,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> tuple[float, float]:
    """백분위수 기반 경계 계산"""
    lower = profile.quantiles.get(lower_pct, profile.min_value)
    upper = profile.quantiles.get(upper_pct, profile.max_value)
    return lower, upper
```

## 빠른 튜닝

```python
from truthound.profiler.auto_threshold import tune_thresholds
from truthound.profiler.base import Strictness

# 간편 함수 - 전략과 strictness 지정
thresholds = tune_thresholds(
    profile,
    strategy="adaptive",
    strictness=Strictness.MEDIUM,  # LOOSE, MEDIUM, STRICT
)

# 테이블 수준 임계값
print(f"Duplicate threshold: {thresholds.duplicate_threshold:.2%}")
print(f"Global null threshold: {thresholds.global_null_threshold:.2%}")

# 컬럼별 임계값
for col_name, col_thresh in thresholds.columns.items():
    print(f"{col_name}: null <= {col_thresh.null_threshold:.1%}")
```

## A/B 테스팅

```python
from truthound.profiler.auto_threshold import ThresholdTester, ThresholdTuner
import polars as pl

tester = ThresholdTester()

# 두 임계값 설정 비교
tuner_a = ThresholdTuner(strategy="conservative")
tuner_b = ThresholdTuner(strategy="permissive")

threshold_a = tuner_a.tune(profile)
threshold_b = tuner_b.tune(profile)

# DataFrame으로 테스트
df = pl.read_csv("test_data.csv")
result = tester.compare(
    data=df,
    threshold_a=threshold_a,
    threshold_b=threshold_b,
)

print(f"Recommendation: {result.recommendation}")
print(f"Violations A: {result.violations_a}")
print(f"Violations B: {result.violations_b}")
```

## 임계값 내보내기

```python
import json

# JSON으로 저장
with open("thresholds.json", "w") as f:
    json.dump(thresholds.to_dict(), f, indent=2)

# YAML로 저장
import yaml
with open("thresholds.yaml", "w") as f:
    yaml.dump(thresholds.to_dict(), f)
```

## CLI 사용법

```bash
# 자동 임계값 튜닝
th tune-thresholds profile.json -o thresholds.yaml

# 전략 지정
th tune-thresholds profile.json -o thresholds.yaml --strategy statistical

# 신뢰수준 지정
th tune-thresholds profile.json -o thresholds.yaml --strategy statistical --confidence 0.99

# 튜닝된 임계값으로 규칙 생성
th generate-suite profile.json -o rules.yaml --thresholds thresholds.yaml
```

## 통합 예제

```python
from truthound.profiler import TableProfiler, generate_suite
from truthound.profiler.auto_threshold import ThresholdTuner, TuningStrategy

# 프로파일링
profiler = TableProfiler()
profile = profiler.profile_file("data.csv")

# 임계값 튜닝
tuner = ThresholdTuner(strategy=TuningStrategy.ADAPTIVE)
thresholds = tuner.tune(profile)

# 튜닝된 임계값으로 규칙 생성
suite = generate_suite(
    profile,
    thresholds=thresholds,
)

# 저장
save_suite(suite, "rules.yaml", format="yaml")
```

## 다음 단계

- [품질 스코어링](quality-scoring.md) - 튜닝된 임계값 품질 평가
- [규칙 생성](rule-generation.md) - 임계값 기반 규칙 생성
