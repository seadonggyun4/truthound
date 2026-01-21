# ML-based Type Inference

이 문서는 ML 기반 시맨틱 타입 추론 시스템을 설명합니다.

## 개요

`src/truthound/profiler/ml_inference.py`에 구현된 ML 타입 추론 시스템은 컬럼 이름, 값 패턴, 통계적 특성을 결합하여 시맨틱 타입을 추론합니다.

## Feature

피처 정의를 위한 데이터 클래스입니다.

```python
@dataclass
class Feature:
    """단일 피처"""

    name: str
    value: float
    feature_type: FeatureType  # NAME, VALUE, STATISTICAL, CONTEXTUAL
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

## FeatureVector

피처 벡터 컬렉션입니다.

```python
@dataclass
class FeatureVector:
    """피처 벡터"""

    column_name: str
    features: list[Feature]
    raw_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """딕셔너리로 변환"""
        return {f.name: f.value for f in self.features}

    def to_array(self) -> list[float]:
        """배열로 변환"""
        return [f.value for f in self.features]

    def get_feature(self, name: str) -> Feature | None:
        """이름으로 피처 조회"""
        for f in self.features:
            if f.name == name:
                return f
        return None
```

## InferenceResult

```python
@dataclass
class InferenceResult:
    """타입 추론 결과"""

    column_name: str                  # 컬럼 이름
    inferred_type: DataType           # 추론된 타입
    confidence: float                 # 신뢰도 (0.0-1.0)
    alternatives: list[tuple[DataType, float]] = field(default_factory=list)  # 대안 타입들
    reasoning: list[str] = field(default_factory=list)  # 추론 근거
    features_used: list[str] = field(default_factory=list)  # 사용된 피처
    model_version: str = "1.0"        # 모델 버전
    inference_time_ms: float = 0.0
```

## ContextFeatureExtractor

다양한 유형의 피처를 추출합니다. 모든 추출기를 조합하여 종합적인 피처 벡터를 생성합니다.

```python
from truthound.profiler.ml_inference import ContextFeatureExtractor
import polars as pl

extractor = ContextFeatureExtractor()

# 컬럼(Series)에서 피처 추출
column = pl.Series("email", ["user@example.com", "test@domain.org"])
context = {"table_name": "users"}
features = extractor.extract(column, context)

for feature in features.features:
    print(f"{feature.name}: {feature.value:.4f} (type: {feature.feature_type})")
```

### 추출되는 피처 유형

| 카테고리 | 피처 | 설명 |
|----------|------|------|
| **NAME** | `name_has_email` | 컬럼명에 'email' 포함 |
| **NAME** | `name_has_phone` | 컬럼명에 'phone' 포함 |
| **NAME** | `name_has_date` | 컬럼명에 'date' 포함 |
| **NAME** | `name_has_id` | 컬럼명에 'id' 포함 |
| **NAME** | `name_has_name` | 컬럼명에 'name' 포함 |
| **NAME** | `name_has_address` | 컬럼명에 'address' 포함 |
| **NAME** | `name_has_url` | 컬럼명에 'url' 포함 |
| **VALUE** | `value_has_at_symbol` | '@' 기호 포함 비율 |
| **VALUE** | `value_has_dot` | '.' 포함 비율 |
| **VALUE** | `value_digit_ratio` | 숫자 문자 비율 |
| **VALUE** | `value_alpha_ratio` | 알파벳 비율 |
| **VALUE** | `value_has_dash` | '-' 포함 비율 |
| **VALUE** | `value_has_slash` | '/' 포함 비율 |
| **STATISTICAL** | `stat_avg_length` | 평균 길이 |
| **STATISTICAL** | `stat_std_length` | 길이 표준편차 |
| **STATISTICAL** | `stat_unique_ratio` | 고유값 비율 |
| **STATISTICAL** | `stat_null_ratio` | null 비율 |
| **STATISTICAL** | `stat_min_length` | 최소 길이 |
| **STATISTICAL** | `stat_max_length` | 최대 길이 |

## MLTypeInferrer

ML 기반 타입 추론기입니다. 여러 모델(RuleBasedModel, NaiveBayesModel, EnsembleModel)을 지원합니다.

```python
from truthound.profiler.ml_inference import MLTypeInferrer
import polars as pl

inferrer = MLTypeInferrer()  # 기본값: ensemble 모델

# 단일 컬럼 타입 추론
column = pl.Series("email", ["user@example.com", "test@domain.org"])
context = {"table_name": "users"}
result = inferrer.infer(column, context)

print(f"Column: {result.column_name}")
print(f"Type: {result.inferred_type}")
print(f"Confidence: {result.confidence:.2%}")

# 대안 타입들
for dtype, prob in result.alternatives[:3]:
    print(f"  {dtype}: {prob:.2%}")

# 추론 근거
for reason in result.reasoning:
    print(f"  - {reason}")
```

### 모델 타입

3가지 내장 모델을 지원하며, InferenceModel 프로토콜을 구현하여 커스텀 모델을 추가할 수 있습니다.

```python
class MLTypeInferrer:
    """ML 타입 추론기"""

    def __init__(
        self,
        model: str = "ensemble",  # "rule", "naive_bayes", "ensemble"
        config: InferrerConfig | None = None,
    ):
        self._config = config or InferrerConfig()
        self._model = model_registry.get(model)
        self._extractor = ContextFeatureExtractor()
        self._cache: dict[str, InferenceResult] = {}

    def infer(
        self,
        column: pl.Series,
        context: dict[str, Any] | None = None,
    ) -> InferenceResult:
        context = context or {}
        features = self._extractor.extract(column, context)

        # 캐싱 지원
        if self._config.use_caching:
            cache_key = self._get_cache_key(column, context)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # 모델 예측
        result = self._model.predict(features)
        return result
```

## 테이블 전체 타입 추론

```python
from truthound.profiler.ml_inference import infer_table_types_ml
import polars as pl

# DataFrame으로 모든 컬럼 타입 추론
df = pl.DataFrame({
    "email": ["user@example.com", "test@domain.org"],
    "phone": ["010-1234-5678", "02-123-4567"],
    "age": [25, 30],
})

results = infer_table_types_ml(df, table_name="users", model="ensemble")

for column, result in results.items():
    print(f"{column}: {result.inferred_type} ({result.confidence:.0%})")
```

## 피처 추출기 종류

모든 추출기는 `FeatureExtractor` 프로토콜을 구현하며, `extract(column: pl.Series, context: dict) -> list[Feature]` 메서드를 제공합니다.

### NameFeatureExtractor

컬럼 이름에서 피처를 추출합니다.

```python
from truthound.profiler.ml_inference import NameFeatureExtractor
import polars as pl

extractor = NameFeatureExtractor()
column = pl.Series("customer_email_address", ["user@example.com"])
features = extractor.extract(column, {})

# 추출된 피처:
# - name_has_email: 1.0
# - name_has_customer: 1.0
# - name_has_address: 1.0
```

### ValueFeatureExtractor

컬럼 값에서 피처를 추출합니다.

```python
from truthound.profiler.ml_inference import ValueFeatureExtractor
import polars as pl

extractor = ValueFeatureExtractor()
column = pl.Series("email", ["user@example.com", "test@domain.org"])
features = extractor.extract(column, {})

# 추출된 피처:
# - value_has_at_symbol: 1.0 (100%가 @ 포함)
# - value_avg_length: 18.5
# - value_digit_ratio: 0.0
```

### StatisticalFeatureExtractor

통계적 피처를 추출합니다.

```python
from truthound.profiler.ml_inference import StatisticalFeatureExtractor
import polars as pl

extractor = StatisticalFeatureExtractor()
column = pl.Series("email", ["user@example.com", "test@domain.org", None])
features = extractor.extract(column, {})

# 추출된 피처:
# - stat_unique_ratio: 0.67
# - stat_null_ratio: 0.33
# - stat_count: 3
```

## 커스텀 모델 등록

```python
from truthound.profiler.ml_inference import model_registry, InferenceModel, FeatureVector, InferenceResult

# InferenceModel 프로토콜 구현
class MyCustomModel:
    """커스텀 추론 모델"""

    @property
    def name(self) -> str:
        return "my_model"

    @property
    def version(self) -> str:
        return "1.0.0"

    def predict(self, features: FeatureVector) -> InferenceResult:
        # 커스텀 추론 로직
        ...

# 모델 레지스트리에 등록
model_registry.register(MyCustomModel())

# 등록된 모델 사용
inferrer = MLTypeInferrer(model="my_model")
```

## 추론 설정

```python
from truthound.profiler.ml_inference import InferrerConfig, MLTypeInferrer

config = InferrerConfig(
    model="ensemble",              # 모델 타입 ("rule", "naive_bayes", "ensemble")
    confidence_threshold=0.5,      # 최소 신뢰도
    use_caching=True,              # 캐싱 활성화
    cache_size=1000,               # 캐시 크기
    enable_learning=True,          # 학습 모드 활성화
    model_path=None,               # 커스텀 모델 경로 (선택)
)

inferrer = MLTypeInferrer(config=config)
```

## CLI 사용법

```bash
# ML 기반 타입 추론 활성화
th profile data.csv --ml-inference

# 모델 타입 지정
th profile data.csv --ml-inference --model ensemble

# 신뢰도 임계값 설정
th profile data.csv --ml-inference --confidence-threshold 0.8
```

## RuleBasedModel (규칙 기반 추론)

기본 제공되는 규칙 기반 모델입니다. 피처 값을 기반으로 타입을 추론합니다.

```python
class RuleBasedModel:
    """규칙 기반 타입 추론 모델"""

    @property
    def name(self) -> str:
        return "rule"

    def predict(self, features: FeatureVector) -> InferenceResult:
        """피처 기반 규칙으로 타입 추론"""
        scores: dict[DataType, float] = {}

        # 이메일 패턴 체크
        at_feature = features.get_feature("value_has_at_symbol")
        dot_feature = features.get_feature("value_has_dot")
        if at_feature and at_feature.value > 0.8:
            if dot_feature and dot_feature.value > 0.9:
                scores[DataType.EMAIL] = 0.85

        # 전화번호 패턴 체크
        digit_feature = features.get_feature("value_digit_ratio")
        if digit_feature and digit_feature.value > 0.7:
            scores[DataType.PHONE] = 0.75

        # 최고 점수 타입 반환
        best_type = max(scores, key=scores.get, default=DataType.STRING)
        return InferenceResult(
            column_name=features.column_name,
            inferred_type=best_type,
            confidence=scores.get(best_type, 0.5),
            # ...
        )
```

## 다음 단계

- [임계값 튜닝](threshold-tuning.md) - 추론 임계값 최적화
- [패턴 매칭](patterns.md) - 패턴 기반 타입 감지
