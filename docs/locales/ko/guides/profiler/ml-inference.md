# ML-based Type Inference

실무 운영 가이드에서 ML-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 `src/truthound/profiler/ml_inference.py`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Feature

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
@dataclass
class Feature:
    """Single feature"""

    name: str
    value: float
    feature_type: FeatureType  # NAME, VALUE, STATISTICAL, CONTEXTUAL
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

## FeatureVector

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
@dataclass
class FeatureVector:
    """Feature vector"""

    column_name: str
    features: list[Feature]
    raw_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary"""
        return {f.name: f.value for f in self.features}

    def to_array(self) -> list[float]:
        """Convert to array"""
        return [f.value for f in self.features]

    def get_feature(self, name: str) -> Feature | None:
        """Retrieve feature by name"""
        for f in self.features:
            if f.name == name:
                return f
        return None
```

## InferenceResult

```python
@dataclass
class InferenceResult:
    """Type inference result"""

    column_name: str                  # Column name
    inferred_type: DataType           # Inferred type
    confidence: float                 # Confidence (0.0-1.0)
    alternatives: list[tuple[DataType, float]] = field(default_factory=list)  # Alternative types
    reasoning: list[str] = field(default_factory=list)  # Inference reasoning
    features_used: list[str] = field(default_factory=list)  # Features used
    model_version: str = "1.0"        # Model version
    inference_time_ms: float = 0.0
```

## ContextFeatureExtractor

실무 운영 가이드에서 Extracts, Combines을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.ml_inference import ContextFeatureExtractor
import polars as pl

extractor = ContextFeatureExtractor()

# Extract features from column (Series)
column = pl.Series("email", ["user@example.com", "test@domain.org"])
context = {"table_name": "users"}
features = extractor.extract(column, context)

for feature in features.features:
    print(f"{feature.name}: {feature.value:.4f} (type: {feature.feature_type})")
```

### Extracted Feature Types

| 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|---------|-------------|
| 실무 운영 가이드에서 NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `name_has_email`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name contains 'email' |
| 실무 운영 가이드에서 NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `name_has_phone`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name contains 'phone' |
| 실무 운영 가이드에서 NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `name_has_date`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name contains 'date' |
| 실무 운영 가이드에서 NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `name_has_id`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name contains 'id' |
| 실무 운영 가이드에서 NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `name_has_name`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name contains 'name' |
| 실무 운영 가이드에서 NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `name_has_address`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name contains 'address' |
| 실무 운영 가이드에서 NAME을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `name_has_url`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 컬럼 name contains 'url' |
| 실무 운영 가이드에서 VALUE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `value_has_at_symbol`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ratio을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 VALUE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `value_has_dot`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ratio을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 VALUE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `value_digit_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 VALUE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `value_alpha_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Alphabetic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 VALUE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `value_has_dash`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ratio을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 VALUE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `value_has_slash`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Ratio을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stat_avg_length`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Average을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stat_std_length`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Length을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stat_unique_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Unique을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stat_null_ratio`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stat_min_length`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 STATISTICAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 `stat_max_length`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

## MLTypeInferrer

실무 운영 가이드에서 ML-based, Supports, RuleBasedModel, NaiveBayesModel, EnsembleModel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.ml_inference import MLTypeInferrer
import polars as pl

inferrer = MLTypeInferrer()  # Default: ensemble model

# Single column type inference
column = pl.Series("email", ["user@example.com", "test@domain.org"])
context = {"table_name": "users"}
result = inferrer.infer(column, context)

print(f"Column: {result.column_name}")
print(f"Type: {result.inferred_type}")
print(f"Confidence: {result.confidence:.2%}")

# Alternative types
for dtype, prob in result.alternatives[:3]:
    print(f"  {dtype}: {prob:.2%}")

# Inference reasoning
for reason in result.reasoning:
    print(f"  - {reason}")
```

### Model Types

실무 운영 가이드에서 Three, InferenceModel을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class MLTypeInferrer:
    """ML type inferrer"""

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

        # Caching support
        if self._config.use_caching:
            cache_key = self._get_cache_key(column, context)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Model prediction
        result = self._model.predict(features)
        return result
```

## Full 테이블 Type Inference

```python
from truthound.profiler.ml_inference import infer_table_types_ml
import polars as pl

# Infer types for all columns from DataFrame
df = pl.DataFrame({
    "email": ["user@example.com", "test@domain.org"],
    "phone": ["010-1234-5678", "02-123-4567"],
    "age": [25, 30],
})

results = infer_table_types_ml(df, table_name="users", model="ensemble")

for column, result in results.items():
    print(f"{column}: {result.inferred_type} ({result.confidence:.0%})")
```

## Feature Extractor Types

실무 운영 가이드에서 `FeatureExtractor`, `extract(column: pl.Series, context: dict) -> list[Feature]`, FeatureExtractor, Series, Feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### NameFeatureExtractor

실무 운영 가이드에서 Extracts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.ml_inference import NameFeatureExtractor
import polars as pl

extractor = NameFeatureExtractor()
column = pl.Series("customer_email_address", ["user@example.com"])
features = extractor.extract(column, {})

# Extracted features:
# - name_has_email: 1.0
# - name_has_customer: 1.0
# - name_has_address: 1.0
```

### ValueFeatureExtractor

실무 운영 가이드에서 Extracts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.ml_inference import ValueFeatureExtractor
import polars as pl

extractor = ValueFeatureExtractor()
column = pl.Series("email", ["user@example.com", "test@domain.org"])
features = extractor.extract(column, {})

# Extracted features:
# - value_has_at_symbol: 1.0 (100% contain @)
# - value_avg_length: 18.5
# - value_digit_ratio: 0.0
```

### StatisticalFeatureExtractor

실무 운영 가이드에서 Extracts을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.profiler.ml_inference import StatisticalFeatureExtractor
import polars as pl

extractor = StatisticalFeatureExtractor()
column = pl.Series("email", ["user@example.com", "test@domain.org", None])
features = extractor.extract(column, {})

# Extracted features:
# - stat_unique_ratio: 0.67
# - stat_null_ratio: 0.33
# - stat_count: 3
```

## Custom Model Registration

```python
from truthound.profiler.ml_inference import model_registry, InferenceModel, FeatureVector, InferenceResult

# Implement InferenceModel protocol
class MyCustomModel:
    """Custom inference model"""

    @property
    def name(self) -> str:
        return "my_model"

    @property
    def version(self) -> str:
        return "1.0.0"

    def predict(self, features: FeatureVector) -> InferenceResult:
        # Custom inference logic
        ...

# Register in model registry
model_registry.register(MyCustomModel())

# Use registered model
inferrer = MLTypeInferrer(model="my_model")
```

## Inference 설정

```python
from truthound.profiler.ml_inference import InferrerConfig, MLTypeInferrer

config = InferrerConfig(
    model="ensemble",              # Model type ("rule", "naive_bayes", "ensemble")
    confidence_threshold=0.5,      # Minimum confidence
    use_caching=True,              # Enable caching
    cache_size=1000,               # Cache size
    enable_learning=True,          # Enable learning mode
    model_path=None,               # Custom model path (optional)
)

inferrer = MLTypeInferrer(config=config)
```

## CLI Usage

```bash
# Enable ML-based type inference
th profile data.csv --ml-inference

# Specify model type
th profile data.csv --ml-inference --model ensemble

# Set confidence threshold
th profile data.csv --ml-inference --confidence-threshold 0.8
```

## RuleBasedModel (Rule-Based Inference)

실무 운영 가이드에서 Infers을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
class RuleBasedModel:
    """Rule-based type inference model"""

    @property
    def name(self) -> str:
        return "rule"

    def predict(self, features: FeatureVector) -> InferenceResult:
        """Infer type using feature-based rules"""
        scores: dict[DataType, float] = {}

        # Check email pattern
        at_feature = features.get_feature("value_has_at_symbol")
        dot_feature = features.get_feature("value_has_dot")
        if at_feature and at_feature.value > 0.8:
            if dot_feature and dot_feature.value > 0.9:
                scores[DataType.EMAIL] = 0.85

        # Check phone number pattern
        digit_feature = features.get_feature("value_digit_ratio")
        if digit_feature and digit_feature.value > 0.7:
            scores[DataType.PHONE] = 0.75

        # Return highest scoring type
        best_type = max(scores, key=scores.get, default=DataType.STRING)
        return InferenceResult(
            column_name=features.column_name,
            inferred_type=best_type,
            confidence=scores.get(best_type, 0.5),
            # ...
        )
```

## 다음 단계

- 실무 운영 가이드에서 Threshold, Tuning, Optimize을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Pattern, Matching, Pattern-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
