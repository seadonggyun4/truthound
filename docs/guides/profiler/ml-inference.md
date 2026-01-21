# ML-based Type Inference

This document describes the ML-based semantic type inference system.

## Overview

The ML type inference system implemented in `src/truthound/profiler/ml_inference.py` combines column names, value patterns, and statistical characteristics to infer semantic types.

## Feature

A dataclass for feature definitions.

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

A feature vector collection.

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

Extracts various types of features. Combines all extractors to generate a comprehensive feature vector.

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

| Category | Feature | Description |
|----------|---------|-------------|
| **NAME** | `name_has_email` | Column name contains 'email' |
| **NAME** | `name_has_phone` | Column name contains 'phone' |
| **NAME** | `name_has_date` | Column name contains 'date' |
| **NAME** | `name_has_id` | Column name contains 'id' |
| **NAME** | `name_has_name` | Column name contains 'name' |
| **NAME** | `name_has_address` | Column name contains 'address' |
| **NAME** | `name_has_url` | Column name contains 'url' |
| **VALUE** | `value_has_at_symbol` | Ratio containing '@' symbol |
| **VALUE** | `value_has_dot` | Ratio containing '.' |
| **VALUE** | `value_digit_ratio` | Numeric character ratio |
| **VALUE** | `value_alpha_ratio` | Alphabetic ratio |
| **VALUE** | `value_has_dash` | Ratio containing '-' |
| **VALUE** | `value_has_slash` | Ratio containing '/' |
| **STATISTICAL** | `stat_avg_length` | Average length |
| **STATISTICAL** | `stat_std_length` | Length standard deviation |
| **STATISTICAL** | `stat_unique_ratio` | Unique value ratio |
| **STATISTICAL** | `stat_null_ratio` | Null ratio |
| **STATISTICAL** | `stat_min_length` | Minimum length |
| **STATISTICAL** | `stat_max_length` | Maximum length |

## MLTypeInferrer

The ML-based type inferrer. Supports multiple models (RuleBasedModel, NaiveBayesModel, EnsembleModel).

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

Three built-in models are supported, and custom models can be added by implementing the InferenceModel protocol.

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

## Full Table Type Inference

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

All extractors implement the `FeatureExtractor` protocol and provide the `extract(column: pl.Series, context: dict) -> list[Feature]` method.

### NameFeatureExtractor

Extracts features from column names.

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

Extracts features from column values.

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

Extracts statistical features.

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

## Inference Configuration

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

The default rule-based model. Infers types based on feature values.

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

## Next Steps

- [Threshold Tuning](threshold-tuning.md) - Optimize inference thresholds
- [Pattern Matching](patterns.md) - Pattern-based type detection
