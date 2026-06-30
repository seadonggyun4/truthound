# ML 이상치 Detection

실무 운영 가이드에서 Truthound, ML-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 테이블 of Contents

1. [개요](#overview)
2. [이상치 Detection](#anomaly-detection)
3. [드리프트 Detection](#drift-detection)
4. [Model 모니터링](#model-monitoring)
5. 실무 운영 가이드에서 Rule, Learning을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
6. [설정 레퍼런스](#configuration-reference)

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 개요

실무 운영 가이드에서 `truthound.ml`을(를) 다루는 항목입니다:

- 실무 운영 가이드에서 Anomaly, Detectors, Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Drift, Detectors, Distribution을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Model, Monitoring, Performance을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Rule, Learning, Automatic을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 `src/truthound/ml/`, Location을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```
ml/
├── __init__.py              # Public API exports
├── base.py                  # Core abstractions (1,183 lines)
├── anomaly_models/          # Anomaly detection algorithms
├── drift_detection/         # Drift detection methods
├── monitoring/              # ML model monitoring
└── rule_learning/           # Automatic rule generation
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 이상치 Detection

### Supported Algorithms

| 실무 운영 가이드에서 Detector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Category을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Algorithm을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|----------|-----------|----------|
| 실무 운영 가이드에서 `ZScoreAnomalyDetector`, ZScoreAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Z-score을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Normally을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `IQRAnomalyDetector`, IQRAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Interquartile을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Robust을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MADAnomalyDetector`, MADAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Median, Absolute, Deviation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Extremely을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `IsolationForestDetector`, IsolationForestDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Tree-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Isolation, Forest을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Non-linear을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `EnsembleAnomalyDetector`, EnsembleAnomalyDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Meta을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multiple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Best을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Basic Usage

```python
from truthound import ml
import polars as pl

# Load data
train_data = pl.scan_parquet("train.parquet")
test_data = pl.scan_parquet("test.parquet")

# Statistical anomaly detection
detector = ml.ZScoreAnomalyDetector(z_threshold=3.0)
detector.fit(train_data)
result = detector.predict(test_data)

print(f"Found {result.anomaly_count} anomalies ({result.anomaly_ratio:.1%})")
```

### Statistical Detectors

실무 운영 가이드에서 Z-Score, Detector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.anomaly_models.statistical import (
    StatisticalAnomalyDetector,
    StatisticalConfig,
)

config = StatisticalConfig(
    z_threshold=3.0,           # Standard deviations from mean
    iqr_multiplier=1.5,        # IQR multiplier for bounds
    use_robust_stats=False,    # Use median/MAD instead of mean/std
    per_column=True,           # Compute per-column statistics
    columns=None,              # Specific columns (None = all numeric)
)

detector = StatisticalAnomalyDetector(config)
detector.fit(train_data)

# Get computed statistics
stats = detector.get_statistics()
# {'column1': {'mean': 100.5, 'std': 15.2, ...}, ...}

# Score data (0-1 normalized)
scores = detector.score(test_data)
```

실무 운영 가이드에서 Z-Score, Computation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
```
z_score = |value - mean| / std
normalized_score = min(1.0, z_score / threshold)
```

실무 운영 가이드에서 IQR, Computation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
```
bounds = [Q1 - k*IQR, Q3 + k*IQR]  where k=1.5 (default)
normalized_score = min(1.0, distance_from_bounds / IQR)
```

실무 운영 가이드에서 MAD, Computation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
```
MAD = median(|Xi - median(X)|)
modified_z = 0.6745 * |value - median| / MAD
normalized_score = min(1.0, modified_z / threshold)
```

### Isolation Forest Detector

실무 운영 가이드에서 Pure, Python, Isolation, Forest을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.anomaly_models.isolation_forest import (
    IsolationForestDetector,
    IsolationForestConfig,
)

config = IsolationForestConfig(
    n_estimators=100,          # Number of trees
    max_samples=256,           # Samples per tree
    max_depth=None,            # Tree depth (auto if None)
    bootstrap=True,            # Use bootstrap sampling
    columns=None,              # Specific columns
)

detector = IsolationForestDetector(config)
detector.fit(train_data)

# Get anomaly scores (0-1 where 1 = most anomalous)
scores = detector.score(test_data)

# Get feature importance
importance = detector.get_feature_importance()
# {'column1': 0.25, 'column2': 0.35, ...}
```

실무 운영 가이드에서 Score, Computation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
```
score = 2^(-E(h(x)) / c(n))

Where:
- E(h(x)) = average path length to isolate sample
- c(n) = average path length for unsuccessful search
- c(n) = 2 * (ln(n-1) + 0.5772) - (2 * (n-1) / n)
```

### Ensemble Detector

실무 운영 가이드에서 Combine을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.anomaly_models.ensemble import (
    EnsembleAnomalyDetector,
    EnsembleConfig,
    EnsembleStrategy,
)

config = EnsembleConfig(
    strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
    weights=[0.3, 0.3, 0.4],   # Detector weights
    vote_threshold=0.5,         # For VOTE strategy
)

ensemble = EnsembleAnomalyDetector(config)

# Add detectors
ensemble.add_detector(ml.ZScoreAnomalyDetector(), weight=0.3)
ensemble.add_detector(ml.IQRAnomalyDetector(), weight=0.3)
ensemble.add_detector(ml.IsolationForestDetector(), weight=0.4)

ensemble.fit(train_data)
result = ensemble.predict(test_data)
```

실무 운영 가이드에서 Ensemble, Strategies을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Strategy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|-------------|
| 실무 운영 가이드에서 `AVERAGE`, AVERAGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Simple을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `WEIGHTED_AVERAGE`, WEIGHTED_AVERAGE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Weighted을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MAX`, MAX을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Conservative을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MIN`, MIN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Very을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `VOTE`, VOTE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Majority을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `UNANIMOUS`, UNANIMOUS을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### 결과 Types

```python
from truthound.ml.base import AnomalyResult, AnomalyScore, AnomalyType

# AnomalyResult contains detection results
result = detector.predict(test_data)

print(f"Anomaly count: {result.anomaly_count}")
print(f"Anomaly ratio: {result.anomaly_ratio:.1%}")
print(f"Total points: {result.total_points}")
print(f"Threshold used: {result.threshold_used}")
print(f"Detection time: {result.detection_time_ms:.2f}ms")

# Get only anomalous points
anomalies = result.get_anomalies()

# Iterate through scores
for score in result:
    if score.is_anomaly:
        print(f"Index {score.index}: score={score.score:.3f}, type={score.anomaly_type}")
        print(f"  Contributing features: {score.contributing_features}")
```

**이상치 Types:**

| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|------|-------------|
| 실무 운영 가이드에서 `POINT`, POINT을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Single point 이상치 |
| 실무 운영 가이드에서 `CONTEXTUAL`, CONTEXTUAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Anomaly을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `COLLECTIVE`, COLLECTIVE을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Group을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `PATTERN`, PATTERN을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Pattern을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `TREND`, TREND을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Trend을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `SEASONAL`, SEASONAL을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Seasonal을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 드리프트 Detection

### Quick 드리프트 Detection with `truthound.drift.compare()`

실무 운영 가이드에서 `compare()`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
실무 운영 가이드에서 `truthound.drift`을(를) 다루는 항목입니다:

```python
from truthound.drift import compare

# Auto-select best method based on column type
drift = compare("baseline.csv", "current.csv", method="auto")

# Statistical tests (numeric columns)
drift = compare("baseline.csv", "current.csv", method="ks")          # Kolmogorov-Smirnov
drift = compare("baseline.csv", "current.csv", method="psi")         # Population Stability Index
drift = compare("baseline.csv", "current.csv", method="cvm")         # Cramér-von Mises
drift = compare("baseline.csv", "current.csv", method="anderson")    # Anderson-Darling

# Divergence metrics
drift = compare("baseline.csv", "current.csv", method="kl")          # Kullback-Leibler divergence
drift = compare("baseline.csv", "current.csv", method="js")          # Jensen-Shannon divergence

# Distance metrics (any column type)
drift = compare("baseline.csv", "current.csv", method="wasserstein")   # Earth Mover's Distance
drift = compare("baseline.csv", "current.csv", method="hellinger")     # Hellinger distance (bounded)
drift = compare("baseline.csv", "current.csv", method="bhattacharyya") # Bhattacharyya distance
drift = compare("baseline.csv", "current.csv", method="tv")            # Total Variation distance

# Distance metrics (numeric only)
drift = compare("baseline.csv", "current.csv", method="energy")      # Energy distance
drift = compare("baseline.csv", "current.csv", method="mmd")         # Maximum Mean Discrepancy

# For categorical columns
drift = compare("baseline.csv", "current.csv", method="chi2")        # Chi-squared

# Check results
if drift.has_drift:
    for col in drift.columns:
        if col.result.drifted:
            print(f"{col.column}: {col.result.method} = {col.result.statistic:.4f}")
```

실무 운영 가이드에서 API, `compare()`, `truthound.drift`, See, Python을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### ML-Based 드리프트 Detection

실무 운영 가이드에서 ML-based을(를) 다루는 항목입니다:

| 실무 운영 가이드에서 Detector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Methods을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Case을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|----------|------|---------|----------|
| 실무 운영 가이드에서 `DistributionDriftDetector`, DistributionDriftDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Univariate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PSI, Jensen-Shannon, Wasserstein을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Input을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `FeatureDriftDetector`, FeatureDriftDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Feature-level을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Per-feature을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Individual feature 모니터링 |
| 실무 운영 가이드에서 `ConceptDriftDetector`, ConceptDriftDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Concept을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 DDM, ADWIN, Page-Hinkley을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Model을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `MultivariateDriftDetector`, MultivariateDriftDetector을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Multivariate을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 PCA, Correlation, Mahalanobis을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Joint을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Distribution 드리프트 Detection

실무 운영 가이드에서 Detects을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.drift_detection.distribution import (
    DistributionDriftDetector,
    DistributionDriftConfig,
)

config = DistributionDriftConfig(
    method="psi",              # psi, ks, jensen_shannon, wasserstein
    n_bins=10,                 # Histogram bins
    min_samples=30,            # Minimum required samples
)

detector = DistributionDriftDetector(config)
detector.fit(reference_data)

result = detector.detect(reference_data, current_data)

print(f"Drift detected: {result.is_drifted}")
print(f"Drift score: {result.drift_score:.3f}")
print(f"Drift type: {result.drift_type}")  # gradual, sudden, none

# Per-column scores
for col, score in result.column_scores:
    if score > 0.1:
        print(f"  {col}: {score:.3f}")

# Get drifted columns
drifted = result.get_drifted_columns(threshold=0.1)
```

실무 운영 가이드에서 Distribution, Methods을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Interpretation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|----------------|
| 실무 운영 가이드에서 `psi`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Population, Stability, Index을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | <0.1 stable, 0.1-0.25 small 드리프트, >0.25 significant |
| 실무 운영 가이드에서 `ks`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Kolmogorov-Smirnov을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `jensen_shannon`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Symmetric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `wasserstein`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Earth-Mover을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Minimum을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `kl`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Kullback-Leibler을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Information을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `cvm`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Mises을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 More을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `anderson`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Anderson-Darling을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Most을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `hellinger`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Hellinger을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Bounded을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `bhattacharyya`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Bhattacharyya을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Classification을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `tv`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Total, Variation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `energy`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Energy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Location/scale을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `mmd`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Maximum, Mean, Discrepancy을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 High-dimensional을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

> 실무 운영 가이드에서 `compare()`, `truthound.drift`, Note을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
> 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Feature 드리프트 Detection

실무 운영 가이드에서 Monitors을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.drift_detection.feature import (
    FeatureDriftDetector,
    FeatureDriftConfig,
)

config = FeatureDriftConfig(
    track_stats=["mean", "std", "min", "max", "null_ratio"],
    relative_threshold=True,   # Use relative vs absolute thresholds
    alert_on_new_values=True,  # Flag new categorical values
    categorical_threshold=0.2,
)

detector = FeatureDriftDetector(config)
detector.fit(reference_data)

result = detector.detect(reference_data, current_data)

# Access feature-level drift scores
for score in result.feature_scores:
    print(f"{score.feature}: {score.drift_score:.3f}")
    print(f"  Reference: {score.reference_stats}")
    print(f"  Current: {score.current_stats}")
```

실무 운영 가이드에서 Tracked, Statistics을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

- 실무 운영 가이드에서 Numeric을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Categorical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

### Concept 드리프트 Detection

실무 운영 가이드에서 Detects을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.drift_detection.concept import (
    ConceptDriftDetector,
    ConceptDriftConfig,
)

config = ConceptDriftConfig(
    target_column="label",
    method="ddm",              # ddm, adwin, page_hinkley
    warning_threshold=2.0,     # Std devs for warning
    drift_threshold=3.0,       # Std devs for drift
    min_window=30,             # Minimum samples before check
    feature_columns=None,      # Specific features
)

detector = ConceptDriftDetector(config)
detector.fit(reference_data)

result = detector.detect(reference_data, current_data)
```

**Concept 드리프트 Methods:**

| 실무 운영 가이드에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 실무 운영 가이드에서 `ddm`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Drift, Detection, Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `adwin`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Adaptive, Windowing을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `page_hinkley`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Cumulative을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

### Multivariate 드리프트 Detection

실무 운영 가이드에서 Detects을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.drift_detection.multivariate import (
    MultivariateDriftDetector,
    MultivariateDriftConfig,
)

config = MultivariateDriftConfig(
    method="correlation",      # pca, correlation, mahalanobis
    n_components=None,         # For PCA
    correlation_threshold=0.3,
    columns=None,              # Specific columns
)

detector = MultivariateDriftDetector(config)
detector.fit(reference_data)

result = detector.detect(reference_data, current_data)
```

실무 운영 가이드에서 Multivariate, Methods을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Method을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|--------|-------------|
| 실무 운영 가이드에서 `correlation`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Frobenius을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `pca`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Changes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 `mahalanobis`을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Covariance-aware을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Model 모니터링

### ModelMonitor

실무 운영 가이드에서 Unified을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.monitoring import (
    ModelMonitor,
    MonitorConfig,
)

config = MonitorConfig(
    batch_size=100,
    collect_interval_seconds=60,
    alert_evaluation_interval_seconds=30,
    retention_hours=24,
    enable_drift_detection=True,
    enable_quality_metrics=True,
)

monitor = ModelMonitor(config)

# Register models
monitor.register_model("fraud-detector", config)

# Start monitoring (background tasks)
await monitor.start()

# Record predictions
await monitor.record_prediction(
    model_id="fraud-detector",
    features={"amount": 150.0, "merchant_type": "online"},
    prediction=0.85,
    actual=1,                  # Optional: for quality metrics
    latency_ms=5.2,
)

# Get dashboard data (includes metrics history)
dashboard = await monitor.get_dashboard_data("fraud-detector")

# Access historical metrics from dashboard
metrics_history = dashboard.metrics_history
active_alerts = dashboard.active_alerts
health_score = dashboard.health_score

# Stop monitoring
await monitor.stop()
```

### Metric Collectors

```python
from truthound.ml.monitoring.collectors import (
    PerformanceCollector,
    DriftCollector,
    QualityCollector,
    CompositeCollector,
)

# Performance metrics (latency, throughput, error rate)
perf = PerformanceCollector(
    window_size=1000,
    percentiles=[0.5, 0.95, 0.99],
    throughput_window_seconds=60,
)

# Drift metrics
drift = DriftCollector()

# Quality metrics (accuracy, precision, recall, F1)
quality = QualityCollector()

# Combine collectors
composite = CompositeCollector([perf, drift, quality])
```

### 알림 Rules

```python
from truthound.ml.monitoring.alerting import (
    ThresholdRule,
    AnomalyRule,
    TrendRule,
    AlertSeverity,
)

# Threshold-based alerting
latency_rule = ThresholdRule(
    name="high-latency",
    metric_name="latency_ms",
    threshold=100.0,
    comparison="gt",
    severity=AlertSeverity.WARNING,
    for_duration_seconds=60,
)

# Anomaly-based alerting
anomaly_rule = AnomalyRule(
    name="latency-anomaly",
    metric_name="latency_ms",
    window_size=100,
    std_threshold=3.0,
    severity=AlertSeverity.ERROR,
)

# Trend-based alerting
trend_rule = TrendRule(
    name="degrading-accuracy",
    metric_name="accuracy",
    direction="decreasing",
    slope_threshold=0.01,
    lookback_minutes=60,
    severity=AlertSeverity.CRITICAL,
)

# Add rules to monitor
monitor.add_rule(latency_rule)
monitor.add_rule(anomaly_rule)
monitor.add_rule(trend_rule)
```

### 알림 Handlers

```python
from truthound.ml.monitoring.alerting import (
    SlackAlertHandler,
    PagerDutyAlertHandler,
    WebhookAlertHandler,
)

# Slack notifications
slack = SlackAlertHandler(
    webhook_url="https://hooks.slack.com/...",
    channel="#ml-alerts",
    mention_users=["@oncall"],
)

# PagerDuty incidents
pagerduty = PagerDutyAlertHandler(
    routing_key="...",
    severity_mapping={
        AlertSeverity.CRITICAL: "critical",
        AlertSeverity.ERROR: "error",
    },
)

# Custom webhook
webhook = WebhookAlertHandler(url="https://my-service/alerts")
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Rule Learning

### DataProfileRuleLearner

실무 운영 가이드에서 Automatically을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

```python
from truthound.ml.rule_learning import (
    DataProfileRuleLearner,
    ProfileLearnerConfig,
)

config = ProfileLearnerConfig(
    strictness="medium",           # loose, medium, strict
    include_range_rules=True,
    include_uniqueness_rules=True,
    include_format_rules=True,
    include_null_rules=True,
    include_type_rules=True,
    null_threshold=0.01,           # Gen rule if < 1% nulls
    uniqueness_threshold=0.99,     # Gen rule if > 99% unique
)

learner = DataProfileRuleLearner(config)
result = learner.learn_rules(data)

print(f"Learned {result.total_rules} rules in {result.learning_time_ms:.0f}ms")

for rule in result.rules:
    print(f"{rule.name}: {rule.condition}")
    print(f"  Support: {rule.support:.1%}, Confidence: {rule.confidence:.1%}")
    print(f"  Validator: {rule.validator_config}")

# Convert to validation suite
suite_config = result.to_validation_suite()
```

실무 운영 가이드에서 Generated, Rule, Types을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

| 실무 운영 가이드에서 Rule, Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Description을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|-------------|
| 실무 운영 가이드에서 Range을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Min/max을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Null을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Completeness을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Uniqueness을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Primary을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Type을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Data type 검증 |
| 실무 운영 가이드에서 Format을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Regex을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 설정 레퍼런스

### AnomalyConfig

```python
from truthound.ml.base import AnomalyConfig

config = AnomalyConfig(
    # Core settings
    contamination=0.1,         # Expected outlier ratio
    sensitivity=0.5,           # Detection sensitivity
    min_samples=100,           # Minimum training samples
    window_size=None,          # Temporal window size
    score_threshold=None,      # Override auto threshold
    columns=None,              # Specific columns

    # Common ML settings
    sample_size=None,          # Max training samples
    random_seed=42,            # Reproducibility
    n_jobs=1,                  # Parallel jobs
    cache_predictions=True,    # Cache results
    verbose=0,                 # 0=silent, 1=progress, 2=debug
)
```

### DriftConfig

```python
from truthound.ml.base import DriftConfig

config = DriftConfig(
    reference_window=1000,     # Reference data size
    detection_window=100,      # Detection data size
    threshold=0.05,            # Drift threshold
    min_samples_per_window=30, # Minimum samples
    n_bins=10,                 # Histogram bins
    detect_gradual=True,       # Detect gradual drift
    detect_sudden=True,        # Detect sudden drift
)
```

### RuleLearningConfig

```python
from truthound.ml.base import RuleLearningConfig

config = RuleLearningConfig(
    min_support=0.1,           # Rule support
    min_confidence=0.8,        # Rule confidence
    max_rules=100,             # Max rules
    max_antecedent_length=3,   # Max conditions
    include_negations=False,   # Include negations
)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Model Registry

```python
from truthound.ml.base import model_registry, register_model

# List available models
print(model_registry.list_all())

# Get model by name
detector_class = model_registry.get("IsolationForestDetector")

# Register custom model
@register_model("my_detector")
class MyDetector(AnomalyDetector):
    ...
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Model Persistence

```python
# Save trained model
detector.save("my_detector.json")

# Load model
loaded = IsolationForestDetector.load("my_detector.json")

# Use loaded model
result = loaded.predict(new_data)
```

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 성능 Characteristics

| 실무 운영 가이드에서 Operation을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Complexity을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Notes을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
|-----------|------------|-------|
| 실무 운영 가이드에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Single을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Statistical을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Isolation, Forest을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| 실무 운영 가이드에서 Isolation, Forest을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Distribution 드리프트 | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | 실무 운영 가이드에서 Both을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. |
| Feature 드리프트 | 실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다. | Per-컬럼 |

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## Thread Safety

- 실무 운영 가이드에서 `threading.RLock()`, RLock을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 `UNTRAINED → TRAINING → TRAINED → READY`, Model, UNTRAINED, TRAINING, TRAINED, READY을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- 실무 운영 가이드에서 Global을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

실무 운영 가이드에서 관련 설정과 실행 흐름을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.

## 함께 보기

- 실무 운영 가이드에서 Drift, Detection, Validator-based을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
- [Model 모니터링](../ci-cd/index.md) - CI/CD 통합
- 실무 운영 가이드에서 Performance, Tuning, Optimization을(를) 기준으로 데이터 품질 검증, 워크플로우 자동화, 결과 해석 방법을 설명합니다.
